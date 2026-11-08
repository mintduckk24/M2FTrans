"""Production Inference Pipeline for M2FTrans on MyoPS380 Test Volumes.

Strictly conforms to Section 7 of Promt_train_test_no_val.txt:
- Loads completed checkpoint: m2ftrans_seed<seed>.pt
- Evaluates exactly 76 test cases from test_vol.txt
- Model in eval() mode, single-pass forward + argmax slice-by-slice
- Assembles predictions into 3D volume (128, 128, D) with canonical labels {0, 1, 2, 3}
- Zero morphological postprocessing (no connected components filtering, no LCC)
- Exports test_predictions/<CaseID>_pred.npz with key 'prediction' as uint8 array
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.myops380_dataset import MyoPS380TestVolumeDataset
from models.m2ftrans_2d import M2FTrans2D
from utils.paths import resolve_data_root, resolve_split_dir, resolve_save_dir


def resolve_amp_dtype(mode: str = "auto", device: torch.device | None = None) -> torch.dtype | None:
    """Resolves automatic mixed precision dtype matching SCAR."""
    if device is None or device.type != "cuda" or mode == "none":
        return None
    if mode == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if mode in ("auto", "fp16"):
        return torch.bfloat16 if (mode == "auto" and torch.cuda.is_bf16_supported()) else torch.float16
    return None


@torch.inference_mode()
def single_pass_volume_inference(
    model: nn.Module,
    c0_vol: np.ndarray,
    lge_vol: np.ndarray,
    t2_vol: np.ndarray,
    device: torch.device,
    batch_size: int = 8,
    img_size: int = 128,
    amp_dtype: torch.dtype | None = None,
) -> np.ndarray:
    """Runs slice-by-slice batched single-pass forward + argmax on 3D volume (H, W, D).

    Strictly conforms to Section 7 of Promt_train_test_no_val.txt:
    - Default batch size 8 matching SCAR reference.
    - If input spatial resolution differs from (img_size, img_size),
      bilinearly interpolate images to (img_size, img_size), and bilinearly
      interpolate output logits back to (H, W) before argmax.
    - Strictly avoids morphological post-processing.
    - Returns uint8 array (H, W, D) with canonical classes in {0, 1, 2, 3}.
    """
    model.eval()
    H, W, D = c0_vol.shape
    prediction = np.empty((H, W, D), dtype=np.uint8)

    for start in range(0, D, batch_size):
        stop = min(start + batch_size, D)

        slice_c0 = np.ascontiguousarray(c0_vol[:, :, start:stop].transpose(2, 0, 1))
        slice_lge = np.ascontiguousarray(lge_vol[:, :, start:stop].transpose(2, 0, 1))
        slice_t2 = np.ascontiguousarray(t2_vol[:, :, start:stop].transpose(2, 0, 1))

        t_c0 = torch.from_numpy(slice_c0).unsqueeze(1).float().to(device)
        t_lge = torch.from_numpy(slice_lge).unsqueeze(1).float().to(device)
        t_t2 = torch.from_numpy(slice_t2).unsqueeze(1).float().to(device)

        if (H, W) != (img_size, img_size):
            t_c0 = F.interpolate(t_c0, (img_size, img_size), mode="bilinear", align_corners=False)
            t_lge = F.interpolate(t_lge, (img_size, img_size), mode="bilinear", align_corners=False)
            t_t2 = F.interpolate(t_t2, (img_size, img_size), mode="bilinear", align_corners=False)

        batch_input = torch.cat([t_c0, t_lge, t_t2], dim=1)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None and device.type == "cuda")):
            model_out = model(batch_input)
            if isinstance(model_out, (tuple, list)):
                logits = model_out[0]
            else:
                logits = model_out

        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"Non-finite inference outputs at slice range [{start}:{stop}]")

        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits.float(), (H, W), mode="bilinear", align_corners=False)

        pred_batch = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        prediction[:, :, start:stop] = pred_batch.transpose(1, 2, 0)

    return prediction


def run_prediction(
    checkpoint_path: str | Path,
    data_dir: str | Path | None = None,
    split_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    device_name: str | None = None,
    allow_incomplete: bool = False,
    batch_size: int = 8,
    amp: str = "auto",
) -> Path:
    """Generates canonical test predictions for all 76 cases in test_vol.txt."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    resolved_data = resolve_data_root(data_dir)
    resolved_split = resolve_split_dir(split_dir)

    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        out_dir = ckpt_path.parent.parent / "test_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[M2FTrans Predict] Execution device: {device}")

    amp_dtype = resolve_amp_dtype(amp, device)

    # 1. Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not allow_incomplete and not ckpt.get("training_complete", False):
        completed = ckpt.get("completed_epochs", ckpt.get("epoch", 0))
        raise ValueError(
            f"Checkpoint is incomplete ({completed} completed epochs). "
            "Evaluation strictly requires completed 300 epochs under protocol myops380_train304_test76_last_v1. "
            "Use --allow_incomplete to override for dry-run testing."
        )

    # 2. Build model and load weights
    basic_dims = ckpt.get("basic_dims")
    if basic_dims is None and "model_state_dict" in ckpt:
        basic_dims = ckpt["model_state_dict"]["c0_encoder.e1_c1.conv.weight"].shape[0]
    if basic_dims is None:
        basic_dims = 16
    model = M2FTrans2D(
        in_channels=1,
        num_classes=4,
        img_size=128,
        basic_dims=basic_dims,
        num_modals=3,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 3. Load 76 test cases
    test_dataset = MyoPS380TestVolumeDataset(
        data_dir=resolved_data,
        split_dir=resolved_split,
        split_file="test_vol.txt",
    )
    print(f"[M2FTrans Predict] Running inference on {len(test_dataset)} test volumes (batch_size={batch_size})...")

    t0 = time.perf_counter()
    for idx in range(len(test_dataset)):
        item = test_dataset[idx]
        case_id = item["case_id"]

        pred_3d = single_pass_volume_inference(
            model=model,
            c0_vol=item["c0"],
            lge_vol=item["lge"],
            t2_vol=item["t2"],
            device=device,
            batch_size=batch_size,
            img_size=128,
            amp_dtype=amp_dtype,
        )

        # Validate prediction array
        if pred_3d.shape[:2] != (128, 128) or pred_3d.ndim != 3:
            raise ValueError(f"Invalid prediction shape {pred_3d.shape} for case {case_id}")
        if not np.isin(pred_3d, [0, 1, 2, 3]).all():
            raise ValueError(f"Non-canonical classes in prediction for case {case_id}")

        out_path = out_dir / f"{case_id}_pred.npz"
        np.savez_compressed(
            out_path,
            prediction=pred_3d,
            case_id=case_id,
            exported_label_order="canonical",
            model_id="m2ftrans",
        )

        if (idx + 1) % 10 == 0 or (idx + 1) == len(test_dataset):
            print(f"  [{idx+1:02d}/{len(test_dataset):02d}] Saved {out_path.name}")

    total_time = time.perf_counter() - t0
    print(f"[M2FTrans Predict] Finished prediction for {len(test_dataset)} cases in {total_time:.2f}s.")
    return out_dir


def parse_args():
    parser = argparse.ArgumentParser(description="M2FTrans MyoPS380 Test Predictor")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to completed checkpoint (.pt)")
    parser.add_argument("--data_dir", type=str, default=None, help="Root path to MyoPS380 Processed_data")
    parser.add_argument("--split_dir", type=str, default=None, help="Path to splits/myops380_no_val")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save prediction .npz files")
    parser.add_argument("--device", type=str, default=None, help="Execution device ('cuda:0' or 'cpu')")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for slice inference (default: 8)")
    parser.add_argument("--amp", type=str, default="auto", choices=("auto", "none", "fp16", "bf16"), help="AMP mode")
    parser.add_argument("--allow_incomplete", action="store_true", help="Allow inference on incomplete checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_prediction(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        device_name=args.device,
        allow_incomplete=args.allow_incomplete,
        batch_size=args.batch_size,
        amp=args.amp,
    )

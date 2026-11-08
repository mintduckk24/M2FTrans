"""Production Inference & Shared Evaluation for M2FTrans on MyoPS380 Benchmark.

Strictly adheres to:
- Benchmark Protocol: myops380_train304_test76_last_v1
- Metric Protocol: myops380_voxel_v3
- Shared Evaluator: Reuses SCAR surface_distance.py (benchmark_rows, summarize_rows)
- Requirements:
  - Loads atomic m2ftrans_seed<seed>.pt at epoch 300 (rejects incomplete checkpoints unless overridden)
  - Single-pass forward + argmax on all 76 test patients from test_vol.txt
  - Zero morphological postprocessing (no LCC, no small connected components filtering)
  - Exports test_predictions/{CaseID}_pred.npz (uint8, 128x128xD, key 'prediction')
  - Produces metrics.json and per_case.csv with complete protocol metadata
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from dataset.myops380_dataset import MyoPS380TestVolumeDataset, extract_patient_id
from models.m2ftrans_2d import M2FTrans2D
from predict import single_pass_volume_inference, resolve_amp_dtype
from utils.paths import resolve_data_root, resolve_split_dir, resolve_save_dir


def load_surface_distance_module(scar_root: str | Path | None = None):
    """Loads SCAR's surface_distance.py or bundled local copy."""
    candidates = []
    if scar_root is not None:
        candidates.append(Path(scar_root) / "training" / "metrics" / "surface_distance.py")
    candidates.extend([
        Path(__file__).resolve().parent / "utils" / "surface_distance.py",
        Path(r"D:\nckh_model\SCAR\training\metrics\surface_distance.py"),
        Path(r"../SCAR/training/metrics/surface_distance.py"),
    ])

    for p in candidates:
        if p.is_file():
            spec = importlib.util.spec_from_file_location("scar_surface_distance", str(p.resolve()))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod

    raise FileNotFoundError(
        f"Cannot locate surface_distance.py in candidates: {[str(c) for c in candidates]}"
    )


def compute_file_sha256(path: str | Path) -> str:
    """Computes SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def evaluate(
    checkpoint: str | Path | None = None,
    predictions_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    split_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    scar_root: str | Path | None = None,
    device_name: str | None = None,
    allow_incomplete: bool = False,
    batch_size: int = 8,
    amp: str = "auto",
) -> dict[str, Any]:
    """Runs prediction and shared evaluation for M2FTrans on the 76 MyoPS380 test patients."""
    resolved_data = resolve_data_root(data_dir)
    resolved_split = resolve_split_dir(split_dir)
    surface_dist_mod = load_surface_distance_module(scar_root)

    test_manifest = resolved_split / "test_vol.txt"
    if not test_manifest.is_file():
        raise FileNotFoundError(f"Missing test manifest: {test_manifest}")

    test_dataset = MyoPS380TestVolumeDataset(
        data_dir=resolved_data,
        split_dir=resolved_split,
        split_file="test_vol.txt",
    )
    expected_cases = len(test_dataset)
    if expected_cases != 76:
        print(f"[WARNING] Expected 76 test cases, but test_vol.txt has {expected_cases}")

    # Determine execution device
    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    amp_dtype = resolve_amp_dtype(amp, device)

    ckpt_metadata = {}
    ckpt_path = None
    if checkpoint is not None:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if not allow_incomplete and not ckpt.get("training_complete", False):
            completed = ckpt.get("completed_epochs", ckpt.get("epoch", 0))
            raise ValueError(
                f"Checkpoint {ckpt_path.name} is incomplete ({completed}/300 epochs). "
                "Official evaluation strictly requires 300 completed epochs under protocol "
                "'myops380_train304_test76_last_v1'. Use --allow_incomplete for testing."
            )

        ckpt_metadata = {
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": compute_file_sha256(ckpt_path),
            "completed_epochs": ckpt.get("completed_epochs", 300),
            "planned_epochs": ckpt.get("planned_epochs", 300),
            "model_id": ckpt.get("model_id", "m2ftrans"),
            "seed": ckpt.get("seed", 1234),
        }

        # Determine output directory
        if output_dir is not None:
            resolved_out = Path(output_dir)
        else:
            resolved_out = ckpt_path.parent
        resolved_pred_dir = resolved_out / "test_predictions"
        resolved_pred_dir.mkdir(parents=True, exist_ok=True)

        # Build model and load weights
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

        print(f"[M2FTrans Evaluator] Loaded checkpoint: {ckpt_path.name}")
        print(f"[M2FTrans Evaluator] Running single-pass inference on {expected_cases} test patients...")
    else:
        # Precomputed predictions directory
        if predictions_dir is None:
            raise ValueError("Either --checkpoint or --predictions_dir must be supplied.")
        resolved_pred_dir = Path(predictions_dir)
        resolved_out = resolved_pred_dir.parent if output_dir is None else Path(output_dir)
        model = None
        print(f"[M2FTrans Evaluator] Using precomputed predictions from: {resolved_pred_dir}")

    all_rows = []
    cases_processed = 0
    t0 = time.time()

    for idx in range(expected_cases):
        item = test_dataset[idx]
        case_id = item["case_id"]
        gt = item["label"]  # (128, 128, D) canonical classes 0..3

        pred_file = resolved_pred_dir / f"{case_id}_pred.npz"

        if model is not None:
            # Single-pass forward + argmax
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
            # Save prediction
            np.savez_compressed(pred_file, prediction=pred_3d, case_id=case_id)
        else:
            if not pred_file.is_file():
                raise FileNotFoundError(f"Missing prediction file: {pred_file}")
            with np.load(pred_file) as data:
                pred_3d = np.asarray(data["prediction"], dtype=np.uint8)

        # Validate format
        if pred_3d.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {case_id}: pred {pred_3d.shape} != gt {gt.shape}")
        if not np.isin(pred_3d, [0, 1, 2, 3]).all():
            raise ValueError(f"Non-canonical classes in prediction for case {case_id}")

        # Compute metric rows using SCAR's surface_distance dataset_rows
        case_rows = surface_dist_mod.dataset_rows(
            pred_3d, gt, case=case_id, compute_distance=True, dataset_id="myops380"
        )
        all_rows.extend(case_rows)
        cases_processed += 1

        if cases_processed % 10 == 0 or cases_processed == expected_cases:
            print(f"  Evaluated [{cases_processed:02d}/{expected_cases:02d}] cases...")

    eval_time = time.time() - t0
    print(f"[M2FTrans Evaluator] Evaluation completed in {eval_time:.2f}s")

    # Summarize rows using SCAR's summarize_dataset_rows
    summary = surface_dist_mod.summarize_dataset_rows(all_rows, dataset_id="myops380")

    # Write per_case.csv
    csv_path = resolved_out / "per_case.csv"
    if all_rows:
        fieldnames = list(all_rows[0].keys())
    else:
        fieldnames = [
            "case", "region", "dice", "iou", "hd95", "asd", "status",
            "precision", "recall", "prediction_voxels", "target_voxels",
            "hd95_unit", "hd95_voxel", "asd_voxel",
        ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)
    print(f"[M2FTrans Evaluator] Saved per-case results to: {csv_path}")

    # Write metrics.json with full protocol provenance
    json_path = resolved_out / "metrics.json"
    metrics_report = {
        "benchmark_protocol": "myops380_train304_test76_last_v1",
        "metric_protocol": "myops380_voxel_v3",
        "checkpoint_selection": "last_completed_epoch",
        "checkpoint_selection_metric": None,
        "split_manifest_sha256": compute_file_sha256(test_manifest),
        "cases_evaluated": cases_processed,
        "loss_objective": "M2FTrans_native_composite_WCE_SoftDice_PRM_Sep",
        "exported_label_order": "canonical",
        "regions": {
            "normal_myocardium": [1],
            "edema": [2],
            "scar": [3],
            "edema_inclusive": [2, 3],
            "myocardial_ring": [1, 2, 3],
        },
        **ckpt_metadata,
        "summary": summary,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)
    print(f"[M2FTrans Evaluator] Saved summary report to: {json_path}")

    # Print summary highlights
    print("\n================== BENCHMARK EVALUATION SUMMARY ==================")
    for region_name in ("scar", "edema", "edema_inclusive", "normal_myocardium", "myocardial_ring"):
        if region_name in summary:
            reg = summary[region_name]
            dice_val = f"{reg['mean_dice']:.4f}" if reg.get("mean_dice") is not None else "N/A"
            hd95_val = f"{reg['mean_hd95_voxel']:.2f}" if reg.get("mean_hd95_voxel") is not None else "N/A"
            asd_val = f"{reg['mean_asd_voxel']:.2f}" if reg.get("mean_asd_voxel") is not None else "N/A"
            print(f"  {region_name:<20}: Dice={dice_val:<8} | HD95={hd95_val:<8} | ASD={asd_val:<8}")

    avg_path = summary.get("avg_pathology_dice")
    avg_path_inc = summary.get("avg_pathology_inclusive_dice")
    print(f"  avg_pathology_dice          : {f'{avg_path:.4f}' if avg_path is not None else 'N/A'}")
    print(f"  avg_pathology_inclusive_dice: {f'{avg_path_inc:.4f}' if avg_path_inc is not None else 'N/A'}")
    print("==================================================================\n")

    return metrics_report


def parse_args():
    parser = argparse.ArgumentParser(description="M2FTrans MyoPS380 Shared Evaluator")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file (.pt)")
    parser.add_argument("--predictions_dir", type=str, default=None, help="Directory containing .npz prediction files")
    parser.add_argument("--data_dir", type=str, default=None, help="Root path to MyoPS380 Processed_data")
    parser.add_argument("--split_dir", type=str, default=None, help="Path to splits/myops380_no_val")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for reports")
    parser.add_argument("--scar_root", type=str, default=None, help="Path to SCAR repository root")
    parser.add_argument("--device", type=str, default=None, help="Execution device ('cuda:0' or 'cpu')")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for slice inference (default: 8)")
    parser.add_argument("--amp", type=str, default="auto", choices=("auto", "none", "fp16", "bf16"), help="AMP mode")
    parser.add_argument("--allow_incomplete", action="store_true", help="Allow evaluation on incomplete checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        checkpoint=args.checkpoint,
        predictions_dir=args.predictions_dir,
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        scar_root=args.scar_root,
        device_name=args.device,
        allow_incomplete=args.allow_incomplete,
        batch_size=args.batch_size,
        amp=args.amp,
    )

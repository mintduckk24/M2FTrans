"""Production Training Engine for M2FTrans on MyoPS380 Benchmark.

Strictly adheres to:
- Benchmark Protocol: myops380_train304_test76_last_v1
- Model Identifier: m2ftrans
- Checkpoint: <save_dir>/m2ftrans_seed<seed>.pt
- Requirements:
  1. 304 train patients (1,972 slices) from train.txt, zero validation (no val split, no val loader)
  2. Data augmentation: SCAR RandomGenerator on (bSSFP, LGE, T2w) aligned slices in [0.0, 1.0]
  3. Fixed 300 epochs budget (max_epochs = 300), no early stopping, no best.pt
  4. Target effective batch size 16 with sample-weighted accumulation and drop_last=False
  5. AdamW optimizer (weight_decay=0.0001, lr=0.0003333), step-level Polynomial LR Decay (power=0.9)
  6. Native M2FTrans composite loss: L_fuse + L_sep + L_prm
  7. Deterministic execution (seed=1234), atomic checkpoint saving to m2ftrans_seed<seed>.pt
  8. Strict test set isolation (no test evaluation during training)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from dataset.myops380_dataset import MyoPS380TrainDataset
from models.m2ftrans_2d import M2FTrans2D
from utils.criterions import M2FTransCompositeLoss
from utils.lr_scheduler import build_optimizer, build_scheduler, OPTIMIZATION_PROTOCOL
from utils.paths import resolve_data_root, resolve_split_dir, resolve_save_dir


def resolve_amp(mode: str, device: torch.device) -> torch.dtype | None:
    """Resolves AMP precision policy matching SCAR reference."""
    if mode == "none" or (mode == "auto" and device.type == "cpu"):
        return None
    if device.type != "cuda":
        raise ValueError("Mixed precision is enabled only on CUDA; use --amp none on CPU.")
    if mode not in {"auto", "none", "fp16", "bf16"}:
        raise ValueError(f"Unknown AMP mode: {mode}")
    if mode == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if mode == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("This CUDA device does not support bfloat16.")
        return torch.bfloat16
    return torch.float16


def set_seed(seed: int = 1234, deterministic: bool = True) -> None:
    """Sets random seeds across Python, NumPy, PyTorch, and CUDA for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def seed_worker(worker_id: int) -> None:
    """Seeds DataLoader worker processes using PyTorch initial seed modulo 2**32."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def compute_manifest_hash(path: Path) -> str:
    """Computes SHA256 of manifest file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def count_parameters(model: nn.Module) -> int:
    """Calculates total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    criterion: M2FTransCompositeLoss,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    accum_steps: int = 1,
    epoch: int = 0,
    global_step: int = 0,
    successful_updates: int = 0,
    log_interval: int = 20,
    max_batches: int | None = None,
) -> tuple[dict[str, float], int, int, int]:
    """Executes one training epoch with sample-weighted gradient accumulation and AMP."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    epoch_loss = 0.0
    epoch_fuse = 0.0
    epoch_sep = 0.0
    epoch_prm = 0.0
    total_samples = 0

    group_samples = 0
    skipped_steps = 0
    accumulated_loss = torch.tensor(0.0, device=device)

    use_amp = (amp_dtype is not None and device.type == "cuda")
    t0 = time.perf_counter()
    num_batches = len(loader) if max_batches is None else min(len(loader), max_batches)

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        # Ingest inputs
        image = batch["image"].to(device, non_blocking=True)  # (B, 3, 128, 128)
        target = batch["label"].to(device, non_blocking=True)  # (B, 128, 128)
        batch_size = image.size(0)

        # Forward pass in training mode with optional AMP
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            fuse_pred, sep_preds, prm_preds = model(image)
            loss, loss_metrics = criterion(fuse_pred, sep_preds, prm_preds, target)

        # Sample-weighted accumulation: backward loss * batch_size
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss * batch_size).backward()
        else:
            (loss * batch_size).backward()

        group_samples += batch_size
        accumulated_loss = accumulated_loss + loss.detach() * batch_size
        total_samples += batch_size
        epoch_loss += float(loss.detach().item()) * batch_size
        epoch_fuse += loss_metrics["fuse_loss"] * batch_size
        epoch_sep += loss_metrics["sep_loss"] * batch_size
        epoch_prm += loss_metrics["prm_loss"] * batch_size

        is_accum_boundary = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == num_batches)

        if is_accum_boundary:
            # Unscale before dividing by actual group sample count
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)

            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(group_samples)

            if scaler is not None and scaler.is_enabled():
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < old_scale
            else:
                optimizer.step()
                skipped = False

            if skipped:
                skipped_steps += 1
            else:
                scheduler.step()
                global_step += 1
                successful_updates += 1

            optimizer.zero_grad(set_to_none=True)
            group_samples = 0

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches:
            current_lr = scheduler.get_last_lr()[0]
            step_loss = float(loss.detach().item())
            print(
                f"  Epoch [{epoch+1:03d}] Batch [{batch_idx+1:03d}/{num_batches:03d}] "
                f"Loss: {step_loss:.4f} (Fuse: {loss_metrics['fuse_loss']:.4f}, "
                f"Sep: {loss_metrics['sep_loss']:.4f}, PRM: {loss_metrics['prm_loss']:.4f}) | "
                f"LR: {current_lr:.6f}"
            )

    elapsed = time.perf_counter() - t0
    avg_loss = epoch_loss / max(1, total_samples)

    metrics = {
        "loss": avg_loss,
        "fuse_loss": epoch_fuse / max(1, total_samples),
        "sep_loss": epoch_sep / max(1, total_samples),
        "prm_loss": epoch_prm / max(1, total_samples),
        "epoch_time_seconds": elapsed,
        "samples_per_second": total_samples / max(1e-5, elapsed),
        "amp_skipped_steps": skipped_steps,
    }

    return metrics, global_step, successful_updates, skipped_steps


def train(
    data_dir: str | Path | None = None,
    split_dir: str | Path | None = None,
    save_dir: str | Path | None = None,
    batch_size: int = 16,
    accum_steps: int = 1,
    max_epochs: int = 300,
    base_lr: float = 0.0003333,
    weight_decay: float = 0.0001,
    basic_dims: int = 16,
    seed: int = 1234,
    resume: str | None = None,
    device_name: str | None = None,
    amp: str = "auto",
    num_workers: int = 2,
    log_interval: int = 20,
    max_batches: int | None = None,
    config_file: str | Path | None = None,
) -> Path:
    """Runs standard 300-epoch training strictly conforming to Promt_train_test_no_val.txt."""
    resolved_data = resolve_data_root(data_dir)
    resolved_split = resolve_split_dir(split_dir)
    resolved_save = resolve_save_dir(save_dir)

    train_manifest = resolved_split / "train.txt"
    if not train_manifest.is_file():
        raise FileNotFoundError(f"Missing training manifest: {train_manifest}")

    manifest_hash = compute_manifest_hash(train_manifest)

    # 1. Device and precision resolution
    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp(amp, device)
    print(f"[M2FTrans] Execution device: {device} | AMP: {amp} (dtype: {amp_dtype})")

    # 2. Seed & Determinism
    set_seed(seed=seed, deterministic=True)
    g = torch.Generator()
    g.manual_seed(seed)

    # 3. Training Dataset & DataLoader (Strictly NO validation loader)
    train_dataset = MyoPS380TrainDataset(
        data_dir=resolved_data,
        split_dir=resolved_split,
        split="train",
        img_size=(128, 128),
    )
    print(f"[M2FTrans] Loaded train dataset: {len(train_dataset)} slices (304 patients, zero validation)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,  # Keep all 1,972 slices
        worker_init_fn=seed_worker,
        generator=g,
    )

    batches_per_epoch = len(train_loader)
    updates_per_epoch = math.ceil(batches_per_epoch / accum_steps)
    total_updates = max_epochs * updates_per_epoch
    effective_bs = batch_size * accum_steps

    print(
        f"[M2FTrans] Training Protocol & Budget:\n"
        f"  - protocol_id: myops380_train304_test76_last_v1\n"
        f"  - max_epochs: {max_epochs}\n"
        f"  - batch_size: {batch_size}, accum_steps: {accum_steps} (Effective BS: {effective_bs})\n"
        f"  - batches_per_epoch: {batches_per_epoch}, total_updates: {total_updates}\n"
        f"  - base_lr: {base_lr}, weight_decay: {weight_decay}\n"
        f"  - manifest_hash (train.txt): {manifest_hash[:16]}..."
    )

    # 4. Model initialization
    model = M2FTrans2D(
        in_channels=1,
        num_classes=4,
        img_size=128,
        basic_dims=basic_dims,
        num_modals=3,
    ).to(device)
    num_params = count_parameters(model)
    print(f"[M2FTrans] Initialized M2FTrans2D model with {num_params:,} trainable parameters.")

    # 5. Optimizer, Scheduler, and GradScaler
    optimizer = build_optimizer(model, base_lr=base_lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, total_updates=total_updates, power=0.9)
    criterion = M2FTransCompositeLoss(num_cls=4).to(device)
    use_scaler = (amp_dtype == torch.float16 and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # 6. Checkpoint naming strictly following Promt_train_test_no_val.txt: m2ftrans_seed<seed>.pt
    model_id = "m2ftrans"
    checkpoint_name = f"{model_id}_seed{seed}.pt"
    checkpoint_path = resolved_save / checkpoint_name

    start_epoch = 0
    global_step = 0
    successful_updates = 0
    cumulative_skipped = 0

    resume_target = Path(resume) if (resume and resume != "auto") else checkpoint_path

    if resume_target.is_file():
        print(f"[M2FTrans] Found checkpoint for resume: {resume_target}")
        ckpt = torch.load(resume_target, map_location=device, weights_only=False)
        model_weights = ckpt.get("model_state_dict", ckpt.get("model"))
        model.load_state_dict(model_weights)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt.get("completed_epochs", ckpt.get("epoch", 0) + 1)
        global_step = ckpt.get("global_step", start_epoch * updates_per_epoch)
        successful_updates = ckpt.get("successful_updates", global_step)
        cumulative_skipped = ckpt.get("amp_skipped_updates", 0)

        if "rng_state" in ckpt:
            rng = ckpt["rng_state"]
            if "random" in rng:
                random.setstate(rng["random"])
            if "numpy" in rng:
                np.random.set_state(rng["numpy"])
            if "torch" in rng:
                torch.set_rng_state(rng["torch"].cpu())
            if "cuda" in rng and torch.cuda.is_available() and rng["cuda"] is not None:
                torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])

        print(
            f"[M2FTrans] Resumed successfully from epoch {start_epoch}/{max_epochs} "
            f"(step {global_step}, successful {successful_updates}, skipped {cumulative_skipped})"
        )

    if start_epoch >= max_epochs:
        print(f"[M2FTrans] Checkpoint already completed full budget ({start_epoch}/{max_epochs} epochs).")
        return checkpoint_path

    # 7. Training Loop (No Validation split, No early stopping)
    print(f"[M2FTrans] Starting training for {max_epochs - start_epoch} remaining epochs...")
    total_training_start = time.perf_counter()

    for epoch in range(start_epoch, max_epochs):
        metrics, global_step, successful_updates, skipped_steps = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            amp_dtype=amp_dtype,
            scaler=scaler,
            accum_steps=accum_steps,
            epoch=epoch,
            global_step=global_step,
            successful_updates=successful_updates,
            log_interval=log_interval,
            max_batches=max_batches,
        )
        cumulative_skipped += skipped_steps
        completed_epochs = epoch + 1
        is_final_epoch = (completed_epochs == max_epochs)

        # Atomic checkpoint payload
        checkpoint_data = {
            "model_id": model_id,
            "architecture": "m2ftrans_2d",
            "seed": seed,
            "basic_dims": basic_dims,
            "epoch": epoch,
            "completed_epochs": completed_epochs,
            "planned_epochs": max_epochs,
            "training_complete": is_final_epoch,
            "checkpoint_selection": "last_completed_epoch",
            "checkpoint_selection_metric": None,
            "global_step": global_step,
            "total_updates": total_updates,
            "successful_updates": successful_updates,
            "amp_skipped_updates": cumulative_skipped,
            "amp_dtype": str(amp_dtype),
            "num_parameters": num_params,
            "model": model.state_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "optimization_protocol": OPTIMIZATION_PROTOCOL,
            "training_protocol": {
                "id": "myops380_train304_test76_last_v1",
                "dataset_id": "myops380",
                "train_slices": 1972,
                "train_patients": 304,
                "validation": False,
                "test_patients": 76,
            },
            "manifest_hash": manifest_hash,
            "source_label_order": "legacy",
            "internal_label_order": "canonical",
            "native_output_label_order": "canonical",
            "exported_label_order": "canonical",
            "metrics": metrics,
            "rng_state": {
                "random": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }

        # Atomic write via temporary file
        tmp_checkpoint = checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint_data, tmp_checkpoint)
        if tmp_checkpoint.is_file():
            tmp_checkpoint.replace(checkpoint_path)

        print(
            f"--> Epoch [{completed_epochs:03d}/{max_epochs:03d}] Completed in {metrics['epoch_time_seconds']:.1f}s | "
            f"Loss: {metrics['loss']:.4f} | Saved: {checkpoint_path.name}"
        )

    total_training_time = time.perf_counter() - total_training_start
    print(
        f"[M2FTrans] Training complete! Total duration: {total_training_time / 3600:.2f} hours.\n"
        f"[M2FTrans] Final checkpoint: {checkpoint_path} (training_complete = True)"
    )

    # Save training summary JSON alongside checkpoint
    summary_path = resolved_save / f"training_summary_seed{seed}.json"
    summary_data = {
        "model_id": model_id,
        "benchmark_protocol": "myops380_train304_test76_last_v1",
        "seed": seed,
        "max_epochs": max_epochs,
        "completed_epochs": max_epochs,
        "total_updates": total_updates,
        "successful_updates": successful_updates,
        "amp_skipped_updates": cumulative_skipped,
        "trainable_parameters": num_params,
        "total_training_seconds": total_training_time,
        "checkpoint_file": str(checkpoint_path.resolve()),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return checkpoint_path


def parse_args():
    parser = argparse.ArgumentParser(description="M2FTrans MyoPS380 Training Engine")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yml file")
    parser.add_argument("--data_dir", type=str, default=None, help="Root path to MyoPS380 Processed_data")
    parser.add_argument("--split_dir", type=str, default=None, help="Path to splits/myops380_no_val")
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (default: 16)")
    parser.add_argument("--accum_steps", type=int, default=None, help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--max_epochs", "--epochs", type=int, dest="max_epochs", default=None, help="Maximum epochs (default: 300)")
    parser.add_argument("--lr", "--base_lr", type=float, dest="lr", default=None, help="Base learning rate (default: 0.0003333)")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay (default: 0.0001)")
    parser.add_argument("--basic_dims", type=int, default=None, help="Basic feature dimensions (default: 16)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: 1234)")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        type=str,
        default=None,
        help="Path to checkpoint to resume from, or pass without value to resume from default checkpoint"
    )
    parser.add_argument("--device", type=str, default=None, help="Execution device ('cuda:0' or 'cpu')")
    parser.add_argument("--amp", type=str, default=None, choices=("auto", "none", "fp16", "bf16"), help="AMP mode (default: auto)")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of DataLoader workers (default: 2)")
    parser.add_argument("--log_interval", type=int, default=None, help="Logging frequency in batches (default: 20)")
    parser.add_argument("--max_batches", type=int, default=None, help="Optional limit on batches per epoch for testing")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()

    # Load defaults from config.yml if present
    cfg_data: dict[str, Any] = {}
    config_path = cli_args.config
    if config_path is None:
        local_cfg = Path(__file__).resolve().parent / "config.yml"
        if local_cfg.is_file():
            config_path = str(local_cfg)

    if config_path and Path(config_path).is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_data = yaml.safe_load(f) or {}

    train(
        data_dir=cli_args.data_dir or cfg_data.get("path_to_data"),
        split_dir=cli_args.split_dir or cfg_data.get("path_to_splits"),
        save_dir=cli_args.save_dir or cfg_data.get("path_to_ckpts"),
        batch_size=cli_args.batch_size or cfg_data.get("batch_size", 16),
        accum_steps=cli_args.accum_steps or cfg_data.get("accum_steps", 1),
        max_epochs=cli_args.max_epochs or cfg_data.get("epochs", 300),
        base_lr=cli_args.lr or cfg_data.get("lr", 0.0003333),
        weight_decay=cli_args.weight_decay or cfg_data.get("weight_decay", 0.0001),
        basic_dims=cli_args.basic_dims or cfg_data.get("basic_dims", 16),
        seed=cli_args.seed or cfg_data.get("seed", 1234),
        resume=cli_args.resume or cfg_data.get("path_to_resume"),
        device_name=cli_args.device,
        amp=cli_args.amp or cfg_data.get("amp", "auto"),
        num_workers=cli_args.num_workers if cli_args.num_workers is not None else 2,
        log_interval=cli_args.log_interval or 20,
        max_batches=cli_args.max_batches,
        config_file=config_path,
    )

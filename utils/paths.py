"""Path resolution utilities for M2FTrans on local and server environments."""
from __future__ import annotations

import os
from pathlib import Path


def resolve_split_dir(candidate: str | Path | None = None) -> Path:
    """Resolves split directory containing train.txt and test_vol.txt."""
    if candidate and str(candidate).strip():
        p = Path(candidate)
        if p.is_dir() and (p / "train.txt").is_file():
            return p

    fallbacks = [
        Path(r"D:\nckh_model\M2FTrans\dataset\splits\myops380_no_val"),
        Path(r"D:\nckh_model\SCAR\preprocessing\splits\myops380_no_val"),
        Path(r"../SCAR/preprocessing/splits/myops380_no_val"),
        Path(r"D:\nckh_model\splits\myops380_no_val"),
        Path(r"/content/SCAR/preprocessing/splits/myops380_no_val"),
    ]
    for fb in fallbacks:
        if fb.is_dir() and (fb / "train.txt").is_file():
            return fb

    return Path(r"D:\nckh_model\M2FTrans\dataset\splits\myops380_no_val")


def resolve_data_root(candidate: str | Path | None = None) -> Path:
    """Resolves dataset root containing bSSFP, LGE, T2w directories."""
    if candidate and str(candidate).strip():
        p = Path(candidate)
        if p.is_dir():
            if (p / "bSSFP").is_dir():
                return p
            if (p / "Processed_data" / "bSSFP").is_dir():
                return p / "Processed_data"

    search_dirs = [
        Path(r"D:\nckh_model\Dataset\MyoPS380\Processed_data"),
        Path(r"D:\nckh_model\Dataset\MyoPS380"),
        Path(r"../Dataset/MyoPS380/Processed_data"),
        Path(r"/content/Dataset/MyoPS380/Processed_data"),
        Path(r"/content/MyoPS380_dataset/Processed_data"),
        Path(r"/content/MyoPS380_dataset"),
    ]
    for d in search_dirs:
        if d.is_dir():
            if (d / "bSSFP").is_dir():
                return d
            if (d / "Processed_data" / "bSSFP").is_dir():
                return d / "Processed_data"

    return Path(r"D:\nckh_model\Dataset\MyoPS380\Processed_data")


def resolve_save_dir(candidate: str | Path | None = None) -> Path:
    """Resolves checkpoint output directory."""
    if candidate and str(candidate).strip():
        p = Path(candidate)
    else:
        p = Path(r"D:\nckh_model\M2FTrans\checkpoints")
    p.mkdir(parents=True, exist_ok=True)
    return p

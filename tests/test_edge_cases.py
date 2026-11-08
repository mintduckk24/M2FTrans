"""Explicit edge-case and boundary value tests for M2FTrans adaptation."""
from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import tempfile
import unittest
import numpy as np
import torch

from models.m2ftrans_2d import M2FTrans2D
from dataset.myops380_dataset import (
    MyoPS380TrainDataset,
    read_split_names,
    canonicalize_label,
    ingest_image_array,
)
from utils.criterions import dice_loss, softmax_weighted_loss, M2FTransCompositeLoss
from evaluate import load_surface_distance_module


class TestM2FTransEdgeCases(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = M2FTrans2D(
            in_channels=1,
            num_classes=4,
            img_size=128,
            basic_dims=8,
            num_modals=3,
        ).to(self.device)

    def test_forbid_validation_split(self):
        """Verifies that requesting a validation split is strictly rejected."""
        with self.assertRaises(ValueError) as ctx:
            MyoPS380TrainDataset(split="val")
        self.assertIn("Validation split is forbidden", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            MyoPS380TrainDataset(split="validation")
        self.assertIn("Validation split is forbidden", str(ctx.exception))

    def test_manifest_edge_cases(self):
        """Tests manifest reader with empty, duplicate, or invalid lines."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 1. Non-existent file
            with self.assertRaises(FileNotFoundError):
                read_split_names(tmp_path / "non_existent.txt")

            # 2. Empty file
            empty_file = tmp_path / "empty.txt"
            empty_file.write_text("\n\n  \n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_split_names(empty_file)

            # 3. Duplicate entries
            dup_file = tmp_path / "duplicates.txt"
            dup_file.write_text("case0001\ncase0002\ncase0001\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_split_names(dup_file)

            # 4. Entry with file extension (.npz or .h5)
            ext_file = tmp_path / "extensions.txt"
            ext_file.write_text("case0001.npz\ncase0002\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_split_names(ext_file)

    def test_invalid_input_shapes(self):
        """Tests model rejection of invalid input shapes."""
        self.model.eval()

        # Wrong number of dimensions (3D instead of 4D)
        with self.assertRaises(ValueError):
            self.model(torch.randn(3, 128, 128))

        # Wrong channel count (2 channels or 5 channels)
        with self.assertRaises(ValueError):
            self.model(torch.randn(1, 2, 128, 128))
        with self.assertRaises(ValueError):
            self.model(torch.randn(1, 5, 128, 128))

        # Wrong number of positional image arguments (e.g. 2 or 4 tensors)
        with self.assertRaises(ValueError):
            self.model(torch.randn(1, 1, 128, 128), torch.randn(1, 1, 128, 128))

    def test_non_finite_image_array(self):
        """Tests that ingest_image_array raises ValueError on NaN or Inf."""
        bad_array = np.array([[1.0, np.nan], [0.5, 0.2]], dtype=np.float32)
        with self.assertRaises(ValueError):
            ingest_image_array(bad_array)

        inf_array = np.array([[1.0, np.inf], [0.5, 0.2]], dtype=np.float32)
        with self.assertRaises(ValueError):
            ingest_image_array(inf_array)

    def test_all_empty_masks_metrics(self):
        """Tests surface distance behavior when both prediction and ground truth are empty."""
        sd_mod = load_surface_distance_module()
        # Both empty foreground (all zeros)
        empty_pred = np.zeros((128, 128, 4), dtype=np.uint8)
        empty_gt = np.zeros((128, 128, 4), dtype=np.uint8)

        rows = sd_mod.benchmark_rows(empty_pred, empty_gt, case="case_empty", compute_distance=True)
        for r in rows:
            self.assertEqual(r["status"], "both_empty")
            self.assertIsNone(r["dice"])
            self.assertIsNone(r["hd95"])
            self.assertIsNone(r["asd"])

    def test_one_empty_mask_metrics(self):
        """Tests surface distance behavior when only one mask has foreground."""
        sd_mod = load_surface_distance_module()
        empty_pred = np.zeros((128, 128, 4), dtype=np.uint8)
        gt_with_fg = np.zeros((128, 128, 4), dtype=np.uint8)
        gt_with_fg[50:60, 50:60, 1:3] = 3  # Scar foreground

        rows = sd_mod.benchmark_rows(empty_pred, gt_with_fg, case="case_one_empty", compute_distance=True)
        scar_row = [r for r in rows if r["region"] == "scar"][0]
        self.assertEqual(scar_row["status"], "prediction_empty")
        self.assertEqual(scar_row["dice"], 0.0)
        self.assertIsNone(scar_row["hd95"])

    def test_invalid_label_range(self):
        """Tests that benchmark_rows rejects non-canonical or out-of-range labels."""
        sd_mod = load_surface_distance_module()
        pred = np.zeros((128, 128, 2), dtype=np.uint8)
        pred[10, 10, 0] = 5  # Label 5 is invalid (only 0..3 permitted)
        gt = np.zeros((128, 128, 2), dtype=np.uint8)

        with self.assertRaises(ValueError):
            sd_mod.benchmark_rows(pred, gt, case="case_invalid")

    def test_all_missing_modalities_mask(self):
        """Tests robustness when missing modality mask is applied."""
        self.model.eval()
        x = torch.randn(1, 3, 128, 128)
        # 2 modalities missing (only c0 present)
        mask = torch.tensor([[True, False, False]])
        with torch.no_grad():
            out = self.model(x, mask=mask)
        self.assertEqual(out.shape, (1, 4, 128, 128))
        self.assertTrue(torch.isfinite(out).all())

    def test_amp_resolution(self):
        """Tests AMP resolution matching SCAR precision policy."""
        from train import resolve_amp
        cpu_dev = torch.device("cpu")
        self.assertIsNone(resolve_amp("auto", cpu_dev))
        self.assertIsNone(resolve_amp("none", cpu_dev))
        with self.assertRaises(ValueError):
            resolve_amp("fp16", cpu_dev)

    def test_predict_rejects_incomplete_checkpoint(self):
        """Tests that evaluate/predict rejects checkpoint with training_complete=False."""
        from predict import run_prediction
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ckpt_path = tmp_path / "m2ftrans_seed1234.pt"
            torch.save({
                "model_id": "m2ftrans",
                "completed_epochs": 50,
                "planned_epochs": 300,
                "training_complete": False,
                "basic_dims": 8,
                "model_state_dict": self.model.state_dict(),
            }, ckpt_path)

            with self.assertRaises(ValueError) as ctx:
                run_prediction(checkpoint_path=ckpt_path, allow_incomplete=False)
            self.assertIn("incomplete", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

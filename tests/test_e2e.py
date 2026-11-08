"""End-to-End integration tests for M2FTrans on MyoPS380."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure M2FTrans package root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from models.m2ftrans_2d import M2FTrans2D
from dataset.myops380_dataset import MyoPS380TrainDataset, MyoPS380TestVolumeDataset, read_split_names
from utils.criterions import M2FTransCompositeLoss
from utils.lr_scheduler import build_optimizer, build_scheduler
from predict import single_pass_volume_inference
from evaluate import load_surface_distance_module


class TestM2FTransE2E(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_split_manifests_integrity(self):
        """Verifies train.txt has 1,972 slices and test_vol.txt has 76 patients."""
        split_dir = Path(r"D:\nckh_model\M2FTrans\dataset\splits\myops380_no_val")
        train_names = read_split_names(split_dir / "train.txt")
        test_names = read_split_names(split_dir / "test_vol.txt")

        self.assertEqual(len(train_names), 1972)
        self.assertEqual(len(test_names), 76)

        # Ensure no overlap
        train_patients = {n.split("_slice")[0] for n in train_names}
        test_patients = set(test_names)
        self.assertEqual(len(train_patients), 304)
        self.assertEqual(len(train_patients.intersection(test_patients)), 0)

    def test_dataset_sample_loading(self):
        """Tests reading actual sample from MyoPS380 preprocessed dataset."""
        data_dir = Path(r"D:\nckh_model\Dataset\MyoPS380\Processed_data")
        split_dir = Path(r"D:\nckh_model\M2FTrans\dataset\splits\myops380_no_val")

        if data_dir.is_dir() and (split_dir / "train.txt").is_file():
            ds = MyoPS380TrainDataset(data_dir=data_dir, split_dir=split_dir, split="train")
            self.assertEqual(len(ds), 1972)
            sample = ds[0]
            self.assertEqual(sample["image"].shape, (3, 128, 128))
            self.assertEqual(sample["label"].shape, (128, 128))
            # Verify values are finite (augmented values can be slightly negative or exceed 1.0 due to gaussian noise & brightness up to 1.2 per Prompt Section 3)
            self.assertTrue(torch.isfinite(sample["image"]).all())

    def test_checkpoint_save_and_resume(self):
        """Tests that checkpoint saves all state dictionaries and restores exactly."""
        model = M2FTrans2D(in_channels=1, num_classes=4, basic_dims=8).to(self.device)
        optimizer = build_optimizer(model, base_lr=1e-4)
        scheduler = build_scheduler(optimizer, total_updates=100)

        ckpt_file = self.temp_path / "m2ftrans_seed1234.pt"
        payload = {
            "model_id": "m2ftrans",
            "seed": 1234,
            "completed_epochs": 10,
            "planned_epochs": 300,
            "training_complete": False,
            "global_step": 1240,
            "basic_dims": 8,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        torch.save(payload, ckpt_file)
        self.assertTrue(ckpt_file.is_file())

        # Load back
        loaded = torch.load(ckpt_file, map_location=self.device, weights_only=False)
        self.assertEqual(loaded["model_id"], "m2ftrans")
        self.assertEqual(loaded["completed_epochs"], 10)
        self.assertEqual(loaded["global_step"], 1240)

        model_new = M2FTrans2D(in_channels=1, num_classes=4, basic_dims=8).to(self.device)
        model_new.load_state_dict(loaded["model_state_dict"])

        # Compare model weights
        for p1, p2 in zip(model.parameters(), model_new.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_volume_inference_and_surface_distance(self):
        """Tests single-pass volume inference and shared evaluator surface distance metrics."""
        model = M2FTrans2D(in_channels=1, num_classes=4, basic_dims=8).to(self.device)
        model.eval()

        # Create synthetic volume (128, 128, 4)
        c0 = np.random.rand(128, 128, 4).astype(np.float32)
        lge = np.random.rand(128, 128, 4).astype(np.float32)
        t2 = np.random.rand(128, 128, 4).astype(np.float32)
        gt = np.random.randint(0, 4, (128, 128, 4), dtype=np.int64)

        pred_3d = single_pass_volume_inference(model, c0, lge, t2, self.device, batch_size=2)
        self.assertEqual(pred_3d.shape, (128, 128, 4))
        self.assertEqual(pred_3d.dtype, np.uint8)
        self.assertTrue(np.isin(pred_3d, [0, 1, 2, 3]).all())

        # Test evaluation with surface_distance dataset_rows and summarize_dataset_rows
        sd_mod = load_surface_distance_module()
        rows = sd_mod.dataset_rows(pred_3d, gt, case="case_synth_001", compute_distance=True, dataset_id="myops380")
        self.assertEqual(len(rows), 5)  # 5 regions
        regions = {r["region"] for r in rows}
        self.assertEqual(
            regions,
            {"normal_myocardium", "edema", "scar", "edema_inclusive", "myocardial_ring"}
        )

        summary = sd_mod.summarize_dataset_rows(rows, dataset_id="myops380")
        self.assertIn("scar", summary)
        self.assertIn("avg_pathology_dice", summary)
        self.assertIn("avg_pathology_inclusive_dice", summary)

    def test_volume_inference_arbitrary_resolution(self):
        """Tests inference on non-(128, 128) volume with automatic bilinear interpolation."""
        model = M2FTrans2D(in_channels=1, num_classes=4, basic_dims=8).to(self.device)
        model.eval()

        # Input with spatial size (160, 192, 3)
        c0 = np.random.rand(160, 192, 3).astype(np.float32)
        lge = np.random.rand(160, 192, 3).astype(np.float32)
        t2 = np.random.rand(160, 192, 3).astype(np.float32)

        pred = single_pass_volume_inference(model, c0, lge, t2, self.device, batch_size=2, img_size=128)
        self.assertEqual(pred.shape, (160, 192, 3))
        self.assertEqual(pred.dtype, np.uint8)
        self.assertTrue(np.isin(pred, [0, 1, 2, 3]).all())

    def test_single_epoch_train_with_accumulation(self):
        """Tests single training epoch with accum_steps=2 and sample-weighted accumulation."""
        from train import train_epoch
        model = M2FTrans2D(in_channels=1, num_classes=4, basic_dims=8).to(self.device)
        opt = build_optimizer(model, base_lr=1e-4)
        sched = build_scheduler(opt, total_updates=10)
        crit = M2FTransCompositeLoss(num_cls=4)

        # Create dummy dataloader with 3 batches of varying sizes: 4, 4, 2
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10
            def __getitem__(self, idx):
                return {
                    "image": torch.randn(3, 128, 128),
                    "label": torch.randint(0, 4, (128, 128)),
                    "case_id": f"dummy_{idx}",
                }

        loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=4, shuffle=False)
        metrics, g_step, succ_updates, skipped = train_epoch(
            model=model,
            loader=loader,
            optimizer=opt,
            scheduler=sched,
            criterion=crit,
            device=self.device,
            amp_dtype=None,
            scaler=None,
            accum_steps=2,
            epoch=0,
            global_step=0,
            successful_updates=0,
        )
        self.assertGreater(metrics["loss"], 0.0)
        self.assertEqual(g_step, 2)  # 3 batches with accum_steps=2 -> 2 updates
        self.assertEqual(succ_updates, 2)
        self.assertEqual(skipped, 0)


if __name__ == "__main__":
    unittest.main()

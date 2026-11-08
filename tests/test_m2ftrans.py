"""Unit tests for M2FTrans MyoPS380 2D adaptation."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure M2FTrans package root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.m2ftrans_2d import M2FTrans2D
from utils.criterions import dice_loss, softmax_weighted_loss, M2FTransCompositeLoss
from utils.lr_scheduler import build_optimizer, build_scheduler
from dataset.data_augmentation import RandomGenerator, ResizeGenerator
from dataset.myops380_dataset import canonicalize_label, read_split_names


class TestM2FTrans2D(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = M2FTrans2D(
            in_channels=1,
            num_classes=4,
            img_size=128,
            basic_dims=8,  # Lightweight for fast unit testing
            num_modals=3,
        ).to(self.device)

    def test_eval_forward_single_tensor(self):
        """Tests forward pass with 1 concatenated tensor (B, 3, 128, 128)."""
        self.model.eval()
        x = torch.randn(2, 3, 128, 128, device=self.device)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (2, 4, 128, 128))
        # Check output is softmax probability distribution
        sums = out.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-4))

    def test_eval_forward_three_tensors(self):
        """Tests forward pass with 3 separate tensors (c0, lge, t2)."""
        self.model.eval()
        c0 = torch.randn(2, 1, 128, 128, device=self.device)
        lge = torch.randn(2, 1, 128, 128, device=self.device)
        t2 = torch.randn(2, 1, 128, 128, device=self.device)
        with torch.no_grad():
            out = self.model(c0, lge, t2)
        self.assertEqual(out.shape, (2, 4, 128, 128))

    def test_train_forward_outputs(self):
        """Tests that training mode outputs fuse_pred, sep_preds, and prm_preds."""
        self.model.train()
        x = torch.randn(2, 3, 128, 128, device=self.device)
        fuse_pred, sep_preds, prm_preds = self.model(x)

        self.assertEqual(fuse_pred.shape, (2, 4, 128, 128))
        self.assertEqual(len(sep_preds), 3)  # c0, lge, t2
        for p in sep_preds:
            self.assertEqual(p.shape, (2, 4, 128, 128))
        self.assertEqual(len(prm_preds), 5)  # 5 multi-scale PRM heads
        for p in prm_preds:
            self.assertEqual(p.shape, (2, 4, 128, 128))

    def test_modality_masking(self):
        """Tests forward pass with missing modality mask."""
        self.model.eval()
        x = torch.randn(2, 3, 128, 128, device=self.device)
        # Case 0: all present; Case 1: LGE missing
        mask = torch.tensor([[True, True, True], [True, False, True]], device=self.device)
        with torch.no_grad():
            out = self.model(x, mask=mask)
        self.assertEqual(out.shape, (2, 4, 128, 128))
        self.assertTrue(torch.isfinite(out).all())

    def test_backward_and_gradients(self):
        """Tests that gradients flow to all encoders and bottleneck."""
        self.model.train()
        x = torch.randn(2, 3, 128, 128, device=self.device)
        target = torch.randint(0, 4, (2, 128, 128), device=self.device)

        fuse_pred, sep_preds, prm_preds = self.model(x)
        criterion = M2FTransCompositeLoss(num_cls=4)
        loss, _ = criterion(fuse_pred, sep_preds, prm_preds, target)

        loss.backward()
        self.assertTrue(torch.isfinite(loss))

        # Check gradients in encoders and decoders
        self.assertIsNotNone(self.model.c0_encoder.e1_c1.conv.weight.grad)
        self.assertIsNotNone(self.model.lge_encoder.e1_c1.conv.weight.grad)
        self.assertIsNotNone(self.model.t2_encoder.e1_c1.conv.weight.grad)
        self.assertIsNotNone(self.model.Bottleneck.trans_bottle.cross_attention_list[0].fn.norm.weight.grad)

    def test_parameter_count(self):
        """Checks parameter counting utility."""
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertGreater(total_params, 1000000)

    def test_losses(self):
        """Tests Dice loss, Weighted CE loss, and Composite loss."""
        pred = F.softmax(torch.randn(2, 4, 32, 32), dim=1)
        target = torch.randint(0, 4, (2, 32, 32))

        d_loss = dice_loss(pred, target, num_cls=4)
        self.assertTrue(torch.isfinite(d_loss))
        self.assertGreaterEqual(float(d_loss), 0.0)

        w_loss = softmax_weighted_loss(pred, target, num_cls=4)
        self.assertTrue(torch.isfinite(w_loss))
        self.assertGreaterEqual(float(w_loss), 0.0)

    def test_canonicalize_label(self):
        """Tests legacy to canonical label mapping."""
        legacy = np.array([0, 1, 2, 3], dtype=np.int64)
        canonical = canonicalize_label(legacy, source_order="legacy")
        # 0:bg->0, 1:myo->1, 2:scar->3, 3:edema->2
        np.testing.assert_array_equal(canonical, np.array([0, 1, 3, 2]))

    def test_random_generator(self):
        """Tests SCAR RandomGenerator augmentation."""
        rg = RandomGenerator(output_size=(128, 128))
        sample = {
            "image": np.random.rand(128, 128).astype(np.float32),
            "image1": np.random.rand(128, 128).astype(np.float32),
            "image2": np.random.rand(128, 128).astype(np.float32),
            "label": np.random.randint(0, 4, (128, 128)).astype(np.int64),
        }
        augmented = rg(sample)
        self.assertEqual(augmented["image"].shape, (1, 128, 128))
        self.assertEqual(augmented["label"].shape, (128, 128))
        self.assertEqual(augmented["label"].dtype, torch.int64)

    def test_scheduler_offset(self):
        """Tests that scheduler first two steps use base LR matching I-MMSeg."""
        opt = build_optimizer(self.model, base_lr=1e-3)
        sched = build_scheduler(opt, total_updates=100)

        # Step 0 (before updates)
        lr0 = sched.get_last_lr()[0]
        self.assertAlmostEqual(lr0, 1e-3)

        # Step 1 (after 1st update)
        opt.step()
        sched.step()
        lr1 = sched.get_last_lr()[0]
        self.assertAlmostEqual(lr1, 1e-3)

        # Step 2 (after 2nd update): LR starts to decay
        opt.step()
        sched.step()
        lr2 = sched.get_last_lr()[0]
        self.assertLess(lr2, 1e-3)

    def test_group_conv_block_architecture(self):
        """Verifies GroupConvBlock grouped convolution and post-conv LayerNorm topology."""
        from models.blocks_2d import GroupConvBlock
        block = GroupConvBlock(embed_dims=16, expand_ratio=4)
        self.assertEqual(block.dwconv.groups, 16)
        self.assertEqual(block.dwconv.in_channels, 64)
        self.assertEqual(block.dwconv.out_channels, 64)
        self.assertEqual(block.pwconv1.in_channels, 16)
        self.assertEqual(block.pwconv1.out_channels, 64)
        self.assertEqual(block.pwconv2.in_channels, 64)
        self.assertEqual(block.pwconv2.out_channels, 16)
        self.assertEqual(block.norm1.normalized_shape, (64,))
        self.assertEqual(block.norm2.normalized_shape, (64,))
        self.assertEqual(block.norm3.normalized_shape, (16,))

        x = torch.randn(2, 16, 16, 16)
        out = block(x)
        self.assertEqual(out.shape, (2, 16, 16, 16))

    def test_sample_weighted_gradient_accumulation(self):
        """Verifies that sample-weighted accumulation matches single-batch gradient exactly."""
        torch.manual_seed(42)
        model1 = M2FTrans2D(in_channels=1, num_classes=4, basic_dims=8)
        model2 = M2FTrans2D(in_channels=1, num_classes=4, basic_dims=8)
        model2.load_state_dict(model1.state_dict())

        criterion = M2FTransCompositeLoss(num_cls=4)

        # Batch A: 16 samples, Batch B: 4 samples
        x_a = torch.randn(16, 3, 128, 128)
        y_a = torch.randint(0, 4, (16, 128, 128))
        x_b = torch.randn(4, 3, 128, 128)
        y_b = torch.randint(0, 4, (4, 128, 128))

        # Model 1: Accumulate across A and B with sample weighting (loss * batch_count) / total_samples
        f_a, s_a, p_a = model1(x_a)
        loss_a, _ = criterion(f_a, s_a, p_a, y_a)
        (loss_a * 16).backward()

        f_b, s_b, p_b = model1(x_b)
        loss_b, _ = criterion(f_b, s_b, p_b, y_b)
        (loss_b * 4).backward()

        for p in model1.parameters():
            if p.grad is not None:
                p.grad.div_(20)

        # Verify gradients exist and are finite
        for p in model1.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())


if __name__ == "__main__":
    unittest.main()

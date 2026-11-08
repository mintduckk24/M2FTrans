"""Native Loss Functions for M2FTrans MyoPS380 Adaptation.

Conforms to Section 5 of Promt_train_test_no_val.txt:
Preserves M2FTrans's native objective (SoftDice + SoftmaxWeightedCrossEntropy)
across the fused output, auxiliary unimodal decoders, and multi-scale PRM heads.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_one_hot(target: torch.Tensor, num_cls: int = 4) -> torch.Tensor:
    """Converts (B, H, W) label tensor to (B, num_cls, H, W) one-hot float tensor."""
    if target.dim() == 4 and target.shape[1] == num_cls:
        return target.float()
    if target.dim() == 3:
        # Shape: (B, H, W)
        return F.one_hot(target.long(), num_classes=num_cls).permute(0, 3, 1, 2).float()
    raise ValueError(f"Unsupported target shape for one-hot conversion: {target.shape}")


def dice_loss(output: torch.Tensor, target: torch.Tensor, num_cls: int = 4, eps: float = 1e-7) -> torch.Tensor:
    """Computes soft Dice loss across all classes (including background).

    Args:
        output: Probability tensor of shape (B, num_cls, H, W) from Softmax.
        target: Target tensor of shape (B, H, W) or one-hot (B, num_cls, H, W).
    """
    target_oh = to_one_hot(target, num_cls=num_cls)
    total_dice = torch.zeros(1, device=output.device, dtype=output.dtype)

    for i in range(num_cls):
        out_i = output[:, i, :, :]
        tgt_i = target_oh[:, i, :, :]
        num = 2.0 * torch.sum(out_i * tgt_i)
        den = torch.sum(out_i) + torch.sum(tgt_i) + eps
        total_dice = total_dice + (num / den)

    return 1.0 - (total_dice / num_cls).squeeze()


def softmax_weighted_loss(output: torch.Tensor, target: torch.Tensor, num_cls: int = 4, eps: float = 1e-7) -> torch.Tensor:
    """Computes class-frequency weighted cross-entropy loss.

    Args:
        output: Probability tensor of shape (B, num_cls, H, W) from Softmax.
        target: Target tensor of shape (B, H, W) or one-hot (B, num_cls, H, W).
    """
    target_oh = to_one_hot(target, num_cls=num_cls)
    B, _, H, W = output.shape
    total_target_voxels = torch.sum(target_oh, dim=(1, 2, 3), keepdim=True) + eps

    cross_loss = torch.zeros(1, device=output.device, dtype=output.dtype)

    for i in range(num_cls):
        out_i = output[:, i, :, :]
        tgt_i = target_oh[:, i, :, :]

        # Class frequency weight: 1.0 - (sum(class_voxels) / sum(all_voxels))
        class_voxels = torch.sum(tgt_i, dim=(1, 2), keepdim=True).unsqueeze(1)
        weighted = 1.0 - (class_voxels / total_target_voxels)  # (B, 1, 1, 1)

        # Clamped log probabilities
        clamped_log = torch.log(torch.clamp(out_i, min=0.005, max=1.0))
        loss_i = -1.0 * weighted.squeeze(1) * tgt_i * clamped_log
        cross_loss = cross_loss + torch.mean(loss_i)

    return cross_loss.squeeze()


class M2FTransCompositeLoss(nn.Module):
    """Composite Loss Module for M2FTrans:

    L_total = L_fuse + L_sep + L_prm
    where:
      L_fuse = WeightedCE(fuse_pred, y) + Dice(fuse_pred, y)
      L_sep = sum_{m in {c0, lge, t2}} (WeightedCE(sep_m, y) + Dice(sep_m, y))
      L_prm = sum_{s in 1..5} (WeightedCE(prm_s, y) + Dice(prm_s, y))
    """

    def __init__(self, num_cls: int = 4):
        super().__init__()
        self.num_cls = num_cls

    def forward(
        self,
        fuse_pred: torch.Tensor,
        sep_preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | list[torch.Tensor] | None,
        prm_preds: Sequence[torch.Tensor] | None,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict[str, float]]:
        # 1. Primary Fused Loss
        fuse_cross = softmax_weighted_loss(fuse_pred, target, num_cls=self.num_cls)
        fuse_dice = dice_loss(fuse_pred, target, num_cls=self.num_cls)
        fuse_loss = fuse_cross + fuse_dice

        total_loss = fuse_loss
        sep_loss_val = 0.0
        prm_loss_val = 0.0

        # 2. Auxiliary Unimodal Decoder Loss
        if sep_preds is not None:
            sep_loss = torch.zeros(1, device=fuse_pred.device, dtype=fuse_pred.dtype)
            for p in sep_preds:
                sep_loss = sep_loss + softmax_weighted_loss(p, target, num_cls=self.num_cls) + dice_loss(p, target, num_cls=self.num_cls)
            total_loss = total_loss + sep_loss.squeeze()
            sep_loss_val = float(sep_loss.item())

        # 3. Auxiliary PRM Multi-scale Loss
        if prm_preds is not None:
            prm_loss = torch.zeros(1, device=fuse_pred.device, dtype=fuse_pred.dtype)
            for p in prm_preds:
                prm_loss = prm_loss + softmax_weighted_loss(p, target, num_cls=self.num_cls) + dice_loss(p, target, num_cls=self.num_cls)
            total_loss = total_loss + prm_loss.squeeze()
            prm_loss_val = float(prm_loss.item())

        metrics = {
            "loss": float(total_loss.item()),
            "fuse_loss": float(fuse_loss.item()),
            "fuse_cross": float(fuse_cross.item()),
            "fuse_dice": float(fuse_dice.item()),
            "sep_loss": sep_loss_val,
            "prm_loss": prm_loss_val,
        }

        return total_loss, metrics

"""2D Neural Network Layers for M2FTrans MyoPS380 Adaptation.

Converts original 3D layers of M2FTrans into self-contained 2D operations.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalization(planes: int, norm: str = "in") -> nn.Module:
    """Creates 2D normalization layer."""
    if norm == "bn":
        return nn.BatchNorm2d(planes)
    elif norm == "gn":
        groups = min(4, planes)
        while planes % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, planes)
    elif norm == "in":
        return nn.InstanceNorm2d(planes, affine=True)
    else:
        raise ValueError(f"Normalization type {norm} is not supported")


class general_conv2d_prenorm(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pad_type: str = "zeros",
        norm: str = "in",
        act_type: str = "lrelu",
        relufactor: float = 0.2,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=k_size,
            stride=stride,
            padding=padding,
            padding_mode=pad_type,
            bias=True,
        )
        self.norm = normalization(in_ch, norm=norm)
        if act_type == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif act_type == "lrelu":
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)
        else:
            self.activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        return x


class general_conv2d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pad_type: str = "reflect",
        norm: str = "in",
        act_type: str = "lrelu",
        relufactor: float = 0.2,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=k_size,
            stride=stride,
            padding=padding,
            padding_mode=pad_type,
            bias=True,
        )
        self.norm = normalization(out_ch, norm=norm)
        if act_type == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif act_type == "lrelu":
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)
        else:
            self.activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class prm_fusion2d(nn.Module):
    """PRM head producing multi-scale auxiliary segmentation predictions."""

    def __init__(self, in_channel: int = 64, basic_dim: int = 16, norm: str = "in", num_cls: int = 4):
        super().__init__()
        self.prm_layer = nn.Sequential(
            general_conv2d(in_channel, basic_dim, k_size=1, stride=1, padding=0, pad_type="zeros", norm=norm),
            nn.Conv2d(basic_dim, num_cls, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prm_layer(x)


class fusion_postnorm2d(nn.Module):
    """Region-aware feature modulation (RFM) combining stacked modalities."""

    def __init__(self, in_channel: int = 64, num_modals: int = 3):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            general_conv2d(in_channel * num_modals, in_channel, k_size=1, padding=0, stride=1, pad_type="zeros"),
            general_conv2d(in_channel, in_channel, k_size=3, padding=1, stride=1, pad_type="reflect"),
            general_conv2d(in_channel, in_channel, k_size=1, padding=0, stride=1, pad_type="zeros"),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x shape: (B, num_modals, C, H, W)
        B, K, C, H, W = x.size()
        y = x.clone()
        if mask is not None:
            # mask: (B, K)
            for b in range(B):
                for m in range(K):
                    if not mask[b, m]:
                        y[b, m] = 0.0
        y = y.view(B, K * C, H, W)
        return self.fusion_layer(y)

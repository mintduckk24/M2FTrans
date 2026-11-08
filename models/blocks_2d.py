"""2D Transformer and Attention Blocks for M2FTrans MyoPS380 Adaptation.

Converts original 3D blocks into self-contained 2D operations for 3 modalities (bSSFP, LGE, T2w).
"""
from __future__ import annotations

import math
from functools import partial
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def nchw2nlc2nchw(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Applies a 1D normalization module (LayerNorm) to 2D NCHW tensor."""
    B, C, H, W = x.shape
    x = x.flatten(2).transpose(1, 2)  # B, H*W, C
    x = module(x)
    x = x.transpose(1, 2).reshape(B, C, H, W).contiguous()
    return x


def mask_gen_fusion(
    batch_size: int,
    num_heads: int,
    patches: int,
    num_modals: int,
    mask: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generates attention mask for Bottleneck MaskedTransformer with fusion token."""
    total_tokens = patches * (num_modals + 1)
    if mask is None:
        return torch.ones((batch_size, num_heads, total_tokens, total_tokens), dtype=torch.bool, device=device)

    bs = mask.size(0)
    self_mask_batch = torch.zeros((bs, num_heads, total_tokens, total_tokens), dtype=torch.bool, device=device)

    for j in range(bs):
        self_mask = torch.zeros((total_tokens, total_tokens), dtype=torch.bool, device=device)
        # Modality self-attention within each modality's token block
        for i in range(num_modals):
            self_mask[patches * i : patches * (i + 1), patches * i : patches * (i + 1)] = True

        # Fusion token attends to itself
        self_mask[patches * num_modals : patches * (num_modals + 1), patches * num_modals : patches * (num_modals + 1)] = True

        # Fusion token attends to present modalities
        for i in range(num_modals):
            if mask[j, i]:
                self_mask[patches * num_modals : patches * (num_modals + 1), patches * i : patches * (i + 1)] = True

        self_mask_batch[j] = self_mask.unsqueeze(0).repeat(num_heads, 1, 1)

    return self_mask_batch


def mask_gen_cross(
    batch_size: int,
    query_channels: int,
    key_channels: int,
    num_modals: int,
    mask: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generates cross-attention mask across modalities."""
    if mask is None:
        return torch.ones((batch_size, query_channels, key_channels), dtype=torch.bool, device=device)

    bs = mask.size(0)
    self_mask = torch.ones((bs, query_channels, key_channels), dtype=torch.bool, device=device)
    mod_len = key_channels // num_modals

    for j in range(bs):
        for i in range(num_modals):
            if not mask[j, i]:
                self_mask[j, :, mod_len * i : mod_len * (i + 1)] = False

    return self_mask


class DepthWiseConvBlock(nn.Module):
    """2D Depthwise Separable Convolution block with LayerNorm and GELU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, 1)
        layer_norm = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = layer_norm(mid_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels)
        self.norm2 = layer_norm(mid_channels)
        self.act2 = nn.GELU()

        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, 1)
        self.norm3 = layer_norm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = nchw2nlc2nchw(self.norm1, x)
        x = self.act1(x)

        x = self.conv2(x)
        x = nchw2nlc2nchw(self.norm2, x)
        x = self.act2(x)

        x = self.conv3(x)
        x = nchw2nlc2nchw(self.norm3, x)
        return x


class GroupConvBlock(nn.Module):
    """Grouped convolution block with residual matching original M2FTrans blocks.py."""

    def __init__(self, embed_dims: int, expand_ratio: int = 4, proj_drop: float = 0.0):
        super().__init__()
        mid_dims = embed_dims * expand_ratio
        self.pwconv1 = nn.Conv2d(embed_dims, mid_dims, 1, 1)
        self.norm1 = nn.LayerNorm(mid_dims, eps=1e-6)
        self.act1 = nn.GELU()

        self.dwconv = nn.Conv2d(mid_dims, mid_dims, 3, 1, 1, groups=embed_dims)
        self.norm2 = nn.LayerNorm(mid_dims, eps=1e-6)
        self.act2 = nn.GELU()

        self.pwconv2 = nn.Conv2d(mid_dims, embed_dims, 1, 1)
        self.norm3 = nn.LayerNorm(embed_dims, eps=1e-6)

        self.final_act = nn.GELU()
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, identity: torch.Tensor | None = None) -> torch.Tensor:
        input_tensor = x
        x = self.pwconv1(x)
        x = nchw2nlc2nchw(self.norm1, x)
        x = self.act1(x)

        x = self.dwconv(x)
        x = nchw2nlc2nchw(self.norm2, x)
        x = self.act2(x)

        x = self.pwconv2(x)
        x = nchw2nlc2nchw(self.norm3, x)

        if identity is None:
            x = input_tensor + self.proj_drop(x)
        else:
            x = identity + self.proj_drop(x)

        x = self.final_act(x)
        return x


class MultiMaskAttentionLayer(nn.Module):
    """Cross-attention layer between query kernels and 3 input modalities."""

    def __init__(
        self,
        kv_dim: int,
        query_dim: int,
        num_modals: int = 3,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_modals = num_modals
        self.attn_drop = nn.Dropout(attn_drop)
        self.query_map = DepthWiseConvBlock(query_dim, query_dim)

        self.key_map_c0 = DepthWiseConvBlock(kv_dim, kv_dim)
        self.value_map_c0 = DepthWiseConvBlock(kv_dim, kv_dim)
        self.key_map_lge = DepthWiseConvBlock(kv_dim, kv_dim)
        self.value_map_lge = DepthWiseConvBlock(kv_dim, kv_dim)
        self.key_map_t2 = DepthWiseConvBlock(kv_dim, kv_dim)
        self.value_map_t2 = DepthWiseConvBlock(kv_dim, kv_dim)

        self.out_project = DepthWiseConvBlock(query_dim, query_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        key: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        value: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        identity = query
        c0, lge, t2 = key
        qb, qc, qh, qw = query.shape

        q = self.query_map(query).flatten(2)  # (qb, qc, qh*qw)

        key_c0 = self.key_map_c0(c0).flatten(2)
        value_c0 = self.value_map_c0(c0).flatten(2)
        key_lge = self.key_map_lge(lge).flatten(2)
        value_lge = self.value_map_lge(lge).flatten(2)
        key_t2 = self.key_map_t2(t2).flatten(2)
        value_t2 = self.value_map_t2(t2).flatten(2)

        k = torch.cat((key_c0, key_lge, key_t2), dim=1)  # (qb, 3*kv_dim, qh*qw)
        v = torch.cat((value_c0, value_lge, value_t2), dim=1)

        attn = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)

        if mask is not None:
            self_mask = mask_gen_cross(qb, qc, k.shape[1], self.num_modals, mask, device=query.device)
            attn = attn.masked_fill(~self_mask, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.reshape(qb, qc, qh, qw)
        x = self.out_project(x)
        return identity + self.proj_drop(x)


class MultiMaskCrossBlock(nn.Module):
    """Cross-attention block with FFN for 3 modalities."""

    def __init__(
        self,
        feature_channels: int,
        num_classes: int,
        num_modals: int = 3,
        expand_ratio: int = 4,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        ffn_feature_maps: bool = True,
    ):
        super().__init__()
        self.ffn_feature_maps = ffn_feature_maps
        self.cross_attn = MultiMaskAttentionLayer(
            kv_dim=feature_channels,
            query_dim=num_classes,
            num_modals=num_modals,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
        )

        if ffn_feature_maps:
            self.ffn2 = GroupConvBlock(embed_dims=feature_channels, expand_ratio=expand_ratio, proj_drop=drop_rate)
        self.ffn1 = GroupConvBlock(embed_dims=num_classes, expand_ratio=expand_ratio, proj_drop=drop_rate)

    def forward(
        self,
        kernels: torch.Tensor,
        feature_maps: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        c0, lge, t2 = feature_maps
        kernels = self.cross_attn(query=kernels, key=feature_maps, value=feature_maps, mask=mask)
        kernels = self.ffn1(kernels, identity=kernels)

        if self.ffn_feature_maps:
            c0 = self.ffn2(c0, identity=c0)
            lge = self.ffn2(lge, identity=lge)
            t2 = self.ffn2(t2, identity=t2)
            feature_maps = (c0, lge, t2)

        return kernels, feature_maps


class MultiCrossToken(nn.Module):
    """Multi-layer Cross-Attention Token block across modalities."""

    def __init__(
        self,
        num_layers: int = 2,
        mlp_ratio: int = 4,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.0,
        channel: int = 256,
        num_modals: int = 3,
    ):
        super().__init__()
        self.channels = channel
        self.layers = nn.ModuleList([
            MultiMaskCrossBlock(
                feature_channels=self.channels,
                num_classes=self.channels,
                num_modals=num_modals,
                expand_ratio=mlp_ratio,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                ffn_feature_maps=(i != num_layers - 1),
            )
            for i in range(num_layers)
        ])

    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        kernels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feature_maps = inputs
        for layer in self.layers:
            kernels, feature_maps = layer(kernels, feature_maps, mask)
        return kernels

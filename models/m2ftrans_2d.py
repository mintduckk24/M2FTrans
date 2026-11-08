"""M2FTrans 2D Architecture for Multimodal CMR Image Segmentation.

Converts the official PyTorch implementation of M2FTrans (Shi et al., IEEE-JBHI 2024)
from 3D BraTS formulation to 2D MyoPS380 formulation:
- 3 Dedicated 2D Encoders for aligned bSSFP, LGE, T2w modalities
- Bottleneck MaskedTransformer with modality masking and learnable fusion token
- Weight_Attention for modality-aware spatial skip-feature modulation
- Multi-scale Decoder_fusion with MultiCrossToken, RFM, and deep supervision PRM heads
- Auxiliary unimodal Decoder_sep for training supervision
- Canonical 4-class output: 0: Background, 1: Normal Myocardium, 2: Edema, 3: Scar
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers_2d import (
    general_conv2d,
    general_conv2d_prenorm,
    normalization,
    prm_fusion2d,
    fusion_postnorm2d,
)
from models.blocks_2d import (
    DepthWiseConvBlock,
    GroupConvBlock,
    MultiCrossToken,
    mask_gen_fusion,
)


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class PreNormDrop(nn.Module):
    def __init__(self, dim: int, dropout_rate: float, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout_rate)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.dropout(self.fn(self.norm(x), **kwargs))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskedResidual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        res, attn = self.fn(x, mask=mask)
        return res + x, attn


class MaskedPreNormDrop(nn.Module):
    def __init__(self, dim: int, dropout_rate: float, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout_rate)
        self.fn = fn

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, attn = self.fn(self.norm(x), mask=mask)
        return self.dropout(out), attn


class MaskedAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        dropout_rate: float = 0.0,
        num_modals: int = 3,
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_modals = num_modals

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        patches = N // (self.num_modals + 1)
        self_mask = mask_gen_fusion(B, self.num_heads, patches, self.num_modals, mask, device=x.device)
        attn = attn.masked_fill(~self_mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_drop(attn)

        x_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out, attn


class MaskedTransformer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        depth: int = 3,
        heads: int = 8,
        mlp_dim: int = 1024,
        dropout_rate: float = 0.1,
        num_modals: int = 3,
    ):
        super().__init__()
        self.depth = depth
        self.cross_attention_list = nn.ModuleList([
            MaskedResidual(
                MaskedPreNormDrop(
                    embedding_dim,
                    dropout_rate,
                    MaskedAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate, num_modals=num_modals),
                )
            )
            for _ in range(self.depth)
        ])
        self.cross_ffn_list = nn.ModuleList([
            Residual(PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate)))
            for _ in range(self.depth)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        attn_list = []
        for j in range(self.depth):
            x, attn = self.cross_attention_list[j](x, mask=mask)
            attn_list.append(attn.detach())
            x = self.cross_ffn_list[j](x)
        return x, attn_list


class Encoder2D(nn.Module):
    """5-stage 2D Convolutional Encoder for a single MRI modality."""

    def __init__(self, in_ch: int = 1, basic_dims: int = 16):
        super().__init__()
        self.basic_dims = basic_dims

        # Stage 1: 128x128
        self.e1_c1 = general_conv2d(in_ch, basic_dims, pad_type="reflect")
        self.e1_c2 = general_conv2d(basic_dims, basic_dims, pad_type="reflect")
        self.e1_c3 = general_conv2d(basic_dims, basic_dims, pad_type="reflect")

        # Stage 2: 64x64
        self.e2_c1 = general_conv2d(basic_dims, basic_dims * 2, stride=2, pad_type="reflect")
        self.e2_c2 = general_conv2d(basic_dims * 2, basic_dims * 2, pad_type="reflect")
        self.e2_c3 = general_conv2d(basic_dims * 2, basic_dims * 2, pad_type="reflect")

        # Stage 3: 32x32
        self.e3_c1 = general_conv2d(basic_dims * 2, basic_dims * 4, stride=2, pad_type="reflect")
        self.e3_c2 = general_conv2d(basic_dims * 4, basic_dims * 4, pad_type="reflect")
        self.e3_c3 = general_conv2d(basic_dims * 4, basic_dims * 4, pad_type="reflect")

        # Stage 4: 16x16
        self.e4_c1 = general_conv2d(basic_dims * 4, basic_dims * 8, stride=2, pad_type="reflect")
        self.e4_c2 = general_conv2d(basic_dims * 8, basic_dims * 8, pad_type="reflect")
        self.e4_c3 = general_conv2d(basic_dims * 8, basic_dims * 8, pad_type="reflect")

        # Stage 5: 8x8
        self.e5_c1 = general_conv2d(basic_dims * 8, basic_dims * 16, stride=2, pad_type="reflect")
        self.e5_c2 = general_conv2d(basic_dims * 16, basic_dims * 16, pad_type="reflect")
        self.e5_c3 = general_conv2d(basic_dims * 16, basic_dims * 16, pad_type="reflect")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))

        return x1, x2, x3, x4, x5


class Decoder_sep2D(nn.Module):
    """Auxiliary unimodal decoder for training supervision."""

    def __init__(self, num_cls: int = 4, basic_dims: int = 16):
        super().__init__()
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.d4_c1 = general_conv2d(basic_dims * 16, basic_dims * 8, pad_type="reflect")
        self.d4_c2 = general_conv2d(basic_dims * 16, basic_dims * 8, pad_type="reflect")
        self.d4_out = general_conv2d(basic_dims * 8, basic_dims * 8, k_size=1, padding=0, pad_type="zeros")

        self.d3_c1 = general_conv2d(basic_dims * 8, basic_dims * 4, pad_type="reflect")
        self.d3_c2 = general_conv2d(basic_dims * 8, basic_dims * 4, pad_type="reflect")
        self.d3_out = general_conv2d(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type="zeros")

        self.d2_c1 = general_conv2d(basic_dims * 4, basic_dims * 2, pad_type="reflect")
        self.d2_c2 = general_conv2d(basic_dims * 4, basic_dims * 2, pad_type="reflect")
        self.d2_out = general_conv2d(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type="zeros")

        self.d1_c1 = general_conv2d(basic_dims * 2, basic_dims, pad_type="reflect")
        self.d1_c2 = general_conv2d(basic_dims * 2, basic_dims, pad_type="reflect")
        self.d1_out = general_conv2d(basic_dims, basic_dims, k_size=1, padding=0, pad_type="zeros")

        self.seg_layer = nn.Conv2d(basic_dims, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
    ) -> torch.Tensor:
        de_x5 = self.d4_c1(self.up2(x5))
        cat_x4 = torch.cat((de_x5, x4), dim=1)
        de_x4 = self.d4_out(self.d4_c2(cat_x4))

        de_x4 = self.d3_c1(self.up2(de_x4))
        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))

        de_x3 = self.d2_c1(self.up2(de_x3))
        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))

        de_x2 = self.d1_c1(self.up2(de_x2))
        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        logits = self.seg_layer(de_x1)
        return self.softmax(logits)


class Bottleneck2D(nn.Module):
    """Bottleneck Transformer layer with modality masking and token fusion."""

    def __init__(
        self,
        basic_dims: int = 16,
        depth: int = 3,
        num_heads: int = 8,
        mlp_dim: int = 1024,
        num_modals: int = 3,
    ):
        super().__init__()
        self.num_modals = num_modals
        self.embedding_dim = basic_dims * 16
        self.trans_bottle = MaskedTransformer(
            embedding_dim=self.embedding_dim,
            depth=depth,
            heads=num_heads,
            mlp_dim=mlp_dim,
            num_modals=num_modals,
        )

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        mask: torch.Tensor | None,
        fusion: torch.Tensor,
        pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        c0, lge, t2 = x
        embed_c0 = c0.flatten(2).transpose(1, 2).contiguous()
        embed_lge = lge.flatten(2).transpose(1, 2).contiguous()
        embed_t2 = t2.flatten(2).transpose(1, 2).contiguous()

        embed_cat = torch.cat((embed_c0, embed_lge, embed_t2, fusion), dim=1)
        embed_cat = embed_cat + pos
        embed_cat_trans, attn = self.trans_bottle(embed_cat, mask)
        c0_trans, lge_trans, t2_trans, fusion_trans = torch.chunk(embed_cat_trans, self.num_modals + 1, dim=1)

        return c0_trans, lge_trans, t2_trans, fusion_trans, attn


class Weight_Attention2D(nn.Module):
    """Extracts attention from fusion token and modulates skip connections across modalities."""

    def __init__(self, num_modals: int = 3, spatial_size: int = 8):
        super().__init__()
        self.num_modals = num_modals
        self.spatial_size = spatial_size
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(
        self,
        de_x1: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        de_x2: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        de_x3: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        de_x4: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        de_x5: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn: list[torch.Tensor],
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        c0_tra, lge_tra, t2_tra = de_x5
        c0_x4, lge_x4, t2_x4 = de_x4
        c0_x3, lge_x3, t2_x3 = de_x3
        c0_x2, lge_x2, t2_x2 = de_x2
        c0_x1, lge_x1, t2_x1 = de_x1

        patches = self.spatial_size * self.spatial_size
        attn_0 = attn[0]  # (B, heads, 4*patches, 4*patches)
        attn_fusion = attn_0[:, :, patches * self.num_modals :, :]
        attn_c0, attn_lge, attn_t2, _ = torch.chunk(attn_fusion, self.num_modals + 1, dim=-1)

        B = c0_tra.size(0)
        attn_c0 = torch.sum(torch.sum(attn_c0, dim=1), dim=-2).reshape(B, 1, self.spatial_size, self.spatial_size)
        attn_lge = torch.sum(torch.sum(attn_lge, dim=1), dim=-2).reshape(B, 1, self.spatial_size, self.spatial_size)
        attn_t2 = torch.sum(torch.sum(attn_t2, dim=1), dim=-2).reshape(B, 1, self.spatial_size, self.spatial_size)

        dex5 = (c0_tra * attn_c0, lge_tra * attn_lge, t2_tra * attn_t2)

        attn_c0, attn_lge, attn_t2 = self.upsample(attn_c0), self.upsample(attn_lge), self.upsample(attn_t2)
        dex4 = (c0_x4 * attn_c0, lge_x4 * attn_lge, t2_x4 * attn_t2)

        attn_c0, attn_lge, attn_t2 = self.upsample(attn_c0), self.upsample(attn_lge), self.upsample(attn_t2)
        dex3 = (c0_x3 * attn_c0, lge_x3 * attn_lge, t2_x3 * attn_t2)

        attn_c0, attn_lge, attn_t2 = self.upsample(attn_c0), self.upsample(attn_lge), self.upsample(attn_t2)
        dex2 = (c0_x2 * attn_c0, lge_x2 * attn_lge, t2_x2 * attn_t2)

        attn_c0, attn_lge, attn_t2 = self.upsample(attn_c0), self.upsample(attn_lge), self.upsample(attn_t2)
        dex1 = (c0_x1 * attn_c0, lge_x1 * attn_lge, t2_x1 * attn_t2)

        return dex1, dex2, dex3, dex4, dex5


class Decoder_fusion2D(nn.Module):
    """Multi-scale multimodal fusion decoder with CT, RFM, and PRM heads."""

    def __init__(self, num_cls: int = 4, basic_dims: int = 16, num_modals: int = 3):
        super().__init__()
        self.num_cls = num_cls
        self.num_modals = num_modals

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.up8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)
        self.up16 = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=False)

        # Stage 5
        self.d5_c2 = general_conv2d(basic_dims * 32, basic_dims * 16, pad_type="reflect")
        self.d5_out = general_conv2d(basic_dims * 16, basic_dims * 16, k_size=1, padding=0, pad_type="zeros")
        self.CT5 = MultiCrossToken(channel=basic_dims * 16, num_modals=num_modals)

        # Stage 4
        self.CT4 = MultiCrossToken(channel=basic_dims * 8, num_modals=num_modals)
        self.d4_c1 = general_conv2d(basic_dims * 16, basic_dims * 8, pad_type="reflect")
        self.d4_c2 = general_conv2d(basic_dims * 16, basic_dims * 8, pad_type="reflect")
        self.d4_out = general_conv2d(basic_dims * 8, basic_dims * 8, k_size=1, padding=0, pad_type="zeros")

        # Stage 3
        self.d3_c1 = general_conv2d(basic_dims * 8, basic_dims * 4, pad_type="reflect")
        self.d3_c2 = general_conv2d(basic_dims * 8, basic_dims * 4, pad_type="reflect")
        self.d3_out = general_conv2d(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type="zeros")

        # Stage 2
        self.d2_c1 = general_conv2d(basic_dims * 4, basic_dims * 2, pad_type="reflect")
        self.d2_c2 = general_conv2d(basic_dims * 4, basic_dims * 2, pad_type="reflect")
        self.d2_out = general_conv2d(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type="zeros")

        # Stage 1
        self.d1_c1 = general_conv2d(basic_dims * 2, basic_dims, pad_type="reflect")
        self.d1_c2 = general_conv2d(basic_dims * 2, basic_dims, pad_type="reflect")
        self.d1_out = general_conv2d(basic_dims, basic_dims, k_size=1, padding=0, pad_type="zeros")

        # Segmentation layer
        self.seg_layer = nn.Conv2d(basic_dims, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        # Region-aware feature modulation (RFM)
        self.RFM3 = fusion_postnorm2d(in_channel=basic_dims * 4, num_modals=num_modals)
        self.RFM2 = fusion_postnorm2d(in_channel=basic_dims * 2, num_modals=num_modals)
        self.RFM1 = fusion_postnorm2d(in_channel=basic_dims * 1, num_modals=num_modals)

        # PRM auxiliary prediction heads
        self.prm_fusion5 = prm_fusion2d(in_channel=basic_dims * 16, basic_dim=basic_dims, num_cls=num_cls)
        self.prm_fusion4 = prm_fusion2d(in_channel=basic_dims * 8, basic_dim=basic_dims, num_cls=num_cls)
        self.prm_fusion3 = prm_fusion2d(in_channel=basic_dims * 4, basic_dim=basic_dims, num_cls=num_cls)
        self.prm_fusion2 = prm_fusion2d(in_channel=basic_dims * 2, basic_dim=basic_dims, num_cls=num_cls)
        self.prm_fusion1 = prm_fusion2d(in_channel=basic_dims * 1, basic_dim=basic_dims, num_cls=num_cls)

    def forward(
        self,
        dx1: torch.Tensor,
        dx2: torch.Tensor,
        dx3: torch.Tensor,
        dx4: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        dx5: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        fusion: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        prm_pred5 = self.prm_fusion5(fusion)
        de_x5 = self.CT5(dx5, fusion, mask)
        de_x5 = torch.cat((de_x5, fusion), dim=1)
        de_x5 = self.d5_out(self.d5_c2(de_x5))
        de_x5 = self.d4_c1(self.up2(de_x5))

        prm_pred4 = self.prm_fusion4(de_x5)
        de_x4 = self.CT4(dx4, de_x5, mask)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))

        prm_pred3 = self.prm_fusion3(de_x4)
        de_x3 = self.RFM3(dx3, mask)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        prm_pred2 = self.prm_fusion2(de_x3)
        de_x2 = self.RFM2(dx2, mask)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        prm_pred1 = self.prm_fusion1(de_x2)
        de_x1 = self.RFM1(dx1, mask)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        prm_preds = (
            prm_pred1,
            self.up2(prm_pred2),
            self.up4(prm_pred3),
            self.up8(prm_pred4),
            self.up16(prm_pred5),
        )
        return pred, prm_preds


class M2FTrans2D(nn.Module):
    """M2FTrans 2D Model for MyoPS380 CMR Multimodal Segmentation.

    Interface adheres to SCAR standard contracts:
    - Accepts 3 individual modality tensors (cine, lge, t2w) of shape (B, 1, 128, 128)
      or 1 concatenated tensor of shape (B, 3, 128, 128).
    - During training (model.training = True):
      returns tuple: (fuse_pred, (c0_pred, lge_pred, t2_pred), prm_preds)
    - During evaluation (model.eval() / training = False):
      returns tensor: fuse_pred of shape (B, 4, 128, 128)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        img_size: int = 128,
        basic_dims: int = 16,
        depth: int = 3,
        num_heads: int = 8,
        mlp_dim: int = 1024,
        num_modals: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.basic_dims = basic_dims
        self.num_modals = num_modals
        self.spatial_size = img_size // 16  # 128 // 16 = 8
        self.patches = self.spatial_size * self.spatial_size  # 64

        # 1. 3 Modality Encoders
        self.c0_encoder = Encoder2D(in_ch=in_channels, basic_dims=basic_dims)
        self.lge_encoder = Encoder2D(in_ch=in_channels, basic_dims=basic_dims)
        self.t2_encoder = Encoder2D(in_ch=in_channels, basic_dims=basic_dims)

        # 2. Bottleneck Transformer
        self.Bottleneck = Bottleneck2D(
            basic_dims=basic_dims,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            num_modals=num_modals,
        )

        # 3. Skip Modulation & Decoders
        self.weight_attention = Weight_Attention2D(num_modals=num_modals, spatial_size=self.spatial_size)
        self.decoder_fusion = Decoder_fusion2D(num_cls=num_classes, basic_dims=basic_dims, num_modals=num_modals)
        self.decoder_sep = Decoder_sep2D(num_cls=num_classes, basic_dims=basic_dims)

        # 4. Learnable Positional Embedding and Fusion Token
        total_tokens = self.patches * (num_modals + 1)  # 64 * 4 = 256
        embed_dim = basic_dims * 16  # 16 * 16 = 256
        self.pos = nn.Parameter(torch.zeros(1, total_tokens, embed_dim))
        self.fusion = nn.Parameter(torch.randn(1, self.patches, embed_dim) * 0.02)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        *images: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        """Forward pass.

        Args:
            *images: Either 3 tensors (c0, lge, t2) each (B, 1, 128, 128)
                     or 1 tensor (B, 3, 128, 128).
            mask: Optional tensor (B, 3) indicating modality availability.
        """
        if len(images) == 1:
            x = images[0]
            if x.dim() != 4:
                raise ValueError(f"Input tensor must have 4 dimensions (B, C, H, W); got shape {x.shape}")
            if x.shape[1] == 3:
                c0 = x[:, 0:1, :, :]
                lge = x[:, 1:2, :, :]
                t2 = x[:, 2:3, :, :]
            elif x.shape[1] == 1:
                # Replicated single channel if passed
                c0 = lge = t2 = x
            else:
                raise ValueError(f"Expected 1 or 3 input channels; received {x.shape[1]}")
        elif len(images) == 3:
            c0, lge, t2 = images
        else:
            raise ValueError(f"Expected 1 or 3 image tensors; received {len(images)}")

        B = c0.size(0)
        device = c0.device

        # Default mask: all 3 modalities present
        if mask is None:
            mask = torch.ones((B, self.num_modals), dtype=torch.bool, device=device)

        # 1. Multi-scale feature extraction
        c0_x1, c0_x2, c0_x3, c0_x4, c0_x5 = self.c0_encoder(c0)
        lge_x1, lge_x2, lge_x3, lge_x4, lge_x5 = self.lge_encoder(lge)
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(t2)

        # 2. Bottleneck token fusion
        x_bottle = (c0_x5, lge_x5, t2_x5)
        fusion_tokens = self.fusion.expand(B, -1, -1)
        c0_trans, lge_trans, t2_trans, fusion_trans, attn = self.Bottleneck(
            x_bottle, mask, fusion_tokens, self.pos
        )

        # Reshape tokens back to spatial feature maps (B, C, 8, 8)
        C_bottle = self.basic_dims * 16
        S = self.spatial_size
        c0_tra = c0_trans.transpose(1, 2).reshape(B, C_bottle, S, S).contiguous()
        lge_tra = lge_trans.transpose(1, 2).reshape(B, C_bottle, S, S).contiguous()
        t2_tra = t2_trans.transpose(1, 2).reshape(B, C_bottle, S, S).contiguous()
        fusion_tra = fusion_trans.transpose(1, 2).reshape(B, C_bottle, S, S).contiguous()

        # 3. Skip attention weighting
        de_x5 = (c0_tra, lge_tra, t2_tra)
        de_x4 = (c0_x4, lge_x4, t2_x4)
        de_x3 = (c0_x3, lge_x3, t2_x3)
        de_x2 = (c0_x2, lge_x2, t2_x2)
        de_x1 = (c0_x1, lge_x1, t2_x1)

        de_x1, de_x2, de_x3, de_x4, de_x5 = self.weight_attention(de_x1, de_x2, de_x3, de_x4, de_x5, attn)

        # Stack modality representations for RFM layers
        stack_x3 = torch.stack(de_x3, dim=1)  # (B, 3, C, 32, 32)
        stack_x2 = torch.stack(de_x2, dim=1)  # (B, 3, C, 64, 64)
        stack_x1 = torch.stack(de_x1, dim=1)  # (B, 3, C, 128, 128)

        # 4. Decoder fusion
        fuse_pred, prm_preds = self.decoder_fusion(
            stack_x1, stack_x2, stack_x3, de_x4, de_x5, fusion_tra, mask
        )

        if self.training:
            c0_pred = self.decoder_sep(c0_x1, c0_x2, c0_x3, c0_x4, c0_x5)
            lge_pred = self.decoder_sep(lge_x1, lge_x2, lge_x3, lge_x4, lge_x5)
            t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5)
            return fuse_pred, (c0_pred, lge_pred, t2_pred), prm_preds

        return fuse_pred

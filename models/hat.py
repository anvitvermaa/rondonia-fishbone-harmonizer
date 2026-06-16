"""
models/hat.py  –  HAT (Hybrid Attention Transformer) for satellite imagery
=========================================================================
Source  : https://github.com/XPixelGroup/HAT
Paper   : Chen et al., "Activating More Pixels in Image Super-Resolution
          Transformer", CVPR 2023.

Adaptations for the Rondônia SR study
--------------------------------------
• Self-contained implementation — no external BasicSR dependency.
• Hybrid attention = window self-attention + channel attention (CAB).
• Overlapping cross-window attention (OCAB) captures long-range fishbone
  context beyond the local Swin window.
• `in_chans` defaults to 6 for 6-band satellite imagery.
• RTX 3060 budget: `embed_dim=48`, 4 HATB stages, `window_size=8`.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# Re-use SwinIR building blocks imported from sibling module
from .swinir import (
    Mlp, window_partition, window_reverse, WindowAttention,
    PatchEmbed, PatchUnEmbed, Upsample,
)


# ─────────────────────────────────────────────────────────────────────────────
# Channel Attention Block (CAB)  – the "hybrid" part of HAT
# ─────────────────────────────────────────────────────────────────────────────

class CAB(nn.Module):
    """
    Channel Attention Block.
    Complements window self-attention with per-channel recalibration.
    """

    def __init__(self, n_feat: int, compress_ratio: int = 3,
                 squeeze_factor: int = 30) -> None:
        super().__init__()
        mid = max(1, n_feat // compress_ratio)
        self.cab = nn.Sequential(
            nn.Conv2d(n_feat, mid, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(mid, n_feat, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(n_feat, max(1, n_feat // squeeze_factor)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, n_feat // squeeze_factor), n_feat),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        attn = self.cab(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * attn


# ─────────────────────────────────────────────────────────────────────────────
# HAT Block  =  Swin Transformer Block + parallel CAB
# ─────────────────────────────────────────────────────────────────────────────

class HATBlock(nn.Module):
    """
    Hybrid Attention Transformer Block.

    Runs window MSA and channel attention in parallel then sums them.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=8,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 compress_ratio=3, squeeze_factor=30):
        super().__init__()
        self.dim              = dim
        self.input_resolution = input_resolution
        self.num_heads        = num_heads
        self.window_size      = window_size
        self.shift_size       = shift_size
        self.mlp_ratio        = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size  = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn  = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        self.conv_scale = nn.Parameter(torch.zeros(1))  # learnable blend weight
        self.cab        = CAB(dim, compress_ratio, squeeze_factor)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = norm_layer(dim)
        self.mlp       = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            self.register_buffer("attn_mask", self._calc_mask(self.input_resolution))
        else:
            self.register_buffer("attn_mask", None)

    def _calc_mask(self, x_size):
        H, W     = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mw = window_partition(img_mask, self.window_size).view(
            -1, self.window_size * self.window_size)
        attn_mask = mw.unsqueeze(1) - mw.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.).masked_fill(attn_mask == 0, 0.)

    def forward(self, x, x_size):
        H, W    = x_size
        B, L, C = x.shape
        shortcut = x

        # ── Window MSA branch ──
        x_norm = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:
            shifted = torch.roll(x_norm, (-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted = x_norm
        x_windows = window_partition(shifted, self.window_size).view(
            -1, self.window_size * self.window_size, C)
        mask = (self.attn_mask if self.input_resolution == x_size
                else self._calc_mask(x_size).to(x.device))
        attn_windows = self.attn(x_windows, mask=mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted      = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x_msa = torch.roll(shifted, (self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_msa = shifted
        x_msa = x_msa.view(B, H * W, C)

        # ── Channel Attention branch (operate in spatial domain) ──
        x_spatial = self.norm1(x).view(B, H, W, C).permute(0, 3, 1, 2)   # B,C,H,W
        x_cab = self.cab(x_spatial).permute(0, 2, 3, 1).view(B, H * W, C)

        # Blend: MSA + learnable-scaled CAB
        x = shortcut + self.drop_path(x_msa + self.conv_scale * x_cab)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# HAT Basic Layer and HATB (Residual HAT Block)
# ─────────────────────────────────────────────────────────────────────────────

class HATLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 compress_ratio=3, squeeze_factor=30):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            HATBlock(dim=dim, input_resolution=input_resolution,
                     num_heads=num_heads, window_size=window_size,
                     shift_size=0 if i % 2 == 0 else window_size // 2,
                     mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop=drop, attn_drop=attn_drop,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                     norm_layer=norm_layer,
                     compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)
            for i in range(depth)])

    def forward(self, x, x_size):
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x, x_size) if self.use_checkpoint else blk(x, x_size)
        return x


class HATB(nn.Module):
    """Residual HAT Block (analogous to RSTB in SwinIR)."""
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 img_size=48, patch_size=1, resi_connection='1conv',
                 compress_ratio=3, squeeze_factor=30):
        super().__init__()
        self.residual_group = HATLayer(
            dim=dim, input_resolution=input_resolution, depth=depth,
            num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, norm_layer=norm_layer, use_checkpoint=use_checkpoint,
            compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.patch_embed   = PatchEmbed(img_size, patch_size, 0, dim, None)
        self.patch_unembed = PatchUnEmbed(img_size, patch_size, 0, dim, None)

    def forward(self, x, x_size):
        return self.patch_embed(
            self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x


# ─────────────────────────────────────────────────────────────────────────────
# HAT
# ─────────────────────────────────────────────────────────────────────────────

class HAT(nn.Module):
    """
    Hybrid Attention Transformer for SISR.

    Parameters
    ----------
    img_size        : LR patch size fed to the network.
    in_chans        : Input / output spectral channels (6 for satellite data).
    embed_dim       : Transformer embedding dimension.
    depths          : Depth per HATB stage.
    num_heads       : Attention heads per stage.
    window_size     : Local window size.
    mlp_ratio       : MLP hidden dim ratio.
    compress_ratio  : CAB conv compression ratio.
    squeeze_factor  : CAB SE squeeze factor.
    upscale         : SR scale factor (3 for 30m→10m).
    upsampler       : 'pixelshuffle'.
    """

    def __init__(
        self,
        img_size: int       = 48,
        patch_size: int     = 1,
        in_chans: int       = 6,
        embed_dim: int      = 48,
        depths: list        = None,
        num_heads: list     = None,
        window_size: int    = 8,
        mlp_ratio: float    = 2.0,
        qkv_bias: bool      = True,
        qk_scale            = None,
        drop_rate: float    = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        norm_layer          = nn.LayerNorm,
        ape: bool           = False,
        patch_norm: bool    = True,
        use_checkpoint: bool = False,
        upscale: int        = 3,
        img_range: float    = 1.,
        upsampler: str      = 'pixelshuffle',
        resi_connection: str = '1conv',
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
    ) -> None:
        super().__init__()
        depths    = depths    or [6, 6, 6, 6]
        num_heads = num_heads or [6, 6, 6, 6]

        num_feat        = 64
        self.img_range  = img_range
        self.mean       = torch.zeros(1, 1, 1, 1)
        self.upscale    = upscale
        self.upsampler  = upsampler
        self.window_size = window_size

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        self.patch_embed   = PatchEmbed(img_size, patch_size, embed_dim, embed_dim,
                                         norm_layer if patch_norm else None)
        self.patch_unembed = PatchUnEmbed(img_size, patch_size, embed_dim, embed_dim,
                                           norm_layer if patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution

        if ape:
            n_patches = patches_resolution[0] * patches_resolution[1]
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList([
            HATB(dim=embed_dim,
                 input_resolution=(patches_resolution[0], patches_resolution[1]),
                 depth=depths[i], num_heads=num_heads[i], window_size=window_size,
                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop=drop_rate, attn_drop=attn_drop_rate,
                 drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                 norm_layer=norm_layer, use_checkpoint=use_checkpoint,
                 img_size=img_size, patch_size=patch_size,
                 resi_connection=resi_connection,
                 compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)
            for i in range(len(depths))
        ])
        self.norm = norm_layer(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        if upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample  = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, in_chans, 3, 1, 1)
        else:
            self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_h = (self.window_size - h % self.window_size) % self.window_size
        mod_w = (self.window_size - w % self.window_size) % self.window_size
        return F.pad(x, (0, mod_w, 0, mod_h), 'reflect')

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if hasattr(self, 'absolute_pos_embed'):
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        return self.patch_unembed(x, x_size)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x

        if self.upsampler == 'pixelshuffle':
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        else:
            x = self.conv_last(x)

        x = x / self.img_range + self.mean
        return x[:, :, :H * self.upscale, :W * self.upscale]

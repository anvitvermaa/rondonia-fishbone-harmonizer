"""
models/swinir.py  –  SwinIR adapted for multi-band satellite imagery
====================================================================
Source  : https://github.com/JingyunLiang/SwinIR
Paper   : Liang et al., "SwinIR: Image Restoration Using Swin Transformer",
          ICCVW 2021.

Adaptations for the Rondônia SR study
--------------------------------------
• Self-contained implementation ported directly from the official repo.
• Mean shift (RGB-specific) replaced with a neutral identity path — external
  normalisation is performed by the Smart Scaling / standard dataloader.
• `in_chans` defaults to 6 for 6-band Landsat / Sentinel-2.
• RTX 3060 12 GB budget: `embed_dim=60`, `depths=[6,6,6,6]`, `window_size=8`.
• Uses `timm.models.layers` helpers (DropPath, trunc_normal_, to_2tuple).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# ─────────────────────────────────────────────────────────────────────────────
# Utility modules
# ─────────────────────────────────────────────────────────────────────────────

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


def window_partition(x, window_size):
    """(B,H,W,C) → (num_windows*B, Ws, Ws, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    """(num_windows*B, Ws, Ws, C) → (B, H, W, C)"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# ─────────────────────────────────────────────────────────────────────────────
# Window-based Multi-head Self-Attention
# ─────────────────────────────────────────────────────────────────────────────

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = qk_scale or head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        trunc_normal_(self.relative_position_bias_table, std=.02)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords   = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        rel = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += self.window_size[0] - 1
        rel[:, :, 1] += self.window_size[1] - 1
        rel[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer("relative_position_index", rel.sum(-1))

        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax   = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q   = q * self.scale
        attn = q @ k.transpose(-2, -1)

        rpe = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1],
               self.window_size[0] * self.window_size[1], -1)
        rpe = rpe.permute(2, 0, 1).contiguous()
        attn = attn + rpe.unsqueeze(0)

        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) \
                   + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.attn_drop(self.softmax(attn))

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


# ─────────────────────────────────────────────────────────────────────────────
# Swin Transformer Block
# ─────────────────────────────────────────────────────────────────────────────

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
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
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = norm_layer(dim)
        self.mlp       = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self._calc_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def _calc_mask(self, x_size):
        H, W = x_size
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
        mask_windows = window_partition(img_mask, self.window_size).view(
            -1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.).masked_fill(attn_mask == 0, 0.)

    def forward(self, x, x_size):
        H, W   = x_size
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, (-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows,
                                     mask=self._calc_mask(x_size).to(x.device))

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x    = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, (self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Basic Swin Layer, RSTB, Patch Embed/UnEmbed
# ─────────────────────────────────────────────────────────────────────────────

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x, x_size):
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x, x_size) if self.use_checkpoint else blk(x, x_size)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=48, patch_size=1, in_chans=0, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size   = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=48, patch_size=1, in_chans=0, embed_dim=96, norm_layer=None):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        return x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])


class RSTB(nn.Module):
    """Residual Swin Transformer Block."""
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 img_size=48, patch_size=1, resi_connection='1conv'):
        super().__init__()
        self.residual_group = BasicLayer(
            dim=dim, input_resolution=input_resolution, depth=depth,
            num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.conv = (nn.Conv2d(dim, dim, 3, 1, 1) if resi_connection == '1conv' else
                     nn.Sequential(
                         nn.Conv2d(dim, dim // 4, 3, 1, 1),
                         nn.LeakyReLU(0.2, True),
                         nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                         nn.LeakyReLU(0.2, True),
                         nn.Conv2d(dim // 4, dim, 3, 1, 1)))

        self.patch_embed   = PatchEmbed(img_size, patch_size, 0, dim, None)
        self.patch_unembed = PatchUnEmbed(img_size, patch_size, 0, dim, None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(
            self.patch_unembed(self.residual_group(x, x_size), x_size))) + x


# ─────────────────────────────────────────────────────────────────────────────
# Upsample helpers
# ─────────────────────────────────────────────────────────────────────────────

class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m += [nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1), nn.PixelShuffle(2)]
        elif scale == 3:
            m += [nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1), nn.PixelShuffle(3)]
        else:
            raise ValueError(f"Unsupported scale: {scale}")
        super().__init__(*m)


# ─────────────────────────────────────────────────────────────────────────────
# SwinIR
# ─────────────────────────────────────────────────────────────────────────────

class SwinIR(nn.Module):
    """
    SwinIR: Image Restoration Using Swin Transformer.

    Configured for classical SR mode ('pixelshuffle' upsampler).

    Parameters
    ----------
    img_size    : LR patch size fed to the transformer.
    in_chans    : Input / output spectral channels (6 for Landsat/S2).
    embed_dim   : Transformer embedding dimension.
    depths      : Transformer depth per stage.
    num_heads   : Attention heads per stage.
    window_size : Local attention window size.
    mlp_ratio   : MLP hidden dim ratio.
    upscale     : SR scale factor.
    upsampler   : 'pixelshuffle' (classical SR).
    """

    def __init__(
        self,
        img_size: int       = 48,
        patch_size: int     = 1,
        in_chans: int       = 6,
        embed_dim: int      = 60,
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
    ) -> None:
        super().__init__()
        depths    = depths    or [6, 6, 6, 6]
        num_heads = num_heads or [6, 6, 6, 6]

        num_feat   = 64
        self.img_range  = img_range
        self.mean       = torch.zeros(1, 1, 1, 1)   # neutral (external normalisation)
        self.upscale    = upscale
        self.upsampler  = upsampler
        self.window_size = window_size

        # Shallow feature
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # Deep feature
        self.num_layers = len(depths)
        self.embed_dim  = embed_dim
        self.ape        = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim, embed_dim,
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
            RSTB(dim=embed_dim,
                 input_resolution=(patches_resolution[0], patches_resolution[1]),
                 depth=depths[i], num_heads=num_heads[i], window_size=window_size,
                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop=drop_rate, attn_drop=attn_drop_rate,
                 drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                 norm_layer=norm_layer, use_checkpoint=use_checkpoint,
                 img_size=img_size, patch_size=patch_size,
                 resi_connection=resi_connection)
            for i in range(self.num_layers)
        ])
        self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        else:
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(0.2, True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(0.2, True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # Reconstruction
        if upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
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
        if self.ape:
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

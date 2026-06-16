"""
models/edsr.py  –  EDSR adapted for multi-band satellite imagery
================================================================
Source: https://github.com/sanghyun-son/EDSR-PyTorch
Paper : Lim et al., "Enhanced Deep Residual Networks for Single
        Image Super-Resolution", CVPRW 2017.

Adaptations
-----------
• Self-contained implementation (no external `model.common` dependency).
• `n_colors` defaults to 6 for 6-band Landsat / Sentinel-2.
• MeanShift disabled (we normalise externally with Smart Scaling / standard).
• Configurable `n_resblocks` and `n_feats` for GPU-budget experiments.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

def default_conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=kernel_size // 2, bias=bias
    )


class ResBlock(nn.Module):
    """
    Single residual block: Conv → ReLU → Conv, with res_scale.
    Removing BatchNorm is the key EDSR modification over VDSR.
    """

    def __init__(
        self,
        conv,
        n_feat: int,
        kernel_size: int = 3,
        act: nn.Module = nn.ReLU(True),
        res_scale: float = 0.1,
    ) -> None:
        super().__init__()
        layers = [
            conv(n_feat, n_feat, kernel_size),
            act,
            conv(n_feat, n_feat, kernel_size),
        ]
        self.body = nn.Sequential(*layers)
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x).mul(self.res_scale)


class Upsampler(nn.Sequential):
    """Sub-pixel convolution upsampler (PixelShuffle)."""

    def __init__(self, conv, scale: int, n_feat: int, act: bool = False) -> None:
        m = []
        if (scale & (scale - 1)) == 0:   # 2^n scales
            for _ in range(int(math.log(scale, 2))):
                m += [conv(n_feat, 4 * n_feat, 3), nn.PixelShuffle(2)]
                if act:
                    m.append(nn.ReLU(True))
        elif scale == 3:
            m += [conv(n_feat, 9 * n_feat, 3), nn.PixelShuffle(3)]
            if act:
                m.append(nn.ReLU(True))
        else:
            raise ValueError(f"Unsupported scale: {scale}")
        super().__init__(*m)


# ─────────────────────────────────────────────────────────────────────────────
# EDSR
# ─────────────────────────────────────────────────────────────────────────────

class EDSR(nn.Module):
    """
    Enhanced Deep Residual Network for SISR.

    Parameters
    ----------
    scale : int         Upscale factor (3 for 30 m → 10 m).
    n_resblocks : int   Number of residual blocks (paper: 32 for EDSR+).
    n_feats : int       Number of feature channels (paper: 256 for EDSR+).
    n_colors : int      Number of input / output image channels.
    res_scale : float   Residual scaling factor (paper: 0.1).
    """

    def __init__(
        self,
        scale: int = 3,
        n_resblocks: int = 16,
        n_feats: int = 64,
        n_colors: int = 6,
        res_scale: float = 0.1,
    ) -> None:
        super().__init__()
        conv = default_conv
        kernel_size = 3
        act = nn.ReLU(True)

        # Head
        self.head = conv(n_colors, n_feats, kernel_size)

        # Body: residual blocks
        body = [ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
                for _ in range(n_resblocks)]
        body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*body)

        # Tail: upsample + reconstruct
        self.tail = nn.Sequential(
            Upsampler(conv, scale, n_feats),
            conv(n_feats, n_colors, kernel_size),
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        return self.tail(res)

"""
models/rcan.py  –  RCAN adapted for multi-band satellite imagery
================================================================
Source: https://github.com/yulunzhang/RCAN
Paper : Zhang et al., "Image Super-Resolution Using Very Deep Residual
        Channel Attention Networks", ECCV 2018.

Adaptations
-----------
• Self-contained (no external common.py dependency).
• `n_colors` defaults to 6 for 6-band data.
• MeanShift removed; normalisation done by Smart Scaling / standard loader.
"""

import math
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (mirrors EDSR helpers to keep independence)
# ─────────────────────────────────────────────────────────────────────────────

def default_conv(in_c: int, out_c: int, ks: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, ks, padding=ks // 2, bias=bias)


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale: int, n_feat: int) -> None:
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m += [conv(n_feat, 4 * n_feat, 3), nn.PixelShuffle(2)]
        elif scale == 3:
            m += [conv(n_feat, 9 * n_feat, 3), nn.PixelShuffle(3)]
        else:
            raise ValueError(f"Unsupported scale: {scale}")
        super().__init__(*m)


# ─────────────────────────────────────────────────────────────────────────────
# Channel Attention Layer  (CALayer)
# ─────────────────────────────────────────────────────────────────────────────

class CALayer(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.
    GAP → FC(÷reduction) → ReLU → FC → Sigmoid → scale.
    """

    def __init__(self, channel: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.avg_pool(x)
        scale = self.conv_du(scale)
        return x * scale


# ─────────────────────────────────────────────────────────────────────────────
# Residual Channel Attention Block (RCAB)
# ─────────────────────────────────────────────────────────────────────────────

class RCAB(nn.Module):
    def __init__(
        self,
        conv,
        n_feat: int,
        kernel_size: int = 3,
        reduction: int = 16,
        act: nn.Module = nn.ReLU(True),
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            conv(n_feat, n_feat, kernel_size),
            act,
            conv(n_feat, n_feat, kernel_size),
            CALayer(n_feat, reduction),
        )

    def forward(self, x):
        return x + self.body(x)


# ─────────────────────────────────────────────────────────────────────────────
# Residual Group (RG)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualGroup(nn.Module):
    def __init__(
        self,
        conv,
        n_feat: int,
        kernel_size: int = 3,
        reduction: int = 16,
        n_resblocks: int = 10,
    ) -> None:
        super().__init__()
        body = [RCAB(conv, n_feat, kernel_size, reduction) for _ in range(n_resblocks)]
        body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return x + self.body(x)


# ─────────────────────────────────────────────────────────────────────────────
# RCAN
# ─────────────────────────────────────────────────────────────────────────────

class RCAN(nn.Module):
    """
    Residual Channel Attention Network.

    Parameters
    ----------
    scale       : Upscale factor (3 for 30 m → 10 m).
    n_resgroups : Number of Residual Groups.
    n_resblocks : RCABs per Residual Group.
    n_feats     : Feature channel count.
    reduction   : Channel reduction ratio in CALayer.
    n_colors    : Input / output spectral bands.
    """

    def __init__(
        self,
        scale: int       = 3,
        n_resgroups: int = 5,
        n_resblocks: int = 10,
        n_feats: int     = 64,
        reduction: int   = 16,
        n_colors: int    = 6,
    ) -> None:
        super().__init__()
        conv = default_conv
        kernel_size = 3

        # Head
        self.head = conv(n_colors, n_feats, kernel_size)

        # Body: stacked Residual Groups + long skip
        body = [
            ResidualGroup(conv, n_feats, kernel_size, reduction, n_resblocks)
            for _ in range(n_resgroups)
        ]
        body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*body)

        # Tail
        self.tail = nn.Sequential(
            Upsampler(conv, scale, n_feats),
            conv(n_feats, n_colors, kernel_size),
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        return self.tail(res)

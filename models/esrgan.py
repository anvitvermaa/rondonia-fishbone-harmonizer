"""
models/esrgan.py  –  ESRGAN / Real-ESRGAN for satellite imagery
================================================================
Source  : https://github.com/xinntao/ESRGAN
          https://github.com/xinntao/Real-ESRGAN
Paper   : Wang et al., "ESRGAN: Enhanced Super-Resolution Generative
          Adversarial Networks", ECCVW 2018.
          Wang et al., "Real-ESRGAN: Training Real-World Blind SR with
          Pure Synthetic Data", ICCVW 2021.

Adaptations
-----------
• Self-contained RRDB (Residual-in-Residual Dense Block) generator — the
  core ESRGANimprovement over SRGAN (no batch norm, dense connections).
• UNet discriminator with spectral normalisation (Real-ESRGAN style) for
  more perceptually realistic hallucination of logging-road textures.
• `n_channels` defaults to 6 for multi-band satellite imagery.
• RRDB count reduced to 16 by default to fit RTX 3060 12 GB VRAM.
"""

from __future__ import annotations

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Dense Block → Residual Dense Block → RRDB
# ─────────────────────────────────────────────────────────────────────────────

class DenseLayer(nn.Module):
    def __init__(self, in_c: int, growth_c: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, growth_c, 3, 1, 1)
        self.act  = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class ResidualDenseBlock(nn.Module):
    """
    5-layer dense block with residual scaling (β = 0.2).
    Each layer concatenates all previous feature maps.
    """

    def __init__(self, n_feat: int = 64, growth_c: int = 32, beta: float = 0.2) -> None:
        super().__init__()
        self.beta = beta
        self.d1 = nn.Conv2d(n_feat,             growth_c, 3, 1, 1)
        self.d2 = nn.Conv2d(n_feat + growth_c,  growth_c, 3, 1, 1)
        self.d3 = nn.Conv2d(n_feat + 2*growth_c, growth_c, 3, 1, 1)
        self.d4 = nn.Conv2d(n_feat + 3*growth_c, growth_c, 3, 1, 1)
        self.d5 = nn.Conv2d(n_feat + 4*growth_c, n_feat,   3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # Initialise with small weights for training stability
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x1  = self.act(self.d1(x))
        x2  = self.act(self.d2(torch.cat([x,  x1], dim=1)))
        x3  = self.act(self.d3(torch.cat([x,  x1, x2], dim=1)))
        x4  = self.act(self.d4(torch.cat([x,  x1, x2, x3], dim=1)))
        x5  = self.d5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x5 * self.beta + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block: 3 RDB with outer residual."""

    def __init__(self, n_feat: int = 64, growth_c: int = 32, beta: float = 0.2) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(n_feat, growth_c, beta)
        self.rdb2 = ResidualDenseBlock(n_feat, growth_c, beta)
        self.rdb3 = ResidualDenseBlock(n_feat, growth_c, beta)
        self.beta = beta

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * self.beta + x


# ─────────────────────────────────────────────────────────────────────────────
# RRDBNet  –  ESRGAN Generator
# ─────────────────────────────────────────────────────────────────────────────

class RRDBNet(nn.Module):
    """
    ESRGAN / Real-ESRGAN generator.

    Parameters
    ----------
    scale       : Upscale factor.  Power-of-2 or 3.
    n_rrdb      : Number of RRDB blocks (paper: 23; use 16 for RTX 3060).
    n_feat      : Feature channels (paper: 64).
    growth_c    : Growth channels in dense blocks (paper: 32).
    n_channels  : Input / output spectral channels.
    """

    def __init__(
        self,
        scale: int     = 3,
        n_rrdb: int    = 16,
        n_feat: int    = 64,
        growth_c: int  = 32,
        n_channels: int = 6,
    ) -> None:
        super().__init__()

        self.scale = scale
        self.conv_first = nn.Conv2d(n_channels, n_feat, 3, 1, 1)

        # RRDB trunk
        trunk = [RRDB(n_feat, growth_c) for _ in range(n_rrdb)]
        trunk.append(nn.Conv2d(n_feat, n_feat, 3, 1, 1))
        self.trunk = nn.Sequential(*trunk)

        # Upsampling via nearest-neighbour + conv (Real-ESRGAN approach: less ringing)
        upsample_layers: list[nn.Module] = []
        n_ups = _n_upsample_steps(scale)
        for _ in range(n_ups):
            upsample_layers += [
                nn.Conv2d(n_feat, n_feat, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.upsample = nn.Sequential(*upsample_layers)

        self.conv_hr   = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(n_feat, n_channels, 3, 1, 1)
        self.act       = nn.LeakyReLU(0.2, inplace=True)

    def _upsample(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        return F.interpolate(x, scale_factor=factor, mode="nearest")

    def forward(self, x):
        feat = self.conv_first(x)
        trunk = self.trunk(feat)
        feat = feat + trunk

        # Nearest-neighbour upsample between each conv layer
        for i, layer in enumerate(self.upsample):
            if i % 2 == 0:    # before each conv, do NN upsample
                # For scale=3 we do a single ×3 upsample; for scale=4 do two ×2
                factor = self.scale if _n_upsample_steps(self.scale) == 1 else 2
                feat = self._upsample(feat, factor)
            feat = layer(feat)

        feat = self.act(self.conv_hr(feat))
        return self.conv_last(feat)


def _n_upsample_steps(scale: int) -> int:
    import math
    if scale == 3:
        return 1
    return int(math.log2(scale))


# ─────────────────────────────────────────────────────────────────────────────
# UNet Discriminator with Spectral Norm  (Real-ESRGAN discriminator)
# ─────────────────────────────────────────────────────────────────────────────

class UNetDiscriminatorSN(nn.Module):
    """
    U-Net discriminator with spectral normalisation.

    Equivalent to the Real-ESRGAN discriminator.  The skip connections allow
    both global (low-freq) and local (high-freq logging road) feedback to the
    generator.

    Parameters
    ----------
    n_channels : Spectral channels (6 for 6-band imagery).
    base_feat  : Base feature channels.
    """

    def __init__(self, n_channels: int = 6, base_feat: int = 64) -> None:
        super().__init__()
        SN = functools.partial(nn.utils.spectral_norm)
        f = base_feat

        # Encoder
        self.conv0   = nn.Conv2d(n_channels, f, 3, 1, 1)
        self.conv1   = SN(nn.Conv2d(f,     f * 2, 4, 2, 1))
        self.conv2   = SN(nn.Conv2d(f * 2, f * 4, 4, 2, 1))
        self.conv3   = SN(nn.Conv2d(f * 4, f * 8, 4, 2, 1))

        # Decoder / skip connections
        self.convd3  = SN(nn.Conv2d(f * 8, f * 4, 3, 1, 1))
        self.convd2  = SN(nn.Conv2d(f * 4, f * 2, 3, 1, 1))
        self.convd1  = SN(nn.Conv2d(f * 2, f,     3, 1, 1))
        self.conv_last = nn.Conv2d(f, 1, 3, 1, 1)

        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        f0 = self.act(self.conv0(x))
        f1 = self.act(self.conv1(f0))
        f2 = self.act(self.conv2(f1))
        f3 = self.act(self.conv3(f2))

        # Decoder with skip connections (bilinear upsample)
        d3 = self.act(self.convd3(f3))
        d3 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False) + f2
        d2 = self.act(self.convd2(d3))
        d2 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False) + f1
        d1 = self.act(self.convd1(d2))
        d1 = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False) + f0

        return self.conv_last(d1)

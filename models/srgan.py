"""
models/srgan.py  –  SRGAN generator + discriminator for satellite imagery
=========================================================================
Source  : https://github.com/ESAOpenSR/SRGAN  (ESA adaptation)
Paper   : Ledig et al., "Photo-Realistic Single Image Super-Resolution Using
          a Generative Adversarial Network", CVPR 2017.

Adaptations for the Rondônia SR study
--------------------------------------
• `SRResNet` generator works on N-band (default 6) satellite imagery.
• `Discriminator` is a PatchGAN variant (more stable on texture-rich imagery).
• Perceptual loss uses band-agnostic feature maps from a 1×1 projection layer
  since VGG is RGB-only; we project N bands → 3 channels for VGG features.
• Spectral normalisation on discriminator for training stability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights


# ─────────────────────────────────────────────────────────────────────────────
# Residual Block (no BatchNorm in the generator per SRGAN paper)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, n_feat: int = 64) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 3, 1, 1),
            nn.BatchNorm2d(n_feat),
            nn.PReLU(),
            nn.Conv2d(n_feat, n_feat, 3, 1, 1),
            nn.BatchNorm2d(n_feat),
        )

    def forward(self, x):
        return x + self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
# SRResNet / SRGAN Generator
# ─────────────────────────────────────────────────────────────────────────────

class SRResNet(nn.Module):
    """
    SRGAN Generator backbone (SRResNet).

    Parameters
    ----------
    scale         : Upscale factor.
    n_resblocks   : Number of residual blocks (paper: 16).
    n_feat        : Feature channels (paper: 64).
    n_channels    : Input / output spectral channels.
    """

    def __init__(
        self,
        scale: int = 3,
        n_resblocks: int = 16,
        n_feat: int = 64,
        n_channels: int = 6,
    ) -> None:
        super().__init__()

        # Initial conv
        self.conv_in = nn.Sequential(
            nn.Conv2d(n_channels, n_feat, 9, 1, 4),
            nn.PReLU(),
        )

        # Residual trunk
        trunk = [ResidualBlock(n_feat) for _ in range(n_resblocks)]
        trunk += [nn.Conv2d(n_feat, n_feat, 3, 1, 1), nn.BatchNorm2d(n_feat)]
        self.trunk = nn.Sequential(*trunk)

        # PixelShuffle upsample blocks
        upsample = []
        for _ in range(_log2_steps(scale)):
            upsample += [
                nn.Conv2d(n_feat, n_feat * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.PReLU(),
            ]
        if scale == 3:
            upsample = [
                nn.Conv2d(n_feat, n_feat * 9, 3, 1, 1),
                nn.PixelShuffle(3),
                nn.PReLU(),
            ]
        self.upsample = nn.Sequential(*upsample)

        self.conv_out = nn.Conv2d(n_feat, n_channels, 9, 1, 4)

    def forward(self, x):
        feat = self.conv_in(x)
        trunk = self.trunk(feat)
        feat = feat + trunk
        feat = self.upsample(feat)
        return self.conv_out(feat)


def _log2_steps(scale: int) -> int:
    """Number of ×2 PixelShuffle steps for power-of-2 scales."""
    import math
    return int(math.log2(scale)) if (scale & (scale - 1)) == 0 else 0


# ─────────────────────────────────────────────────────────────────────────────
# PatchGAN Discriminator with Spectral Normalisation
# ─────────────────────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    PatchGAN discriminator with spectral normalisation.

    Operates on HR-spatial patches; outputs a score map rather than a
    single scalar → more granular gradient signal for fishbone texture.

    Parameters
    ----------
    n_channels : Input spectral channels.
    base_feat  : Base feature count (doubles per stride-2 layer).
    """

    def __init__(self, n_channels: int = 6, base_feat: int = 64) -> None:
        super().__init__()

        def disc_block(in_c, out_c, stride=2, first=False):
            layers: list[nn.Module] = [
                nn.utils.spectral_norm(
                    nn.Conv2d(in_c, out_c, 4, stride, 1, bias=False)
                )
            ]
            if not first:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        f = base_feat
        self.net = nn.Sequential(
            *disc_block(n_channels, f,     stride=2, first=True),
            *disc_block(f,          f,     stride=1),
            *disc_block(f,          f * 2, stride=2),
            *disc_block(f * 2,      f * 2, stride=1),
            *disc_block(f * 2,      f * 4, stride=2),
            *disc_block(f * 4,      f * 4, stride=1),
            *disc_block(f * 4,      f * 8, stride=2),
            *disc_block(f * 8,      f * 8, stride=1),
            nn.Conv2d(f * 8, 1, 4, 1, 1),   # patch output map
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# VGG Perceptual Loss with N-band projection
# ─────────────────────────────────────────────────────────────────────────────

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 feature maps (relu3_4 layer).

    Because VGG19 expects 3-channel RGB input we project the N-band image
    to 3 channels with a learned 1×1 convolution.  This keeps the perceptual
    loss spectral-band-agnostic.
    """

    def __init__(self, n_channels: int = 6, freeze_vgg: bool = True) -> None:
        super().__init__()
        # Learnable spectral → RGB projection
        self.band_proj = nn.Conv2d(n_channels, 3, 1, bias=False)
        nn.init.xavier_uniform_(self.band_proj.weight)

        # VGG19 feature extractor up to relu3_4 (layer index 18)
        vgg = vgg19(weights=VGG19_Weights.DEFAULT)
        self.feat_extractor = nn.Sequential(*list(vgg.features)[:18])
        if freeze_vgg:
            for p in self.feat_extractor.parameters():
                p.requires_grad = False

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        sr_rgb = self.band_proj(sr)
        hr_rgb = self.band_proj(hr)
        # Normalise for VGG statistics (ImageNet mean/std)
        mean = torch.tensor([0.485, 0.456, 0.406], device=sr.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=sr.device).view(1, 3, 1, 1)
        sr_rgb = (sr_rgb - mean) / std
        hr_rgb = (hr_rgb - mean) / std
        feat_sr = self.feat_extractor(sr_rgb)
        feat_hr = self.feat_extractor(hr_rgb.detach())
        return F.l1_loss(feat_sr, feat_hr)

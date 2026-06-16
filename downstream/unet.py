"""
downstream/unet.py  –  Rondônia SR Study  –  Lightweight U-Net Segmenter
=========================================================================

A 4-level encoder-decoder U-Net (~500K parameters) for semantic segmentation
of super-resolved satellite patches.

Architecture
------------
Input:   (B, 6, H, W)   6-band multispectral SR / HR patch
Output:  (B, n_classes, H, W)  class logits

Encoder:
  Level 0: 6  → 32  → 32   (no pooling — spatial resolution preserved)
  Level 1: 32 → 64  → 64   (pool → H/2,  W/2)
  Level 2: 64 → 128 → 128  (pool → H/4,  W/4)
  Level 3: 128→ 256 → 256  (pool → H/8,  W/8)

Bottleneck:
  256 → 512 → 512

Decoder:
  Level 3: upsample + skip(256) → 512+256=768 → 256
  Level 2: upsample + skip(128) → 256+128=384 → 128
  Level 1: upsample + skip(64)  → 128+64=192  → 64
  Level 0: upsample + skip(32)  → 64+32=96    → 32

Head:  32 → n_classes (1×1 conv, no activation)

Total parameters (6-band input, 3 classes): ~640K

Notes
-----
• BatchNorm used throughout (safe with batch_size >= 4).
• Designed to train in < 30 minutes per SR model on RTX 3060 / T4.
• dropout=0.1 on bottleneck for mild regularisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    """Two Conv2d-BN-ReLU blocks (the basic U-Net building block)."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpConv(nn.Module):
    """Bilinear upsample + 1×1 channel-halving conv (avoids checkerboard artefacts)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class UNet(nn.Module):
    """
    Lightweight 4-level U-Net for multispectral SR patch segmentation.

    Parameters
    ----------
    in_channels : int    Number of input spectral bands (default 6).
    n_classes   : int    Number of output classes (default 3: forest/deforestation/road).
    base_feat   : int    Feature channels at the first level (default 32).
    dropout     : float  Dropout probability on bottleneck (default 0.1).
    """

    def __init__(
        self,
        in_channels: int = 6,
        n_classes:   int = 3,
        base_feat:   int = 32,
        dropout:     float = 0.1,
    ) -> None:
        super().__init__()

        f = base_feat  # 32

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc0 = _DoubleConv(in_channels, f)          # → (B, f,   H,   W)
        self.enc1 = _DoubleConv(f,     f * 2)            # → (B, 2f,  H/2, W/2)
        self.enc2 = _DoubleConv(f * 2, f * 4)            # → (B, 4f,  H/4, W/4)
        self.enc3 = _DoubleConv(f * 4, f * 8)            # → (B, 8f,  H/8, W/8)

        self.pool = nn.MaxPool2d(2, 2)

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = _DoubleConv(f * 8, f * 16, dropout=dropout)  # → (B, 16f, H/16, W/16)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up3   = _UpConv(f * 16, f * 8)              # → (B, 8f,  H/8, W/8)
        self.dec3  = _DoubleConv(f * 16, f * 8)          # skip(8f) + up(8f) → 8f

        self.up2   = _UpConv(f * 8,  f * 4)              # → (B, 4f,  H/4, W/4)
        self.dec2  = _DoubleConv(f * 8,  f * 4)          # skip(4f) + up(4f) → 4f

        self.up1   = _UpConv(f * 4,  f * 2)              # → (B, 2f,  H/2, W/2)
        self.dec1  = _DoubleConv(f * 4,  f * 2)          # skip(2f) + up(2f) → 2f

        self.up0   = _UpConv(f * 2,  f)                  # → (B, f,   H,   W)
        self.dec0  = _DoubleConv(f * 2,  f)              # skip(f)  + up(f)  → f

        # ── Segmentation Head ─────────────────────────────────────────────────
        self.head = nn.Conv2d(f, n_classes, 1)           # 1×1 conv, no activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s0 = self.enc0(x)                    # (B, f,   H,   W)
        s1 = self.enc1(self.pool(s0))        # (B, 2f,  H/2, W/2)
        s2 = self.enc2(self.pool(s1))        # (B, 4f,  H/4, W/4)
        s3 = self.enc3(self.pool(s2))        # (B, 8f,  H/8, W/8)

        # Bottleneck
        b = self.bottleneck(self.pool(s3))   # (B, 16f, H/16, W/16)

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b),  s3], dim=1))   # → 8f
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))   # → 4f
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))   # → 2f
        d0 = self.dec0(torch.cat([self.up0(d1), s0], dim=1))   # → f

        return self.head(d0)                 # (B, n_classes, H, W)

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check (run: python -m downstream.unet)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = UNet(in_channels=6, n_classes=3, base_feat=32)
    x = torch.randn(2, 6, 144, 144)
    y = model(x)
    assert y.shape == (2, 3, 144, 144), f"Output shape mismatch: {y.shape}"
    print(f"  U-Net output shape : {y.shape}")
    print(f"  Trainable params   : {model.count_parameters():,}")
    print("  ✓ UNet sanity check passed.")

"""
models/srcnn.py  –  SRCNN adapted for multi-band satellite imagery
===================================================================
Source: https://github.com/yjn870/SRCNN-pytorch
Paper : Dong et al., "Image Super-Resolution Using Deep Convolutional
        Networks", ECCV 2014.

Adaptations for the Rondônia SR study
--------------------------------------
• `num_channels` defaults to 6 to handle 6-band Landsat / Sentinel-2 stacks.
• The network operates on bicubic-upsampled LR inputs (expects HR-size input).
• Kernel sizes kept at 9-5-5 per the paper.
"""

import torch.nn as nn


class SRCNN(nn.Module):
    """
    Three-layer super-resolution CNN.

    Architecture
    ------------
    Conv(9) → ReLU → Conv(5) → ReLU → Conv(5)
    Input : bicubic-upsampled LR image, shape (B, C, H_hr, W_hr)
    Output: refined SR image, same shape.
    """

    def __init__(self, num_channels: int = 6) -> None:
        super().__init__()
        # Patch extraction & representation
        self.conv1 = nn.Conv2d(num_channels, 64,  kernel_size=9, padding=4)
        # Non-linear mapping
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        # Reconstruction
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.conv3(x)

"""
downstream/__init__.py  –  Rondônia SR Downstream Segmentation Task
"""
from .unet import UNet
from .labels import compute_ndvi_pseudo_labels

__all__ = ["UNet", "compute_ndvi_pseudo_labels"]

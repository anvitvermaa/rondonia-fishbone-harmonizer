"""
downstream/labels.py  –  Rondônia SR Study  –  Pseudo-label Generator
======================================================================

Generates pixel-level semantic labels from spectral indices on Sentinel-2
HR patches.  Used to create the training/evaluation labels for the downstream
segmentation task WITHOUT requiring external shapefiles.

Classes
-------
0  intact_forest    NDVI >= 0.5  (dense Amazonian canopy)
1  deforestation    0.1 <= NDVI < 0.5  (cleared/degraded, bare soil)
2  logging_road     NDVI < 0.1 AND high SWIR edge response (linear features)

Usage
-----
    from downstream.labels import compute_ndvi_pseudo_labels
    seg_mask = compute_ndvi_pseudo_labels(hr_patch)  # (H, W) uint8
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import sobel


# Band indices (0-based) in the 6-band Sentinel-2 stack
# Stack order from download_data.py: Blue(0), Green(1), Red(2), NIR(3), SWIR1(4), SWIR2(5)
DEFAULT_NIR_BAND   = 3
DEFAULT_RED_BAND   = 2
DEFAULT_GREEN_BAND = 1
DEFAULT_SWIR1_BAND = 4

# NDVI thresholds calibrated for dry-season Amazonian imagery
NDVI_FOREST_THRESH = 0.50   # dense forest (OFC: intact)
NDVI_DEFOR_THRESH  = 0.10   # bare soil / pasture (OFC: deforestation)
# NDVI < NDVI_DEFOR_THRESH + high SWIR edge = logging road (class 2)

# SWIR edge percentile: pixels above this edge magnitude are classified as roads
EDGE_PERCENTILE = 85


def compute_ndvi_pseudo_labels(
    hr_patch: np.ndarray,
    nir_band:          int   = DEFAULT_NIR_BAND,
    red_band:          int   = DEFAULT_RED_BAND,
    green_band:        int   = DEFAULT_GREEN_BAND,
    swir1_band:        int   = DEFAULT_SWIR1_BAND,
    ndvi_forest_thresh: float = NDVI_FOREST_THRESH,
    ndvi_defor_thresh:  float = NDVI_DEFOR_THRESH,
    edge_percentile:   float = EDGE_PERCENTILE,
) -> np.ndarray:
    """
    Generate pixel-level pseudo-labels from NDVI/SWIR thresholds.

    Parameters
    ----------
    hr_patch : np.ndarray
        Shape (C, H, W), raw (unnormalised) Sentinel-2 reflectance values.
        Must have at least 5 bands (Blue/Green/Red/NIR/SWIR1).
    nir_band, red_band, green_band, swir1_band : int
        0-based band indices.
    ndvi_forest_thresh : float
        NDVI >= this threshold → class 0 (intact forest).
    ndvi_defor_thresh : float
        NDVI in [ndvi_defor_thresh, ndvi_forest_thresh) → class 1 (deforestation).
        NDVI < ndvi_defor_thresh → class 1 or 2 depending on edge magnitude.
    edge_percentile : float
        SWIR1 edge magnitude percentile above which a pixel is classified as
        a logging road (class 2) rather than bare soil (class 1).

    Returns
    -------
    np.ndarray
        Shape (H, W), dtype uint8. Values: 0 (forest), 1 (deforestation), 2 (road).

    Notes
    -----
    • These are PSEUDO-labels derived from spectral indices, not ground truth.
      They are suitable for demonstrating SR benefit on downstream tasks, but
      results should be validated against PRODES/MapBiomas for final publication.
    • Water bodies (NDWI > 0.2) are assigned class 1 (deforestation proxy)
      since the SR study focuses on terrestrial land cover.
    """
    C = hr_patch.shape[0]

    # Guard: clamp band indices to valid range
    nir_b   = min(nir_band,   C - 1)
    red_b   = min(red_band,   C - 1)
    green_b = min(green_band, C - 1)
    swir1_b = min(swir1_band, C - 1)

    eps = 1e-8
    nir   = hr_patch[nir_b].astype(np.float32)
    red   = hr_patch[red_b].astype(np.float32)
    green = hr_patch[green_b].astype(np.float32)
    swir1 = hr_patch[swir1_b].astype(np.float32)

    # ── NDVI (Normalized Difference Vegetation Index) ────────────────────────
    ndvi = (nir - red) / (nir + red + eps)

    # ── SWIR1 Edge Magnitude (Sobel filter) ──────────────────────────────────
    # Logging roads appear as high-contrast linear features in SWIR1.
    swir_norm = (swir1 - swir1.min()) / (swir1.max() - swir1.min() + eps)
    sx = sobel(swir_norm, axis=0)
    sy = sobel(swir_norm, axis=1)
    edge_mag  = np.sqrt(sx ** 2 + sy ** 2)
    high_edge = edge_mag > np.percentile(edge_mag, edge_percentile)

    # ── Label Assignment ─────────────────────────────────────────────────────
    # Order matters: later assignments overwrite earlier ones.
    labels = np.ones(hr_patch.shape[1:], dtype=np.uint8)  # default: deforestation

    # Class 0: intact forest
    labels[ndvi >= ndvi_forest_thresh] = 0

    # Class 1: deforestation / bare soil (default for mid-NDVI)
    labels[(ndvi >= ndvi_defor_thresh) & (ndvi < ndvi_forest_thresh)] = 1

    # Class 2: logging road (very low NDVI + high SWIR edge response)
    labels[(ndvi < ndvi_defor_thresh) & high_edge] = 2

    # Class 1: bare soil (very low NDVI but no road signature)
    labels[(ndvi < ndvi_defor_thresh) & ~high_edge] = 1

    return labels


def compute_class_distribution(seg_mask: np.ndarray) -> dict:
    """
    Return fractional distribution of classes in a label mask.

    Parameters
    ----------
    seg_mask : np.ndarray  Shape (H, W), dtype uint8.

    Returns
    -------
    dict  { 'forest': float, 'deforestation': float, 'logging_road': float }
    """
    total = seg_mask.size
    return {
        "forest":        float(np.sum(seg_mask == 0) / total),
        "deforestation": float(np.sum(seg_mask == 1) / total),
        "logging_road":  float(np.sum(seg_mask == 2) / total),
    }

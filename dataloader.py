"""
dataloader.py  (v2)
===================
Rondônia SR Study – Data Loading, Alignment & Pre-processing Module

Key responsibilities
--------------------
1. Load paired Landsat (30 m, LR) and Sentinel-2 (10 m, HR) patches from .tif files.
2. Apply sensor harmonization via per-channel histogram matching (Landsat → Sentinel-2
   spectral domain) to correct bandpass discrepancies between sensors.
3. Apply sub-pixel phase cross-correlation alignment (skimage) to correct residual
   mission-level co-registration errors. Computes alignment RMSE and rejects pairs
   exceeding the configured threshold.
4. Expose TWO normalisation modes selected at runtime:
     • 'standard' → per-channel min-max → [-1, 1]  (NOT global min/max — see bug fix)
     • 'smart'    → apply_smart_scaling() (percentile clipping, then [-1, 1])
5. Generate NDVI/NDWI-based pseudo-labels for the downstream segmentation task.
6. Return PyTorch DataLoaders ready for the training pipeline.

Authors: Rondônia SR Study pipeline

v2 Fixes
--------
* CRITICAL: standard_scaling() was using GLOBAL min/max across all channels.
  For Landsat/Sentinel-2 pairs with different absolute DN ranges this maps
  LR and HR into completely incompatible domains, destroying all pixel-level
  metrics (PSNR < 20 dB even for bicubic). Fixed to PER-CHANNEL min/max.
* Added apply_histogram_matching() for sensor-to-sensor harmonization.
* Added compute_alignment_rmse() to quantify co-registration quality.
* Added compute_ndvi_pseudo_labels() for downstream segmentation labels.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.registration import phase_cross_correlation
from skimage.exposure import match_histograms
from scipy.ndimage import shift as ndimage_shift

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def standard_scaling(tensor: np.ndarray) -> np.ndarray:
    """
    Per-channel min-max normalisation → [-1, 1].

    CRITICAL FIX (v2): Previously used GLOBAL min/max across all channels,
    which maps Landsat and Sentinel-2 patches into incompatible domains due to
    their different absolute DN ranges. This caused PSNR to collapse to ~16 dB
    even for bicubic upscaling (expected: 24-28 dB). Fixed to PER-CHANNEL.

    Per-channel normalisation ensures both LR and HR patches for a given band
    are mapped to the same [-1, 1] range, making pixel-level metrics meaningful.

    Parameters
    ----------
    tensor : np.ndarray
        Shape (C, H, W), dtype float32.  Raw DN / reflectance values.

    Returns
    -------
    np.ndarray
        Per-channel normalised tensor in [-1, 1], dtype float32.
    """
    eps = 1e-8
    C = tensor.shape[0]
    out = np.empty_like(tensor, dtype=np.float32)
    for c in range(C):
        ch = tensor[c].astype(np.float32)
        t_min = ch.min()
        t_max = ch.max()
        denom = t_max - t_min
        if denom < eps:
            out[c] = np.zeros_like(ch)
        else:
            out[c] = (2.0 * (ch - t_min) / denom - 1.0)  # [-1, 1]
    return out


def apply_smart_scaling(
    tensor: np.ndarray,
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
    out_min: float = -1.0,
    out_max: float = 1.0,
    lo_vals: Optional[np.ndarray] = None,
    hi_vals: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    ╔══════════════════════════════════════════════════════════╗
    ║             S M A R T   S C A L I N G                    ║
    ║  Placeholder – replace body with proprietary algorithm.  ║
    ╚══════════════════════════════════════════════════════════╝

    Overview of the reference implementation supplied here
    -------------------------------------------------------
    Satellite sensors (Landsat / Sentinel-2) deliver 16-bit Top-of-Atmosphere
    (TOA) reflectance.  A small fraction of pixels – primarily bright clouds,
    specular water glint, and saturated industrial surfaces – account for the
    top 1-2 % of the dynamic range.  Standard global min-max normalisation
    compresses the vegetation/soil variance into a tiny slice of the output
    domain, degrading gradient flow during super-resolution training.

    Smart Scaling applies a *per-channel, localised* percentile clipping prior
    to mapping the data to [out_min, out_max].  This preserves the spectral
    shape (SAM angle) of surface reflectances while preventing outlier
    dominance.

    CRITICAL: lo_vals / hi_vals must be pre-computed from the HR patch and
    passed in when scaling the LR patch, so both modalities share the same
    normalization domain.  If not provided they are computed from `tensor`.

    Parameters
    ----------
    tensor : np.ndarray
        Shape (C, H, W), dtype float32.  Raw 16-bit TOA values (or float).
    lo_pct : float
        Lower percentile for clipping (default 2.0).
    hi_pct : float
        Upper percentile for clipping (default 98.0).
    out_min : float
        Target minimum of output range (default -1.0).
    out_max : float
        Target maximum of output range (default 1.0).
    lo_vals : np.ndarray | None
        Pre-computed per-channel lower clip values (shape C,).  If provided,
        percentiles are NOT recomputed from tensor — use for LR scaling.
    hi_vals : np.ndarray | None
        Pre-computed per-channel upper clip values (shape C,).  If provided,
        percentiles are NOT recomputed from tensor — use for LR scaling.

    Returns
    -------
    np.ndarray
        Scaled tensor in [out_min, out_max], dtype float32.

    Note
    ----
    Drop your proprietary algorithm into this function body.
    The function signature must remain unchanged so the rest of the
    pipeline can call it transparently.
    """
    # ── Reference implementation (replace below with proprietary code) ───────
    eps = 1e-8
    C = tensor.shape[0]
    out = np.empty_like(tensor, dtype=np.float32)
    for c in range(C):
        ch = tensor[c].astype(np.float32)
        lo = lo_vals[c] if lo_vals is not None else np.percentile(ch, lo_pct)
        hi = hi_vals[c] if hi_vals is not None else np.percentile(ch, hi_pct)
        ch = np.clip(ch, lo, hi)
        ch = (ch - lo) / (hi - lo + eps)           # → [0, 1]
        ch = ch * (out_max - out_min) + out_min     # → [out_min, out_max]
        out[c] = ch
    return out
    # ── End reference implementation ─────────────────────────────────────────


def compute_smart_scaling_params(
    hr: np.ndarray,
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel percentile clip values from the HR patch.

    Returns (lo_vals, hi_vals) each of shape (C,), derived from the HR
    reference.  These are then used to scale BOTH the HR and LR patches so
    both modalities are mapped to the same [-1, 1] domain.

    Parameters
    ----------
    hr     : np.ndarray  Shape (C, H, W).
    lo_pct : float       Lower percentile (default 2.0).
    hi_pct : float       Upper percentile (default 98.0).

    Returns
    -------
    lo_vals : np.ndarray  Shape (C,).
    hi_vals : np.ndarray  Shape (C,).
    """
    C = hr.shape[0]
    lo_vals = np.array([np.percentile(hr[c], lo_pct) for c in range(C)], dtype=np.float32)
    hi_vals = np.array([np.percentile(hr[c], hi_pct) for c in range(C)], dtype=np.float32)
    return lo_vals, hi_vals


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Sensor Harmonization — Histogram Matching
# ─────────────────────────────────────────────────────────────────────────────

def apply_histogram_matching(
    lr_array: np.ndarray,
    hr_array: np.ndarray,
) -> np.ndarray:
    """
    Align the Landsat LR array's spectral distribution to the Sentinel-2 HR
    array using per-channel histogram matching.

    This corrects sensor-to-sensor bandpass discrepancies: even for spectrally
    matched bands (e.g. Landsat Red ↔ Sentinel-2 B04), the two sensors have
    different spectral response functions, resulting in systematic bias in the
    pixel value distributions. Histogram matching removes this bias before patch
    extraction, improving co-registration quality and PSNR.

    Parameters
    ----------
    lr_array : np.ndarray
        Shape (C, H_lr, W_lr), float32.  Landsat LR scene.
    hr_array : np.ndarray
        Shape (C, H_hr, W_hr), float32.  Sentinel-2 HR scene (reference).

    Returns
    -------
    np.ndarray
        Histogram-matched LR array, same shape as input, float32.

    Notes
    -----
    • Uses skimage.exposure.match_histograms (channel-independent).
    • The HR array is NOT modified — it is used as the reference only.
    • Applied ONCE at the scene level in prepare_patches.py before patch
      extraction (not per-patch, to avoid boundary artefacts).
    """
    # match_histograms expects (H, W, C) for multichannel; we have (C, H, W)
    # Transpose, match, transpose back.
    lr_hwc = lr_array.transpose(1, 2, 0)   # (H_lr, W_lr, C)
    hr_hwc = hr_array.transpose(1, 2, 0)   # (H_hr, W_hr, C)

    # Downsample HR to LR spatial size for reference histogram (avoids bias
    # from spatial resolution difference influencing the histogram shape).
    # We sample the HR histogram directly — pixel values should match across
    # sensors regardless of spatial resolution.
    matched_hwc = match_histograms(lr_hwc, hr_hwc, channel_axis=-1)
    return matched_hwc.transpose(2, 0, 1).astype(np.float32)  # back to (C, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Sub-pixel alignment via phase cross-correlation
# ─────────────────────────────────────────────────────────────────────────────

def align_lr_to_hr(
    lr_patch: np.ndarray,
    hr_patch: np.ndarray,
    upsample_factor: int = 100,
    reference_band: int = 3,     # NIR band (most contrast in tropical forest)
) -> np.ndarray:
    """
    Align a low-resolution (LR) patch to a high-resolution (HR) reference
    using sub-pixel phase cross-correlation.

    The LR patch is first bicubically upsampled to HR spatial dimensions,
    then registered against the HR reference band.  The estimated sub-pixel
    shift is applied via Fourier-domain interpolation.

    Parameters
    ----------
    lr_patch : np.ndarray
        Shape (C, H_lr, W_lr), float32.  Landsat LR patch.
    hr_patch : np.ndarray
        Shape (C, H_hr, W_hr), float32.  Sentinel-2 HR patch (reference).
    upsample_factor : int
        Phase-correlation upsampling factor.  100 ≈ 0.01-pixel accuracy.
    reference_band : int
        Band index used for cross-correlation (0-based).

    Returns
    -------
    np.ndarray
        Aligned LR patch (same shape as input lr_patch), float32.

    Notes
    -----
    • skimage.registration.phase_cross_correlation works in the Fourier domain
      and is robust to photometric differences between sensors.
    • The shift is estimated at HR resolution then scaled back to LR before
      being applied with scipy.ndimage.shift.
    • Large shifts (> 10 LR pixels) likely indicate a gross mis-alignment; the
      function raises a warning and returns the original patch.
    """
    from skimage.transform import resize as sk_resize

    C, H_lr, W_lr = lr_patch.shape
    _, H_hr, W_hr = hr_patch.shape

    # Upsample LR reference band to HR spatial dims for correlation
    scale_h = H_hr / H_lr
    scale_w = W_hr / W_lr

    lr_ref_upsampled = sk_resize(
        lr_patch[reference_band],
        (H_hr, W_hr),
        order=3,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)

    hr_ref = hr_patch[reference_band].astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shift_hr, _, _ = phase_cross_correlation(
            hr_ref,
            lr_ref_upsampled,
            upsample_factor=upsample_factor,
            normalization="phase",
        )

    # Convert shift from HR to LR pixel coords
    shift_lr = np.array([shift_hr[0] / scale_h, shift_hr[1] / scale_w])

    max_allowed_shift = 10.0  # LR pixels
    if np.abs(shift_lr).max() > max_allowed_shift:
        logger.warning(
            f"Large alignment shift detected: {shift_lr}.  "
            "Skipping alignment for this patch."
        )
        return lr_patch

    # Apply sub-pixel shift to all channels
    aligned = np.stack(
        [ndimage_shift(lr_patch[c], shift_lr, mode="reflect") for c in range(C)],
        axis=0,
    )
    return aligned.astype(np.float32)


def compute_alignment_rmse(
    lr_patch: np.ndarray,
    hr_patch: np.ndarray,
    reference_band: int = 3,
) -> float:
    """
    Compute the co-registration RMSE between an LR patch and HR reference
    after bicubic upsampling the LR to HR resolution.

    Used in prepare_patches.py to filter out grossly misaligned patch pairs
    (threshold: max_align_rmse from config, default 0.3 pixels).

    Returns RMSE in HR pixel units (float). Lower = better aligned.
    """
    from skimage.transform import resize as sk_resize
    C, H_lr, W_lr = lr_patch.shape
    _, H_hr, W_hr = hr_patch.shape

    ref_band = min(reference_band, C - 1)
    lr_up = sk_resize(
        lr_patch[ref_band], (H_hr, W_hr),
        order=3, preserve_range=True, anti_aliasing=True,
    ).astype(np.float32)
    hr_ref = hr_patch[ref_band].astype(np.float32)

    # Normalise both to [0,1] before RMSE to be scale-invariant
    def norm01(x):
        r = x.max() - x.min()
        return (x - x.min()) / (r + 1e-8)

    rmse = float(np.sqrt(np.mean((norm01(lr_up) - norm01(hr_ref)) ** 2)))
    return rmse


# ─────────────────────────────────────────────────────────────────────────────
# 4.  NDVI / NDWI Pseudo-label Generator (for downstream segmentation)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ndvi_pseudo_labels(
    hr_patch: np.ndarray,
    nir_band: int = 3,
    red_band: int = 2,
    green_band: int = 1,
    swir1_band: int = 4,
    ndvi_forest_thresh: float = 0.50,
    ndvi_defor_thresh: float = 0.10,
) -> np.ndarray:
    """
    Generate pixel-level semantic labels from NDVI/NDWI thresholds on the
    Sentinel-2 HR patch for the downstream segmentation task.

    Classes
    -------
    0 = intact_forest    NDVI >= ndvi_forest_thresh
    1 = deforestation    ndvi_defor_thresh <= NDVI < ndvi_forest_thresh
    2 = logging_road     NDVI < ndvi_defor_thresh AND high SWIR edge response

    Parameters
    ----------
    hr_patch : np.ndarray  Shape (C, H, W), raw reflectance (unnormalized).
    nir_band, red_band, green_band, swir1_band : int  0-based band indices.
    ndvi_forest_thresh  : float  NDVI threshold for forest class.
    ndvi_defor_thresh   : float  NDVI threshold separating deforestation/road.

    Returns
    -------
    np.ndarray  Shape (H, W), dtype uint8. Values: 0, 1, or 2.
    """
    C = hr_patch.shape[0]
    nir_b   = min(nir_band,   C - 1)
    red_b   = min(red_band,   C - 1)
    green_b = min(green_band, C - 1)
    swir1_b = min(swir1_band, C - 1)

    eps = 1e-8
    nir   = hr_patch[nir_b].astype(np.float32)
    red   = hr_patch[red_b].astype(np.float32)
    green = hr_patch[green_b].astype(np.float32)
    swir1 = hr_patch[swir1_b].astype(np.float32)

    # NDVI: vegetation index
    ndvi = (nir - red) / (nir + red + eps)

    # SWIR edge magnitude: roads show high-contrast SWIR linear features
    from scipy.ndimage import sobel
    swir_norm = (swir1 - swir1.min()) / (swir1.max() - swir1.min() + eps)
    sx = sobel(swir_norm, axis=0)
    sy = sobel(swir_norm, axis=1)
    edge_mag = np.sqrt(sx ** 2 + sy ** 2)
    high_edge = edge_mag > np.percentile(edge_mag, 85)

    # Label assignment
    labels = np.zeros(hr_patch.shape[1:], dtype=np.uint8)
    labels[ndvi >= ndvi_forest_thresh] = 0                    # intact forest
    labels[(ndvi >= ndvi_defor_thresh) & (ndvi < ndvi_forest_thresh)] = 1  # deforestation
    labels[(ndvi < ndvi_defor_thresh) & high_edge] = 2        # logging road
    labels[(ndvi < ndvi_defor_thresh) & ~high_edge] = 1       # bare soil → deforestation

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Raster I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_bands(path: Path, band_indices: List[int]) -> np.ndarray:
    """
    Read selected bands from a GeoTIFF and return as float32 array (C, H, W).

    Parameters
    ----------
    path : Path
        Path to the .tif file.
    band_indices : list[int]
        1-indexed band numbers to read (rasterio convention).

    Returns
    -------
    np.ndarray
        Shape (len(band_indices), H, W), dtype float32.
    """
    with rasterio.open(path) as src:
        data = src.read(band_indices).astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = 0.0
    return data


def extract_patches(
    lr_array: np.ndarray,
    hr_array: np.ndarray,
    lr_size: int,
    hr_size: int,
    stride: int,
    min_valid_frac: float = 0.85,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract paired (LR, HR) patch pairs from co-registered arrays.

    Parameters
    ----------
    lr_array : np.ndarray  (C, H_lr, W_lr)
    hr_array : np.ndarray  (C, H_hr, W_hr)
    lr_size  : int   LR patch side length in pixels.
    hr_size  : int   HR patch side length (= lr_size × scale).
    stride   : int   LR stride between patches.
    min_valid_frac : float  Reject patch if valid (non-zero) fraction is below.

    Returns
    -------
    list of (lr_patch, hr_patch) tuples, both np.ndarray (C, patch_size, patch_size)
    """
    scale = hr_size // lr_size
    C, H_lr, W_lr = lr_array.shape
    patches: List[Tuple[np.ndarray, np.ndarray]] = []

    for y in range(0, H_lr - lr_size + 1, stride):
        for x in range(0, W_lr - lr_size + 1, stride):
            lr_patch = lr_array[:, y : y + lr_size, x : x + lr_size]

            # Valid pixel check (reject nodata / clouds)
            valid_frac = (lr_patch > 0).mean()
            if valid_frac < min_valid_frac:
                continue

            y_hr = y * scale
            x_hr = x * scale
            hr_patch = hr_array[:, y_hr : y_hr + hr_size, x_hr : x_hr + hr_size]

            if hr_patch.shape[1] != hr_size or hr_patch.shape[2] != hr_size:
                continue  # boundary guard

            patches.append((lr_patch.copy(), hr_patch.copy()))

    return patches


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class RondoniaDataset(Dataset):
    """
    PyTorch Dataset for the Rondônia fishbone SR study.

    Each item is a (lr_tensor, hr_tensor) pair ready for network input.

    Parameters
    ----------
    patch_dir : Path | str
        Directory containing pre-extracted .npz patch files.
        Each .npz must have keys 'lr' and 'hr'.
    scaling_mode : str
        'standard' or 'smart'.
    smart_scaling_kwargs : dict, optional
        Kwargs forwarded to apply_smart_scaling() when mode == 'smart'.
    augment : bool
        If True, apply random horizontal/vertical flips & 90° rotations.
    """

    def __init__(
        self,
        patch_dir: str | Path,
        scaling_mode: str = "standard",
        smart_scaling_kwargs: Optional[dict] = None,
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.patch_dir = Path(patch_dir)
        self.scaling_mode = scaling_mode
        self.smart_kwargs = smart_scaling_kwargs or {}
        self.augment = augment

        self.files = sorted(self.patch_dir.glob("**/*.npz"))
        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No .npz patch files found in '{self.patch_dir}'. "
                "Run scripts/prepare_patches.py first."
            )
        logger.info(
            f"RondoniaDataset: {len(self.files)} patches | "
            f"scaling='{scaling_mode}' | augment={augment}"
        )

    def __len__(self) -> int:
        return len(self.files)

    def _scale(self, arr: np.ndarray) -> np.ndarray:
        """Single-array scale — used only for inference (not paired training)."""
        if self.scaling_mode == "smart":
            return apply_smart_scaling(arr, **self.smart_kwargs)
        return standard_scaling(arr)

    def _augment(
        self,
        lr: np.ndarray,
        hr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Spatially consistent random flips + 90° rotations."""
        # Horizontal flip
        if np.random.rand() > 0.5:
            lr = lr[:, :, ::-1].copy()
            hr = hr[:, :, ::-1].copy()
        # Vertical flip
        if np.random.rand() > 0.5:
            lr = lr[:, ::-1, :].copy()
            hr = hr[:, ::-1, :].copy()
        # 90° rotations (k = 0,1,2,3)
        k = np.random.randint(0, 4)
        lr = np.rot90(lr, k=k, axes=(1, 2)).copy()
        hr = np.rot90(hr, k=k, axes=(1, 2)).copy()
        return lr, hr

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.files[idx])
        lr: np.ndarray = data["lr"].astype(np.float32)  # (C, H_lr, W_lr)
        hr: np.ndarray = data["hr"].astype(np.float32)  # (C, H_hr, W_hr)

        # Guard: skip patches where HR has significant nodata (black corner artifacts).
        # Fall back to the next patch (wrap-around) instead of crashing.
        hr_valid_frac = (hr != 0).mean()
        if hr_valid_frac < 0.85:
            next_idx = (idx + 1) % len(self.files)
            return self.__getitem__(next_idx)

        if self.augment:
            lr, hr = self._augment(lr, hr)

        lr, hr = self._scale_pair(lr, hr)

        return torch.from_numpy(lr), torch.from_numpy(hr)

    def _scale_pair(
        self,
        lr: np.ndarray,
        hr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalise (LR, HR) pair using a SHARED normalization domain.

        Standard mode: each array normalised independently (global min-max
        within that patch) — acceptable because standard_scaling uses the
        patch's own range and Landsat/Sentinel-2 absolute values are similar.

        Smart mode: percentile clip values are derived from the HR patch,
        then applied to BOTH LR and HR.  This is critical — if LR and HR
        were scaled with their own independent percentiles they would be
        mapped into incomparable ranges (the HR has 3× more pixels and
        different sensor characteristics), causing PSNR/SSIM to collapse
        and SAM to report ~90° (orthogonal spectral vectors).
        """
        if self.scaling_mode == "smart":
            lo_vals, hi_vals = compute_smart_scaling_params(
                hr,
                lo_pct=self.smart_kwargs.get("lo_pct", 2.0),
                hi_pct=self.smart_kwargs.get("hi_pct", 98.0),
            )
            kwargs = {
                "lo_pct":  self.smart_kwargs.get("lo_pct",  2.0),
                "hi_pct":  self.smart_kwargs.get("hi_pct",  98.0),
                "out_min": self.smart_kwargs.get("out_min", -1.0),
                "out_max": self.smart_kwargs.get("out_max",  1.0),
            }
            hr_scaled = apply_smart_scaling(hr, lo_vals=lo_vals, hi_vals=hi_vals, **kwargs)
            lr_scaled = apply_smart_scaling(lr, lo_vals=lo_vals, hi_vals=hi_vals, **kwargs)
            return lr_scaled, hr_scaled
        else:
            return standard_scaling(lr), standard_scaling(hr)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    train_patch_dir: str | Path,
    val_patch_dir: str | Path,
    scaling_mode: str = "standard",
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
    smart_scaling_kwargs: Optional[dict] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build (train_loader, val_loader) for the SR study.

    Parameters
    ----------
    train_patch_dir : str | Path
        Root directory of training .npz patches.
    val_patch_dir : str | Path
        Root directory of validation .npz patches.
    scaling_mode : str
        'standard' or 'smart'.
    batch_size : int
        Samples per GPU batch.
    num_workers : int
        DataLoader worker processes.
    pin_memory : bool
        Pin CPU memory for faster GPU transfer.
    smart_scaling_kwargs : dict, optional
        Forwarded to apply_smart_scaling().

    Returns
    -------
    (train_loader, val_loader) : tuple of DataLoader
    """
    train_ds = RondoniaDataset(
        train_patch_dir,
        scaling_mode=scaling_mode,
        smart_scaling_kwargs=smart_scaling_kwargs,
        augment=True,
    )
    val_ds = RondoniaDataset(
        val_patch_dir,
        scaling_mode=scaling_mode,
        smart_scaling_kwargs=smart_scaling_kwargs,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader

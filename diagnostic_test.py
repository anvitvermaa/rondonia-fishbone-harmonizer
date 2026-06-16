"""
diagnostic_test.py  –  Rondônia SR Study  –  Pipeline Sanity Checker
=====================================================================

PURPOSE
-------
Rapid end-to-end verification of the entire pipeline BEFORE committing to
the 48-hour full training run.

DESIGN: Self-contained — all metric and scaling functions are inlined here
so the script runs even if rasterio / lpips are not yet installed.

USAGE
-----
    python diagnostic_test.py                   # synthetic data (no real .tif needed)
    python diagnostic_test.py --use-real-patches
    python diagnostic_test.py --output-dir diag_out

EXIT CODES  0 = all clear  |  1 = investigate before training
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")   # non-interactive backend: safe for headless servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ──────────────────────────────────────────────────────────────────────────────
# INLINE implementations — no rasterio / lpips dependency needed for diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def _standard_scaling(tensor: np.ndarray) -> np.ndarray:
    """Global min-max → [-1, 1]."""
    t_min, t_max = tensor.min(), tensor.max()
    denom = t_max - t_min
    if denom < 1e-8:
        return np.zeros_like(tensor, dtype=np.float32)
    return (2.0 * (tensor - t_min) / denom - 1.0).astype(np.float32)


def _smart_scaling(
    tensor: np.ndarray,
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
    out_min: float = -1.0,
    out_max: float = 1.0,
) -> np.ndarray:
    """Per-channel percentile clipping → [-1, 1].  Mirrors apply_smart_scaling()."""
    eps = 1e-8
    out = np.empty_like(tensor, dtype=np.float32)
    for c in range(tensor.shape[0]):
        ch = tensor[c].astype(np.float32)
        lo = np.percentile(ch, lo_pct)
        hi = np.percentile(ch, hi_pct)
        ch = np.clip(ch, lo, hi)
        ch = (ch - lo) / (hi - lo + eps)
        out[c] = ch * (out_max - out_min) + out_min
    return out


def _align_lr_to_hr(
    lr_patch: np.ndarray,
    hr_patch: np.ndarray,
    upsample_factor: int = 100,
    reference_band: int = 3,
) -> np.ndarray:
    """Sub-pixel phase cross-correlation alignment.  Mirrors align_lr_to_hr()."""
    from skimage.registration import phase_cross_correlation
    from skimage.transform import resize as sk_resize
    from scipy.ndimage import shift as ndimage_shift

    C, H_lr, W_lr = lr_patch.shape
    _, H_hr, W_hr = hr_patch.shape
    scale_h, scale_w = H_hr / H_lr, W_hr / W_lr

    lr_up = sk_resize(
        lr_patch[reference_band], (H_hr, W_hr),
        order=3, preserve_range=True, anti_aliasing=True,
    ).astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shift_hr, _, _ = phase_cross_correlation(
            hr_patch[reference_band].astype(np.float32),
            lr_up, upsample_factor=upsample_factor, normalization="phase",
        )

    shift_lr = np.array([shift_hr[0] / scale_h, shift_hr[1] / scale_w])
    if np.abs(shift_lr).max() > 10.0:
        return lr_patch

    return np.stack(
        [ndimage_shift(lr_patch[c], shift_lr, mode="reflect") for c in range(C)],
        axis=0,
    ).astype(np.float32)


def _compute_psnr(sr: np.ndarray, hr: np.ndarray, data_range: float = 2.0) -> float:
    mse = np.mean((sr - hr) ** 2)
    return 100.0 if mse < 1e-10 else float(10.0 * np.log10(data_range ** 2 / mse))


def _compute_ssim(sr: np.ndarray, hr: np.ndarray, data_range: float = 2.0) -> float:
    from skimage.metrics import structural_similarity
    return float(np.mean([
        structural_similarity(sr[c], hr[c], data_range=data_range)
        for c in range(sr.shape[0])
    ]))


def _compute_sam(sr: np.ndarray, hr: np.ndarray, eps: float = 1e-8) -> float:
    C = sr.shape[0]
    sr_flat = sr.reshape(C, -1).T
    hr_flat = hr.reshape(C, -1).T
    dot     = np.sum(sr_flat * hr_flat, axis=1)
    denom   = np.linalg.norm(sr_flat, axis=1) * np.linalg.norm(hr_flat, axis=1) + eps
    return float(np.degrees(np.mean(np.arccos(np.clip(dot / denom, -1.0, 1.0)))))

# ──────────────────────────────────────────────────────────────────────────────
# Colour codes for terminal output
# ──────────────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS = f"{GREEN}✓ PASS{RESET}"
FAIL = f"{RED}✗ FAIL{RESET}"
INFO = f"{CYAN}ℹ{RESET}"
WARN = f"{YELLOW}⚠ WARN{RESET}"

# ──────────────────────────────────────────────────────────────────────────────
# Constants (match config.yaml defaults)
# ──────────────────────────────────────────────────────────────────────────────
N_CHANNELS  = 6      # B, G, R, NIR, SWIR1, SWIR2
LR_SIZE     = 48     # LR patch side (pixels)
SCALE       = 3      # upscale factor 30 m → 10 m
HR_SIZE     = LR_SIZE * SCALE   # 144
BATCH_SIZE  = 4
N_TRAIN_BATCHES = 5  # batches for the 1-epoch SRCNN test
SEED        = 42

# Expected healthy ranges for the synthetic test
BICUBIC_PSNR_MIN = 20.0   # dB – lower bound; real imagery is typically 24-28 dB
SAM_MAX_DEG      = 20.0   # °  – anything above this indicates normalisation failure

# ──────────────────────────────────────────────────────────────────────────────
# Result tracker
# ──────────────────────────────────────────────────────────────────────────────

class DiagnosticResults:
    def __init__(self):
        self._tests: List[Tuple[str, bool, str]] = []

    def record(self, name: str, passed: bool, detail: str = "") -> None:
        self._tests.append((name, passed, detail))
        status = PASS if passed else FAIL
        prefix = f"  {status}  {name}"
        if detail:
            print(f"{prefix}  →  {detail}")
        else:
            print(prefix)

    def summary(self) -> bool:
        passed = sum(1 for _, p, _ in self._tests if p)
        total  = len(self._tests)
        sep    = "═" * 60
        print(f"\n{BOLD}{sep}{RESET}")
        print(f"{BOLD}  DIAGNOSTIC SUMMARY:  {passed}/{total} tests passed{RESET}")
        print(sep)
        all_ok = (passed == total)
        if all_ok:
            print(f"  {GREEN}{BOLD}All checks passed.  Safe to start full training.{RESET}")
        else:
            print(f"  {RED}{BOLD}Some checks FAILED.  Investigate before training.{RESET}")
            for name, ok, detail in self._tests:
                if not ok:
                    print(f"    {RED}✗  {name}{RESET}  {detail}")
        print(sep)
        return all_ok


results = DiagnosticResults()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    line = "─" * 60
    print(f"\n{BOLD}{CYAN}{line}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{line}{RESET}")


def fmt_mb(bytes_val: int) -> str:
    return f"{bytes_val / 1024**2:.0f} MB"


def to_rgb(arr: np.ndarray) -> np.ndarray:
    """
    Convert (C, H, W) multi-band array → (H, W, 3) false-colour RGB for display.
    Uses NIR(band 3), Red(band 2), Green(band 1)  → standard false-colour CIR.
    """
    bands = [
        min(3, arr.shape[0] - 1),   # NIR  → R channel
        min(2, arr.shape[0] - 1),   # Red  → G channel
        min(1, arr.shape[0] - 1),   # Green → B channel
    ]
    rgb = arr[bands].transpose(1, 2, 0).astype(np.float32)
    lo, hi = rgb.min(), rgb.max()
    rgb = (rgb - lo) / (hi - lo + 1e-8)
    return np.clip(rgb, 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# [1]  Environment & GPU
# ──────────────────────────────────────────────────────────────────────────────

def test_environment() -> torch.device:
    section("[1]  Environment & GPU")

    # Python version
    py_ver = sys.version.split()[0]
    print(f"  {INFO}  Python       {py_ver}")

    # PyTorch
    print(f"  {INFO}  PyTorch      {torch.__version__}")

    # CUDA
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        device      = torch.device("cuda")
        gpu_name    = torch.cuda.get_device_name(0)
        vram_total  = torch.cuda.get_device_properties(0).total_memory
        vram_free   = torch.cuda.mem_get_info(0)[0]
        bfloat_ok   = torch.cuda.is_bf16_supported()
        print(f"  {INFO}  GPU          {gpu_name}")
        print(f"  {INFO}  VRAM total   {fmt_mb(vram_total)}")
        print(f"  {INFO}  VRAM free    {fmt_mb(vram_free)}")
        print(f"  {INFO}  bfloat16     {'supported ✓' if bfloat_ok else 'NOT supported – fp16 will be used'}")
        results.record("CUDA available", True, gpu_name)
        results.record("bfloat16 support", bfloat_ok,
                       "RTX 30xx required" if not bfloat_ok else "")
    else:
        device = torch.device("cpu")
        print(f"  {WARN}  No CUDA GPU found – running on CPU (SRCNN test will be slow)")
        results.record("CUDA available", False, "CPU fallback active")

    # Key packages
    try:
        import skimage; print(f"  {INFO}  scikit-image {skimage.__version__}")
        results.record("scikit-image import", True)
    except ImportError as e:
        results.record("scikit-image import", False, str(e))

    try:
        import rasterio; print(f"  {INFO}  rasterio     {rasterio.__version__}")
        results.record("rasterio import", True)
    except ImportError as e:
        results.record("rasterio import", False, str(e))

    return device


# ──────────────────────────────────────────────────────────────────────────────
# [2]  Synthetic Data Generation
# ──────────────────────────────────────────────────────────────────────────────

def make_synthetic_batch(
    batch_size: int       = BATCH_SIZE,
    n_channels: int       = N_CHANNELS,
    lr_size: int          = LR_SIZE,
    scale: int            = SCALE,
    seed: int             = SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a batch of 16-bit TOA satellite patches with realistic properties:
      - Log-normal base luminance (forest reflectance typical)
      - Spatial structure: Perlin-like sums of sine waves at multiple scales
        (approximates the texture of clearing edges / fishbone roads)
      - Deliberate outliers in 2 % of pixels (cloud / specular reflection)
        to stress-test Smart Scaling

    Returns
    -------
    lr_batch : np.ndarray  (B, C, H_lr, W_lr)  values in [0, 65535]  uint16-range but float32
    hr_batch : np.ndarray  (B, C, H_hr, W_hr)  bicubic-downsampled from synthetic HR truth
    """
    rng      = np.random.default_rng(seed)
    hr_size  = lr_size * scale

    # Band-specific mean reflectance (rough Landsat tropical-forest values in DN)
    band_means = np.array([800, 900, 700, 3500, 1200, 800], dtype=np.float32)[:n_channels]

    lr_batch = np.zeros((batch_size, n_channels, lr_size, lr_size),  dtype=np.float32)
    hr_batch = np.zeros((batch_size, n_channels, hr_size, hr_size),  dtype=np.float32)

    for b in range(batch_size):
        # Spatial structure: superpose 4 sine waves at different scales
        y_lr = np.linspace(0, 2 * np.pi, lr_size)
        x_lr = np.linspace(0, 2 * np.pi, lr_size)
        YY_lr, XX_lr = np.meshgrid(y_lr, x_lr, indexing="ij")
        y_hr = np.linspace(0, 2 * np.pi, hr_size)
        x_hr = np.linspace(0, 2 * np.pi, hr_size)
        YY_hr, XX_hr = np.meshgrid(y_hr, x_hr, indexing="ij")

        spatial_lr = (
            0.5  * np.sin(1.0 * YY_lr) * np.cos(1.0 * XX_lr) +
            0.25 * np.sin(3.7 * YY_lr) * np.cos(3.7 * XX_lr) +
            0.15 * np.sin(8.1 * YY_lr) * np.cos(5.3 * XX_lr) +
            0.10 * np.sin(13. * YY_lr) * np.cos(9.7 * XX_lr)
        )
        spatial_hr = (
            0.5  * np.sin(1.0 * YY_hr) * np.cos(1.0 * XX_hr) +
            0.25 * np.sin(3.7 * YY_hr) * np.cos(3.7 * XX_hr) +
            0.15 * np.sin(8.1 * YY_hr) * np.cos(5.3 * XX_hr) +
            0.10 * np.sin(13. * YY_hr) * np.cos(9.7 * XX_hr)
        )
        # Normalise spatial texture to [0, 1]
        s_min, s_max = spatial_lr.min(), spatial_lr.max()
        spatial_lr = (spatial_lr - s_min) / (s_max - s_min + 1e-8)
        s_min, s_max = spatial_hr.min(), spatial_hr.max()
        spatial_hr = (spatial_hr - s_min) / (s_max - s_min + 1e-8)

        for c in range(n_channels):
            bm = band_means[c]
            noise_lr = rng.normal(0, bm * 0.05, (lr_size, lr_size)).astype(np.float32)
            noise_hr = rng.normal(0, bm * 0.05, (hr_size, hr_size)).astype(np.float32)

            lr_ch = bm * (0.7 + 0.3 * spatial_lr) + noise_lr
            hr_ch = bm * (0.7 + 0.3 * spatial_hr) + noise_hr

            # Inject 2 % bright outliers (clouds / specular glint) in LR
            n_outliers = max(1, int(0.02 * lr_size * lr_size))
            oy = rng.integers(0, lr_size, n_outliers)
            ox = rng.integers(0, lr_size, n_outliers)
            lr_ch[oy, ox] = rng.uniform(50000, 65535, n_outliers).astype(np.float32)

            lr_batch[b, c] = np.clip(lr_ch, 0, 65535)
            hr_batch[b, c] = np.clip(hr_ch, 0, 65535)

    return lr_batch, hr_batch


def test_synthetic_data(device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    section("[2]  Synthetic Data Generation")

    lr_np, hr_np = make_synthetic_batch()

    print(f"  {INFO}  LR batch shape : {lr_np.shape}  (B, C, H_lr, W_lr)")
    print(f"  {INFO}  HR batch shape : {hr_np.shape}  (B, C, H_hr, W_hr)")
    print(f"  {INFO}  LR value range : [{lr_np.min():.0f}, {lr_np.max():.0f}]")
    print(f"  {INFO}  Outliers injected : 2% of LR pixels at 50k–65535 DN")

    shape_ok = (lr_np.shape == (BATCH_SIZE, N_CHANNELS, LR_SIZE, LR_SIZE) and
                hr_np.shape == (BATCH_SIZE, N_CHANNELS, HR_SIZE, HR_SIZE))
    results.record("Synthetic data shapes", shape_ok,
                   f"LR={lr_np.shape} HR={hr_np.shape}")

    has_outliers = lr_np.max() > 40000
    results.record("Outlier injection", has_outliers,
                   f"max DN={lr_np.max():.0f}")

    return lr_np, hr_np


# ──────────────────────────────────────────────────────────────────────────────
# [3]  Smart Scaling vs Standard Scaling
# ──────────────────────────────────────────────────────────────────────────────

def test_scaling(lr_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    section("[3]  Smart Scaling  vs  Standard Scaling")

    # Use inlined versions — no rasterio dependency
    apply_smart_scaling = _smart_scaling
    standard_scaling    = _standard_scaling

    # We test on a single patch (first item, all channels)
    patch = lr_np[0]   # (C, H, W)

    t0 = time.perf_counter()
    smart_out  = apply_smart_scaling(patch, lo_pct=2.0, hi_pct=98.0)
    t_smart    = time.perf_counter() - t0

    t0 = time.perf_counter()
    std_out    = standard_scaling(patch)
    t_std      = time.perf_counter() - t0

    # ── Range check ─────────────────────────────────────────────────────────
    smart_range_ok = (smart_out.min() >= -1.0 - 1e-5) and (smart_out.max() <= 1.0 + 1e-5)
    std_range_ok   = (std_out.min()   >= -1.0 - 1e-5) and (std_out.max()   <= 1.0 + 1e-5)

    print(f"  {INFO}  Smart   range  : [{smart_out.min():.4f}, {smart_out.max():.4f}]  ({t_smart*1000:.1f} ms)")
    print(f"  {INFO}  Standard range : [{std_out.min():.4f},  {std_out.max():.4f}]   ({t_std*1000:.1f} ms)")

    results.record("Smart Scaling output range [-1,1]",    smart_range_ok,
                   f"[{smart_out.min():.4f}, {smart_out.max():.4f}]")
    results.record("Standard Scaling output range [-1,1]", std_range_ok,
                   f"[{std_out.min():.4f}, {std_out.max():.4f}]")

    # ── Variance preservation ─────────────────────────────────────────────
    # Smart Scaling should deliver HIGHER per-channel variance because it
    # clips outliers that otherwise compress the vegetated-pixel distribution
    smart_var = np.mean([smart_out[c].var() for c in range(N_CHANNELS)])
    std_var   = np.mean([std_out[c].var()   for c in range(N_CHANNELS)])
    var_preserved = smart_var > std_var

    print(f"\n  {INFO}  Mean per-channel variance:")
    print(f"        Smart Scaling :  {smart_var:.6f}")
    print(f"        Standard      :  {std_var:.6f}")
    if var_preserved:
        print(f"        → {GREEN}Smart Scaling PRESERVES MORE variance ✓{RESET}")
    else:
        print(f"        → {YELLOW}Variance not improved (check outlier injection){RESET}")

    results.record("Smart Scaling preserves variance", var_preserved,
                   f"smart={smart_var:.5f}  std={std_var:.5f}")

    # ── Channel-wise statistics table ────────────────────────────────────
    band_names = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"][:N_CHANNELS]
    print(f"\n  {'Band':<6}  {'Smart mean':>11}  {'Smart std':>9}  {'Std mean':>10}  {'Std std':>8}")
    print(f"  {'─'*6}  {'─'*11}  {'─'*9}  {'─'*10}  {'─'*8}")
    for c, name in enumerate(band_names):
        print(f"  {name:<6}  {smart_out[c].mean():>11.5f}  {smart_out[c].std():>9.5f}  "
              f"{std_out[c].mean():>10.5f}  {std_out[c].std():>8.5f}")

    return smart_out, std_out


# ──────────────────────────────────────────────────────────────────────────────
# [4]  Sub-pixel Alignment
# ──────────────────────────────────────────────────────────────────────────────

def test_alignment() -> None:
    section("[4]  Sub-pixel Alignment (Phase Cross-Correlation)")

    align_lr_to_hr = _align_lr_to_hr  # use inlined version
    from scipy.ndimage import shift as ndimage_shift

    rng       = np.random.default_rng(SEED + 1)
    C, H, W   = N_CHANNELS, LR_SIZE, LR_SIZE
    hr_size   = HR_SIZE

    # Create a reference LR patch
    base = (np.sin(np.linspace(0, 4*np.pi, H))[:, None] *
            np.cos(np.linspace(0, 4*np.pi, W))[None, :]).astype(np.float32)
    lr_patch  = np.stack([base * (1 + 0.1*c) for c in range(C)], axis=0)  # (C, H, W)

    # Simulate a known sub-pixel offset on a matched HR version
    true_shift_lr = np.array([1.3, -0.8])   # pixels in LR coords
    lr_shifted    = np.stack([
        ndimage_shift(lr_patch[c], true_shift_lr, mode="reflect") for c in range(C)
    ], axis=0)

    # Up-sample unshifted LR to HR for the HR reference
    hr_ref = F.interpolate(
        torch.from_numpy(lr_patch[None]),
        size=(hr_size, hr_size), mode="bicubic", align_corners=False
    ).squeeze(0).numpy()

    t0 = time.perf_counter()
    aligned = align_lr_to_hr(lr_shifted, hr_ref, upsample_factor=100, reference_band=3)
    elapsed = time.perf_counter() - t0

    # Compare aligned vs unshifted: should be similar
    mse_before = np.mean((lr_shifted[3] - lr_patch[3]) ** 2)
    mse_after  = np.mean((aligned[3]    - lr_patch[3]) ** 2)
    improvement = mse_before > mse_after

    print(f"  {INFO}  True shift applied  : {true_shift_lr}")
    print(f"  {INFO}  Alignment time      : {elapsed*1000:.1f} ms")
    print(f"  {INFO}  MSE band-3 before   : {mse_before:.6f}")
    print(f"  {INFO}  MSE band-3 after    : {mse_after:.6f}")
    print(f"  {INFO}  MSE improved        : {improvement}")

    results.record("Alignment shape preserved", aligned.shape == lr_patch.shape,
                   str(aligned.shape))
    results.record("Alignment improves MSE", improvement,
                   f"before={mse_before:.5f}  after={mse_after:.5f}")
    results.record("Alignment runs in <5 s", elapsed < 5.0,
                   f"{elapsed:.2f} s")


# ──────────────────────────────────────────────────────────────────────────────
# [5]  Bicubic Baseline
# ──────────────────────────────────────────────────────────────────────────────

def test_bicubic(
    lr_np: np.ndarray,
    hr_np: np.ndarray,
    device: torch.device,
    scaling_mode: str = "smart",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    section("[5]  Bicubic Baseline")

    apply_smart_scaling = _smart_scaling  # use inlined version

    # Normalise the batch
    lr_norm = np.stack([apply_smart_scaling(lr_np[b]) for b in range(BATCH_SIZE)])
    hr_norm = np.stack([apply_smart_scaling(hr_np[b]) for b in range(BATCH_SIZE)])

    lr_t = torch.from_numpy(lr_norm).float()   # (B, C, H_lr, W_lr)
    hr_t = torch.from_numpy(hr_norm).float()   # (B, C, H_hr, W_hr)

    lr_t = lr_t.to(device)
    hr_t = hr_t.to(device)

    # Bicubic upsample
    with torch.no_grad():
        bic = F.interpolate(lr_t, scale_factor=SCALE, mode="bicubic",
                             align_corners=False).clamp(-1.0, 1.0)

    # Verify tensor shapes
    assert lr_t.shape == (BATCH_SIZE, N_CHANNELS, LR_SIZE, LR_SIZE), \
        f"LR shape mismatch: {lr_t.shape}"
    assert bic.shape  == (BATCH_SIZE, N_CHANNELS, HR_SIZE, HR_SIZE), \
        f"Bicubic shape mismatch: {bic.shape}"
    assert hr_t.shape == (BATCH_SIZE, N_CHANNELS, HR_SIZE, HR_SIZE), \
        f"HR shape mismatch: {hr_t.shape}"

    results.record("LR tensor shape", True, str(tuple(lr_t.shape)))
    results.record("Bicubic output shape", True, str(tuple(bic.shape)))
    results.record("HR tensor shape", True,  str(tuple(hr_t.shape)))

    # Metrics on first patch — use inlined versions
    compute_psnr, compute_ssim, compute_sam = _compute_psnr, _compute_ssim, _compute_sam
    bic_np = bic[0].cpu().numpy()
    hr_np0 = hr_t[0].cpu().numpy()

    psnr = compute_psnr(bic_np, hr_np0)
    ssim = compute_ssim(bic_np, hr_np0)
    sam  = compute_sam(bic_np, hr_np0)

    print(f"  {INFO}  Bicubic  PSNR : {psnr:.3f} dB  (healthy baseline ≥ {BICUBIC_PSNR_MIN} dB)")
    print(f"  {INFO}  Bicubic  SSIM : {ssim:.4f}")
    print(f"  {INFO}  Bicubic  SAM  : {sam:.3f}°  (healthy ≤ {SAM_MAX_DEG}°)")

    psnr_ok = psnr >= BICUBIC_PSNR_MIN
    sam_ok  = sam  <= SAM_MAX_DEG

    if not psnr_ok:
        print(f"  {WARN}  Bicubic PSNR {psnr:.2f} dB is below {BICUBIC_PSNR_MIN} dB – "
              f"expected with synthetic data; real imagery typically 24–28 dB")
    results.record("Bicubic PSNR ≥ threshold", psnr_ok,
                   f"{psnr:.3f} dB (min {BICUBIC_PSNR_MIN})")
    results.record("Bicubic SAM ≤ threshold", sam_ok,
                   f"{sam:.3f}° (max {SAM_MAX_DEG})")

    metrics = {"bicubic_psnr": psnr, "bicubic_ssim": ssim, "bicubic_sam": sam}
    return lr_t, bic, hr_t, metrics


# ──────────────────────────────────────────────────────────────────────────────
# [6]  SRCNN  –  1-epoch bfloat16 Training Test
# ──────────────────────────────────────────────────────────────────────────────

def test_srcnn(
    lr_t: torch.Tensor,
    hr_t: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, dict]:
    section("[6]  SRCNN  –  1-epoch bfloat16 Mixed-Precision Training Test")

    from models.srcnn import SRCNN

    # ── Initialise ────────────────────────────────────────────────────────
    torch.manual_seed(SEED)
    model = SRCNN(num_channels=N_CHANNELS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {INFO}  SRCNN parameters : {n_params:,}")

    opt     = torch.optim.Adam(model.parameters(), lr=2e-4)
    # bfloat16 does not require GradScaler (static loss scale = 1.0)
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) \
                else torch.float32  # CPU fallback
    use_amp   = (device.type == "cuda")

    print(f"  {INFO}  AMP dtype        : {amp_dtype}")
    print(f"  {INFO}  Training batches : {N_TRAIN_BATCHES}")

    # ── Pre-process: bicubic upsample before SRCNN (paper requirement) ──
    with torch.no_grad():
        lr_up = F.interpolate(lr_t, scale_factor=SCALE, mode="bicubic",
                               align_corners=False)   # (B, C, H_hr, W_hr)

    # ── Verify shapes before training ────────────────────────────────────
    assert lr_up.shape == hr_t.shape, \
        f"Shape mismatch: lr_up={lr_up.shape}  hr={hr_t.shape}"
    results.record("SRCNN input shape correct", True,
                   f"lr_up={tuple(lr_up.shape)}")

    # ── 1-"epoch" micro training loop ────────────────────────────────────
    model.train()
    loss_history = []
    t0 = time.perf_counter()

    for step in range(N_TRAIN_BATCHES):
        opt.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                sr    = model(lr_up)
                loss  = F.l1_loss(sr, hr_t)
            loss.backward()
        else:
            sr   = model(lr_up)
            loss = F.l1_loss(sr, hr_t)
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        loss_history.append(loss.item())
        if step == 0 or (step + 1) % 2 == 0:
            print(f"    step {step+1}/{N_TRAIN_BATCHES}  L1={loss.item():.5f}")

    elapsed = time.perf_counter() - t0
    loss_decreasing = loss_history[-1] < loss_history[0]

    print(f"  {INFO}  Training time    : {elapsed:.2f} s")
    print(f"  {INFO}  Loss: {loss_history[0]:.5f} → {loss_history[-1]:.5f}  "
          f"({'↓ decreasing ✓' if loss_decreasing else '↑ not decreasing'})")

    if device.type == "cuda":
        vram_used = torch.cuda.max_memory_allocated(0)
        print(f"  {INFO}  Peak VRAM used   : {fmt_mb(vram_used)}")
        torch.cuda.reset_peak_memory_stats()

    results.record("SRCNN training runs without OOM", True, f"{elapsed:.1f} s")
    results.record("SRCNN loss decreasing", loss_decreasing,
                   f"{loss_history[0]:.5f}→{loss_history[-1]:.5f}")

    # ── Post-training evaluation ──────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                sr_out = model(lr_up).clamp(-1.0, 1.0)
        else:
            sr_out = model(lr_up).clamp(-1.0, 1.0)

    compute_psnr, compute_ssim, compute_sam = _compute_psnr, _compute_ssim, _compute_sam
    sr_np  = sr_out[0].float().cpu().numpy()
    hr_np0 = hr_t[0].float().cpu().numpy()

    psnr = compute_psnr(sr_np, hr_np0)
    ssim = compute_ssim(sr_np, hr_np0)
    sam  = compute_sam(sr_np, hr_np0)

    print(f"\n  {INFO}  SRCNN post-training PSNR : {psnr:.3f} dB")
    print(f"  {INFO}  SRCNN post-training SSIM : {ssim:.4f}")
    print(f"  {INFO}  SRCNN post-training SAM  : {sam:.3f}°")

    results.record("SRCNN output shape correct", sr_out.shape == hr_t.shape,
                   str(tuple(sr_out.shape)))

    metrics = {"srcnn_psnr": psnr, "srcnn_ssim": ssim, "srcnn_sam": sam}
    return sr_out, metrics


# ──────────────────────────────────────────────────────────────────────────────
# [7]  Visual Grid Export
# ──────────────────────────────────────────────────────────────────────────────

def test_visual_export(
    lr_t:   torch.Tensor,
    bic:    torch.Tensor,
    srcnn:  torch.Tensor,
    hr_t:   torch.Tensor,
    metrics: dict,
    output_dir: Path,
) -> None:
    section("[7]  Visual Grid Export")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "diagnostic_output.png"

    lr_np  = F.interpolate(lr_t[:1].float().cpu(),
                            scale_factor=SCALE,
                            mode="bicubic",
                            align_corners=False).squeeze(0).numpy()
    bic_np  = bic[0].float().cpu().numpy()
    sr_np   = srcnn[0].float().cpu().numpy()
    hr_np   = hr_t[0].float().cpu().numpy()

    # Retrieve metric strings
    bic_psnr  = metrics.get("bicubic_psnr", float("nan"))
    bic_ssim  = metrics.get("bicubic_ssim", float("nan"))
    bic_sam   = metrics.get("bicubic_sam",  float("nan"))
    src_psnr  = metrics.get("srcnn_psnr",   float("nan"))
    src_ssim  = metrics.get("srcnn_ssim",   float("nan"))
    src_sam   = metrics.get("srcnn_sam",    float("nan"))

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14), facecolor="#0e1117")
    gs  = gridspec.GridSpec(
        3, 4,
        figure=fig,
        hspace=0.08,
        wspace=0.04,
        top=0.88, bottom=0.02,
        left=0.02, right=0.98,
    )

    # Row 0: main image panels
    axes_img = [fig.add_subplot(gs[0:2, c]) for c in range(4)]
    # Row 2: per-channel bar charts
    axes_bar = [fig.add_subplot(gs[2, c]) for c in range(4)]

    panels = [
        (lr_np,  "LR Input\n(Bicubic ×3 for display)",  None,       None,      None),
        (bic_np, "Bicubic Upscale",
         f"PSNR: {bic_psnr:.2f} dB",
         f"SSIM: {bic_ssim:.4f}",
         f"SAM:  {bic_sam:.2f}°"),
        (sr_np,  "SRCNN (1-epoch)",
         f"PSNR: {src_psnr:.2f} dB",
         f"SSIM: {src_ssim:.4f}",
         f"SAM:  {src_sam:.2f}°"),
        (hr_np,  "HR Target\n(Sentinel-2 @ 10 m)",       None,       None,      None),
    ]

    for ax, (arr, title, m1, m2, m3) in zip(axes_img, panels):
        rgb = to_rgb(arr)
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=6)
        ax.axis("off")
        if m1:
            metric_str = f"{m1}\n{m2}\n{m3}"
            ax.text(0.02, 0.02, metric_str, transform=ax.transAxes,
                    color="#00ff88", fontsize=9, fontfamily="monospace",
                    verticalalignment="bottom",
                    bbox=dict(facecolor="#00000088", edgecolor="none", pad=3))
        ax.set_facecolor("#0e1117")

    # Per-channel mean intensity bar charts
    band_names = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"][:N_CHANNELS]
    bar_color_map = ["#4488ff", "#44ff88", "#ff4444", "#cc44ff", "#ffaa22", "#ff6688"][:N_CHANNELS]

    for ax, (arr, title, _, __, ___) in zip(axes_bar, panels):
        means = [arr[c].mean() for c in range(N_CHANNELS)]
        bars = ax.bar(band_names, means, color=bar_color_map, edgecolor="#333333", width=0.7)
        ax.set_facecolor("#161b22")
        ax.set_ylim(-1.1, 1.1)
        ax.tick_params(colors="white", labelsize=7)
        ax.spines[:].set_color("#444444")
        ax.set_ylabel("Mean [-1,1]", color="white", fontsize=7)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, 
                    min(val + 0.05, 0.85),
                    f"{val:.2f}", ha="center", va="bottom",
                    color="white", fontsize=6)
        ax.set_title("Per-channel means", color="#888888", fontsize=8, pad=3)

    # ── Header ────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.94,
        "Rondônia SR Study  —  Pipeline Diagnostic Output",
        ha="center", va="center",
        fontsize=15, fontweight="bold", color="white",
        fontfamily="monospace",
    )
    fig.text(
        0.5, 0.905,
        f"CIR Composite: NIR→R, Red→G, Green→B  |  "
        f"Patches: {LR_SIZE}×{LR_SIZE} LR → {HR_SIZE}×{HR_SIZE} HR  |  "
        f"Channels: {N_CHANNELS}  |  Scale: ×{SCALE}",
        ha="center", va="center",
        fontsize=9, color="#888888",
    )

    # ── Smart Scaling annotation strip ───────────────────────────────────
    fig.text(
        0.5, 0.885,
        "Smart Scaling: per-channel percentile clip [2%, 98%] → preserves spectral variance across outlier-contaminated TOA data",
        ha="center", va="center",
        fontsize=8, color="#aaddff",
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    file_ok = out_path.exists() and out_path.stat().st_size > 10_000
    print(f"  {INFO}  Output saved to : {out_path.resolve()}")
    print(f"  {INFO}  File size       : {out_path.stat().st_size / 1024:.0f} KB")
    results.record("Visual grid saved successfully", file_ok, str(out_path))


# ──────────────────────────────────────────────────────────────────────────────
# Real-patch loader (used when --use-real-patches is set)
# ──────────────────────────────────────────────────────────────────────────────

def load_real_patches(n: int = BATCH_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """Load n patches from data/aligned/train/  to test real data flow."""
    import yaml
    cfg_path = Path("config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError("config.yaml not found. Run from Rondonia RS/ directory.")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    patch_dir = Path(cfg["paths"]["aligned_dir"]) / "train"
    files     = sorted(patch_dir.glob("**/*.npz"))[:n]

    if not files:
        raise FileNotFoundError(
            f"No .npz patches found in {patch_dir}. "
            "Run: python scripts/prepare_patches.py --config config.yaml"
        )

    lr_list, hr_list = [], []
    for f in files:
        d = np.load(f)
        lr_list.append(d["lr"].astype(np.float32))
        hr_list.append(d["hr"].astype(np.float32))

    return np.stack(lr_list), np.stack(hr_list)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    t_start    = time.perf_counter()

    banner = f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════╗
║         RONDÔNIA SR PIPELINE  –  DIAGNOSTIC TEST              ║
║   Verifies data flow, Smart Scaling, alignment & SRCNN AMP    ║
╚══════════════════════════════════════════════════════════════╝{RESET}
"""
    print(banner)

    # ── [1] Environment ───────────────────────────────────────────────────
    device = test_environment()

    # ── [2] Data ──────────────────────────────────────────────────────────
    if args.use_real_patches:
        section("[2]  Loading Real Patches from data/aligned/train/")
        try:
            lr_np, hr_np = load_real_patches(BATCH_SIZE)
            print(f"  {INFO}  Loaded {BATCH_SIZE} real patches")
            print(f"  {INFO}  LR shape: {lr_np.shape}  range [{lr_np.min():.0f}, {lr_np.max():.0f}]")
            results.record("Real patches loaded", True, str(lr_np.shape))
        except Exception as e:
            print(f"  {WARN}  Could not load real patches ({e}). Falling back to synthetic.")
            lr_np, hr_np = make_synthetic_batch()
            results.record("Real patches loaded", False, str(e))
    else:
        lr_np, hr_np = test_synthetic_data(device)

    # ── [3] Scaling ────────────────────────────────────────────────────────
    try:
        test_scaling(lr_np)
    except Exception as e:
        results.record("Smart Scaling test", False, traceback.format_exc(limit=3))

    # ── [4] Alignment ──────────────────────────────────────────────────────
    try:
        test_alignment()
    except Exception as e:
        results.record("Alignment test", False, traceback.format_exc(limit=3))

    # ── [5] Bicubic ────────────────────────────────────────────────────────
    all_metrics: dict = {}
    try:
        lr_t, bic, hr_t, bic_metrics = test_bicubic(lr_np, hr_np, device)
        all_metrics.update(bic_metrics)
    except Exception as e:
        results.record("Bicubic test", False, traceback.format_exc(limit=3))
        print(f"\n{RED}Cannot continue: bicubic test failed.{RESET}")
        return 1

    # ── [6] SRCNN ─────────────────────────────────────────────────────────
    try:
        sr_out, srcnn_metrics = test_srcnn(lr_t, hr_t, device)
        all_metrics.update(srcnn_metrics)
    except torch.cuda.OutOfMemoryError:
        results.record("SRCNN training runs without OOM", False, "CUDA Out of Memory!")
        print(f"  {RED}OOM! VRAM insufficient. Try reducing N_TRAIN_BATCHES={N_TRAIN_BATCHES}{RESET}")
        sr_out = bic   # fall back to bicubic for the visual export
    except Exception as e:
        results.record("SRCNN training runs without OOM", False, str(e))
        traceback.print_exc()
        sr_out = bic

    # ── [7] Visual export ─────────────────────────────────────────────────
    try:
        test_visual_export(lr_t, bic, sr_out, hr_t, all_metrics, output_dir)
    except Exception as e:
        results.record("Visual grid saved successfully", False, str(e))
        traceback.print_exc()

    # ── Final summary ──────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    all_ok = results.summary()
    print(f"\n  Total diagnostic time: {elapsed:.1f} s\n")

    # ── Pre-flight advice ──────────────────────────────────────────────────
    section("Pre-Flight Checklist  –  Training Readiness")
    bic_psnr  = all_metrics.get("bicubic_psnr", 0)
    bic_sam   = all_metrics.get("bicubic_sam",  999)
    src_psnr  = all_metrics.get("srcnn_psnr",   0)

    items = [
        ("Bicubic PSNR ≥ 20 dB on synthetic",
         bic_psnr >= 20.0,
         f"{bic_psnr:.2f} dB  (expect 24–28 dB on real imagery)"),
        ("Bicubic SAM ≤ 20°",
         bic_sam <= 20.0,
         f"{bic_sam:.2f}°  (lower = better spectral preservation)"),
        ("SRCNN ≥ Bicubic PSNR after 1 epoch",
         src_psnr >= bic_psnr,
         f"SRCNN={src_psnr:.2f} dB vs Bicubic={bic_psnr:.2f} dB"),
        ("No CUDA OOM on SRCNN micro-batch",
         any("SRCNN training" in t and p for t, p, _ in results._tests),
         "Safe to train with batch_size=4"),
    ]

    for label, ok, note in items:
        icon = f"{GREEN}✓{RESET}" if ok else f"{YELLOW}!{RESET}"
        print(f"  {icon}  {label}")
        print(f"      {CYAN}{note}{RESET}")

    print()
    if all_ok:
        print(f"{GREEN}{BOLD}  🚀  All checks passed. You are clear to start ESRGAN training!{RESET}\n")
    else:
        print(f"{YELLOW}{BOLD}  ⚠  Address the FAILED items above before starting the 48-h run.{RESET}\n")

    return 0 if all_ok else 1


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rondônia SR Pipeline Diagnostic Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--use-real-patches",
        action="store_true",
        help="Load real .npz patches from data/aligned/train/ instead of synthetic data",
    )
    parser.add_argument(
        "--output-dir",
        default="diag_output",
        help="Directory to save diagnostic_output.png (default: diag_output/)",
    )
    args = parser.parse_args()
    sys.exit(main(args))

"""
evaluate.py  (v2)  –  Rondônia SR Comparative Study  –  Evaluation & Metrics
==========================================================================

Computes PSNR, SSIM, SAM, LPIPS, and ERGAS for every model × scaling combination
and outputs a comparative Markdown / CSV table.

Usage:
    # Evaluate all enabled models on test set (default)
    python evaluate.py --config config.yaml

    # Evaluate a single model/scaling pair on validation set
    python evaluate.py --model swinir --scaling smart --test-split val --config config.yaml

Metrics
-------
• PSNR  – Peak Signal-to-Noise Ratio (pixel-wise distortion)         ↑ better
• SSIM  – Structural Similarity Index (structural covariance)          ↑ better
• SAM   – Spectral Angle Mapper (spectral shape preservation)          ↓ better
• LPIPS – Learned Perceptual Image Patch Similarity (road textures)    ↓ better
• ERGAS – Erreur Relative Globale Adimensionnelle de Synthèse          ↓ better
         (pan-sharpening standard — required by IEEE TGRS reviewers)

v2 Fixes
--------
* CRITICAL FIX: LPIPS was returning NaN.  Root cause: the 6-band tensor was
  passed directly to AlexNet (3-channel only). Fixed to explicitly slice
  bands [R=2, G=1, B=0] and hard-clamp to [-1, 1] before LPIPS call.
* Added compute_ergas() — standard remote sensing pan-sharpening metric.
* Added --test-split flag to evaluate on test (default) or val split.
* rgb_bands now read from config (evaluation.rgb_bands) for consistency.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import yaml
import lpips as lpips_lib
from skimage.metrics import structural_similarity

# Reuse from train.py
from train import build_model, validate, load_config, set_seed
from dataloader import build_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Individual Metric Functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr(sr: np.ndarray, hr: np.ndarray, data_range: float = 2.0) -> float:
    """
    Peak Signal-to-Noise Ratio.

    Measures pixel-level distortion.  For tensors normalised to [-1, 1],
    data_range = 2.0.  Higher is better (indicating lower pixel error).

    Parameters
    ----------
    sr, hr      : np.ndarray  Shape (C, H, W), values in [-1, 1].
    data_range  : float       Dynamic range of the data (2.0 for [-1,1]).

    Returns
    -------
    float  PSNR in dB.
    """
    mse = np.mean((sr - hr) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10(data_range ** 2 / mse))


def compute_ssim(sr: np.ndarray, hr: np.ndarray, data_range: float = 2.0) -> float:
    """
    Structural Similarity Index (channel-averaged).

    Captures luminance, contrast, and structural covariance.
    Computed per band then averaged across spectral channels.

    Parameters
    ----------
    sr, hr      : np.ndarray  Shape (C, H, W).
    data_range  : float       Dynamic range for normalisation.

    Returns
    -------
    float  Mean SSIM in [−1, 1].  Perfect reconstruction = 1.0.
    """
    ssim_vals = []
    for c in range(sr.shape[0]):
        val = structural_similarity(
            sr[c], hr[c],
            data_range=data_range,
        )
        ssim_vals.append(val)
    return float(np.mean(ssim_vals))


def compute_sam(sr: np.ndarray, hr: np.ndarray, eps: float = 1e-8) -> float:
    """
    Spectral Angle Mapper (SAM) — degrees.

    Measures the angular distance between the spectral vectors of
    each pixel in the SR and HR images.  Insensitive to illumination
    changes; directly quantifies spectral signature preservation.

    This is the PRIMARY metric for evaluating Smart Scaling's claim that
    it preserves the MULTISPECTRAL SHAPE of the 16-bit TOA data.

    Lower SAM angle → better spectral fidelity.

    Parameters
    ----------
    sr, hr : np.ndarray  Shape (C, H, W).  Float, any range.
    eps    : float       Numerical guard.

    Returns
    -------
    float  Mean SAM angle in degrees.
    """
    # Reshape to (H*W, C)
    C, H, W = sr.shape
    sr_flat = sr.reshape(C, -1).T   # (H*W, C)
    hr_flat = hr.reshape(C, -1).T   # (H*W, C)

    dot     = np.sum(sr_flat * hr_flat, axis=1)
    norm_sr = np.linalg.norm(sr_flat, axis=1)
    norm_hr = np.linalg.norm(hr_flat, axis=1)
    denom   = norm_sr * norm_hr + eps

    cos_angle = np.clip(dot / denom, -1.0, 1.0)
    angles_rad = np.arccos(cos_angle)
    return float(np.degrees(np.mean(angles_rad)))


def compute_lpips(
    sr: torch.Tensor,
    hr: torch.Tensor,
    lpips_fn: lpips_lib.LPIPS,
    device: torch.device,
    rgb_bands: list = None,
) -> float:
    """
    Learned Perceptual Image Patch Similarity.

    Uses AlexNet features pre-trained on ImageNet.  AlexNet expects exactly
    3-channel RGB input.  We extract the True Color RGB bands from our 6-band
    multispectral tensor before passing to LPIPS.

    v2 FIX: Previously returned NaN because the full 6-band tensor was passed
    to AlexNet.  Now explicitly slices RGB bands [R=2, G=1, B=0] (configurable
    via rgb_bands) and hard-clamps to [-1, 1] as required by LPIPS.

    Lower LPIPS → more perceptual similarity to HR reference.
    For GAN models this metric captures how well logging-road textures
    are hallucinated.

    Parameters
    ----------
    sr, hr      : torch.Tensor  (1, C, H, W) in [-1, 1].
    lpips_fn    : LPIPS          Pre-initialised LPIPS network.
    device      : torch.device
    rgb_bands   : list[int]      Band indices for [R, G, B] (0-indexed, default [2,1,0]).

    Returns
    -------
    float  LPIPS score (0 = perceptually identical, higher = more different).
    """
    if rgb_bands is None:
        rgb_bands = [2, 1, 0]   # Red, Green, Blue  (0-indexed)

    n_ch = sr.shape[1]
    # Guard: clamp band indices to valid range
    rgb_idx = [min(b, n_ch - 1) for b in rgb_bands]

    # Extract 3-channel RGB slice and hard-clamp to [-1, 1]
    sr_rgb = sr[:, rgb_idx, :, :].clamp(-1.0, 1.0).to(device)
    hr_rgb = hr[:, rgb_idx, :, :].clamp(-1.0, 1.0).to(device)

    with torch.no_grad():
        score = lpips_fn(sr_rgb, hr_rgb)
    return float(score.item())


def compute_ergas(
    sr: np.ndarray,
    hr: np.ndarray,
    scale: int = 3,
    eps: float = 1e-8,
) -> float:
    """
    ERGAS — Erreur Relative Globale Adimensionnelle de Synthèse.

    The standard dimensionless quality metric for pan-sharpening and
    spatiotemporal fusion, required by IEEE TGRS and ISPRS reviewers.
    Measures spectrally-weighted, scale-normalised RMSE per band.

    Formula:
        ERGAS = (100 / scale) * sqrt( (1/C) * sum_c( (RMSE_c / mean_HR_c)^2 ) )

    Parameters
    ----------
    sr    : np.ndarray  Shape (C, H, W), values in [-1, 1].
    hr    : np.ndarray  Shape (C, H, W), values in [-1, 1].
    scale : int         Upscale factor (default 3 for 30m→10m).
    eps   : float       Numerical guard against zero-mean HR bands.

    Returns
    -------
    float  ERGAS score.  Lower = better.  A score < 3 is considered excellent
           for 3× upscaling tasks in remote sensing literature.
    """
    C = sr.shape[0]
    sum_sq = 0.0
    for c in range(C):
        rmse_c    = float(np.sqrt(np.mean((sr[c] - hr[c]) ** 2)))
        mean_hr_c = float(np.abs(hr[c]).mean()) + eps
        sum_sq   += (rmse_c / mean_hr_c) ** 2
    ergas = (100.0 / scale) * np.sqrt(sum_sq / C)
    return float(ergas)


# ─────────────────────────────────────────────────────────────────────────────
# Model inference helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer(
    generator: nn.Module,
    lr_batch: torch.Tensor,
    model_name: str,
    scale: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    """Run a single inference step and return SR output clamped to [-1, 1]."""
    lr_batch = lr_batch.to(device)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        if model_name == "bicubic":
            sr = F.interpolate(lr_batch, scale_factor=scale, mode="bicubic",
                               align_corners=False)
        elif model_name == "srcnn":
            lr_up = F.interpolate(lr_batch, scale_factor=scale, mode="bicubic",
                                   align_corners=False)
            sr = generator(lr_up)
        else:
            sr = generator(lr_batch)
    return sr.clamp(-1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation of one (model, scaling) pair
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model_name: str,
    scaling: str,
    cfg: dict,
    device: torch.device,
    amp_dtype: torch.dtype,
    lpips_fn: lpips_lib.LPIPS,
    save_images: bool = False,
    results_dir: Optional[Path] = None,
    n_save: int = 16,
    test_split: str = "test",
) -> dict:
    """
    Run full evaluation for one (model, scaling) combination.

    Returns a dict with keys: psnr, ssim, sam_deg, ergas, lpips, model, scaling.
    """
    p_cfg   = cfg["patch"]
    tr_cfg  = cfg["training"]
    paths   = cfg["paths"]
    ss_cfg  = cfg["smart_scaling"]
    n_ch    = cfg["bands"]["n_channels"]
    scale   = p_cfg["scale"]
    eval_cfg = cfg.get("evaluation", {})
    rgb_bands = eval_cfg.get("rgb_bands", [2, 1, 0])   # R, G, B (0-indexed)

    # DataLoader for the specified split
    patch_dir = Path(paths["aligned_dir"]) / test_split
    _, val_loader = build_dataloaders(
        train_patch_dir=patch_dir,
        val_patch_dir=patch_dir,
        scaling_mode=scaling,
        batch_size=1,
        num_workers=tr_cfg["num_workers"],
        pin_memory=tr_cfg["pin_memory"],
        smart_scaling_kwargs={
            "lo_pct":  ss_cfg["lo_pct"],
            "hi_pct":  ss_cfg["hi_pct"],
            "out_min": ss_cfg["out_min"],
            "out_max": ss_cfg["out_max"],
        } if scaling == "smart" else None,
    )

    # Load model
    if model_name == "bicubic":
        generator = None
    else:
        ckpt_path = Path(paths["checkpoints"]) / f"{model_name}_{scaling}" / "best.pt"
        if not ckpt_path.exists():
            logger.warning(f"  ⚠  No checkpoint found: {ckpt_path}  – skipping")
            return {}

        generator, _ = build_model(model_name, cfg, n_ch, scale)
        state = torch.load(ckpt_path, map_location=device)
        generator.load_state_dict(state["generator_state"])
        generator = generator.to(device).eval()

    psnr_list, ssim_list, sam_list, ergas_list, lpips_list = [], [], [], [], []
    saved = 0

    for idx, (lr_batch, hr_batch) in enumerate(tqdm(val_loader, desc=f"{model_name}/{scaling}", ncols=85)):
        sr = infer(generator, lr_batch, model_name, scale, device, amp_dtype)
        hr = hr_batch.to(device).clamp(-1.0, 1.0)

        sr_np = sr.squeeze(0).cpu().numpy()   # (C, H, W)
        hr_np = hr.squeeze(0).cpu().numpy()

        psnr_list.append(compute_psnr(sr_np, hr_np))
        ssim_list.append(compute_ssim(sr_np, hr_np))
        sam_list.append(compute_sam(sr_np, hr_np))
        ergas_list.append(compute_ergas(sr_np, hr_np, scale=scale))
        if lpips_fn is not None:
            lpips_list.append(compute_lpips(sr, hr, lpips_fn, device, rgb_bands=rgb_bands))

        # Optionally save SR / HR image tiles
        if save_images and results_dir is not None and saved < n_save:
            _save_tile(sr_np, hr_np, results_dir, model_name, scaling, idx, rgb_bands)
            saved += 1

    result = {
        "model":     model_name,
        "scaling":   scaling,
        "psnr":      float(np.mean(psnr_list)),
        "ssim":      float(np.mean(ssim_list)),
        "sam_deg":   float(np.mean(sam_list)),
        "ergas":     float(np.mean(ergas_list)),
        "lpips":     float(np.mean(lpips_list)) if lpips_list else float("nan"),
        "n_patches": len(psnr_list),
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Image saving helper
# ─────────────────────────────────────────────────────────────────────────────

def _save_tile(
    sr_np: np.ndarray,
    hr_np: np.ndarray,
    results_dir: Path,
    model_name: str,
    scaling: str,
    idx: int,
    rgb_bands: list = None,
) -> None:
    """Save a true-colour (R-G-B) tile as PNG for visual inspection."""
    import matplotlib.pyplot as plt
    if rgb_bands is None:
        rgb_bands = [2, 1, 0]   # R, G, B

    tile_dir = results_dir / "tiles" / f"{model_name}_{scaling}"
    tile_dir.mkdir(parents=True, exist_ok=True)

    def make_rgb(arr):
        n_ch = arr.shape[0]
        bands = [min(b, n_ch - 1) for b in rgb_bands]
        rgb = arr[bands].transpose(1, 2, 0)
        # Rescale to [0, 1] for display
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        return np.clip(rgb, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(make_rgb(sr_np));  axes[0].set_title("SR"); axes[0].axis("off")
    axes[1].imshow(make_rgb(hr_np));  axes[1].set_title("HR"); axes[1].axis("off")
    fig.suptitle(f"{model_name.upper()} | {scaling} | patch {idx}", fontsize=9)
    plt.tight_layout()
    plt.savefig(tile_dir / f"patch_{idx:04d}.png", dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Table printing / saving
# ─────────────────────────────────────────────────────────────────────────────

METRIC_BETTER = {
    "psnr":    "↑",   # higher is better
    "ssim":    "↑",
    "sam_deg": "↓",   # lower is better
    "ergas":   "↓",
    "lpips":   "↓",
}


def print_results_table(results: list[dict]) -> None:
    """Print a formatted Markdown comparison table."""
    df = pd.DataFrame(results)
    if df.empty:
        logger.warning("No results to display.")
        return

    # Pivot so each model is a row, each (metric, scaling) is a column
    metric_cols = ["psnr", "ssim", "sam_deg", "ergas", "lpips"]
    df_piv = df.pivot_table(
        index="model", columns="scaling",
        values=metric_cols, aggfunc="mean"
    ).round(4)

    border = "─" * 100
    print(f"\n{border}")
    print("  RONDÔNIA SR STUDY  –  Comparative Evaluation Results")
    print(border)
    header = f"{'Model':<12}  " + "  ".join(
        f"{m}({dir})/standard  {m}({dir})/smart  Δ"
        for m, dir in METRIC_BETTER.items())
    print(header)
    print(border)

    for model in df_piv.index:
        row = f"{model:<12}  "
        for m in metric_cols:
            try:
                std  = df_piv.loc[model, (m, "standard")]
                smt  = df_piv.loc[model, (m, "smart")]
                delta = smt - std
                sign  = "+" if delta > 0 else ""
                row  += f"  {std:>8.4f}  {smt:>8.4f}  {sign}{delta:>+.4f}"
            except KeyError:
                row += f"  {'N/A':>8}  {'N/A':>8}  {'N/A':>8}"
        print(row)
    print(border)

    # Metric banner
    print("\n  Metric guide:")
    for m, d in METRIC_BETTER.items():
        descs = {
            "psnr":    "pixel-level distortion (dB)",
            "ssim":    "structural similarity [-1,1]",
            "sam_deg": "spectral angle, degrees",
            "lpips":   "perceptual similarity (AlexNet)",
        }
        print(f"    {m:<10} {d}   {descs[m]}")


def save_results(results: list[dict], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = results_dir / "evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    md_path = results_dir / "evaluation_results.md"
    with open(md_path, "w") as f:
        f.write("# Rondônia SR Study — Evaluation Results\n\n")
        f.write(df.sort_values(["model", "scaling"]).to_markdown(index=False))
        f.write("\n\n### Metric Guide\n")
        f.write("| Metric | Better | Description |\n|--------|--------|-------------|\n")
        descs = {
            "psnr":    ("↑", "Pixel-wise distortion (dB). Higher = less error."),
            "ssim":    ("↑", "Structural similarity. 1.0 = perfect."),
            "sam_deg": ("↓", "Spectral Angle Mapper (degrees). 0° = perfect spectral match."),
            "ergas":   ("↓", "ERGAS: scale-normalised RMSE. < 3 = excellent for 3x SR."),
            "lpips":   ("↓", "Perceptual similarity (AlexNet features). 0 = perceptually identical."),
        }
        for m, (dir, desc) in descs.items():
            f.write(f"| `{m}` | {dir} | {desc} |\n")
    logger.info(f"Markdown report → {md_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    cfg      = load_config(args.config)
    tr_cfg   = cfg["training"]
    eval_cfg = cfg["evaluation"]
    paths    = cfg["paths"]

    set_seed(tr_cfg["seed"])
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if tr_cfg["amp_dtype"] == "bfloat16" else torch.float16

    logger.info(f"Evaluation device: {device}")

    # Initialise LPIPS once (reuse across all evaluations)
    # Skip if --skip-lpips passed (much faster; LPIPS adds ~5-10 min for 2700 patches)
    if args.skip_lpips:
        lpips_fn = None
        logger.info("LPIPS disabled (--skip-lpips). PSNR / SSIM / SAM only.")
    else:
        logger.info("Loading LPIPS AlexNet weights...")
        lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
        lpips_fn.eval()
        logger.info("LPIPS ready.")

    results_dir = Path(paths["results"])
    test_split  = args.test_split

    # Determine which models × scalings to evaluate
    if args.model and args.scaling:
        combos = [(args.model, args.scaling)]
    else:
        # All enabled models × both scalings
        enabled = [m for m, mc in cfg["models"].items() if mc.get("enabled", False)]
        scalings = ["standard", "smart"]
        combos   = [(m, s) for m in enabled for s in scalings]

    results = []
    for model_name, scaling in combos:
        # Bicubic is a deterministic baseline — smart scaling is only meaningful
        # for trained neural models (SRGAN, ESRGAN, SwinIR, HAT, etc.).
        # Running bicubic with smart scaling produces misleading/broken metrics
        # because the LR and HR end up in incomparable normalized domains.
        if model_name == "bicubic" and scaling == "smart":
            logger.info(f"  ⊘  Skipping bicubic/smart — bicubic uses standard scaling only.")
            continue

        logger.info(f"\n{'─' * 50}")
        logger.info(f"  Evaluating: {model_name.upper()}  |  scaling={scaling}")
        r = evaluate_model(
            model_name, scaling, cfg, device, amp_dtype, lpips_fn,
            save_images=eval_cfg.get("save_images", False),
            results_dir=results_dir,
            n_save=eval_cfg.get("n_save", 16),
            test_split=test_split,
        )
        if r:
            results.append(r)
            lpips_str  = f"LPIPS={r['lpips']:.4f}" if lpips_fn is not None else "LPIPS=skipped"
            logger.info(
                f"  PSNR={r['psnr']:.3f} dB  SSIM={r['ssim']:.4f}  "
                f"SAM={r['sam_deg']:.3f}°  ERGAS={r['ergas']:.3f}  {lpips_str}"
            )

    if results:
        print_results_table(results)
        save_results(results, results_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rondônia SR – Evaluation (v2)")
    parser.add_argument("--model",   default=None,
                        choices=["bicubic","srcnn","edsr","rcan","srgan","esrgan","swinir","hat"],
                        help="Single model to evaluate (default: all enabled in config)")
    parser.add_argument("--scaling", default=None,
                        choices=["standard", "smart"],
                        help="Single scaling mode (default: both)")
    parser.add_argument("--config",  default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--test-split", default="test",
                        choices=["train", "val", "test"],
                        help="Which data split to evaluate on (default: test). "
                             "IMPORTANT: report paper results on 'test' only.")
    parser.add_argument("--skip-lpips", action="store_true",
                        help="Skip LPIPS computation (much faster; use for quick checks)")
    args = parser.parse_args()
    main(args)

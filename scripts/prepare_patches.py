"""
scripts/prepare_patches.py  (v2)  –  Rondônia SR Study
=======================================================

Offline preprocessing pipeline:
  1. Load paired Landsat (LR, 30m) and Sentinel-2 (HR, 10m) GeoTIFF files.
  2. Apply sensor harmonization: per-channel histogram matching (Landsat → Sentinel-2).
  3. Optionally apply sub-pixel phase cross-correlation alignment.
  4. Compute alignment RMSE; discard patch pairs exceeding max_align_rmse threshold.
  5. Extract overlapping patch pairs.
  6. Generate NDVI/NDWI pseudo-labels for downstream segmentation task.
  7. Save as .npz files to data/aligned/{train,val,test}/

v2 Changes
----------
* CRITICAL FIX: split assignment now reads the scene ID prefix from the filename
  (T##=train, V##=val, TE##=test) instead of a naïve positional 70/15/15 split.
  The old positional split violated the spatial holdout design (Ji-Paraná test scenes
  may have ended up in training due to alphabetical sorting).
* Added histogram matching step (sensor harmonization).
* Added RMSE-based patch pair rejection (max_align_rmse from config).
* Added pseudo-label saving (seg_mask key in .npz).
* Reports alignment RMSE statistics at the end.

Run this ONCE before training.

Usage:
    python scripts/prepare_patches.py --config config.yaml
    python scripts/prepare_patches.py --config config.yaml --no-align   # skip alignment
    python scripts/prepare_patches.py --config config.yaml --no-histogram  # skip histogram matching
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm
import yaml

# Allow imports from parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataloader import (
    read_bands,
    extract_patches,
    align_lr_to_hr,
    apply_histogram_matching,
    compute_alignment_rmse,
    compute_ndvi_pseudo_labels,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_scene_split(filepath: Path) -> str:
    """
    Determine train/val/test split from the scene filename stem.

    Naming convention (from download_data.py):
      T01_porto_velho_dry_clear.tif   → 'train'
      V01_cacoal_dry_clear.tif        → 'val'
      TE01_ji_parana_dry_clear.tif    → 'test'

    Falls back to 'train' for unrecognised prefixes.
    """
    stem = filepath.stem.upper()
    if stem.startswith("TE"):
        return "test"
    elif stem.startswith("V"):
        return "val"
    elif stem.startswith("T"):
        return "train"
    else:
        logger.warning(
            f"Cannot determine split from filename '{filepath.name}'. "
            "Defaulting to 'train'. Rename files to T##_*, V##_*, or TE##_*."
        )
        return "train"


def main(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    paths   = cfg["paths"]
    p_cfg   = cfg["patch"]
    b_cfg   = cfg["bands"]
    pp_cfg  = cfg.get("pre_processing", {})

    max_align_rmse  = pp_cfg.get("max_align_rmse", 0.3) if not args.no_align else None
    do_histogram    = pp_cfg.get("histogram_matching", True) and not args.no_histogram
    reference_band  = pp_cfg.get("reference_band", 3)

    lr_root  = Path(paths["landsat_dir"])
    hr_root  = Path(paths["sentinel_dir"])
    out_root = Path(paths["aligned_dir"])

    lr_files = sorted(lr_root.glob("**/*.tif"))
    hr_files = sorted(hr_root.glob("**/*.tif"))

    assert len(lr_files) == len(hr_files), (
        f"Landsat ({len(lr_files)}) and Sentinel-2 ({len(hr_files)}) "
        "file counts do not match. Ensure 1-to-1 correspondence by filename."
    )

    logger.info(f"Found {len(lr_files)} paired scene(s)")
    logger.info(f"LR patch size       : {p_cfg['lr_size']} px")
    logger.info(f"HR patch size       : {p_cfg['hr_size']} px  (scale ×{p_cfg['scale']})")
    logger.info(f"Stride              : {p_cfg['stride']} px")
    logger.info(f"Min valid frac      : {p_cfg['min_valid_frac']}")
    logger.info(f"Sub-pixel alignment : {'DISABLED' if args.no_align else 'ENABLED'}")
    logger.info(f"Histogram matching  : {'DISABLED' if args.no_histogram else 'ENABLED'}")
    logger.info(f"Max align RMSE      : {max_align_rmse if max_align_rmse else 'N/A'}")
    logger.info(f"Split method        : scene-ID prefix (T=train, V=val, TE=test)")
    logger.info(f"Pseudo-labels       : NDVI/NDWI (saved as seg_mask in .npz)")

    # Assign splits by scene-ID prefix
    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    for lr_path, hr_path in zip(lr_files, hr_files):
        split = get_scene_split(lr_path)
        splits[split].append((lr_path, hr_path))

    for split_name, pairs in splits.items():
        logger.info(f"  {split_name:5s}: {len(pairs)} scene(s)")

    total_patches   = 0
    total_discarded = 0
    rmse_all        = []

    for split_name, pairs in splits.items():
        if not pairs:
            logger.info(f"  ⊘  {split_name.upper()} — no scenes, skipping.")
            continue

        split_dir = out_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        split_patches   = 0
        split_discarded = 0
        patch_idx       = 0

        logger.info(f"\n── {split_name.upper()}  ({len(pairs)} scene(s)) ──")

        for lr_path, hr_path in tqdm(pairs, desc=f"  {split_name}"):
            # ── Read bands ──────────────────────────────────────────────────
            lr_array = read_bands(lr_path, b_cfg["landsat"])   # (C, H_lr, W_lr)
            hr_array = read_bands(hr_path, b_cfg["sentinel"])  # (C, H_hr, W_hr)

            # ── Sensor harmonization: histogram matching ─────────────────────
            if do_histogram:
                try:
                    lr_array = apply_histogram_matching(lr_array, hr_array)
                    logger.debug(f"  ✓ Histogram matched: {lr_path.name}")
                except Exception as e:
                    logger.warning(f"  Histogram matching failed for {lr_path.name}: {e}")

            # ── Sub-pixel alignment ──────────────────────────────────────────
            if not args.no_align:
                try:
                    lr_array = align_lr_to_hr(
                        lr_array, hr_array,
                        upsample_factor=100,
                        reference_band=reference_band,
                    )
                except Exception as e:
                    logger.warning(f"  Alignment failed for {lr_path.name}: {e}")

            # ── Extract patches ──────────────────────────────────────────────
            pairs_extracted = extract_patches(
                lr_array, hr_array,
                lr_size=p_cfg["lr_size"],
                hr_size=p_cfg["hr_size"],
                stride=p_cfg["stride"],
                min_valid_frac=p_cfg["min_valid_frac"],
            )

            for lr_patch, hr_patch in pairs_extracted:
                # ── RMSE-based co-registration quality filter ────────────────
                if max_align_rmse is not None:
                    rmse = compute_alignment_rmse(
                        lr_patch, hr_patch, reference_band=reference_band
                    )
                    rmse_all.append(rmse)
                    if rmse > max_align_rmse:
                        split_discarded += 1
                        total_discarded += 1
                        continue

                # ── Generate NDVI pseudo-labels ──────────────────────────────
                seg_mask = compute_ndvi_pseudo_labels(hr_patch)

                # ── Save .npz ────────────────────────────────────────────────
                fname = split_dir / f"{lr_path.stem}_patch{patch_idx:06d}.npz"
                np.savez_compressed(fname, lr=lr_patch, hr=hr_patch, seg_mask=seg_mask)
                patch_idx     += 1
                split_patches += 1

        logger.info(
            f"  → {split_patches} patches saved | "
            f"{split_discarded} discarded (RMSE > {max_align_rmse}) | "
            f"dir: {split_dir}"
        )
        total_patches += split_patches

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Total patches created  : {total_patches}")
    logger.info(f"Total patches discarded: {total_discarded} (RMSE filter)")
    logger.info(f"Output directory       : {out_root.resolve()}")

    if rmse_all:
        rmse_arr = np.array(rmse_all)
        logger.info(f"\nAlignment RMSE statistics (n={len(rmse_arr)}):")
        logger.info(f"  Mean  : {rmse_arr.mean():.4f}")
        logger.info(f"  Median: {np.median(rmse_arr):.4f}")
        logger.info(f"  P95   : {np.percentile(rmse_arr, 95):.4f}")
        logger.info(f"  Max   : {rmse_arr.max():.4f}")
        logger.info(f"  (Threshold used: {max_align_rmse})")

    if total_patches > 0:
        logger.info(f"\nNext step:")
        logger.info(f"  python evaluate.py --model bicubic --scaling standard --config config.yaml")
        logger.info(f"  Expected: PSNR >= 24 dB, SAM <= 10°, LPIPS not NaN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare aligned LR/HR patch pairs (v2 — scene-ID splits, histogram matching)"
    )
    parser.add_argument("--config",        default="config.yaml")
    parser.add_argument("--no-align",      action="store_true",
                        help="Skip phase cross-correlation alignment step")
    parser.add_argument("--no-histogram",  action="store_true",
                        help="Skip histogram matching (sensor harmonization)")
    args = parser.parse_args()
    main(args)

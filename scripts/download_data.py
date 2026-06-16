"""
scripts/download_data.py  –  Rondônia SR Study  –  Automated Data Downloader
=============================================================================

Downloads paired Landsat-8/9 C2 L2  and  Sentinel-2 L2A scenes
for the 12-scene dataset described in docs/TRAINING_STRATEGY.md §0c.

Uses the Element84 Earth Search STAC API (free, no login required).
Reads Cloud-Optimised GeoTIFFs directly via rasterio windowed reads
so only the sub-region of interest is downloaded — NOT full ~1 GB scenes.

USAGE
-----
    python scripts/download_data.py            # full 12-scene run
    python scripts/download_data.py --dry-run  # search only, no download
    python scripts/download_data.py --split train   # only train scenes
    python scripts/download_data.py --scene T01     # only one scene

OUTPUTS
-------
    data/raw/
    ├── landsat/       ← sorted .tif files  (T01..T12)
    ├── sentinel/      ← sorted .tif files  (T01..T12)
    ├── scene_manifest.csv   ← full metadata for every pair
    └── .scene_index.json    ← machine-readable index

STORAGE ESTIMATES (per scene ROI = 0.5° × 0.5°)
    Landsat  6-band  30m  ≈  35–50 MB / scene
    Sentinel 6-band  10m  ≈ 300–450 MB / scene
    12 pairs total         ≈  4–6 GB raw
    Processed patches      ≈  0.8–1.2 GB (.npz)
    Model checkpoints      ≈  1.5–2 GB
    ──────────────────────────────────────────────
    Total project          ≈  8–11 GB  (on top of the .venv 1.4 GB)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pystac_client
import planetary_computer
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# STAC endpoint (Microsoft Planetary Computer)
# ─────────────────────────────────────────────────────────────────────────────
STAC_URL   = "https://planetarycomputer.microsoft.com/api/stac/v1"
LS_COLLECTION = "landsat-c2-l2"      # Landsat Collection 2 Level-2 SR
S2_COLLECTION = "sentinel-2-l2a"     # Sentinel-2 Level-2A


# ─────────────────────────────────────────────────────────────────────────────
# Band mappings
# ─────────────────────────────────────────────────────────────────────────────
# Landsat-8/9 OLI — Surface reflectance asset keys in the STAC item
LANDSAT_BANDS = {
    "blue":  "blue",    # SR_B2 → 30m
    "green": "green",   # SR_B3 → 30m
    "red":   "red",     # SR_B4 → 30m
    "nir08": "nir08",   # SR_B5 → 30m
    "swir16":"swir16",  # SR_B6 → 30m
    "swir22":"swir22",  # SR_B7 → 30m
}
LANDSAT_BAND_ORDER = ["blue", "green", "red", "nir08", "swir16", "swir22"]

# Sentinel-2 L2A — asset keys in the STAC item
# Note: B11, B12 are 20m natively; resampled to 10m on read
SENTINEL_BANDS = {
    "blue":  "blue",    # B02 → 10m
    "green": "green",   # B03 → 10m
    "red":   "red",     # B04 → 10m
    "nir":   "nir",     # B08 → 10m
    "swir16":"swir16",  # B11 → 20m (bicubic up to 10m)
    "swir22":"swir22",  # B12 → 20m (bicubic up to 10m)
}
SENTINEL_BAND_ORDER = ["blue", "green", "red", "nir", "swir16", "swir22"]

TARGET_CRS = "EPSG:4326"   # output in geographic coordinates

# ─────────────────────────────────────────────────────────────────────────────
# Scene Catalog
# 12 paired scenes as designed in docs/TRAINING_STRATEGY.md §0c
# Each entry: (scene_id, split, sub_region, season, cloud_category,
#              landscape_type, bbox [W,S,E,N], date_range, max_cloud_pct)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SceneSpec:
    scene_id:       str            # e.g. "T01"
    split:          str            # "train" | "val" | "test"
    sub_region:     str            # e.g. "porto_velho"
    season:         str            # "dry" | "wet"
    cloud_category: str            # "clear" | "partly_cloudy" | "cloudy"
    landscape:      str            # "active_deforestation" | "fishbone_roads" | etc.
    bbox:           List[float]    # [west, south, east, north]  WGS84
    date_start:     str            # "YYYY-MM-DD"
    date_end:       str            # "YYYY-MM-DD"
    max_cloud_pct:  float          # upper bound for STAC search filter

SCENE_CATALOG: List[SceneSpec] = [
    # ── TRAINING SCENES (8) ─────────────────────────────────────────────────
    SceneSpec("T01", "train", "porto_velho",  "dry", "clear",
              "active_fishbone_roads",
              [-63.8, -9.6, -63.3, -9.1],    # Porto Velho corridor
              "2022-07-01", "2022-08-31", 5.0),

    SceneSpec("T02", "train", "porto_velho",  "dry", "clear",
              "intact_forest",
              [-64.2, -9.8, -63.7, -9.3],    # Dense Amazon, untouched
              "2022-07-01", "2022-09-15", 5.0),

    SceneSpec("T03", "train", "porto_velho",  "dry", "partly_cloudy",
              "active_deforestation",
              [-63.5, -10.0, -63.0, -9.5],   # Active clearing frontier
              "2021-07-01", "2021-08-31", 15.0),

    SceneSpec("T04", "train", "porto_velho",  "dry", "cloudy",
              "mixed_cloud_clearing",
              [-64.0, -9.5, -63.5, -9.0],    # Key Smart Scaling stress test
              "2021-06-15", "2021-08-31", 35.0),

    SceneSpec("T05", "train", "ariquemes",    "wet", "clear",
              "established_clearing",
              [-63.1, -10.4, -62.6, -9.9],   # Ariquemes area
              "2022-01-01", "2022-02-28", 8.0),

    SceneSpec("T06", "train", "ariquemes",    "wet", "partly_cloudy",
              "subtle_roads_regrowth",
              [-62.8, -10.6, -62.3, -10.1],  # Hardest training case
              "2021-11-01", "2022-01-31", 20.0),

    SceneSpec("T07", "train", "porto_velho",  "wet", "cloudy",
              "heavy_cloud_forest",
              [-63.6, -9.3, -63.1, -8.8],    # Smart Scaling stress test 2
              "2021-12-01", "2022-02-28", 40.0),

    SceneSpec("T08", "train", "ariquemes",    "dry", "clear",
              "active_fishbone_roads",
              [-62.9, -9.8, -62.4, -9.3],    # Second fishbone area
              "2020-07-01", "2020-08-31", 5.0),

    # ── VALIDATION SCENES (2) ───────────────────────────────────────────────
    SceneSpec("V01", "val",   "cacoal",       "dry", "clear",
              "mixed_landscape",
              [-61.3, -11.6, -60.8, -11.1],  # Cacoal municipality
              "2022-07-01", "2022-09-15", 10.0),

    SceneSpec("V02", "val",   "cacoal",       "wet", "partly_cloudy",
              "cloud_fishbone",
              [-61.5, -11.8, -61.0, -11.3],  # Cacoal wet season
              "2022-11-01", "2023-01-31", 25.0),

    # ── TEST SCENES (2) — SPATIALLY HELD OUT (Ji-Paraná) ───────────────────
    SceneSpec("TE01", "test", "ji_parana",    "dry", "clear",
              "active_deforestation",
              [-62.2, -11.0, -61.7, -10.5],  # Ji-Paraná, Table 1 (standard)
              "2022-07-01", "2022-09-15", 5.0),

    SceneSpec("TE02", "test", "ji_parana",    "dry", "clear",
              "subtle_fishbone_roads",
              [-62.5, -11.3, -62.0, -10.8],  # Ji-Paraná, Table 2 (hard case)
              "2021-07-01", "2021-09-15", 5.0),
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def banner(text: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─'*60}\n  {text}\n{'─'*60}{RESET}")

def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET}  {msg}")

def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET}  {msg}")

def err(msg: str) -> None:
    print(f"  {RED}✗{RESET}  {msg}")

def fmt_mb(path: Path) -> str:
    if path.exists():
        return f"{path.stat().st_size / 1024**2:.1f} MB"
    return "0 MB"


def days_apart(date_a: str, date_b: str) -> int:
    """Absolute day difference between two YYYY-MM-DD strings."""
    d1 = datetime.strptime(date_a[:10], "%Y-%m-%d")
    d2 = datetime.strptime(date_b[:10], "%Y-%m-%d")
    return abs((d1 - d2).days)


def bbox_to_geojson(bbox: List[float]) -> dict:
    """Convert [W, S, E, N] to GeoJSON Polygon."""
    w, s, e, n = bbox
    return {
        "type": "Polygon",
        "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAC Search
# ─────────────────────────────────────────────────────────────────────────────

def search_stac(
    catalog,
    collection: str,
    bbox: List[float],
    date_start: str,
    date_end: str,
    max_cloud: float,
    limit: int = 50,
) -> list:
    """
    Search STAC catalog and return items sorted by cloud cover.
    Sorts in Python (server-side sortby is not supported on all backends).
    """
    results = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{date_start}/{date_end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        limit=limit,
    )
    items = list(results.items_as_dicts())
    # Sort by cloud cover ascending
    items.sort(key=lambda x: x["properties"].get("eo:cloud_cover", 99))
    return items


def find_best_pair(
    ls_items: list,
    s2_items: list,
    max_days_apart: int = 15,
) -> Optional[Tuple[dict, dict]]:
    """
    Find the best Landsat ↔ Sentinel-2 temporal pair.
    Strategy: minimise cloud_cover_sum + days_apart penalty.
    """
    best_score = float("inf")
    best_pair  = None

    for ls in ls_items:
        ls_date  = ls["properties"].get("datetime", "")[:10]
        ls_cloud = ls["properties"].get("eo:cloud_cover", 99)

        for s2 in s2_items:
            s2_date  = s2["properties"].get("datetime", "")[:10]
            s2_cloud = s2["properties"].get("eo:cloud_cover", 99)

            gap = days_apart(ls_date, s2_date)
            if gap > max_days_apart:
                continue

            # Score: combined cloud % + 1 point per day apart
            score = ls_cloud + s2_cloud + gap
            if score < best_score:
                best_score = best_pair and best_score
                best_score = score
                best_pair  = (ls, s2)

    return best_pair


# ─────────────────────────────────────────────────────────────────────────────
# COG Windowed Read — downloads only the bbox, not the full scene
# ─────────────────────────────────────────────────────────────────────────────

def read_band_window(
    asset_href: str,
    bbox: List[float],
    target_res_m: float,
    nodata_value: float = 0.0,
) -> Optional[np.ndarray]:
    """
    Read a single band from a COG (Cloud Optimised GeoTIFF) using
    windowed reading — only downloads the bytes covering the bbox.

    Returns (H, W) float32 array clipped to bbox, or None on error.
    """
    try:
        with rasterio.open(asset_href, "r") as src:
            from rasterio.warp import transform_bounds
            # Reproject bbox from WGS84 to the image CRS
            bounds_native = transform_bounds(
                "EPSG:4326", src.crs,
                bbox[0], bbox[1], bbox[2], bbox[3]
            )
            window = src.window(*bounds_native)

            # Compute output shape at target resolution
            west, south, east, north = bounds_native
            width_m  = abs(east - west)
            height_m = abs(north - south)
            # approximate metres per deg at equator for native CRS
            if src.crs.is_geographic:
                # geographic: convert deg → m approximately
                width_m  *= 111_320 * math.cos(math.radians((bbox[1]+bbox[3])/2))
                height_m *= 111_320
            out_width  = max(1, int(round(width_m  / target_res_m)))
            out_height = max(1, int(round(height_m / target_res_m)))

            data = src.read(
                1,
                window=window,
                out_shape=(out_height, out_width),
                resampling=Resampling.bilinear,
                fill_value=nodata_value,
            ).astype(np.float32)

        return data
    except Exception as exc:
        warn(f"    COG read failed: {exc}")
        return None


def download_scene(
    item: dict,
    band_order: list,
    band_map: dict,
    bbox: List[float],
    target_res_m: float,
    out_path: Path,
) -> bool:
    """
    Download a multi-band stack for one STAC item.
    Reads each band via COG windowed read, stacks them, saves GeoTIFF.
    Returns True on success.
    """
    if out_path.exists() and out_path.stat().st_size > 50_000:
        ok(f"  Exists → {out_path.name}  ({fmt_mb(out_path)})")
        return True

    assets = item.get("assets", {})
    bands: List[np.ndarray] = []
    first_shape = None

    for band_name in band_order:
        asset_key = band_map[band_name]
        asset     = assets.get(asset_key)

        # Fallback: search by common alternate key names
        if asset is None:
            # Landsat Earth Search uses SR_B2, SR_B3 etc. OR 'blue', 'green' etc.
            # Sentinel-2 uses 'blue', 'B02', 'B2' etc.
            fallback_keys = {
                "blue":   ["SR_B2", "B02", "B2", "blue"],
                "green":  ["SR_B3", "B03", "B3", "green"],
                "red":    ["SR_B4", "B04", "B4", "red"],
                "nir08":  ["SR_B5", "B08", "B8", "nir", "nir08"],
                "nir":    ["SR_B5", "B08", "B8", "nir", "nir08"],
                "swir16": ["SR_B6", "B11", "swir16", "swir_1"],
                "swir22": ["SR_B7", "B12", "swir22", "swir_2"],
            }
            for fk in fallback_keys.get(band_name, []):
                if fk in assets:
                    asset = assets[fk]
                    break

        if asset is None:
            warn(f"    Band '{band_name}' not found. Available: {list(assets.keys())[:10]}")
            return False

        href = asset.get("href", "")
        if not href:
            warn(f"    No href for band '{band_name}'")
            return False

        arr = read_band_window(href, bbox, target_res_m)
        if arr is None:
            return False

        if first_shape is None:
            first_shape = arr.shape
        elif arr.shape != first_shape:
            # Resample to first band shape (handles 20m S2 SWIR → 10m)
            import cv2
            arr = cv2.resize(arr, (first_shape[1], first_shape[0]),
                             interpolation=cv2.INTER_LINEAR)

        bands.append(arr)

    if not bands:
        return False

    stack = np.stack(bands, axis=0)   # (C, H, W)
    C, H, W = stack.shape

    # Build GeoTIFF transform for the bbox in WGS84
    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], W, H)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=H, width=W,
        count=C,
        dtype="float32",
        crs=CRS.from_epsg(4326),
        transform=transform,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(stack)
        # Write band descriptions
        for i, name in enumerate(band_order, 1):
            dst.update_tags(i, name=name)

    ok(f"  Saved → {out_path.name}  ({fmt_mb(out_path)})")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main download loop
# ─────────────────────────────────────────────────────────────────────────────

def run_downloads(args: argparse.Namespace) -> None:
    out_dir     = Path(args.output_dir)
    ls_dir      = out_dir / "landsat"
    s2_dir      = out_dir / "sentinel"
    ls_dir.mkdir(parents=True, exist_ok=True)
    s2_dir.mkdir(parents=True, exist_ok=True)

    # Filter scenes by split / scene_id
    scenes = SCENE_CATALOG
    if args.split:
        scenes = [s for s in scenes if s.split == args.split]
    if args.scene:
        scenes = [s for s in scenes if s.scene_id.upper() == args.scene.upper()]

    banner(f"Rondônia SR — Automated Data Download ({len(scenes)} scenes)")
    print(f"  STAC URL  : {STAC_URL}")
    print(f"  Output    : {out_dir.resolve()}")
    print(f"  Mode      : {'DRY RUN (no download)' if args.dry_run else 'FULL DOWNLOAD'}")
    print()

    # Open STAC catalog
    try:
        catalog = pystac_client.Client.open(
            STAC_URL,
            modifier=planetary_computer.sign_inplace
        )
        ok(f"Connected to {STAC_URL}")
    except Exception as e:
        err(f"Cannot reach STAC endpoint: {e}")
        err("Check your internet connection and try again.")
        sys.exit(1)

    # Manifest records
    manifest_rows = []
    success_count = 0
    fail_count    = 0

    for spec in scenes:
        banner(f"{spec.scene_id} | {spec.split.upper()} | {spec.season} | {spec.cloud_category} | {spec.landscape}")
        print(f"  BBox      : {spec.bbox}  ({spec.date_start} → {spec.date_end}  ≤{spec.max_cloud_pct}% cloud)")

        # ── Search Landsat ──────────────────────────────────────────────────
        print(f"  Searching Landsat C2 L2 ...")
        try:
            ls_items = search_stac(catalog, LS_COLLECTION, spec.bbox,
                                   spec.date_start, spec.date_end,
                                   spec.max_cloud_pct + 5.0)  # slight buffer
        except Exception as e:
            err(f"  Landsat search failed: {e}")
            fail_count += 1
            continue

        print(f"    Found {len(ls_items)} Landsat candidates")

        # ── Search Sentinel-2 ───────────────────────────────────────────────
        print(f"  Searching Sentinel-2 L2A ...")
        try:
            s2_items = search_stac(catalog, S2_COLLECTION, spec.bbox,
                                   spec.date_start, spec.date_end,
                                   spec.max_cloud_pct + 5.0)
        except Exception as e:
            err(f"  Sentinel-2 search failed: {e}")
            fail_count += 1
            continue

        print(f"    Found {len(s2_items)} Sentinel-2 candidates")

        if not ls_items or not s2_items:
            warn(f"  No items found — try widening date range or cloud limit")
            warn(f"  Tip: edit SCENE_CATALOG entry for {spec.scene_id} in this script")
            fail_count += 1
            continue

        # ── Find best temporal pair ─────────────────────────────────────────
        pair = find_best_pair(ls_items, s2_items, max_days_apart=args.max_days_apart)
        if pair is None:
            warn(f"  No pair within {args.max_days_apart} days — try --max-days-apart 30")
            fail_count += 1
            continue

        ls_item, s2_item = pair
        ls_date    = ls_item["properties"].get("datetime", "")[:10]
        s2_date    = s2_item["properties"].get("datetime", "")[:10]
        ls_cloud   = ls_item["properties"].get("eo:cloud_cover", "?")
        s2_cloud   = s2_item["properties"].get("eo:cloud_cover", "?")
        gap        = days_apart(ls_date, s2_date)

        print(f"  Best pair:")
        print(f"    Landsat   {ls_date}  cloud={ls_cloud:.1f}%")
        print(f"    Sentinel  {s2_date}  cloud={s2_cloud:.1f}%  (gap={gap} days)")

        # Filename: scene_id + landscape description
        fname = f"{spec.scene_id}_{spec.sub_region}_{spec.season}_{spec.cloud_category}"
        ls_out = ls_dir / f"{fname}.tif"
        s2_out = s2_dir / f"{fname}.tif"

        if args.dry_run:
            print(f"  [DRY RUN] Would save:")
            print(f"    {ls_out}")
            print(f"    {s2_out}")
            success_count += 1
            continue

        # ── Download Landsat ────────────────────────────────────────────────
        print(f"  Downloading Landsat bands ...")
        ls_ok = download_scene(ls_item, LANDSAT_BAND_ORDER, LANDSAT_BANDS,
                               spec.bbox, 30.0, ls_out)

        # ── Download Sentinel-2 ─────────────────────────────────────────────
        print(f"  Downloading Sentinel-2 bands ...")
        s2_ok = download_scene(s2_item, SENTINEL_BAND_ORDER, SENTINEL_BANDS,
                               spec.bbox, 10.0, s2_out)

        # ── Record manifest ─────────────────────────────────────────────────
        manifest_rows.append({
            "scene_id":       spec.scene_id,
            "split":          spec.split,
            "sub_region":     spec.sub_region,
            "season":         spec.season,
            "cloud_category": spec.cloud_category,
            "landscape":      spec.landscape,
            "landsat_date":   ls_date,
            "sentinel_date":  s2_date,
            "days_apart":     gap,
            "landsat_cloud":  ls_cloud,
            "sentinel_cloud": s2_cloud,
            "landsat_file":   str(ls_out),
            "sentinel_file":  str(s2_out),
            "landsat_size_mb": round(ls_out.stat().st_size / 1024**2, 1) if ls_out.exists() else 0,
            "sentinel_size_mb": round(s2_out.stat().st_size / 1024**2, 1) if s2_out.exists() else 0,
            "bbox":           str(spec.bbox),
        })

        if ls_ok and s2_ok:
            ok(f"  Scene {spec.scene_id} complete ✓")
            success_count += 1
        else:
            warn(f"  Scene {spec.scene_id} partially failed")
            fail_count += 1

        time.sleep(0.5)  # polite rate limiting

    # ── Write manifest ──────────────────────────────────────────────────────
    if manifest_rows:
        manifest_path = out_dir / "scene_manifest.csv"
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
            writer.writeheader()
            writer.writerows(manifest_rows)
        ok(f"Manifest written → {manifest_path}")

    # ── Summary ─────────────────────────────────────────────────────────────
    total_ls = sum(
        p.stat().st_size for p in ls_dir.glob("*.tif") if p.exists()
    ) / 1024**2
    total_s2 = sum(
        p.stat().st_size for p in s2_dir.glob("*.tif") if p.exists()
    ) / 1024**2

    banner("Download Summary")
    print(f"  Scenes completed : {success_count}/{len(scenes)}")
    print(f"  Scenes failed    : {fail_count}")
    print(f"  Landsat total    : {total_ls:.0f} MB  ({ls_dir})")
    print(f"  Sentinel total   : {total_s2:.0f} MB  ({s2_dir})")
    print(f"  Combined raw     : {total_ls + total_s2:.0f} MB")

    print()
    if success_count == len(scenes):
        print(f"  {GREEN}{BOLD}All scenes downloaded. Next step:{RESET}")
        print(f"  python scripts/prepare_patches.py --config config.yaml")
    elif success_count > 0:
        print(f"  {YELLOW}{BOLD}Partial success. Re-run for failed scenes.{RESET}")
    else:
        print(f"  {RED}{BOLD}All failed. Check internet connection and STAC availability.{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Storage estimate (no network required)
# ─────────────────────────────────────────────────────────────────────────────

def print_storage_estimate() -> None:
    banner("Storage Estimate for Full Dataset")

    scenes = SCENE_CATALOG
    # Rough area per scene ROI in km²
    rows = []
    total_ls = 0
    total_s2 = 0
    for spec in scenes:
        w, s, e, n = spec.bbox
        km_x = abs(e - w) * 111.32 * math.cos(math.radians((s+n)/2))
        km_y = abs(n - s) * 111.32
        area = km_x * km_y

        # Landsat: 6 bands × (km/0.03)² pixels × 2 bytes (uint16) / 0.5 compression
        ls_px  = (km_x * 1000 / 30) * (km_y * 1000 / 30)
        ls_mb  = 6 * ls_px * 2 / 1024**2 * 0.5
        # Sentinel: 4 bands × (km/0.01)², 2 SWIR bands × (km/0.02)²
        s2_px4 = (km_x * 1000 / 10) * (km_y * 1000 / 10)
        s2_px2 = (km_x * 1000 / 20) * (km_y * 1000 / 20)
        s2_mb  = (4 * s2_px4 + 2 * s2_px2) * 2 / 1024**2 * 0.4

        rows.append((spec.scene_id, spec.split, f"{area:.0f} km²",
                     f"~{ls_mb:.0f} MB", f"~{s2_mb:.0f} MB"))
        total_ls += ls_mb
        total_s2 += s2_mb

    print(f"  {'ID':<6} {'Split':<6} {'Area':<10} {'Landsat':<12} {'Sentinel-2'}")
    print(f"  {'─'*6} {'─'*6} {'─'*10} {'─'*12} {'─'*12}")
    for row in rows:
        print(f"  {row[0]:<6} {row[1]:<6} {row[2]:<10} {row[3]:<12} {row[4]}")

    print()
    print(f"  {'─'*55}")
    print(f"  Raw data total          : ~{total_ls:.0f} MB Landsat + ~{total_s2:.0f} MB Sentinel")
    print(f"                            = ~{(total_ls + total_s2)/1024:.1f} GB raw")
    print(f"  Processed patches       : ~0.8–1.2 GB  (3,000 × 48²+144² × 6ch float32, compressed)")
    print(f"  Model checkpoints       : ~1.5–2.0 GB  (8 models × 2 scalings × ~120 MB avg)")
    print(f"  Results / logs          : ~0.2 GB")
    print(f"  Python .venv (existing) : ~1.4 GB")
    print(f"  {'─'*55}")
    total_gb = (total_ls + total_s2)/1024 + 1.2 + 2.0 + 0.2 + 1.4
    print(f"  {BOLD}Estimated total on disk : ~{total_gb:.1f} GB{RESET}")
    print(f"  Available now (after cleanup) : ~14 GB")
    headroom = 14 - total_gb
    status = GREEN if headroom > 3 else YELLOW if headroom > 1 else RED
    print(f"  {status}Estimated headroom      : ~{headroom:.1f} GB{RESET}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rondônia SR — Landsat + Sentinel-2 Data Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir", default="data/raw",
        help="Root output directory  (default: data/raw/)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Search STAC and report pairs without downloading"
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default=None,
        help="Download only one data split"
    )
    parser.add_argument(
        "--scene", default=None,
        help="Download a single scene by ID (e.g. T01, V01, TE02)"
    )
    parser.add_argument(
        "--max-days-apart", type=int, default=15,
        help="Maximum days between Landsat and Sentinel-2 acquisition (default: 15)"
    )
    parser.add_argument(
        "--estimate-only", action="store_true",
        help="Print storage estimates and exit (no network required)"
    )
    args = parser.parse_args()

    print_storage_estimate()

    if not args.estimate_only:
        run_downloads(args)

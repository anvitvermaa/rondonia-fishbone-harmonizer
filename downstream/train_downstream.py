"""
downstream/train_downstream.py  –  Rondônia SR Study  –  Downstream Segmentation
==================================================================================

Trains a lightweight U-Net semantic segmenter on SR outputs from each model
and evaluates deforestation / logging-road detection performance.

Scientific motivation
---------------------
SR is ultimately a means to an end — what matters is whether higher-resolution
imagery improves downstream analysis tasks. This script provides the scientific
proof by training the same U-Net architecture on:
  (a) Bicubic 30m → 10m upscaled LR patches   (baseline)
  (b) SR output from each trained model         (per-model)
  (c) Native 10m Sentinel-2 HR patches          (upper bound)

Reporting F1-score and IoU per class per model proves (or disproves) whether
SR actually helps deforestation monitoring — which is the core scientific claim
of the Rondônia SR paper.

Usage
-----
    # Train segmenter on SR outputs from all models
    python downstream/train_downstream.py --config config.yaml

    # Train on a specific SR model (e.g., esrgan / smart scaling)
    python downstream/train_downstream.py --config config.yaml --sr-model esrgan --scaling smart

    # Train on native HR (upper bound)
    python downstream/train_downstream.py --config config.yaml --use-hr

Output
------
    results/downstream_results.csv   — F1-score and IoU per model per class
    results/downstream_results.md    — Markdown table for the paper
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml

# Allow imports from parent directory when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from downstream.unet import UNet
from downstream.labels import compute_ndvi_pseudo_labels, compute_class_distribution
from dataloader import build_dataloaders, standard_scaling, apply_smart_scaling

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: loads pre-extracted patches and generates labels on-the-fly
# ─────────────────────────────────────────────────────────────────────────────

class SegDataset(Dataset):
    """
    PyTorch Dataset for downstream segmentation.

    Loads .npz patch files from the aligned patch directory.  If the patch
    contains a 'seg_mask' key (saved by prepare_patches.py v2), uses that.
    Otherwise, generates NDVI pseudo-labels on-the-fly from the HR patch.

    Parameters
    ----------
    patch_dir   : Path  Directory of .npz patch files.
    input_type  : str   'lr' (use LR patch), 'hr' (use HR patch), or
                        'sr' (use SR reconstruction — requires sr_outputs dict).
    sr_outputs  : dict  {patch_stem: np.ndarray}  Pre-computed SR arrays.
                        Only used when input_type='sr'.
    scaling_mode : str  'standard' or 'smart'.
    smart_kwargs : dict  Arguments for apply_smart_scaling() if mode='smart'.
    """

    def __init__(
        self,
        patch_dir:    Path,
        input_type:   str  = "hr",
        sr_outputs:   Optional[dict] = None,
        scaling_mode: str  = "standard",
        smart_kwargs: Optional[dict] = None,
    ) -> None:
        self.patch_files = sorted(patch_dir.glob("*.npz"))
        self.input_type  = input_type
        self.sr_outputs  = sr_outputs or {}
        self.scaling_mode = scaling_mode
        self.smart_kwargs = smart_kwargs or {}

        if not self.patch_files:
            raise FileNotFoundError(f"No .npz files found in {patch_dir}")
        logger.info(f"  Loaded {len(self.patch_files)} patches from {patch_dir}")

    def __len__(self) -> int:
        return len(self.patch_files)

    def __getitem__(self, idx: int):
        data = np.load(self.patch_files[idx])
        lr   = data["lr"].astype(np.float32)    # (C, H_lr, W_lr)
        hr   = data["hr"].astype(np.float32)    # (C, H_hr, W_hr)

        # ── Semantic label ───────────────────────────────────────────────────
        if "seg_mask" in data:
            mask = data["seg_mask"].astype(np.int64)
        else:
            mask = compute_ndvi_pseudo_labels(hr).astype(np.int64)

        # ── Input image ──────────────────────────────────────────────────────
        if self.input_type == "hr":
            img = hr
        elif self.input_type == "lr":
            # Bicubic upsample LR to HR resolution (3× for 30m → 10m)
            lr_t  = torch.from_numpy(lr).unsqueeze(0)
            hr_t  = F.interpolate(lr_t, size=(hr.shape[1], hr.shape[2]),
                                  mode="bicubic", align_corners=False)
            img   = hr_t.squeeze(0).numpy()
        elif self.input_type == "sr":
            stem = self.patch_files[idx].stem
            if stem in self.sr_outputs:
                img = self.sr_outputs[stem]
            else:
                # Fall back to bicubic if SR not found
                lr_t = torch.from_numpy(lr).unsqueeze(0)
                hr_t = F.interpolate(lr_t, size=(hr.shape[1], hr.shape[2]),
                                     mode="bicubic", align_corners=False)
                img  = hr_t.squeeze(0).numpy()
        else:
            img = hr

        # ── Normalise ────────────────────────────────────────────────────────
        if self.scaling_mode == "smart":
            img = apply_smart_scaling(img, **self.smart_kwargs)
        else:
            img = standard_scaling(img)

        return torch.from_numpy(img), torch.from_numpy(mask)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou_f1(
    preds:  torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
) -> tuple[list[float], list[float]]:
    """
    Compute per-class IoU and F1-score from prediction and label tensors.

    Parameters
    ----------
    preds  : torch.Tensor  Shape (N,) predicted class indices.
    labels : torch.Tensor  Shape (N,) ground truth class indices.

    Returns
    -------
    (iou_per_class, f1_per_class)  Lists of floats, length n_classes.
    """
    iou_per_class = []
    f1_per_class  = []

    for c in range(n_classes):
        pred_c  = (preds  == c)
        label_c = (labels == c)

        tp = float((pred_c & label_c).sum())
        fp = float((pred_c & ~label_c).sum())
        fn = float((~pred_c & label_c).sum())

        denom_iou = tp + fp + fn
        iou = tp / denom_iou if denom_iou > 0 else 0.0

        denom_f1 = 2 * tp + fp + fn
        f1  = (2 * tp) / denom_f1 if denom_f1 > 0 else 0.0

        iou_per_class.append(iou)
        f1_per_class.append(f1)

    return iou_per_class, f1_per_class


# ─────────────────────────────────────────────────────────────────────────────
# Training + evaluation of one U-Net (one SR model)
# ─────────────────────────────────────────────────────────────────────────────

def train_and_eval(
    train_dataset: SegDataset,
    val_dataset:   SegDataset,
    n_classes:     int,
    cfg:           dict,
    device:        torch.device,
    input_label:   str,
) -> dict:
    """
    Train a U-Net on `train_dataset` and evaluate on `val_dataset`.

    Returns a dict with per-class and mean IoU, F1, and model label.
    """
    ds_cfg = cfg.get("downstream", {})
    epochs     = ds_cfg.get("epochs",     50)
    lr         = ds_cfg.get("lr",         1e-3)
    batch_size = ds_cfg.get("batch_size", 8)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    model     = UNet(in_channels=cfg["bands"]["n_channels"], n_classes=n_classes).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    logger.info(f"  U-Net params: {model.count_parameters():,}")

    best_miou = 0.0
    best_state: Optional[dict] = None

    for epoch in range(epochs):
        # ── Training ─────────────────────────────────────────────────────────
        model.train()
        for imgs, masks in train_loader:
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(imgs)
            loss   = criterion(logits, masks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # ── Validation (every 5 epochs) ──────────────────────────────────────
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            iou_list, f1_list = evaluate_segmenter(model, val_loader, n_classes, device)
            miou = float(np.mean(iou_list))

            logger.info(
                f"  Epoch {epoch+1:3d}/{epochs}  mIoU={miou:.4f}  "
                f"[f/d/r] F1={f1_list[0]:.3f}/{f1_list[1]:.3f}/{f1_list[2]:.3f}"
            )

            if miou > best_miou:
                best_miou  = miou
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Final evaluation with best weights ───────────────────────────────────
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    iou_list, f1_list = evaluate_segmenter(model, val_loader, n_classes, device)
    miou = float(np.mean(iou_list))
    mf1  = float(np.mean(f1_list))

    result = {
        "input":                 input_label,
        "miou":                  round(miou, 4),
        "mf1":                   round(mf1,  4),
        "iou_forest":            round(iou_list[0], 4),
        "iou_deforestation":     round(iou_list[1], 4),
        "iou_logging_road":      round(iou_list[2], 4),
        "f1_forest":             round(f1_list[0], 4),
        "f1_deforestation":      round(f1_list[1], 4),
        "f1_logging_road":       round(f1_list[2], 4),
    }
    logger.info(f"  ── {input_label:<30s} mIoU={miou:.4f}  mF1={mf1:.4f}")
    return result


@torch.no_grad()
def evaluate_segmenter(
    model:      UNet,
    val_loader: DataLoader,
    n_classes:  int,
    device:     torch.device,
) -> tuple[list[float], list[float]]:
    """Run inference on val_loader and return per-class IoU and F1."""
    model.eval()
    all_preds  = []
    all_labels = []

    for imgs, masks in val_loader:
        imgs   = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        preds  = logits.argmax(dim=1).cpu().view(-1)
        all_preds.append(preds)
        all_labels.append(masks.view(-1))

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    model.train()
    return compute_iou_f1(all_preds, all_labels, n_classes)


# ─────────────────────────────────────────────────────────────────────────────
# Results saving
# ─────────────────────────────────────────────────────────────────────────────

def save_downstream_results(results: list[dict], results_dir: Path) -> None:
    """Save downstream segmentation results to CSV and Markdown."""
    try:
        import pandas as pd
        results_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(results)

        csv_path = results_dir / "downstream_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"  Results saved → {csv_path}")

        md_path = results_dir / "downstream_results.md"
        with open(md_path, "w") as f:
            f.write("# Rondônia SR Study — Downstream Segmentation Results\n\n")
            f.write(
                "U-Net trained separately on each SR model output (same architecture, "
                "same hyperparameters). Evaluating on held-out test patches.\n\n"
            )
            f.write("## Metrics\n")
            f.write("- **mIoU**: Mean Intersection-over-Union (↑ better)\n")
            f.write("- **mF1**: Mean F1-score (↑ better)\n")
            f.write("- Classes: 0=forest, 1=deforestation, 2=logging_road\n\n")
            f.write(df.sort_values("miou", ascending=False).to_markdown(index=False))
            f.write("\n\n> Note: Labels generated from NDVI/SWIR pseudo-labelling. ")
            f.write("Validate against PRODES for final publication.\n")
        logger.info(f"  Markdown report → {md_path}")

    except ImportError:
        logger.warning("pandas not installed; skipping CSV/Markdown output.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ds_cfg    = cfg.get("downstream", {})
    n_classes = ds_cfg.get("n_classes", 3)
    paths     = cfg["paths"]
    ss_cfg    = cfg["smart_scaling"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Downstream segmentation device: {device}")

    aligned_dir  = Path(paths["aligned_dir"])
    results_dir  = Path(paths["results"])
    ckpt_dir     = Path(paths["checkpoints"])

    scaling_mode = args.scaling
    smart_kwargs = {
        "lo_pct":  ss_cfg["lo_pct"],
        "hi_pct":  ss_cfg["hi_pct"],
        "out_min": ss_cfg["out_min"],
        "out_max": ss_cfg["out_max"],
    } if scaling_mode == "smart" else {}

    results = []

    # ── What to evaluate ─────────────────────────────────────────────────────
    # Build list of (input_label, input_type, sr_model_name_or_None)
    eval_configs = []

    if args.use_hr:
        # Native 10m HR patches (upper bound)
        eval_configs.append(("hr_native", "hr", None))
    elif args.sr_model:
        # Single SR model
        eval_configs.append((f"{args.sr_model}_{scaling_mode}", "sr", args.sr_model))
    else:
        # All enabled models + bicubic baseline + HR upper bound
        eval_configs.append(("bicubic_baseline", "lr", None))
        eval_configs.append(("hr_native",        "hr", None))
        for model_name, mc in cfg["models"].items():
            if mc.get("enabled", False):
                eval_configs.append(
                    (f"{model_name}_{scaling_mode}", "sr", model_name)
                )

    for input_label, input_type, sr_model in eval_configs:
        logger.info(f"\n{'─'*60}")
        logger.info(f"  Training segmenter on: {input_label}")

        # ── Build SR outputs dict (if needed) ────────────────────────────────
        sr_outputs = {}
        if input_type == "sr" and sr_model is not None:
            ckpt_path = ckpt_dir / f"{sr_model}_{scaling_mode}" / "best.pt"
            if not ckpt_path.exists():
                logger.warning(f"  ⚠ No checkpoint: {ckpt_path} — skipping {input_label}")
                continue

            from train import build_model, load_config
            n_ch  = cfg["bands"]["n_channels"]
            scale = cfg["patch"]["scale"]
            gen, _ = build_model(sr_model, cfg, n_ch, scale)
            state  = torch.load(ckpt_path, map_location=device)
            gen.load_state_dict(state["generator_state"])
            gen    = gen.to(device).eval()

            logger.info(f"  Generating SR outputs from {ckpt_path.name}...")
            from dataloader import RondoniaDataset
            for split in ["train", "val", "test"]:
                split_dir = aligned_dir / split
                if not split_dir.exists():
                    continue
                ds  = RondoniaDataset(split_dir, scaling_mode, smart_kwargs or None)
                for idx in tqdm(range(len(ds)), desc=f"  SR {split}", ncols=80):
                    lr_t, _ = ds[idx]
                    lr_t    = lr_t.unsqueeze(0).to(device)
                    with torch.no_grad():
                        sr_t = gen(lr_t).clamp(-1, 1)
                    stem = ds.patch_files[idx].stem
                    sr_outputs[stem] = sr_t.squeeze(0).cpu().numpy()

        # ── Datasets ─────────────────────────────────────────────────────────
        train_ds = SegDataset(
            aligned_dir / "train",
            input_type=input_type, sr_outputs=sr_outputs,
            scaling_mode=scaling_mode, smart_kwargs=smart_kwargs,
        )
        val_ds = SegDataset(
            aligned_dir / "val",
            input_type=input_type, sr_outputs=sr_outputs,
            scaling_mode=scaling_mode, smart_kwargs=smart_kwargs,
        )

        result = train_and_eval(train_ds, val_ds, n_classes, cfg, device, input_label)
        results.append(result)

    if results:
        save_downstream_results(results, results_dir)

        # ── Summary table ─────────────────────────────────────────────────────
        print(f"\n{'─'*70}")
        print("  DOWNSTREAM SEGMENTATION RESULTS")
        print(f"{'─'*70}")
        print(f"  {'Input':<30} {'mIoU':>8} {'mF1':>8} {'Road F1':>8}")
        print(f"{'─'*70}")
        for r in sorted(results, key=lambda x: -x["miou"]):
            print(
                f"  {r['input']:<30} {r['miou']:>8.4f} {r['mf1']:>8.4f} "
                f"{r['f1_logging_road']:>8.4f}"
            )
        print(f"{'─'*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rondônia SR – Downstream Segmentation Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config",    default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--sr-model",  default=None,
                        choices=["bicubic","srcnn","edsr","rcan","srgan","esrgan","swinir","hat"],
                        help="SR model to evaluate (default: all enabled in config)")
    parser.add_argument("--scaling",   default="standard",
                        choices=["standard", "smart"],
                        help="Scaling mode used for SR training (default: standard)")
    parser.add_argument("--use-hr",    action="store_true",
                        help="Train/evaluate on native 10m HR patches (upper bound)")
    args = parser.parse_args()
    main(args)

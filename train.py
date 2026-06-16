"""
train.py  (v2)  –  Rondônia SR Comparative Study  –  Master Training Loop
=========================================================================

Usage:
    # Train EDSR with Smart Scaling
    python train.py --model edsr --scaling smart --config config.yaml

    # Train ESRGAN with standard scaling, resume from checkpoint
    python train.py --model esrgan --scaling standard --resume checkpoints/esrgan_standard/latest.pt

    # Train SRGAN with EMA (auto-enabled for GAN models)
    python train.py --model srgan --scaling smart --config config.yaml

All hardware optimisations targeted at RTX 3060 (12 GB VRAM):
 • torch.cuda.amp.autocast(dtype=torch.bfloat16)  – mixed precision
 • Gradient accumulation (accum_steps from config)
 • GAN discriminator backward uses detached generator output
 • EMA of generator weights (decay=0.999) for SRGAN/ESRGAN stability
 • Robust checkpointing saves generator + discriminator + both optimizers
   + EMA weights (best_ema.pt) for GAN models

v2 Fixes
--------
* CosineAnnealingLR T_max bug: was using tr_cfg["num_epochs"] (global default)
  instead of the resolved per-model `num_epochs` variable. Fixed to use the
  resolved value, ensuring the LR schedule is correct for models with non-default
  epoch counts (e.g., SRCNN=150, HAT=500).
* Added EMAModel for GAN training stability (SRGAN / ESRGAN only).
  EMA decay=0.999; EMA weights saved as best_ema.pt alongside best.pt.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tqdm import tqdm
import yaml

from dataloader import build_dataloaders
from models import (
    SRCNN, EDSR, RCAN,
    SRResNet, SRGANDiscriminator,
    RRDBNet, ESRGANDiscriminator,
    SwinIR, HAT,
)
from models.srgan import VGGPerceptualLoss

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def bicubic_upsample(lr: torch.Tensor, scale: int) -> torch.Tensor:
    """Deterministic bicubic baseline — no parameters to train."""
    return F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)


# ─────────────────────────────────────────────────────────────────────────────
# EMA (Exponential Moving Average) of Generator Weights
# ─────────────────────────────────────────────────────────────────────────────

class EMAModel:
    """
    Exponential Moving Average of generator weights.

    Prevents catastrophic forgetting of good solutions during adversarial
    training oscillations.  After each generator update, the EMA shadow
    weights are updated as:

        ema_w = decay * ema_w + (1 - decay) * current_w

    At inference / validation time, swap in the EMA weights for evaluation.
    Applied only to GAN models (SRGAN / ESRGAN) where training oscillations
    are most pronounced.

    Usage
    -----
        ema = EMAModel(generator, decay=0.999)
        # After each generator optimizer.step():
        ema.update(generator)
        # To evaluate with EMA weights:
        with ema.apply(generator):
            val_psnr = validate(...)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        # Deep-copy the initial weights as the EMA shadow
        self.shadow: dict[str, torch.Tensor] = {
            name: param.data.clone().detach()
            for name, param in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA shadow weights from current model weights."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self, model: nn.Module):
        """Context manager: temporarily swap EMA weights into model."""
        return _EMAContext(model, self.shadow)

    def state_dict(self) -> dict:
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict, device: torch.device) -> None:
        self.shadow = {k: v.to(device) for k, v in state.items()}


class _EMAContext:
    """Internal context manager used by EMAModel.apply()."""

    def __init__(self, model: nn.Module, shadow: dict) -> None:
        self.model  = model
        self.shadow = shadow
        self.backup: dict[str, torch.Tensor] = {}

    def __enter__(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return self

    def __exit__(self, *args):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_name: str, cfg: dict, n_ch: int, scale: int):
    """Return (generator, discriminator_or_None) for the given model name."""
    mc = cfg["models"].get(model_name, {})  # model-specific config

    if model_name == "srcnn":
        return SRCNN(num_channels=n_ch), None

    elif model_name == "edsr":
        return EDSR(
            scale=scale,
            n_resblocks=mc.get("n_resblocks", 16),
            n_feats=mc.get("n_feats", 64),
            n_colors=n_ch,
            res_scale=mc.get("res_scale", 0.1),
        ), None

    elif model_name == "rcan":
        return RCAN(
            scale=scale,
            n_resgroups=mc.get("n_resgroups", 5),
            n_resblocks=mc.get("n_resblocks", 10),
            n_feats=mc.get("n_feats", 64),
            reduction=mc.get("reduction", 16),
            n_colors=n_ch,
        ), None

    elif model_name == "srgan":
        gen  = SRResNet(scale=scale, n_resblocks=mc.get("n_resblocks_g", 16), n_channels=n_ch)
        disc = SRGANDiscriminator(n_channels=n_ch)
        return gen, disc

    elif model_name == "esrgan":
        gen  = RRDBNet(scale=scale, n_rrdb=mc.get("n_rrdb", 16), n_channels=n_ch)
        disc = ESRGANDiscriminator(n_channels=n_ch)
        return gen, disc

    elif model_name == "swinir":
        return SwinIR(
            img_size=mc.get("img_size", 48),
            in_chans=n_ch,
            embed_dim=mc.get("embed_dim", 60),
            depths=mc.get("depths", [6, 6, 6, 6]),
            num_heads=mc.get("num_heads", [6, 6, 6, 6]),
            window_size=mc.get("window_size", 8),
            mlp_ratio=mc.get("mlp_ratio", 2.0),
            upscale=scale,
            upsampler=mc.get("upsampler", "pixelshuffle"),
        ), None

    elif model_name == "hat":
        return HAT(
            img_size=mc.get("img_size", 48),
            in_chans=n_ch,
            embed_dim=mc.get("embed_dim", 48),
            depths=mc.get("depths", [6, 6, 6, 6]),
            num_heads=mc.get("num_heads", [6, 6, 6, 6]),
            window_size=mc.get("window_size", 8),
            mlp_ratio=mc.get("mlp_ratio", 2.0),
            upscale=scale,
            upsampler=mc.get("upsampler", "pixelshuffle"),
            compress_ratio=mc.get("compress_ratio", 3),
            squeeze_factor=mc.get("squeeze_factor", 30),
        ), None

    raise ValueError(f"Unknown model: {model_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    save_dir: Path,
    model_name: str,
    epoch: int,
    generator: nn.Module,
    opt_g: torch.optim.Optimizer,
    discriminator: Optional[nn.Module] = None,
    opt_d: Optional[torch.optim.Optimizer] = None,
    metrics: Optional[dict] = None,
    is_best: bool = False,
    ema: Optional["EMAModel"] = None,
) -> None:
    """
    Save a portable checkpoint containing all state needed to resume training.

    Saves:
      - Generator weights + Adam optimizer state (m/v moments + step count)
      - Discriminator weights + Adam optimizer state (if GAN)
      - EMA shadow weights (if ema is provided — GAN models only)
      - Current epoch and metrics dict
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch":           epoch,
        "model_name":      model_name,
        "generator_state": generator.state_dict(),
        "opt_g_state":     opt_g.state_dict(),
        "metrics":         metrics or {},
    }
    if discriminator is not None:
        state["discriminator_state"] = discriminator.state_dict()
    if opt_d is not None:
        state["opt_d_state"] = opt_d.state_dict()
    if ema is not None:
        state["ema_state"] = ema.state_dict()

    latest_path = save_dir / "latest.pt"
    torch.save(state, latest_path)
    logger.info(f"  ✓ Checkpoint saved → {latest_path}")

    if is_best:
        best_path = save_dir / "best.pt"
        torch.save(state, best_path)
        logger.info(f"  ★ Best model updated → {best_path}")

        # Also save a separate EMA-weights-only checkpoint for evaluation
        if ema is not None:
            ema_state = {
                "epoch":           epoch,
                "model_name":      model_name,
                "generator_state": ema.state_dict(),  # EMA weights as generator_state
                "metrics":         metrics or {},
            }
            ema_path = save_dir / "best_ema.pt"
            torch.save(ema_state, ema_path)
            logger.info(f"  ★ Best EMA weights → {ema_path}")


def load_checkpoint(
    ckpt_path: str,
    generator: nn.Module,
    opt_g: torch.optim.Optimizer,
    discriminator: Optional[nn.Module] = None,
    opt_d: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> int:
    """
    Load checkpoint and restore model + optimizer states.
    Returns the epoch to resume from.
    """
    state = torch.load(ckpt_path, map_location=device)
    generator.load_state_dict(state["generator_state"])
    opt_g.load_state_dict(state["opt_g_state"])
    if discriminator is not None and "discriminator_state" in state:
        discriminator.load_state_dict(state["discriminator_state"])
    if opt_d is not None and "opt_d_state" in state:
        opt_d.load_state_dict(state["opt_d_state"])
    epoch = state.get("epoch", 0)
    logger.info(f"  ✓ Resumed from {ckpt_path} (epoch {epoch})")
    return epoch


# ─────────────────────────────────────────────────────────────────────────────
# SRCNN pre-processes differently (bicubic upsample before forward pass)
# ─────────────────────────────────────────────────────────────────────────────

def generator_forward(
    model_name: str,
    generator: nn.Module,
    lr: torch.Tensor,
    scale: int,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    """Run generator forward pass with AMP autocast."""
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        if model_name == "srcnn":
            lr_up = F.interpolate(lr, scale_factor=scale, mode="bicubic",
                                   align_corners=False)
            return generator(lr_up)
        return generator(lr)


# ─────────────────────────────────────────────────────────────────────────────
# Non-GAN Training Step
# ─────────────────────────────────────────────────────────────────────────────

def train_step_regr(
    model_name: str,
    generator: nn.Module,
    lr_batch: torch.Tensor,
    hr_batch: torch.Tensor,
    opt_g: torch.optim.Optimizer,
    scaler: GradScaler,
    scale: int,
    amp_dtype: torch.dtype,
    accum_steps: int,
    step: int,
) -> float:
    """
    Single optimisation step for regression (L1) models
    (SRCNN / EDSR / RCAN / SwinIR / HAT).

    Uses gradient accumulation and AMP bfloat16 autocast.
    """
    sr = generator_forward(model_name, generator, lr_batch, scale, amp_dtype)
    loss = F.l1_loss(sr, hr_batch) / accum_steps          # scale loss for accumulation

    scaler.scale(loss).backward()

    if (step + 1) % accum_steps == 0:
        scaler.unscale_(opt_g)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        scaler.step(opt_g)
        scaler.update()
        opt_g.zero_grad(set_to_none=True)

    return loss.item() * accum_steps   # return unscaled loss for logging


# ─────────────────────────────────────────────────────────────────────────────
# GAN Training Step
# ─────────────────────────────────────────────────────────────────────────────

def train_step_gan(
    model_name: str,
    generator: nn.Module,
    discriminator: nn.Module,
    lr_batch: torch.Tensor,
    hr_batch: torch.Tensor,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    scaler_g: GradScaler,
    scaler_d: GradScaler,
    percep_loss_fn: VGGPerceptualLoss,
    scale: int,
    amp_dtype: torch.dtype,
    accum_steps: int,
    step: int,
    gan_cfg: dict,
) -> tuple[float, float]:
    """
    GAN training step with:
    • AMP bfloat16 autocast
    • Gradient accumulation for both G and D
    • CRITICAL: sr_detached used in D backward pass to prevent autograd crash

    Returns (g_loss, d_loss) for logging.
    """
    lambda_px   = gan_cfg.get("lambda_pixel", 1e-2)
    lambda_feat = gan_cfg.get("lambda_feat",  1.0)
    lambda_adv  = gan_cfg.get("lambda_adv_g", 5e-3)

    # ── Generator forward ──────────────────────────────────────────────────
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        sr = generator_forward(model_name, generator, lr_batch, scale, amp_dtype)

    # ── Discriminator update ───────────────────────────────────────────────
    # CRITICAL: .detach() stops gradients from flowing into G during D update
    sr_detached = sr.detach()

    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        real_logits = discriminator(hr_batch)
        fake_logits = discriminator(sr_detached)   # ← detached SR image
        # Relativistic average GAN loss (ESRGAN) / standard BCE (SRGAN)
        if model_name == "esrgan":
            d_real = real_logits - fake_logits.mean()
            d_fake = fake_logits - real_logits.mean()
            d_loss = (
                F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) +
                F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            ) / (2 * accum_steps)
        else:
            d_loss = (
                F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) +
                F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            ) / (2 * accum_steps)

    scaler_d.scale(d_loss).backward()
    if (step + 1) % accum_steps == 0:
        scaler_d.unscale_(opt_d)
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        scaler_d.step(opt_d)
        scaler_d.update()
        opt_d.zero_grad(set_to_none=True)

    # ── Generator update ───────────────────────────────────────────────────
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        fake_logits_g = discriminator(sr)   # non-detached for G gradient
        if model_name == "esrgan":
            real_logits_g = discriminator(hr_batch).detach()   # detach real for G update
            g_adv = (
                F.binary_cross_entropy_with_logits(
                    fake_logits_g - real_logits_g.mean(), torch.ones_like(fake_logits_g)) +
                F.binary_cross_entropy_with_logits(
                    real_logits_g - fake_logits_g.mean(), torch.zeros_like(real_logits_g))
            ) / 2
        else:
            g_adv = F.binary_cross_entropy_with_logits(
                fake_logits_g, torch.ones_like(fake_logits_g))

        g_pixel = F.l1_loss(sr, hr_batch)
        g_feat  = percep_loss_fn(sr, hr_batch)
        g_loss  = (lambda_px * g_pixel + lambda_feat * g_feat + lambda_adv * g_adv) / accum_steps

    scaler_g.scale(g_loss).backward()
    if (step + 1) % accum_steps == 0:
        scaler_g.unscale_(opt_g)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        scaler_g.step(opt_g)
        scaler_g.update()
        opt_g.zero_grad(set_to_none=True)

    return g_loss.item() * accum_steps, d_loss.item() * accum_steps


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    cfg      = load_config(args.config)
    tr_cfg   = cfg["training"]
    gan_cfg  = cfg["gan"]
    p_cfg    = cfg["patch"]
    paths    = cfg["paths"]
    ss_cfg   = cfg["smart_scaling"]

    set_seed(tr_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    amp_dtype = torch.bfloat16 if tr_cfg["amp_dtype"] == "bfloat16" else torch.float16
    is_gan    = args.model in ("srgan", "esrgan")
    n_ch      = cfg["bands"]["n_channels"]
    scale     = p_cfg["scale"]
    accum     = tr_cfg["accum_steps"]

    # ── Per-model epoch count (model config overrides global default) ──────
    # Rationale: SRCNN converges in 150 epochs; HAT needs 500.
    # See docs/TRAINING_STRATEGY.md §3 for full calibration table.
    model_cfg    = cfg["models"].get(args.model, {})
    num_epochs   = (
        args.epochs                                # CLI flag (highest priority)
        if args.epochs is not None
        else model_cfg.get("num_epochs",           # per-model config
             tr_cfg.get("num_epochs", 300))        # global default fallback
    )

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        train_patch_dir=Path(paths["aligned_dir"]) / "train",
        val_patch_dir=Path(paths["aligned_dir"]) / "val",
        scaling_mode=args.scaling,
        batch_size=tr_cfg["batch_size"],
        num_workers=tr_cfg["num_workers"],
        pin_memory=tr_cfg["pin_memory"],
        smart_scaling_kwargs={
            "lo_pct":  ss_cfg["lo_pct"],
            "hi_pct":  ss_cfg["hi_pct"],
            "out_min": ss_cfg["out_min"],
            "out_max": ss_cfg["out_max"],
        } if args.scaling == "smart" else None,
    )

    # ── Models ────────────────────────────────────────────────────────────
    generator, discriminator = build_model(args.model, cfg, n_ch, scale)
    generator = generator.to(device)
    if discriminator is not None:
        discriminator = discriminator.to(device)

    # ── Optimizers ────────────────────────────────────────────────────────
    opt_g = Adam(generator.parameters(),     lr=tr_cfg["lr"], betas=(0.9, 0.999))
    opt_d = (Adam(discriminator.parameters(), lr=gan_cfg["lr_d"], betas=(0.9, 0.999))
             if discriminator is not None else None)

    # ── LR schedulers ─────────────────────────────────────────────────────
    if tr_cfg["scheduler"] == "cosine":
        # v2 FIX: was tr_cfg["num_epochs"] (global default), now `num_epochs`
        # (resolved per-model value). Ensures HAT (500 epochs) gets a full
        # cosine cycle, not truncated to the global default of 300.
        sched_g = CosineAnnealingLR(opt_g, T_max=num_epochs, eta_min=tr_cfg["lr_min"])
        sched_d = (CosineAnnealingLR(opt_d, T_max=num_epochs, eta_min=tr_cfg["lr_min"])
                   if opt_d is not None else None)
    else:
        sched_g = MultiStepLR(opt_g, milestones=tr_cfg["step_decay"], gamma=tr_cfg["step_gamma"])
        sched_d = (MultiStepLR(opt_d, milestones=tr_cfg["step_decay"], gamma=tr_cfg["step_gamma"])
                   if opt_d is not None else None)

    # ── AMP GradScalers ───────────────────────────────────────────────────
    # Use enabled=False for bfloat16 (static scaling is not needed for bf16)
    scaler_g = GradScaler(enabled=(amp_dtype == torch.float16))
    scaler_d = GradScaler(enabled=(amp_dtype == torch.float16))

    # ── Perceptual loss (GAN only) ────────────────────────────────────────
    percep_loss_fn = None
    if is_gan:
        percep_loss_fn = VGGPerceptualLoss(n_channels=n_ch).to(device)

    # ── EMA configuration ─────────────────────────────────────────────────
    ema_cfg   = cfg.get("ema", {})
    use_ema   = (is_gan and ema_cfg.get("enabled", True)) or args.ema
    ema_decay = ema_cfg.get("decay", 0.999)
    ema: Optional[EMAModel] = None

    # ── Checkpoint dir ────────────────────────────────────────────────────
    ckpt_dir = Path(paths["checkpoints"]) / f"{args.model}_{args.scaling}"
    start_epoch = 0
    best_psnr   = 0.0

    if args.resume:
        start_epoch = load_checkpoint(
            args.resume, generator, opt_g, discriminator, opt_d, device)

    # Initialise EMA after checkpoint loading (so EMA tracks the resumed weights)
    if use_ema:
        ema = EMAModel(generator, decay=ema_decay)
        logger.info(f"  EMA enabled (decay={ema_decay}). Will save best_ema.pt")

    # ── Training loop ─────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"  Model   : {args.model.upper()}")
    logger.info(f"  Scaling : {args.scaling}")
    logger.info(f"  Epochs  : {num_epochs}  (resume from {start_epoch})")
    logger.info(f"  Eff. BS : {tr_cfg['batch_size']} × {accum} = {tr_cfg['batch_size']*accum}")
    logger.info(f"  AMP     : {amp_dtype}")
    logger.info(f"  EMA     : {'enabled (decay=' + str(ema_decay) + ')' if use_ema else 'disabled'}")
    logger.info(f"  Source  : {'--epochs flag' if args.epochs else 'models.' + args.model + '.num_epochs in config.yaml'}")
    logger.info(f"{'='*60}\n")

    for epoch in range(start_epoch, num_epochs):
        generator.train()
        if discriminator is not None:
            discriminator.train()

        epoch_g_loss   = 0.0
        epoch_d_loss   = 0.0
        opt_g.zero_grad(set_to_none=True)
        if opt_d is not None:
            opt_d.zero_grad(set_to_none=True)

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{tr_cfg['num_epochs']}", ncols=100)

        for step, (lr_batch, hr_batch) in pbar:
            lr_batch = lr_batch.to(device, non_blocking=True)
            hr_batch = hr_batch.to(device, non_blocking=True)

            if is_gan:
                g_loss, d_loss = train_step_gan(
                    args.model, generator, discriminator,
                    lr_batch, hr_batch, opt_g, opt_d,
                    scaler_g, scaler_d, percep_loss_fn,
                    scale, amp_dtype, accum, step, gan_cfg)
                epoch_g_loss += g_loss
                epoch_d_loss += d_loss
                pbar.set_postfix({"G": f"{g_loss:.4f}", "D": f"{d_loss:.4f}"})
            else:
                g_loss = train_step_regr(
                    args.model, generator, lr_batch, hr_batch,
                    opt_g, scaler_g, scale, amp_dtype, accum, step)
                epoch_g_loss += g_loss
                pbar.set_postfix({"L1": f"{g_loss:.4f}"})

            # Update EMA after each generator step
            if ema is not None:
                ema.update(generator)

        sched_g.step()
        if sched_d is not None:
            sched_d.step()

        avg_g = epoch_g_loss / len(train_loader)
        avg_d = epoch_d_loss / len(train_loader) if is_gan else None

        if is_gan:
            logger.info(f"Epoch {epoch+1}: G={avg_g:.5f}  D={avg_d:.5f}")
        else:
            logger.info(f"Epoch {epoch+1}: L1={avg_g:.5f}")

        # ── Validation ────────────────────────────────────────────────────
        if (epoch + 1) % tr_cfg.get("val_interval", 5) == 0:
            # Validate with EMA weights for GANs (converge faster and smoother)
            if ema is not None:
                with ema.apply(generator):
                    val_psnr = validate(generator, val_loader, device, args.model, scale, amp_dtype)
                logger.info(f"  → Val PSNR (EMA): {val_psnr:.3f} dB")
            else:
                val_psnr = validate(generator, val_loader, device, args.model, scale, amp_dtype)
                logger.info(f"  → Val PSNR: {val_psnr:.3f} dB")

            is_best = val_psnr > best_psnr
            if is_best:
                best_psnr = val_psnr
            save_checkpoint(
                ckpt_dir, args.model, epoch + 1,
                generator, opt_g,
                discriminator, opt_d,
                metrics={"psnr": val_psnr, "epoch": epoch + 1},
                is_best=is_best,
                ema=ema,
            )

    logger.info(f"\nTraining complete.  Best PSNR: {best_psnr:.3f} dB")
    logger.info(f"Checkpoints: {ckpt_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Validation (PSNR only, for early stopping)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    generator: nn.Module,
    val_loader,
    device: torch.device,
    model_name: str,
    scale: int,
    amp_dtype: torch.dtype,
) -> float:
    generator.eval()
    total_psnr = 0.0
    for lr_batch, hr_batch in val_loader:
        lr_batch = lr_batch.to(device, non_blocking=True)
        hr_batch = hr_batch.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            if model_name == "srcnn":
                lr_up = F.interpolate(lr_batch, scale_factor=scale,
                                       mode="bicubic", align_corners=False)
                sr = generator(lr_up)
            elif model_name == "bicubic":
                sr = F.interpolate(lr_batch, scale_factor=scale,
                                    mode="bicubic", align_corners=False)
            else:
                sr = generator(lr_batch)

        # PSNR in [-1,1] range
        mse = F.mse_loss(sr.clamp(-1, 1), hr_batch.clamp(-1, 1))
        if mse == 0:
            total_psnr += 100.0
        else:
            total_psnr += 10 * torch.log10(4.0 / mse).item()   # range = 2 ([-1,1])

    generator.train()
    return total_psnr / len(val_loader)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rondônia SR – Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Epoch counts are read from config.yaml models.<name>.num_epochs\n"
            "(per-model) or training.num_epochs (global default).\n\n"
            "GAN two-stage training example:\n"
            "  Stage 1: python train.py --model srgan --scaling smart\n"
            "  Stage 2: python train.py --model srgan --scaling smart \\\n"
            "               --resume checkpoints/srgan_smart/latest.pt\n\n"
            "See docs/TRAINING_STRATEGY.md for full guidance."
        ),
    )
    parser.add_argument("--model",   required=True,
                        choices=["bicubic","srcnn","edsr","rcan","srgan","esrgan","swinir","hat"],
                        help="Model architecture to train")
    parser.add_argument("--scaling", required=True,
                        choices=["standard", "smart"],
                        help="Normalisation strategy")
    parser.add_argument("--config",  default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--resume",  default=None,
                        help="Path to checkpoint .pt file to resume from")
    parser.add_argument("--epochs",  default=None, type=int,
                        help="Override epoch count (default: read from config.yaml per-model)")
    parser.add_argument("--ema",     action="store_true",
                        help="Force EMA on (auto-enabled for srgan/esrgan; override for others)")
    args = parser.parse_args()
    main(args)

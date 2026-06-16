# Rondônia SR Comparative Study

> **Comparative Analysis of State-of-the-Art Single-Image Super-Resolution Algorithms for 30m→10m Landsat-to-Sentinel-2 Upscaling in the Brazilian Amazon Fishbone Deforestation Pattern**

---

## Directory Structure

```
Rondonia RS/
├── config.yaml                   ← Master experiment configuration
├── requirements.txt              ← Python dependencies
├── dataloader.py                 ← Data loading, alignment, Smart Scaling
├── train.py                      ← Training loop (AMP + grad accum + GAN-safe)
├── evaluate.py                   ← PSNR / SSIM / SAM / LPIPS evaluation
├── diagnostic_test.py            ← Pre-flight pipeline sanity checker
│
├── docs/
│   └── TRAINING_STRATEGY.md      ← ★ Epoch counts, VRAM, time estimates,
│                                      academic publishing checklist
│
├── models/
│   ├── __init__.py
│   ├── srcnn.py                  ← SRCNN (CNN baseline)
│   ├── edsr.py                   ← EDSR (deep CNN)
│   ├── rcan.py                   ← RCAN (channel attention CNN)
│   ├── srgan.py                  ← SRGAN generator + PatchGAN discriminator
│   ├── esrgan.py                 ← ESRGAN/Real-ESRGAN (RRDB + UNet disc.)
│   ├── swinir.py                 ← SwinIR (Swin Transformer)
│   ├── hat.py                    ← HAT (Hybrid Attention Transformer)
│   └── diffusionsat_stub.py      ← DiffusionSat (theoretical/optional)
│
├── scripts/
│   ├── prepare_patches.py        ← Offline patch extraction & alignment
│   └── run_all_experiments.py    ← Sequential train+eval for all models
│
├── data/
│   ├── raw/
│   │   ├── landsat/              ← Place 30m Landsat 5/7/8 .tif files here
│   │   └── sentinel/             ← Place 10m Sentinel-2 .tif files here
│   └── aligned/                  ← Generated: train/val/test .npz patches
│
├── checkpoints/                  ← Model checkpoints (auto-created)
├── results/                      ← Evaluation tables + tile images (auto-created)
├── diag_output/                  ← diagnostic_test.py visual output
└── logs/                         ← Training logs (auto-created)
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# PyTorch with CUDA 12.1 for RTX 3060
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Prepare Your Data

Place your paired GeoTIFF files:
- `data/raw/landsat/` — 30m Landsat 5/7/8 scenes (multi-band .tif)
- `data/raw/sentinel/` — 10m Sentinel-2 scenes (multi-band .tif)

Filenames must be in 1-to-1 sorted correspondence.

### 3. Extract & Align Patches

```bash
# With sub-pixel phase cross-correlation alignment (recommended)
python scripts/prepare_patches.py --config config.yaml

# Without alignment (for ablation)
python scripts/prepare_patches.py --config config.yaml --no-align
```

This creates `data/aligned/{train,val,test}/` with compressed `.npz` patch pairs.

### 4. Train a Single Model

```bash
# EDSR with Smart Scaling
python train.py --model edsr --scaling smart

# SwinIR with standard scaling
python train.py --model swinir --scaling standard

# ESRGAN (GAN), resume from checkpoint
python train.py --model esrgan --scaling smart --resume checkpoints/esrgan_smart/latest.pt
```

### 5. Evaluate

```bash
# Evaluate all enabled models (reads existing best.pt checkpoints)
python evaluate.py

# Evaluate one specific combination
python evaluate.py --model hat --scaling smart
```

### 6. Run All Experiments

```bash
# Full pipeline: train + eval all models × both scalings
python scripts/run_all_experiments.py

# Evaluation only (use existing checkpoints)
python scripts/run_all_experiments.py --eval-only
```

---

## Models Compared

| # | Model | Type | Params | VRAM (train) | Epochs | Est. Time |
|---|-------|------|--------|--------------|--------|-----------|
| 0 | Bicubic | Deterministic baseline | – | ~0.1 GB | — | < 1 min |
| 1 | SRCNN | Shallow CNN | 87K | ~0.5 GB | 150 | ~20 min |
| 2 | EDSR | Deep residual CNN | 1.6M | ~1.5 GB | 300 | ~2.1 hrs |
| 3 | RCAN | Channel attention CNN | 4.3M | ~3.5 GB | 300 | ~3.75 hrs |
| 4 | SRGAN | Adversarial CNN | 1.6M G | ~2.5 GB | 200 | ~3.1 hrs |
| 5 | ESRGAN | Adversarial RRDB + UNet disc. | 11.6M G | ~5.5 GB | 300 | ~7.5 hrs |
| 6 | SwinIR | Swin Transformer | 1.3M | ~2.0 GB | 400 | ~7.8 hrs |
| 7 | HAT | Hybrid attention transformer | 1.3M | ~2.5 GB | 500 | ~11.8 hrs |
| 8 | DiffusionSat | Diffusion foundation model | >1B | >12 GB (4-bit) | — | theoretical |

> Parameter counts and VRAM measured on RTX 3060 / bfloat16 / batch=4.
> See **[docs/TRAINING_STRATEGY.md](docs/TRAINING_STRATEGY.md)** for full calibration details.

All models use 6-band input (B, G, R, NIR, SWIR1, SWIR2) to match
Landsat ↔ Sentinel-2 spectral overlap.

---

## Evaluation Metrics

| Metric | ↑/↓ | Description |
|--------|-----|-------------|
| **PSNR** (dB) | ↑ | Pixel-level fidelity. Direct distortion measure. |
| **SSIM** | ↑ | Structural covariance. Luminance + contrast + structure. |
| **SAM** (°) | ↓ | **Spectral Angle Mapper** — the primary metric for proving Smart Scaling preserves 16-bit TOA spectral shape. Lower = better spectral match. |
| **LPIPS** | ↓ | Perceptual similarity (AlexNet). Measures how well logging-road textures are hallucinated. Lower = more perceptually realistic. |

---

## Hardware Optimisations (RTX 3060, 12 GB VRAM)

| Technique | Implementation | Rationale |
|-----------|---------------|-----------|
| **AMP bfloat16** | `torch.autocast("cuda", dtype=torch.bfloat16)` | Avoids float16 gradient underflow; native on Ampere (RTX 30xx). |
| **Gradient accumulation** | `accum_steps=8` → effective batch = 32 | Simulates large-batch training without exceeding VRAM. |
| **GAN graph detachment** | `sr_detached = sr.detach()` in discriminator update | Prevents autograd engine crash; D backward graph must not touch G params. |
| **Robust checkpointing** | Saves generator + discriminator + both Adam states | Allows exact mid-experiment resumption including Adam m/v moments. |
| **Gradient clipping** | `clip_grad_norm_(..., max_norm=1.0)` | Stabilises transformer training under bfloat16. |
| **set_to_none=True** | `opt.zero_grad(set_to_none=True)` | Frees gradient memory buffer immediately. |

---

## Smart Scaling — Academic Contribution

```
Standard Scaling:         x_norm = 2·(x - x_min)/(x_max - x_min) - 1
                                        ↑
                          dominated by <2% of pixels (clouds / specular water)
                          → vegetation variance COLLAPSED

Smart Scaling:            lo = percentile(x_c, 2%)
                          hi = percentile(x_c, 98%)
                          x_clip = clip(x_c, lo, hi)
                          x_norm = 2·(x_clip - lo)/(hi - lo + ε) - 1
                                        ↑  
                          per-channel → spectral SHAPE preserved
                          gradient variance across vegetation / soil RETAINED
```

**Hypothesis**: Smart Scaling will lower SAM scores across ALL models (better
spectral fidelity) while also improving PSNR/SSIM for CNN models (better
gradient signal during training) and improving LPIPS for GAN/Transformer
models (richer texture gradients → better road hallucination).

---

## Dataset Notes

- **Input**: Landsat 5/7/8 Collection 2 Level-2 Surface Reflectance (30 m GSD)
- **Target**: Sentinel-2 Level-2A Surface Reflectance (10 m GSD)
- **Geography**: Rondônia, Brazil; ~62°W–65°W, ~10°S–13°S
- **Key challenge**: "Fishbone" deforestation pattern — narrow (<30 m) dirt
  roads and sub-hectare clearings that fall at or below the Landsat IFOV,
  making them invisible in LR but recoverable in HR if spectrally preserved.
- **Patch size**: 48×48 LR → 144×144 HR (×3 upscale, ~1440m ground footprint)

---

## Running Order (Recommended)

Run fastest models first to catch data issues before committing overnight:

```bash
# 1. Bicubic  — instant — gives real-data PSNR ceiling
python evaluate.py --model bicubic --scaling smart

# 2. SRCNN    — ~20 min — sanity check real-data training
python train.py --model srcnn --scaling smart

# 3. EDSR     — ~2.1 hrs
python train.py --model edsr --scaling smart

# 4. SRGAN    — ~3.1 hrs  (GAN, 2-stage)
python train.py --model srgan --scaling smart
python train.py --model srgan --scaling smart --resume checkpoints/srgan_smart/latest.pt

# 5-8: RCAN, ESRGAN, SwinIR, HAT follow the same pattern
# See docs/TRAINING_STRATEGY.md §5 for full order
```

---

## References

1. Dong et al. (2014) — SRCNN — https://arxiv.org/abs/1501.00092
2. Lim et al. (2017) — EDSR — https://arxiv.org/abs/1707.02921
3. Zhang et al. (2018) — RCAN — https://arxiv.org/abs/1807.02758
4. Ledig et al. (2017) — SRGAN — https://arxiv.org/abs/1609.04802
5. Wang et al. (2018) — ESRGAN — https://arxiv.org/abs/1809.00219
6. Wang et al. (2021) — Real-ESRGAN — https://arxiv.org/abs/2107.10833
7. Liang et al. (2021) — SwinIR — https://arxiv.org/abs/2108.10257
8. Chen et al. (2023) — HAT — https://arxiv.org/abs/2309.05239
9. Khanna et al. (2024) — DiffusionSat — https://arxiv.org/abs/2312.03606
10. Blau & Michaeli (2018) — Perception-Distortion Tradeoff — https://arxiv.org/abs/1711.06077

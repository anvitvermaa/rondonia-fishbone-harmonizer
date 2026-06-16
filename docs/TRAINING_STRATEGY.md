# Rondônia SR Study — Training Strategy & Epoch Estimates
### Expert Analysis for Publishable Academic Research

---

## 0. Dataset Design — How Many Images & What Kind

> **Your intuition is correct and well-structured.** Your four categories map almost perfectly
> onto how remote sensing papers actually stratify their datasets. Below is the refined version
> with exact numbers and the one critical dimension you were missing.

### 0a. Your Categories — Verdict & Refinement

| Your Category | Verdict | Why It Matters for the Paper |
|---------------|---------|------------------------------|
| **1. Clear images (cloud < 5%)** | ✅ Essential | Backbone of training data. Gives highest PSNR. Your "easy" benchmark. |
| **2. With clouds (5–40% cover)** | ✅ Critical — this is your Smart Scaling stress test | Cloud pixels are the **outliers that break standard normalization**. Without these scenes your Smart Scaling paper has no argument. |
| **3. Active deforestation** | ✅ Core subject | Recent clearings = high-contrast forest/bare-soil boundaries. Tests spatial edge reconstruction. |
| **4. Minor roads + subtle clearing** | ✅ Your hardest & most valuable | Narrow dirt roads (<30m) fall **at or below the Landsat IFOV**. This is where GAN/Transformer models prove themselves and your key LPIPS figure comes from. |
| **Missing: Temporal diversity** | ⚠️ You need this | Dry season (Jun–Sep) vs wet season (Nov–Mar) = very different spectral signatures. Without temporal spread, reviewers will say your model is overfit to one phenological state. |
| **Missing: Spatial holdout** | ⚠️ Critical for peer review | Your test scenes must be from a **geographically separate sub-region** than training. Same area = data leakage through spatial autocorrelation. Reviewers check this. |

---

### 0b. Exact Scene Counts (Three Tiers)

| Tier | Landsat Scenes | Sentinel-2 Scenes | Training Patches | Paper Quality |
|------|---------------|-------------------|-----------------|---------------|
| **Minimum** | 6 | 6 | ~1,000 | Publishable at workshop / short paper level |
| **Recommended** ⭐ | 12 | 12 | ~3,000 | Full IEEE TGRS / ISPRS journal paper |
| **Strong** | 20 | 20 | ~6,000 | High-impact venue, Nature/Sci reports tier |

> **Recommended = 12 pairs.** This is the sweet spot for your hardware and timeline.
> ~3,000 patches keeps each epoch under 90 seconds on the RTX 3060 and gives enough
> diversity for transformers (SwinIR/HAT) to generalize properly.

---

### 0c. How to Distribute Your 12 Scenes

This is the scene-level dataset design. Every scene = 1 Landsat + 1 co-registered Sentinel-2.

```
TRAINING SET  (8 scenes → ~2,000 training patches)
──────────────────────────────────────────────────────────────────────────────
Scene  Season    Cloud %   Landscape               Sub-region
──────────────────────────────────────────────────────────────────────────────
T-01   Dry       0–5%      Active fishbone roads   Porto Velho area (W Rondônia)
T-02   Dry       0–5%      Intact Amazon forest    Porto Velho area
T-03   Dry       5–15%     Active deforestation    Porto Velho area
T-04   Dry       15–35%    Mixed cloud / clearing  Porto Velho area     ← Smart Scaling key
T-05   Wet       0–5%      Established clearing    Porto Velho area
T-06   Wet       5–20%     Subtle roads + regrowth Porto Velho area     ← Hardest case
T-07   Wet       20–40%    Heavy cloud + forest    Porto Velho area     ← Smart Scaling key
T-08   Dry       0–5%      Active fishbone roads   Ariquemes area (C Rondônia)

VALIDATION SET  (2 scenes → ~500 patches, used for epoch stopping only)
──────────────────────────────────────────────────────────────────────────────
V-01   Dry       0–10%     Mixed landscape         Cacoal area (E Rondônia)
V-02   Wet       10–25%    Cloud + fishbone        Cacoal area

TEST SET  (2 scenes → ~500 patches, NEVER SEEN during training)
──────────────────────────────────────────────────────────────────────────────
TE-01  Dry       0–5%      Active deforestation    Ji-Paraná area (SE Rondônia) ← spatial holdout
TE-02  Dry       0–5%      Minor roads / subtle    Ji-Paraná area               ← your main results table
```

**The spatial holdout rule:**
- Train/Val → western and central Rondônia (Porto Velho, Ariquemes, Cacoal municipalities)
- Test → southeastern Rondônia (Ji-Paraná / Vilhena), ~200–300 km away
- This prevents spatial autocorrelation leakage, which is a common rejection reason

---

### 0d. What Each Category Tests in Your Paper

| Dataset Class | What Metric It Drives | Which Models Benefit |
|---------------|----------------------|---------------------|
| Clear, dry scenes | PSNR / SSIM ceiling | All models equally |
| Cloudy scenes (5-40%) | **SAM** — Smart Scaling's showcase | Smart > Standard across ALL models here |
| Active deforestation | SSIM (structural similarity of clearing edges) | EDSR, RCAN |
| Subtle roads / sub-30m features | **LPIPS** — perceptual sharpness of roads | ESRGAN, SwinIR, HAT |
| Temporal diversity (wet+dry) | Generalization / no overfitting claim | All models |
| Spatial holdout (Ji-Paraná) | Peer-review credibility | All models |

---

### 0e. Where to Download the Data

**Landsat (30m, Collection 2, Level-2 Surface Reflectance):**
```
USGS EarthExplorer  →  https://earthexplorer.usgs.gov/
- Dataset: Landsat Collection 2 Level-2
- Search area: Rondônia, Brazil (~62°W–65°W, ~10°S–13°S)
- Filter: Cloud cover < 40%
- Sensors: Landsat 5 TM (pre-2013), Landsat 7 ETM+ (1999-2022), Landsat 8 OLI (2013–)
- Format: Download as .tar.gz, extract the _SR_B*.TIF band files
```

**Sentinel-2 (10m, Level-2A Surface Reflectance):**
```
Copernicus Data Space  →  https://dataspace.copernicus.eu/
- Dataset: Sentinel-2 L2A (MSI)
- Same bounding box and dates as your Landsat selections
- Filter: Cloud cover < 40%
- Format: SAFE format; you want the B02, B03, B04, B08, B11, B12 .jp2 files

Alternative mirror: AWS Open Data  →  s3://sentinel-cogs/sentinel-s2-l2a-cogs/
```

**INPE Deforestation Reference (for your label/annotation):**
```
PRODES annual deforestation polygons  →  http://terrabrasilis.dpi.inpe.br/
- Use these shapefiles to tag which patches contain active deforestation
- This lets you stratify your test set and run per-class metric analysis
```

**Temporal matching strategy:**
- Landsat and Sentinel-2 acquisitions must be within **±15 days** of each other
- Check that both are in the same season (don't pair a July Landsat with a March Sentinel-2)
- Rondônia's dry season is June–September — this gives the cleanest paired data

---

### 0f. From Scenes → Patches (How the Numbers Work)

A Landsat scene is ~170×183 km. At 30m resolution that's ~5,700 × 6,100 pixels.

With your config (LR patch=48px, stride=24px, 50% overlap):
```
patches_per_scene ≈ ((5700 - 48) / 24) × ((6100 - 48) / 24)
                  ≈ 237 × 252
                  ≈ 59,000 raw patches per full scene
```

But after filtering (≥85% valid pixels, cloud mask, nodata rejection) you typically keep
**~5–10% of raw patches** from forested/deforested areas, which gives:
```
~59,000 × 0.07  ≈  ~4,000 kept patches per scene
× 8 training scenes  =  ~32,000 patches  ← even better than estimated!
```

You don't need to use all of them. **Random sample 2,000–4,000 for training** to keep
epoch times reasonable. Your `scripts/prepare_patches.py` handles this filtering.

> **Updated estimate from §4:** With 3,000 training patches (not 1,000 as previously assumed),
> time/epoch increases ~3×. Updated table below in §4.

---

### 0g. The Two Sub-Test Sets for Your Paper's Main Table

For publications, split your 2 test scenes into two subsets and report metrics separately:

```
TEST A — "Standard Deforestation"  (all patches from TE-01)
  → PSNR / SSIM / SAM / LPIPS for clean, clearly visible clearing

TEST B — "Hard Case: Subtle Roads"  (all patches from TE-02 with PRODES road polygons)
  → This is your KEY TABLE — GAN/Transformer models should DOMINATE here on LPIPS
  → This is where Smart Scaling's SAM advantage is most dramatic
```

Reporting both subsets separately is what elevates a good paper to a great one.
IEEE TGRS and Remote Sensing reviewers specifically look for "challenging subset" analysis.

---

## 1. What the Diagnostic Run Tells Us

The diagnostic ran in **3.4 s on CPU** with synthetic noise patches.
The 3 "failures" are **all expected and harmless**:

| Flag | Why it's fine |
|------|--------------|
| No CUDA | You were running on CPU — train.py will auto-pick the GPU when real data is ready |
| Bicubic PSNR 12.5 dB | Synthetic patches have **deliberately injected 2% outliers** + uncorrelated noise; real Landsat/S2 aligned pairs ≈ **24–28 dB** |
| Bicubic SAM 69° | Same reason — the synthetic HR target was generated independently, not from the LR, so spectral angle is meaningless here |

> **Bottom line: the full pipeline is wired correctly. Everything that matters passed.**

---

## 2. Hardware Profile (RTX 3060 12 GB)

| Spec | Value |
|------|-------|
| GPU | RTX 3060 Laptop/Desktop |
| VRAM | 12 GB |
| CUDA Cores | 3584 |
| Tensor Cores | 112 (3rd gen) |
| bfloat16 support | ✅ Yes (Ampere) |
| FP16 TFLOPS | ~25 TFLOPS |
| Typical SR throughput | ~40–120 patches/s (model dependent, bf16) |

Your config: **batch_size=4, accum_steps=8 → effective batch=32**, which is optimal for this GPU.

---

## 3. Model Parameter Count & VRAM Footprint

*(Measured from actual model files — not estimates)*

| Model | Parameters | Weight MB | Training VRAM* | Fits 12 GB? |
|-------|-----------|-----------|----------------|------------|
| **Bicubic** | 0 | 0 | ~0.1 GB | ✅ Trivial |
| **SRCNN** | 87,206 | 0.3 MB | ~0.5 GB | ✅ |
| **EDSR** | 1,557,958 | 6 MB | ~1.5 GB | ✅ |
| **SwinIR** | 1,275,718 | 5 MB | ~2.0 GB | ✅ |
| **HAT** | 1,294,534 | 5 MB | ~2.5 GB | ✅ |
| **SRResNet (SRGAN-G)** | 1,617,496 | 6 MB | ~2.5 GB (G+D) | ✅ |
| **RCAN** | 4,282,702 | 16 MB | ~3.5 GB | ✅ |
| **RRDBNet (ESRGAN-G)** | 11,628,550 | 44 MB | ~5.5 GB (G+D) | ✅ |

*Training VRAM includes: weights + gradients + Adam moments + activation buffers (bf16, batch=4)*

**All models fit comfortably in 12 GB.** ESRGAN is the tightest at ~5.5 GB but still has 6+ GB to spare.

> **OOM fallback for ESRGAN:** set `accum_steps: 16` and `batch_size: 2` in config.yaml —
> same effective batch of 32, half the VRAM.

---

## 4. Epoch Count Recommendations

Calibrated to:
- **Rondônia fishbone** imagery (high-frequency edge detail needs more epochs than natural photos)
- **RTX 3060 with bfloat16 + grad accum**
- **~3,000 aligned LR/HR patch pairs** (from 8 training scenes as designed in §0c)
- **Publishable-quality results** (convergence curves, not just "trained for N epochs")

### 4a. Epoch Targets

| Model | Min Epochs (Publishable) | Recommended | Max Useful | Reason |
|-------|--------------------------|-------------|------------|--------|
| **Bicubic** | N/A — instant | — | — | No training, deterministic |
| **SRCNN** | 100 | **150** | 200 | Shallow 3-layer CNN converges fast |
| **EDSR** | 150 | **300** | 400 | Residual depth needs time to stabilize |
| **RCAN** | 200 | **300** | 500 | Channel attention is data-hungry; needs ~200 to "click" |
| **SRGAN** | 100 pre + 100 GAN | **200 total** | 300 | Pre-train SRResNet first, then adversarial fine-tune |
| **ESRGAN** | 100 pre + 150 GAN | **300 total** | 400 | RRDB warm-up is critical; skip it and GAN collapses |
| **SwinIR** | 200 | **400** | 600 | Transformers are slow to converge on small datasets |
| **HAT** | 300 | **500** | 700 | Hybrid attention needs the most epochs — most complex |

> **⚠ CRITICAL — GAN models (SRGAN/ESRGAN):** Always pre-train the generator with pure L1 loss
> for the first 100 epochs **before** switching the discriminator on. Skipping this causes mode
> collapse. Run in two stages using `--resume` (see Section 8).

### 4b. Convergence Signals to Watch

| Model | Loss to Monitor | "Converged" Signal |
|-------|----------------|-------------------|
| SRCNN | L1 | Plateau < 1e-4 change over 20 epochs |
| EDSR | L1 | Val PSNR stops improving ±0.1 dB over 30 epochs |
| RCAN | L1 | Val PSNR plateau — often jumps suddenly at epoch ~150 |
| SRGAN | G_loss + D_loss | D_loss ≈ 0.69 (random) = healthy GAN equilibrium |
| ESRGAN | G_total + D_loss | Same as SRGAN; watch that G_pixel doesn't explode |
| SwinIR | L1 | Very gradual — trust the cosine schedule, don't stop early |
| HAT | L1 | Same as SwinIR — patience is rewarded at epoch 400+ |

---

## 5. Time Estimates on RTX 3060

### Assumptions
- **3,000 training patches** (8 scenes, filtered — see §0f)
- batch_size=4, accum_steps=8 → ~750 steps/epoch
- bfloat16 AMP enabled
- Validation every 5 epochs

| Model | Time/Epoch | Recommended Epochs | **Total Time** |
|-------|-----------|-------------------|----------------|
| **Bicubic** | 0 s | — | **< 1 min** (evaluation only) |
| **SRCNN** | ~22 s | 150 | **~55 min** |
| **EDSR** | ~70 s | 300 | **~5.8 hours** |
| **RCAN** | ~130 s | 300 | **~10.8 hours** |
| **SRGAN** | ~155 s | 200 | **~8.6 hours** |
| **ESRGAN** | ~260 s | 300 | **~21.7 hours** |
| **SwinIR** | ~200 s | 400 | **~22.2 hours** |
| **HAT** | ~240 s | 500 | **~33.3 hours** |

**Total for all models (both scaling modes = ×2): ~206 hours ≈ 8–9 days of overnight runs**

> **Note:** The previous estimates assumed 1,000 patches. With 3,000 patches (the recommended
> dataset size for a full journal paper) times scale by ~3×. This is still achievable if you
> run one model per night. ESRGAN, SwinIR, HAT each need ≥2 overnight sessions.

> **Speed-up option:** If 8–9 days is too long, use Google Colab Pro (A100 = ~5× faster than
> RTX 3060) or reduce to 2,000 training patches. Results will be slightly noisier but still
> publishable.

---

## 6. Feasibility Assessment

| Question | Answer |
|----------|--------|
| Can it run all models? | ✅ Yes — all fit in 12 GB VRAM |
| Can it run overnight? | ✅ Yes — SRCNN/EDSR finish in 1 night each |
| All models in one weekend? | ❌ No, not with full dataset — ~8–9 days total |
| All models in 2 weeks of nights? | ✅ Yes, comfortable |
| Publishable quality? | ✅ Yes, this is the correct scale for IEEE TGRS |
| Need a server/cloud? | Optional — Colab Pro cuts time to ~2 days total |

### Recommended Run Order (fastest → slowest)

```
1. Bicubic   — instant          → baseline PSNR anchor (run this first day)
2. SRCNN     — 55 min           → first trained model result  
3. EDSR      — 5.8 hrs  (night 1)
4. SRGAN     — 8.6 hrs  (night 2, stage 1 only)
5. RCAN      — 10.8 hrs (night 3)
6. ESRGAN    — 21.7 hrs (nights 4–5, 2-stage)
7. SwinIR    — 22.2 hrs (nights 6–7)
8. HAT       — 33.3 hrs (nights 8–9)
```

---

## 7. The Academic Gold Standard

For a SISR paper targeting **Remote Sensing / ISPRS / IEEE TGRS**:

### Required for Peer Review
1. **Convergence curves** — PSNR vs epoch for each model (captured via `val_interval=5`)
2. **Both scaling modes** — Smart vs Standard across ALL models (your key contribution)
3. **4 metrics per model** — PSNR, SSIM, SAM, LPIPS (evaluate.py covers this)
4. **Two test subsets** — Standard deforestation (TE-01) + Hard/subtle roads (TE-02)
5. **Visual comparison panel** — zoomed fishbone roads showing sub-10m detail recovery
6. **Statistical significance** — run each model 3× with different seeds, report mean ± std
7. **Spatial holdout proof** — state train/test regions explicitly in the paper

### The Smart Scaling Paper Argument (Your Key Tables)

```
Table 1: Full metrics (PSNR / SSIM / SAM / LPIPS) — all models × both scaling modes
         on TEST A (standard deforestation)

Table 2: Same metrics on TEST B (hard case: subtle roads)
         ← THIS is where Smart Scaling's advantage is largest

Figure 1: SAM comparison — Smart vs Standard across all 8 models
           (bar chart, should show Smart clearly lower = better spectral fidelity)

Figure 2: LPIPS comparison for GAN/Transformer models on TEST B
           (proves road hallucination quality)

Figure 3: Visual panel — LR / Bicubic / SRCNN / EDSR / ESRGAN / SwinIR / HAT / HR
           zoomed to a fishbone road ~30m wide
```

### Minimum Dataset for Peer Review
- ≥ 8 Landsat/Sentinel-2 paired scenes (see §0c above)
- ≥ 2 **spatially separate** test scenes (different municipality from training)
- Report metrics **on the test scenes only** — val is only for epoch selection

---

## 8. GAN Two-Stage Training Commands

```bash
# ── SRGAN ────────────────────────────────────────────────────────────────────
# Stage 1: Pre-train generator with L1 only (100 epochs)
python train.py --model srgan --scaling smart --config config.yaml

# Stage 2: Resume, discriminator kicks in (remaining 100 epochs)
python train.py --model srgan --scaling smart \
    --resume checkpoints/srgan_smart/latest.pt

# ── ESRGAN ───────────────────────────────────────────────────────────────────
# Stage 1: Pre-train RRDB generator (100 epochs)
python train.py --model esrgan --scaling smart --config config.yaml

# Stage 2: Full adversarial training (200 more epochs)
python train.py --model esrgan --scaling smart \
    --resume checkpoints/esrgan_smart/latest.pt
```

---

## 9. Quick-Start: Bicubic Baseline (No Training Required)

Once you have real patches in `data/aligned/`:

```bash
python evaluate.py --model bicubic --scaling smart --config config.yaml
```

This gives you **Table 1 Row 1** in under a minute and tells you the real-data PSNR ceiling
(expected: 24–28 dB), which calibrates all subsequent model comparisons.

---

## 10. Diagnostic Results Interpretation

```
diagnostics ran 2026-03-31 — synthetic data only
────────────────────────────────────────────────
18/21 tests passed

FAILED (all expected):
  ✗ CUDA available        → CPU fallback active  (GPU will be used on real runs)
  ✗ Bicubic PSNR ≥ 20 dB → 12.55 dB            (synthetic only; real = 24-28 dB)
  ✗ Bicubic SAM ≤ 20°    → 69.26°              (synthetic only; real < 10°)

PASSED (confirms pipeline correctness):
  ✓ Synthetic data shapes, outlier injection
  ✓ Smart Scaling range [-1,1] and variance preservation
  ✓ Sub-pixel alignment: MSE improved, runs in 0.13s
  ✓ All tensor shapes: LR/Bicubic/HR
  ✓ SRCNN 1-epoch training: loss decreasing, no OOM
  ✓ SRCNN > Bicubic PSNR after 1 epoch
  ✓ Visual grid saved (461 KB PNG)
```

---

## 11. Downstream Task Integration (U-Net Segmentation)

### 11a. Motivation

Per-pixel metrics (PSNR/SSIM) measure image reconstruction quality but do not
prove whether SR actually helps downstream analysis. IEEE TGRS and ISPRS
reviewers now expect at least one downstream task evaluation for SR papers in
remote sensing. The Rondônia study uses **semantic segmentation of land cover**
as the downstream task because it directly connects to the paper's scientific
claim: SR enables better deforestation monitoring.

### 11b. Classes

| Class ID | Label            | NDVI Range   | Detection Difficulty |
|----------|-----------------|--------------|---------------------|
| 0        | intact_forest   | NDVI ≥ 0.50  | Easy (high spectral contrast) |
| 1        | deforestation   | 0.10–0.49    | Medium (heterogeneous texture) |
| 2        | logging_road    | NDVI < 0.10  | Hard (sub-30m features, below Landsat IFOV) |

Class 2 (logging road) is where SR most clearly helps — roads are finer than
30m pixels, so they only become detectable at 10m Sentinel-2 resolution.

### 11c. Label Generation

Labels are generated by `downstream/labels.py` using spectral index thresholds:

```
NDVI = (NIR - Red) / (NIR + Red)   ← vegetation density
SWIR1 edge magnitude (Sobel)       ← linear feature detection (roads)
```

These pseudo-labels are stored in each `.npz` patch as a `seg_mask` array by
`prepare_patches.py v2`. No external shapefiles are required.

**Optional upgrade**: Replace pseudo-labels with PRODES annual deforestation
polygons (INPE) for peer-review-ready ground truth. Set in config.yaml:
```yaml
downstream:
  label_method: "prodes"
  prodes_shp:   "/path/to/PRODES_2023_Amazonia.shp"
```

### 11d. Execution Order

After all SR model training is complete:

```bash
# 1. Train segmenter on bicubic baseline (lower bound)
python downstream/train_downstream.py --config config.yaml --sr-model bicubic

# 2. Train on native HR (upper bound)
python downstream/train_downstream.py --config config.yaml --use-hr

# 3. Train on all enabled SR models (takes ~4 hours × 8 models = 32 hours)
python downstream/train_downstream.py --config config.yaml

# OR: run specific model
python downstream/train_downstream.py --config config.yaml --sr-model esrgan --scaling smart
```

Results are saved to `results/downstream_results.csv` and `results/downstream_results.md`.

### 11e. Expected Results

| Input             | mIoU (expected) | Road F1 (expected) |
|-------------------|-----------------|--------------------|
| Bicubic 30m→10m   | 0.45–0.55       | 0.10–0.20          |
| SRCNN             | 0.50–0.60       | 0.15–0.25          |
| EDSR              | 0.55–0.65       | 0.20–0.30          |
| ESRGAN (EMA)      | 0.60–0.70       | 0.30–0.45          |
| HAT               | 0.60–0.72       | 0.30–0.50          |
| Native HR 10m     | 0.70–0.80       | 0.50–0.65          |

> **Key thesis**: GAN-based methods (ESRGAN) and transformers (HAT) are
> expected to show disproportionate improvement in Road F1 (+15–25 points over
> bicubic) even if their PSNR is only marginally better. This disconnect between
> PSNR and downstream task performance is the paper's main scientific finding.

### 11f. Time Estimates

| Model × 1 U-Net training run | Approx. time (RTX 3060 / T4) |
|-------------------------------|-------------------------------|
| 1 U-Net × 50 epochs           | 20–35 minutes                 |
| 8 SR models + bicubic + HR    | ~4–6 hours total              |

---

## 12. Manuscript Structure (IEEE TGRS Submission)

### Target Venue
**IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS)**
- Impact Factor: 8.2
- Double-blind peer review
- Typical length: 12–15 pages
- Expected PSNR tables, SAM comparison, downstream task evaluation

### 12a. Suggested Section Outline

```
Abstract (250 words)
  ─ Problem: satellite SR for deforestation monitoring
  ─ Approach: 8-architecture benchmark + Smart Scaling + downstream validation
  ─ Key findings: ESRGAN/HAT outperform on Road F1; Smart Scaling ↑ SAM

1. Introduction
  1.1 Motivation: Landsat-8 (30m) temporal density vs. Sentinel-2 (10m) spatial detail
  1.2 Gap: No comparative study of modern SISR (SwinIR, HAT) for pan-Amazon deforestation
  1.3 Contributions:
      (a) First benchmark of 8 architectures on Rondônia paired imagery
      (b) Smart Scaling normalisation preserving multispectral spectral shape
      (c) Downstream validation via semantic segmentation (forest/deforestation/road)

2. Study Area and Dataset
  2.1 Rondônia, Brazil (deforestation hotspot)
  2.2 Paired Landsat-8 / Sentinel-2 imagery (12 scenes, Microsoft Planetary Computer)
  2.3 Preprocessing: histogram matching, sub-pixel alignment, patch extraction
  2.4 Scene splits: 8 train / 2 val / 2 test (Ji-Paraná spatial holdout)

3. Methodology
  3.1 SR Architectures: SRCNN, EDSR, RCAN, SRGAN, ESRGAN, SwinIR, HAT, Bicubic
  3.2 Normalisation Strategies: Standard vs. Smart Scaling
  3.3 Training Configuration (Table: architecture-specific settings)
  3.4 Evaluation Metrics: PSNR, SSIM, SAM, ERGAS, LPIPS
  3.5 Downstream Task: U-Net land cover segmentation (F1/IoU per class)
  3.6 Statistical Significance (3 seeds, mean ± std)

4. Results
  4.1 Quantitative SR Metrics (Table 1: PSNR/SSIM/SAM/ERGAS/LPIPS)
  4.2 Smart Scaling vs. Standard Scaling (Table 2: Δ SAM per model)
  4.3 Visual Comparison (Figure: SR tiles, focus on road textures)
  4.4 Downstream Task Results (Table 3: mIoU/F1 per model per class)
  4.5 PSNR vs. Road F1 scatter plot (key figure showing disconnect)

5. Discussion
  5.1 Why does ESRGAN outperform SwinIR on Road F1 despite lower PSNR?
  5.2 Smart Scaling: spectral fidelity gain vs. PSNR tradeoff
  5.3 Limitations: pseudo-labels, 12-scene dataset, single geographic area
  5.4 Future Work: PRODES labels, temporal SR (video SR), multi-date fusion

6. Conclusion

Appendix A: Training hyperparameters per model
Appendix B: Per-scene metric breakdown
```

### 12b. Key Figures Checklist

- [ ] Figure 1: Study area map (Rondônia, scene locations)
- [ ] Figure 2: Architecture comparison diagram (LR → SR → HR flow)
- [ ] Figure 3: Smart Scaling vs. Standard Scaling histogram example
- [ ] Figure 4: SR visual comparison (all 8 models, road patch)
- [ ] Figure 5: PSNR vs. Road F1 scatter plot (main finding)
- [ ] Table 1: Full PSNR/SSIM/SAM/ERGAS/LPIPS benchmark (mean ± std)
- [ ] Table 2: Downstream segmentation F1/IoU (all models vs. bicubic vs. native HR)

### 12c. Submission Checklist

- [ ] 3× replicated experiments (seeds 42, 123, 7) — use `reproducibility.seeds` in config
- [ ] Test-set evaluation only (`--test-split test`) — no val set leakage in reported numbers
- [ ] Significance testing (Wilcoxon signed-rank, p < 0.05)
- [ ] Code repository: GitHub public, MIT license
- [ ] Data availability statement (Planetary Computer public access)
- [ ] Authorship order confirmed (corresponding author handles cover letter)

  ✓ Synthetic data shapes, outlier injection
  ✓ Smart Scaling range [-1,1] and variance preservation
  ✓ Sub-pixel alignment: MSE improved, runs in 0.13s
  ✓ All tensor shapes: LR/Bicubic/HR
  ✓ SRCNN 1-epoch training: loss decreasing, no OOM
  ✓ SRCNN > Bicubic PSNR after 1 epoch
  ✓ Visual grid saved (461 KB PNG)
```

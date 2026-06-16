# Kaggle Setup — Step-by-Step Guide

> **Everything runs on Kaggle. You don't need a GPU locally.**

---

## One-Time Setup (do this first, only takes 5 minutes)

### 1. Create a Kaggle Account
Go to **[kaggle.com](https://kaggle.com)** → Sign Up → use your Google account.  
It's free. No credit card needed.

---

### 2. Enable Phone Verification (required for internet + GPU)
1. Click your profile photo (top right) → **Settings**
2. Scroll to **Phone Verification** → verify your phone number
3. This unlocks: free GPU (30 hrs/week) + internet access in notebooks

---

### 3. Get Your API Key
1. Still in Settings, scroll to **API**
2. Click **"Create New Token"** → downloads `kaggle.json`
3. On your local machine:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

---

### 4. Upload Your Project Code to Kaggle (one-time)
Run this from your project folder:
```bash
# Install the Kaggle CLI
pip install kaggle

# Upload the project code (replace 'your_username' with your Kaggle username)
bash kaggle/push_to_kaggle.sh your_username
```

This creates a private dataset called `rondonia-sr-code` on your Kaggle account.  
**You only do this once** (or when you change the code).

---

## Phase 1 — Data Download + Patch Preparation (no GPU, ~90 min)

### 5. Create the Phase 1 Notebook
1. Go to **[kaggle.com/code](https://kaggle.com/code)** → **New Notebook**
2. Click **File** → **Import Notebook** → upload `kaggle/phase1_data_prep.ipynb`
3. In the right panel:
   - **Accelerator**: None (no GPU needed!)
   - **Internet**: ON ← important
4. Click **"+ Add Data"** (top right) → search `rondonia-sr-code` → Your Datasets → Add

### 6. Edit the Username Cell
In the first code cell, change:
```python
KAGGLE_USER = "your_kaggle_username"   # ← put your actual username here
```

### 7. Run Phase 1
Click **Run All** (or Shift+Enter through each cell).

**What happens:**
- ✓ Downloads 12 Landsat-8 + Sentinel-2 scene pairs (~4–6 GB, ~20–40 min)
- ✓ Applies histogram matching + sub-pixel alignment
- ✓ Extracts ~2,700 training patches
- ✓ Runs bicubic baseline (should show PSNR ≥ 24 dB)
- ✓ Saves `rondonia_sr_patches.zip` as output

### 8. Save Phase 1 Output as a Dataset
1. Click **Save & Run All** (top right) to persist the notebook output
2. When complete, go to **Output** tab in the right panel
3. Find `rondonia_sr_patches.zip` → click **⋮** → **"Create Dataset from output"**
4. Name it: `rondonia-sr-patches` → **Create**

---

## Phase 2 — GPU Training (one model per session)

### 9. Create the Phase 2 Notebook
1. **New Notebook** → **Import** → upload `kaggle/phase2_train_evaluate.ipynb`
2. In the right panel:
   - **Accelerator**: GPU T4 x1 ← required
   - **Internet**: ON
3. **"+ Add Data"**:
   - Add `rondonia-sr-code` (your code dataset)
   - Add `rondonia-sr-patches` (output from Phase 1)

### 10. Choose Your First Model
In the first code cell, set:
```python
MODEL_TO_TRAIN = "srcnn"   # start with the fastest model
SCALING        = "smart"
```

### 11. Run Training
Click **Run All** → it will:
- Train the model (1.5–8h depending on model)
- Evaluate PSNR / SSIM / SAM / ERGAS / LPIPS
- Train downstream U-Net segmentation
- Save a zip of checkpoints + results

### 12. Download Results
Go to **Output** tab → download `rondonia_sr_srcnn_smart_output.zip`

### 13. Repeat for Each Model
Each week: open Phase 2 → change `MODEL_TO_TRAIN` → Run All.

**Recommended order:**
```
Session 1: srcnn  (3h)     ← run both standard + smart in one session
Session 2: edsr   (8h)
Session 3: rcan   (10h)    ← spread across 2 sessions if needed
Session 4: srgan  (8h)
Session 5: esrgan (8h)
Session 6: swinir (8h)
Session 7: hat    (6h)
```

---

## FAQ

**Q: Will my notebook time out?**  
Kaggle sessions run for up to 9 hours. SRCNN and EDSR will finish in one session.  
For RCAN/SwinIR, the checkpoint auto-saves every 5 epochs — if it times out, re-run  
and it will automatically resume from `latest.pt`.

**Q: I ran out of GPU hours this week.**  
CPU notebooks don't count toward GPU quota. You can do evaluation/downstream  
tasks with Accelerator=None if you just want to check results.

**Q: Can I run two models in parallel?**  
No — each Kaggle account gets one GPU session at a time. But you can fork  
the notebook and train with the same dataset (doesn't help with GPU limits).

**Q: Where are my results after training?**  
Download the zip from the Output tab. Place `results/` and `checkpoints/`  
back into your local project folder, then I can generate the paper tables.

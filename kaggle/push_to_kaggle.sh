#!/usr/bin/env bash
# =============================================================================
# push_to_kaggle.sh  –  Rondônia SR Study
# =============================================================================
# Zips your project code and uploads it as a private Kaggle dataset so the
# Phase 1 and Phase 2 notebooks can reference it.
#
# Run this ONCE from the project root:
#   bash kaggle/push_to_kaggle.sh YOUR_KAGGLE_USERNAME
#
# Requirements:
#   pip install kaggle
#   Download kaggle.json from kaggle.com → Settings → API → Create New Token
#   Place it at ~/.kaggle/kaggle.json  (chmod 600 ~/.kaggle/kaggle.json)
# =============================================================================

set -e

KAGGLE_USER="${1:-}"
if [ -z "$KAGGLE_USER" ]; then
    echo ""
    echo "  Usage: bash kaggle/push_to_kaggle.sh YOUR_KAGGLE_USERNAME"
    echo "  Example: bash kaggle/push_to_kaggle.sh johndoe"
    echo ""
    exit 1
fi

DATASET_SLUG="rondonia-sr-code"
STAGING="/tmp/rondonia-sr-upload"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Rondônia SR — Kaggle Dataset Upload"
echo "  Target: kaggle.com/datasets/$KAGGLE_USER/$DATASET_SLUG"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Check kaggle CLI ──────────────────────────────────────────────────────────
if ! command -v kaggle &>/dev/null; then
    echo "  Installing Kaggle CLI..."
    pip install kaggle --quiet
fi



# ── Stage files ────────────────────────────────────────────────────────────────
echo "  Staging project files..."
rm -rf "$STAGING"
mkdir -p "$STAGING/models" "$STAGING/scripts" "$STAGING/downstream" "$STAGING/docs"

# Core Python files
cp dataloader.py train.py evaluate.py config.yaml requirements.txt "$STAGING/"

# Models
cp models/__init__.py models/srcnn.py models/edsr.py models/rcan.py \
   models/srgan.py models/esrgan.py models/swinir.py models/hat.py \
   "$STAGING/models/"

# Scripts (exclude download_data.py — it runs inside notebooks directly)
cp scripts/download_data.py scripts/prepare_patches.py \
   scripts/run_all_experiments.py "$STAGING/scripts/"

# Downstream module
cp downstream/__init__.py downstream/labels.py downstream/unet.py \
   downstream/train_downstream.py "$STAGING/downstream/"

# Docs
cp docs/TRAINING_STRATEGY.md "$STAGING/docs/" 2>/dev/null || true

# ── Dataset metadata ───────────────────────────────────────────────────────────
cat > "$STAGING/dataset-metadata.json" << EOF
{
  "title": "Rondônia SR Study — Project Code",
  "id": "$KAGGLE_USER/$DATASET_SLUG",
  "licenses": [{"name": "CC0-1.0"}],
  "keywords": ["satellite-imagery", "super-resolution", "deep-learning", "remote-sensing"]
}
EOF

# ── Upload ─────────────────────────────────────────────────────────────────────
echo "  Uploading dataset to Kaggle..."
echo ""
kaggle datasets create -p "$STAGING" --dir-mode zip

echo ""
echo "  ✓  Upload complete!"
echo ""
echo "  Your dataset is at: https://www.kaggle.com/datasets/$KAGGLE_USER/$DATASET_SLUG"
echo ""
echo "  NEXT: Update the KAGGLE_USER variable in both notebooks to '$KAGGLE_USER'"
echo "        Then upload kaggle/phase1_data_prep.ipynb to Kaggle and run it."
echo ""

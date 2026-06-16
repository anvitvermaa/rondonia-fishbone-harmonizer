"""
scripts/run_all_experiments.py  –  Rondônia SR Study
=====================================================

Convenience wrapper to sequentially train + evaluate ALL model × scaling
combinations defined in config.yaml (enabled: true).

Runs:  bicubic / srcnn / edsr / rcan / srgan / esrgan / swinir / hat
       × standard / smart scaling

Usage:
    python scripts/run_all_experiments.py --config config.yaml
    python scripts/run_all_experiments.py --config config.yaml --eval-only

Will log progress to logs/run_all.log in addition to stdout.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import yaml

# ─────────────────────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"run_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file),
    ],
)
logger = logging.getLogger(__name__)

PYTHON = sys.executable
ROOT   = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> int:
    """Run a subprocess, stream output live, return exit code."""
    logger.info(f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=ROOT)
    proc.wait()
    return proc.returncode


def main(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    models_cfg = cfg["models"]
    scalings   = ["standard", "smart"]

    # Bicubic is deterministic — evaluate only (no training needed)
    deterministic = {"bicubic"}

    enabled_models = [m for m, mc in models_cfg.items() if mc.get("enabled", False)]
    logger.info(f"Enabled models : {enabled_models}")
    logger.info(f"Scaling modes  : {scalings}")
    logger.info(f"Eval-only flag : {args.eval_only}")
    logger.info(f"Log file       : {log_file}")

    failures = []

    for model in enabled_models:
        for scaling in scalings:
            tag = f"{model}/{scaling}"

            # ── Train ──────────────────────────────────────────────────────
            if not args.eval_only and model not in deterministic:
                ret = run([
                    PYTHON, "train.py",
                    "--model",   model,
                    "--scaling", scaling,
                    "--config",  args.config,
                ])
                if ret != 0:
                    logger.error(f"  ✗ Training FAILED for {tag} (exit code {ret})")
                    failures.append(f"train:{tag}")
                    continue
                logger.info(f"  ✓ Training complete: {tag}")
            else:
                logger.info(f"  ⏭  Skipping training for {tag}")

            # ── Evaluate ───────────────────────────────────────────────────
            ret = run([
                PYTHON, "evaluate.py",
                "--model",   model,
                "--scaling", scaling,
                "--config",  args.config,
            ])
            if ret != 0:
                logger.error(f"  ✗ Evaluation FAILED for {tag} (exit code {ret})")
                failures.append(f"eval:{tag}")
            else:
                logger.info(f"  ✓ Evaluation complete: {tag}")

    # ── Final aggregated evaluation table ─────────────────────────────────
    logger.info("\n── Generating combined results table ──")
    run([PYTHON, "evaluate.py", "--config", args.config])

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    if failures:
        logger.warning(f"  {len(failures)} failure(s):")
        for f in failures:
            logger.warning(f"    • {f}")
    else:
        logger.info("  All experiments completed successfully.")
    logger.info(f"  Full log → {log_file.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all SR experiments")
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; load existing checkpoints and evaluate only")
    args = parser.parse_args()
    main(args)

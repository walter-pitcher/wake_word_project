#!/usr/bin/env python3
"""
Full Pipeline Orchestrator — MagisAI Wake Word v6
===================================================
Runs every stage of the training pipeline in sequence:

  1. generate_noise.py           — expand background-noise pool
  2. generate_phonetic_negatives — create phonetic confusers via TTS
  3. augment_positives.py        — offline positive augmentation
  4. mine_hard_negatives.py      — mine hard negatives from existing model
  5. train.py                    — train the v6 model
  6. evaluate.py                 — comprehensive evaluation

Each stage can be skipped with flags.

Usage:
    python new/run_pipeline.py --model models/hey_magis_v5_2026-03-15_model.tflite
    python new/run_pipeline.py --skip-noise --skip-phonetic --model models/...
"""

import sys, os, argparse, subprocess, datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PROJECT_ROOT, MODELS_DIR, NEW_MODEL_DIR,
    POS_AUGMENTED_DIR, EXPANDED_NOISE_DIR, PHONETIC_NEG_DIR,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def run(script, extra_args=None, desc=""):
    cmd = [PYTHON, str(SCRIPT_DIR / script)] + (extra_args or [])
    print(f"\n{'='*72}")
    print(f"  STAGE: {desc or script}")
    print(f"  CMD:   {' '.join(cmd)}")
    print(f"{'='*72}\n")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"\n** {script} failed (exit {result.returncode}) — stopping pipeline.")
        sys.exit(result.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="",
                    help=".tflite model for hard-neg mining (skip mining if empty)")
    ap.add_argument("--skip-noise", action="store_true")
    ap.add_argument("--skip-phonetic", action="store_true")
    ap.add_argument("--skip-augment", action="store_true")
    ap.add_argument("--skip-mine", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    start = datetime.datetime.now()
    print(f"Pipeline started at {start.isoformat()}")

    # 1 — noise expansion
    if not args.skip_noise:
        if EXPANDED_NOISE_DIR.exists() and len(list(EXPANDED_NOISE_DIR.glob("*.wav"))) >= 15:
            print("Expanded noise already exists — skipping generation.")
        else:
            run("generate_noise.py", desc="Background-noise expansion")
    else:
        print("Skipping noise generation (--skip-noise)")

    # 2 — phonetic confusers
    if not args.skip_phonetic:
        if PHONETIC_NEG_DIR.exists() and len(list(PHONETIC_NEG_DIR.glob("*.wav"))) >= 20:
            print("Phonetic negatives already exist — skipping generation.")
        else:
            run("generate_phonetic_negatives.py", desc="Phonetic confusers")
    else:
        print("Skipping phonetic generation (--skip-phonetic)")

    # 3 — positive augmentation
    if not args.skip_augment:
        if POS_AUGMENTED_DIR.exists() and len(list(POS_AUGMENTED_DIR.glob("*.wav"))) >= 3000:
            print("Augmented positives already exist — skipping.")
        else:
            run("augment_positives.py", desc="Positive augmentation")
    else:
        print("Skipping augmentation (--skip-augment)")

    # 4 — hard-negative mining
    if not args.skip_mine and args.model:
        run("mine_hard_negatives.py",
            ["--model", args.model],
            desc="Hard-negative mining")
    else:
        if not args.model:
            print("No --model provided — skipping hard-neg mining.")
        else:
            print("Skipping mining (--skip-mine)")

    # 5 — training
    if not args.skip_train:
        run("train.py",
            ["--epochs", str(args.epochs)],
            desc="Model training")
    else:
        print("Skipping training (--skip-train)")

    # 6 — evaluation (find the latest v6 tflite)
    if not args.skip_eval:
        # Prefer the new pipeline output directory, then fall back to legacy models/.
        v6_models = sorted(NEW_MODEL_DIR.glob("hey_magis_v6_*_model.tflite"))
        if not v6_models:
            v6_models = sorted(MODELS_DIR.glob("hey_magis_v6_*_model.tflite"))
        if v6_models:
            latest = v6_models[-1]
            run("evaluate.py",
                ["--model", str(latest), "--out-dir", "new/model"],
                desc="Comprehensive evaluation")
        else:
            print("No v6 .tflite found — skipping evaluation.")
    else:
        print("Skipping evaluation (--skip-eval)")

    elapsed = datetime.datetime.now() - start
    print(f"\nPipeline finished in {elapsed}")


if __name__ == "__main__":
    main()

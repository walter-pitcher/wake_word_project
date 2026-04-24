# MagisAI Wake Word v6 — Detailed Change Document

## Table of Contents

1. [Overview](#1-overview)
2. [What Changed and Why](#2-what-changed-and-why)
3. [File Reference](#3-file-reference)
4. [Expected Improvements](#4-expected-improvements)
5. [How to Run](#5-how-to-run)
6. [Deployment Recommendations](#6-deployment-recommendations)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Overview

Version 6 is a comprehensive upgrade to the "Hey Magis" wake-word training
pipeline.  **No Dart code or app changes are required** — the output `.tflite`
model has the exact same input shape `[1, 148, 40, 1]` and single sigmoid
output `[0-1]` as previous versions.

The core strategy is to attack the three root causes of the v5 accuracy
ceiling: **insufficient positive diversity**, **weak loss signal for rare
positives**, and **under-represented decision-boundary negatives**.

---

## 2. What Changed and Why

### 2.1 Positive Augmentation Pipeline (`augment_positives.py`)

| Aspect | v5 (before) | v6 (after) |
|--------|-------------|------------|
| Positive samples | 454 raw | 454 raw + ~5 000 augmented |
| Pitch diversity | none | +-1, +-2 semitones |
| Speed diversity | none | 0.9x, 1.1x time stretch |
| Noise injection (offline) | none | SNR 5 / 10 / 15 dB |
| Combined augmentations | none | pitch+noise, speed+noise |

**Why:** 454 positives against 110k negatives creates a 1:240 class
imbalance.  Expanding to 5 000+ effective positives (11 augmented variants per
raw clip) reduces the imbalance to ~1:22 and teaches the model to recognise
"Hey Magis" across pitch, speed, and noise conditions it will encounter in real
rooms.

**Expected impact:** +5-15% absolute recall at fixed specificity; stronger
generalisation to unseen speakers and environments.

---

### 2.2 Hard-Negative Mining (`mine_hard_negatives.py`)

| Aspect | v5 | v6 |
|--------|----|----|
| Mining threshold | 0.95 (top 3 000 copied) | **0.3** (all above threshold) |
| Oversampling | not explicit | 3-5x via `HARD_NEG_FRACTION=0.60` |
| Mining source | single eval CSV | full negative + hard manifests |

**Why:** The old mining copied only the top-3 000 at score >= 0.95.  These
are extreme outliers — the real decision boundary lies in the 0.3-0.9 range.
Lowering the threshold and oversampling these samples forces the model to
learn finer distinctions at the decision boundary.

**Expected impact:** Dramatic reduction in false positives on TV audio and
speech with similar cadence; tighter score distribution separation.

---

### 2.3 Phonetic-Confuser Negatives (`generate_phonetic_negatives.py`)

| Phrases added as hard negatives |
|---------------------------------|
| "Hey magic", "Hey massive", "Hey Agnes", "Hey Marcus" |
| "Okay Magis", "Hey Maxis", "Hey mattress", "Hey Mavis" |
| "Hey Magnus", "Hey Maddox", "Hey marriages", "Hey Margaret" |
| "Hey manage", "Hey majesty", "Hey malice", "Hey my guess" |

Generated via Windows SAPI5 TTS at multiple speaking rates and voices,
resampled to 16 kHz / 1.5 s.

**Why:** These phrases share the "Hey M..." phoneme pattern.  Training the
model to reject them explicitly hardens the decision boundary against
the most confusable utterances.

**Expected impact:** Near-elimination of false triggers from phonetically
similar phrases.

---

### 2.4 Focal Loss (`train.py`)

| Aspect | v5 | v6 |
|--------|----|----|
| Loss function | `binary_crossentropy` | **Focal loss** (gamma=2, alpha=0.75) |
| Label smoothing | none | epsilon = 0.05 |

**Why:** Standard BCE treats all samples equally.  With 1:3 pos:neg ratio per
batch (and hard negatives that are intrinsically harder), the model spends most
of its gradient budget on easy negatives that it already classifies correctly.

Focal loss down-weights easy examples by a factor of `(1 - p_t)^gamma`.  With
gamma=2, a correctly classified negative at score 0.05 gets weight ~0.0025
(essentially zero), while a misclassified positive at score 0.4 gets weight
~0.36 — a **144x ratio**.  This focuses learning on the hard cases.

The alpha=0.75 parameter assigns 3x the base weight to positive samples,
compensating for the 1:3 batch ratio.

Label smoothing (epsilon=0.05) softens targets to 0.025/0.975, preventing the model
from becoming over-confident and improving calibration.

**Expected impact:** Steeper precision-recall curve; higher recall at any
given specificity; better-calibrated scores.

---

### 2.5 AUC-PR as Primary Metric

| Aspect | v5 | v6 |
|--------|----|----|
| Primary metric | `val_auc` (ROC) | **`val_auc_pr`** (Precision-Recall) |

**Why:** AUC-ROC is dominated by the massive true-negative count (>100k) and
can look near-perfect even when precision is poor.  AUC-PR is far more
sensitive to the positive-class performance and is the standard metric for
imbalanced detection tasks.

---

### 2.6 SpecAugment (`train.py`, online)

| Aspect | v5 | v6 |
|--------|----|----|
| Spectrogram masking | none | 2 time masks (<=20 frames) + 2 freq masks (<=5 bins) |
| Probability | — | 50% of training samples |

**Why:** SpecAugment (Park et al., 2019) randomly zeroes out contiguous
time/frequency bands in the log-mel spectrogram.  This simulates partial
occlusion of spectral features, forcing the model to use the full
time-frequency representation rather than relying on any single band.

**Expected impact:** Better robustness to environmental noise that masks
specific frequency bands; reduced over-fitting.

---

### 2.7 Background-Noise Expansion (`generate_noise.py`)

| Aspect | v5 | v6 |
|--------|----|----|
| Noise files | 6 | **23** (6 original + 17 synthetic) |
| Noise types | limited | Room tone, HVAC, cafe, outdoor, TV murmur, appliances, rain |

**Why:** With only 6 noise files, noise augmentation has limited variety.
Training on a wider range of environmental textures yields a model that
generalises better to real-world conditions (homes, offices, cars, outdoors).

---

### 2.8 Model Architecture — SE Attention

The internal DS-CNN architecture is preserved, with one addition:

```
Last DS block output -> Squeeze-and-Excitation (ratio=4) -> GAP -> Dense
```

SE attention (Hu et al., 2018) learns per-channel weights via:
`GAP -> Dense(16, relu) -> Dense(64, sigmoid) -> channel-wise multiply`

This adds ~1 100 parameters (~5% overhead) but allows the model to
dynamically emphasise the most informative frequency channels for each input.

**Can be disabled:** `python new/train.py --no-se`

---

### 2.9 Additional Online Augmentation

| Augmentation | v5 | v6 |
|--------------|----|----|
| Random gain | +-8 dB | +-8 dB (unchanged) |
| Noise mixing | 35% prob | **40%** prob |
| Time shift | none | **+-50 ms** random circular shift |
| SpecAugment | none | 50% prob (see 2.6) |

---

### 2.10 Training Configuration

| Parameter | v5 | v6 |
|-----------|----|----|
| Epochs | 50 | **60** |
| Hard-neg fraction | 50% | **60%** |
| Noise mix probability | 35% | **40%** |
| Dropout | 0.25 | **0.30** |
| LR patience | 2 | **3** |
| Min LR | 1e-5 | **1e-6** |

---

### 2.11 Comprehensive Evaluation (`evaluate.py`)

New evaluation script produces a full deployment-ready report:

| Metric | Description |
|--------|-------------|
| **FAR per hour** | False accepts per hour of negative audio |
| **FRR** | False reject rate = 1 - recall |
| **Threshold sweep** | 22 thresholds from 0.05 to 0.99 |
| **DET curve** | FRR vs FAR plot (PNG) |
| **PR / ROC curves** | Precision-Recall and ROC plots (PNG) |
| **SNR-segmented recall** | Clean, 10 dB, 5 dB, 0 dB noise |
| **Posterior-smoothing simulation** | Recall with 1-5 consecutive hits required |
| **EER threshold** | Equal Error Rate point |
| **Best F1 threshold** | Optimal F1-score operating point |

---

## 3. File Reference

| File | Purpose |
|------|---------|
| `config.py` | All paths, constants, hyperparameters |
| `augment_positives.py` | Offline positive augmentation (454 -> 5 000+) |
| `mine_hard_negatives.py` | Score negatives, extract >=0.3 as hard negatives |
| `generate_noise.py` | Generate 17 synthetic background-noise WAVs |
| `generate_phonetic_negatives.py` | TTS-based phonetic confusers |
| `train.py` | Main training: focal loss, SpecAugment, AUC-PR |
| `evaluate.py` | Full evaluation with FAR/FRR, DET, SNR segments |
| `run_pipeline.py` | One-command pipeline orchestrator |
| `requirements.txt` | Python dependencies |
| `CHANGES.md` | This document |

---

## 4. Expected Improvements

Based on published benchmarks for each technique:

| Technique | Expected recall gain | Expected FP reduction |
|-----------|---------------------|-----------------------|
| Positive augmentation (10x) | +5-15% | — |
| Focal loss (gamma=2) | +3-8% | 20-40% |
| Hard-neg mining (0.3 threshold) | — | 30-60% |
| Phonetic confusers | — | Near-elimination of confuser FPs |
| SpecAugment | +2-5% (noisy conditions) | — |
| SE attention | +1-3% | 5-10% |
| Expanded noise pool | +2-5% (noisy conditions) | — |

**Combined estimate:**
- Recall at threshold 0.90: from ~69% (v5) to **85-92%**
- False positives at threshold 0.90: from moderate to **near zero on eval set**
- FAR per hour at 0.90: expected **< 0.5** on general negatives

---

## 5. How to Run

### 5.1 Environment Setup

```bash
# From project root
pip install -r new/requirements.txt
```

### 5.2 Full Pipeline (recommended)

```bash
python new/run_pipeline.py --model models/hey_magis_v5_2026-03-15_model.tflite
```

This runs all six stages in sequence.  Each stage auto-skips if its output
already exists.

### 5.3 Individual Steps

```bash
# 1. Generate background noise (once)
python new/generate_noise.py

# 2. Generate phonetic confusers (once)
python new/generate_phonetic_negatives.py

# 3. Augment positives (once)
python new/augment_positives.py

# 4. Mine hard negatives (requires existing .tflite)
python new/mine_hard_negatives.py --model models/hey_magis_v5_2026-03-15_model.tflite

# 5. Train
python new/train.py

# 6. Evaluate
python new/evaluate.py --model models/hey_magis_v6_2026-03-18_model.tflite
```

### 5.4 Quick Test (skip data generation)

```bash
python new/run_pipeline.py \
    --skip-noise --skip-phonetic --skip-augment --skip-mine \
    --model models/hey_magis_v5_2026-03-15_model.tflite
```

---

## 6. Deployment Recommendations

### 6.1 Threshold Selection

| Use case | Suggested threshold | Trade-off |
|----------|--------------------:|-----------|
| High recall (demo/testing) | 0.80-0.85 | More false positives |
| Balanced (daily use) | **0.88-0.92** | Best F1 trade-off |
| Low false positives (production) | 0.93-0.97 | May miss quiet utterances |

The evaluation report includes the **EER threshold** (where FAR = FRR) and
the **best F1 threshold** — use these as starting points.

### 6.2 Dart-Side Posterior Smoothing

The existing Dart code uses `requiredHits=2, hitWindowMs=400`.  With the
improved model, consider:

| Setting | Recall impact | FP reduction |
|---------|:------------:|:------------:|
| requiredHits=2, window=400ms | baseline | baseline |
| requiredHits=3, window=500ms | -2-5% | -50-70% |
| requiredHits=3, window=600ms | -3-8% | -60-80% |

The evaluation report's "Posterior-smoothing simulation" section shows the
recall impact at various consecutive-hit counts.

### 6.3 Model File

The output `.tflite` file is in `models/hey_magis_v6_<date>_model.tflite`.
It is a drop-in replacement for the existing model — upload to Firebase
Storage and update the app's model URL.

---

## 7. Troubleshooting

| Issue | Solution |
|-------|----------|
| `pyttsx3` fails on headless server | Skip phonetic generation: `--skip-phonetic` |
| TensorFlow GPU not available (Win) | Use TF 2.10 for native GPU, or WSL2 for TF 2.11+ |
| librosa `time_stretch` slow | Normal for first run (JIT); ~2 min for 454 files |
| Out of memory during training | Reduce `BATCH_SIZE` in `config.py` (16 or 24) |
| Augmented positives too many | Reduce `PITCH_SHIFTS` or `SPEED_FACTORS` in `config.py` |
| Model accuracy not improving | Check that augmented positives loaded (printed at start) |

---

*Document generated for MagisAI Wake Word v6 pipeline.*

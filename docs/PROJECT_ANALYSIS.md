# MagisAI Wake Word Project — Detailed Project Analysis

> **Document Version:** 1.0  
> **Date:** March 17, 2026  
> **Scope:** Complete file-by-file analysis, classification of core vs. unnecessary files, architecture review, data inventory, known issues, and recommendations.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [File-by-File Analysis](#3-file-by-file-analysis)
   - 3.1 [Training Scripts (Core)](#31-training-scripts-core)
   - 3.2 [Evaluation Scripts (Core)](#32-evaluation-scripts-core)
   - 3.3 [Data Preparation Scripts (Supporting)](#33-data-preparation-scripts-supporting)
   - 3.4 [Audio Processing Scripts (Supporting)](#34-audio-processing-scripts-supporting)
   - 3.5 [Analysis & Utility Scripts (Supporting)](#35-analysis--utility-scripts-supporting)
   - 3.6 [Data Acquisition Scripts (One-Time Use)](#36-data-acquisition-scripts-one-time-use)
   - 3.7 [Data Files & Manifests](#37-data-files--manifests)
   - 3.8 [Model Outputs & Reports](#38-model-outputs--reports)
   - 3.9 [Third-Party / Dataset Files](#39-third-party--dataset-files)
4. [File Classification Summary](#4-file-classification-summary)
5. [Core Architecture Deep-Dive](#5-core-architecture-deep-dive)
6. [Model Evolution History](#6-model-evolution-history)
7. [Data Inventory](#7-data-inventory)
8. [Feature Pipeline Specification](#8-feature-pipeline-specification)
9. [Known Issues & Technical Debt](#9-known-issues--technical-debt)
10. [Dependency Map](#10-dependency-map)

---

## 1. Project Overview

This project trains a **wake word detection model** for the phrase **"Hey Magis"**, part of a Catholic AI mobile assistant called **MagisAI**. The system is designed for on-device inference on iOS and Android via Flutter/FlutterFlow using the `tflite_flutter` plugin.

**Key characteristics:**
- Binary classification: "Hey Magis" (positive) vs. everything else (negative)
- Lightweight CNN with depthwise separable convolutions (~20K parameters)
- Log-mel spectrogram features: input shape `[1, 148, 40, 1]`
- Single sigmoid output: probability in `[0, 1]`
- Training done on Windows using TensorFlow/Python
- Deployment via TensorFlow Lite (.tflite)

---

## 2. Directory Structure

```
wake_word_project/
│
├── docs/                              # Documentation (this folder)
│
├── data/                              # All training/evaluation data
│   ├── positive_manifest_v4.txt       # 468 positive wav paths
│   ├── negative_manifest.txt          # 105,896 negative wav paths
│   ├── negative_manifest_hard.txt     # ~5,163 hard negative paths
│   ├── negative_manifest_hard_merged.txt  # 13,385 merged hard negatives
│   ├── negative_manifest_report.txt   # Build report for negative manifest
│   ├── *.txt (various reports)        # Data quality reports
│   │
│   ├── positive_new_friends/          # Crowdsourced recordings (6 speakers)
│   │   ├── heymagis_cristina_.../     # ~20 clips per session
│   │   ├── heymagis_judy_ramos_.../
│   │   ├── heymagis_kristina_garcia_.../
│   │   ├── heymagis_mat_.../
│   │   ├── heymagis_michael_g_.../
│   │   └── heymagis_wg_.../
│   │
│   ├── positive_trimmed_1p5_A/        # Trimmed/padded positives (1.5s clips)
│   │
│   ├── negative/                      # All negative audio
│   │   ├── _background_noise_/        # 6 background noise files
│   │   ├── backward/, bed/, ...       # Speech Commands dataset (35 word folders)
│   │   ├── Hey__Magis_wrong_ways_to_say/  # Phonetically similar phrases
│   │   └── tv_samples/               # TV audio sliced into 1.5s clips
│   │
│   └── hard_negatives_top/            # Top 3,000 mined false positives
│
├── models/                            # Trained model outputs
│   ├── *_train_report.txt             # Training reports for v3, v4, v5
│   ├── eval_v3_*.csv                  # Evaluation scores and thresholds
│   ├── scores_*.csv                   # Multi-bucket evaluation scores
│   └── Not_Using/                     # Deprecated model metadata
│
├── *.py (29 Python scripts)           # All source code (flat structure)
│
├── validation_list.txt                # Speech Commands validation split
├── testing_list.txt                   # Speech Commands test split
├── LICENSE                            # CC BY 4.0 (Speech Commands)
└── README.md                          # Speech Commands dataset description
```

---

## 3. File-by-File Analysis

### 3.1 Training Scripts (Core)

These are the scripts that actually train the model. They represent the evolution of the project from v1 to v5.

#### `hey_magis_v4_1p5_hard3061.py` — **THE CURRENT CORE TRAINING SCRIPT**

| Property | Value |
|---|---|
| **Status** | **ACTIVE — This is the script you should use and modify** |
| **Version** | v5 (despite the filename suggesting v4) |
| **Input shape** | `(148, 40, 1)` — 1.5s clips |
| **Positives** | `positive_new_friends/` + `positive_trimmed_1p5_A/` |
| **Negatives** | `negative_manifest.txt` + `negative_manifest_hard_merged.txt` + `hard_negatives_top/` |
| **Output** | `models/hey_magis_v5_2026-03-15_model.tflite` |
| **Loss** | Binary cross-entropy (needs upgrade to focal loss) |
| **Key features** | Speaker-aware split, per-epoch negative resampling, 50% hard negative fraction, noise mixing augmentation, ReduceLROnPlateau |

**Architecture defined in this file:**
```
Conv2D(16) → [DW-Conv2D + Conv2D(24)] → [DW-Conv2D + Conv2D(32)] →
[DW-Conv2D + Conv2D(48)] → [DW-Conv2D + Conv2D(64)] →
GlobalAvgPool → Dropout(0.25) → Dense(1, sigmoid)
```

Strides of (2,2) applied in the 1st and 3rd depthwise blocks (spatial downsampling).

**Key config:**
- `SAMPLE_RATE = 16000`
- `CLIP_SECONDS = 1.5`
- `EPOCHS = 50`, `PATIENCE = 10`
- `NEG_MULTIPLIER = 3`
- `HARD_NEG_FRACTION = 0.50`
- `NOISE_MIX_PROB = 0.35`
- `NOISE_MIX_SNR_DB = (0.0, 18.0)`
- `LEARNING_RATE = 5e-4`

---

#### `train_hey_magis_v4_1.py` — Experimental full-negative-exposure variant

| Property | Value |
|---|---|
| **Status** | Experimental / not the primary script |
| **Difference from v5** | Different model architecture (every block has stride=2), `NEG_MULTIPLIER=8`, `LEARNING_RATE=1e-3`, no validation set, no early stopping |
| **Why it exists** | Tried exposing the model to many more negatives per epoch |
| **Recommendation** | **Do NOT use as the base for future work** — it lacks validation, early stopping, and speaker-aware splits |

**Architecture difference:** This file puts stride=2 in ALL 5 blocks (vs. only 2 blocks in the main script). This makes the model much more aggressive at downsampling, which changes its behavior significantly.

---

#### `train_hey_magis_v3.py` — Previous version (1.0s clips)

| Property | Value |
|---|---|
| **Status** | Superseded by v5 |
| **Input shape** | `(98, 40, 1)` — 1.0s clips |
| **Difference** | Uses `positive_trimmed_1p2_A/` only, 1.0s clips, lower thresholds tested |
| **Recommendation** | **Historical reference only — do not use** |

---

#### `train_hey_magis_v2.py` — Earlier version (log-mel, DS-CNN)

| Property | Value |
|---|---|
| **Status** | Superseded |
| **Difference** | 1.0s clips, `LOWER_HZ=20` (vs 80), `UPPER_HZ=7600`, exports both float32 and int8 TFLite |
| **Recommendation** | **Historical reference only** |

---

#### `train_hey_magis.py` — Original v1 (MFCC-based, Conv1D)

| Property | Value |
|---|---|
| **Status** | **OBSOLETE** |
| **Difference** | Completely different architecture (Conv1D, not Conv2D), uses MFCC features (not log-mel), 2.0s clips, uses librosa (not TF ops) |
| **Recommendation** | **Can be deleted or archived.** Does not match current pipeline at all. |

---

#### `train_wakeword_v3.py` — Duplicate of `train_hey_magis_v3.py`

| Property | Value |
|---|---|
| **Status** | **DUPLICATE — Unnecessary** |
| **Recommendation** | Delete this file. It is an exact copy of `train_hey_magis_v3.py`. |

---

### 3.2 Evaluation Scripts (Core)

#### `eval_wakeword_tflite_buckets.py` — **CURRENT PRIMARY EVALUATION SCRIPT**

| Property | Value |
|---|---|
| **Status** | **ACTIVE — Use this for model evaluation** |
| **Purpose** | Scores a .tflite model across 4 separate data buckets: (1) positive_new_friends, (2) positive_trimmed, (3) random 10K negatives, (4) all hard negatives |
| **Key feature** | Auto-adapts to model input shape (reads it from .tflite), matching feature extraction |
| **Output** | 4 CSV files: `scores_pos_newfriends.csv`, `scores_pos_trimmed.csv`, `scores_neg_10k.csv`, `scores_hard.csv` |
| **Feature pipeline** | Matches training: power mel spectrogram, per-example mean/std normalization, 80-7600 Hz mel range |

**Usage:**
```bash
python eval_wakeword_tflite_buckets.py \
  --model models/hey_magis_v5_2026-03-15_model.tflite \
  --pos_newfriends data/positive_new_friends \
  --pos_trimmed data/positive_trimmed_1p5_A \
  --neg_manifest data/negative_manifest.txt \
  --hard_manifest data/negative_manifest_hard_merged.txt
```

---

#### `eval_wakeword_tflite.py` — Older evaluation script (v3-era)

| Property | Value |
|---|---|
| **Status** | Partially outdated |
| **Difference from buckets** | Single-bucket eval, hardcoded to 1.0s/98-frame input, `LOWER_HZ=60` and `UPPER_HZ=7800` (mismatched from training!) |
| **Known bug** | The mel range (60-7800 Hz) does NOT match training (80-7600 Hz). This means scores produced by this script may not reflect real model behavior. |
| **Recommendation** | **Use `eval_wakeword_tflite_buckets.py` instead.** Keep this only as reference. |

---

#### `debug_tflite_outputs.py` — TFLite debugging tool

| Property | Value |
|---|---|
| **Status** | Utility / debugging only |
| **Purpose** | Loads a .tflite model and a few wavs from score CSVs, recomputes features, and prints input/output details (dtype, quantization params, feature stats) |
| **Hardcoded** | Paths point to `C:\Users\miker\...` — needs updating to work |
| **Recommendation** | Useful for debugging but needs path fixes |

---

### 3.3 Data Preparation Scripts (Supporting)

#### `build_positive_manifest.py`

| Property | Value |
|---|---|
| **Status** | Active utility |
| **Purpose** | Recursively lists all .wav files from given directories and writes a manifest file |
| **Usage** | `python build_positive_manifest.py --out data/positive_manifest_v4.txt --dirs data/positive_new_friends data/positive_trimmed_1p5_A` |

---

#### `build_negative_manifest.py`

| Property | Value |
|---|---|
| **Status** | Active utility |
| **Purpose** | Recursively finds .wav files under `data/negative`, writes manifest, optionally writes hard-negative manifest, generates junk/duplicate report |
| **Features** | SHA-1 deduplication, junk file detection, hard negative folder detection by naming conventions |

---

#### `build_negative_manifest_v2.py`

| Property | Value |
|---|---|
| **Status** | Newer version of manifest builder |
| **Difference** | Fast/full dedupe modes, archive detection |
| **Recommendation** | Redundant with v1. Keep whichever one you prefer. |

---

#### `build_hard_manifest_from_folder.py`

| Property | Value |
|---|---|
| **Status** | Active utility |
| **Purpose** | Merges `negative_manifest_hard.txt` with all wavs from `hard_negatives_top/` → writes `negative_manifest_hard_merged.txt` |

---

### 3.4 Audio Processing Scripts (Supporting)

#### `trim_recordings.py` — Original trimmer (v1)

| Property | Value |
|---|---|
| **Status** | Superseded |
| **Purpose** | Interactive tool to trim silence from recordings, create 2.0s clips |
| **Problems** | Uses 2.0s target (not 1.5s), interactive (requires user input), uses librosa |
| **Recommendation** | **Outdated. Use trim_recordings_v2 or v3 instead.** |

---

#### `trim_recordings_v2.py`

| Property | Value |
|---|---|
| **Status** | Superseded by v3 |
| **Purpose** | RMS-based onset/offset detection, 1.0s clips, quality bucketing |

---

#### `trim_recordings_v3.py`

| Property | Value |
|---|---|
| **Status** | Active |
| **Purpose** | 1.2s clips, A/B/C quality buckets by coverage ratio |
| **Note** | Produces the clips that then get padded to 1.5s |

---

#### `pad_to_1p5.py`

| Property | Value |
|---|---|
| **Status** | Active |
| **Purpose** | Pads clips shorter than 1.5s to exactly 1.5s (24,000 samples @ 16kHz) |
| **Modes** | `pad_end` (silence at end) or `pad_both` (split padding) |
| **Usage** | `python pad_to_1p5.py --in_dir data/positive_trimmed_1p2_A --out_dir data/positive_trimmed_1p5_A` |

---

#### `slice_tv.py`

| Property | Value |
|---|---|
| **Status** | One-time use |
| **Purpose** | Slices a long TV recording into 1.5s clips with 50% overlap, skips silent clips |
| **Output** | `data/negative/tv_samples/` |

---

### 3.5 Analysis & Utility Scripts (Supporting)

#### `mine_hard_negatives.py`

| Property | Value |
|---|---|
| **Status** | Active — critical for hard negative mining |
| **Purpose** | Reads `eval_v3_scores_neg.csv`, copies the top 3,000 highest-scoring negatives (score > 0.95) to `data/hard_negatives_top/` |
| **Dependency** | Requires eval script to have been run first to produce score CSVs |

---

#### `count_positives.py`

| Property | Value |
|---|---|
| **Purpose** | Counts wav files in `positive_new_friends/` and `positive_trimmed_1p5_A/` |
| **Recommendation** | Simple utility, keep for reference |

---

#### `count_negatives.py`

| Property | Value |
|---|---|
| **Purpose** | Counts negatives by subfolder, hard_negatives_top, and manifest line counts |

---

#### `check_clipping.py`

| Property | Value |
|---|---|
| **Purpose** | Flags wav files with near-fullscale samples (potential clipping) |

---

#### `quarantine_quiet_positives.py`

| Property | Value |
|---|---|
| **Purpose** | Moves quiet positives (below dBFS threshold) to a quarantine folder |

---

#### `rank_wavs_by_rms.py`

| Property | Value |
|---|---|
| **Purpose** | Ranks wav files by RMS loudness (first 1 second) |

---

#### `inspect_wavs.py`

| Property | Value |
|---|---|
| **Purpose** | Inspects wav metadata (sample rate, channels, duration, RMS) |

---

#### `summarize_wavs.py`

| Property | Value |
|---|---|
| **Purpose** | Summary statistics: sample rate, channels, duration, RMS percentiles |

---

### 3.6 Data Acquisition Scripts (One-Time Use)

#### `download_typeform.py`

| Property | Value |
|---|---|
| **Status** | One-time use / data collection |
| **Purpose** | Downloads audio recordings from Typeform via API |
| **Dependency** | `requests`, `pandas` |

---

#### `download_audio_fixed.py`

| Property | Value |
|---|---|
| **Status** | One-time use / data collection |
| **Purpose** | Downloads from Mux URLs in JSON, converts to WAV with ffmpeg |

---

### 3.7 Data Files & Manifests

#### Manifests (Critical for Training)

| File | Lines | Purpose | Status |
|---|---|---|---|
| `data/positive_manifest_v4.txt` | 468 | All positive wav paths | Active |
| `data/negative_manifest.txt` | 105,896 | All negative wav paths | Active |
| `data/negative_manifest_hard.txt` | ~5,163 | Wrong-ways-to-say negatives | Active |
| `data/negative_manifest_hard_merged.txt` | 13,385 | Hard + mined confusers | **Primary hard neg manifest** |

#### Reports (Informational)

| File | Purpose |
|---|---|
| `data/negative_manifest_report.txt` | Junk/suspicious files found during manifest build |
| `data/positive_trimmed_1p5_A_clipping_report.txt` | Clipping analysis |
| `data/pad_positive_trimmed_1p2_A_to_1p5_report.txt` | Padding operation report |
| `data/positive_quarantine_report.txt` | Quarantined quiet files |
| `data/positive_new_friends_quiet_loud.txt` | Volume analysis |
| `data/positive_new_friends_summary.txt` | Summary stats |
| `data/inspect_positive_new_friends_report.txt` | Wav inspection report |

---

### 3.8 Model Outputs & Reports

| File | Purpose |
|---|---|
| `models/hey_magis_v3_train_report.txt` | v3 training report (val_pos=29) |
| `models/hey_magis_v4_train_report.txt` | v4 training report (perfect val scores — overfitting risk) |
| `models/hey_magis_v5_2026-03-15_train_report.txt` | **Latest training report** (v5) |
| `models/eval_v3_summary.txt` | v3 evaluation summary |
| `models/eval_v3_scores_pos.csv` | v3 positive scores |
| `models/eval_v3_scores_neg.csv` | v3 negative scores |
| `models/eval_v3_thresholds.csv` | v3 threshold sweep |
| `models/scores_pos_newfriends.csv` | Multi-bucket: new friends scores |
| `models/scores_pos_trimmed.csv` | Multi-bucket: trimmed positives scores |
| `models/scores_neg_10k.csv` | Multi-bucket: 10K random negative scores |
| `models/scores_hard.csv` | Multi-bucket: hard negative scores |
| `models/Not_Using/` | Deprecated metadata from v1/v2 |

---

### 3.9 Third-Party / Dataset Files

| File | Purpose | Status |
|---|---|---|
| `README.md` | Speech Commands v0.02 dataset description | From Google dataset — not project-specific |
| `LICENSE` | CC BY 4.0 license | Required for Speech Commands usage |
| `validation_list.txt` | Speech Commands validation partition (9,981 lines) | From dataset |
| `testing_list.txt` | Speech Commands test partition (11,005 lines) | From dataset |

---

## 4. File Classification Summary

### CORE FILES (Must Keep, Actively Used)

| File | Role |
|---|---|
| `hey_magis_v4_1p5_hard3061.py` | **Primary training script** |
| `eval_wakeword_tflite_buckets.py` | **Primary evaluation script** |
| `mine_hard_negatives.py` | Hard negative mining pipeline |
| `build_positive_manifest.py` | Positive manifest builder |
| `build_negative_manifest.py` | Negative manifest builder |
| `build_hard_manifest_from_folder.py` | Merged hard manifest builder |
| `pad_to_1p5.py` | Audio padding utility |
| `data/positive_manifest_v4.txt` | Positive file list |
| `data/negative_manifest.txt` | Negative file list |
| `data/negative_manifest_hard_merged.txt` | Hard negative file list |
| All audio data directories | Training data |

### SUPPORTING FILES (Useful But Not Required for Training)

| File | Role |
|---|---|
| `trim_recordings_v3.py` | Audio trimming (use if re-processing positives) |
| `slice_tv.py` | TV audio slicing (use if adding new TV recordings) |
| `count_positives.py` | Quick stats |
| `count_negatives.py` | Quick stats |
| `check_clipping.py` | Audio quality checking |
| `inspect_wavs.py` | Audio metadata inspection |
| `summarize_wavs.py` | Audio summary statistics |
| `debug_tflite_outputs.py` | TFLite debugging |
| `eval_wakeword_tflite.py` | Older single-bucket evaluation |
| All `data/*.txt` report files | Historical records |
| All `models/*.txt` and `models/*.csv` | Training/eval records |

### UNNECESSARY / REDUNDANT FILES (Safe to Delete or Archive)

| File | Reason |
|---|---|
| `train_hey_magis.py` | **Obsolete** v1 using MFCC + Conv1D (completely different architecture) |
| `train_wakeword_v3.py` | **Duplicate** of `train_hey_magis_v3.py` |
| `train_hey_magis_v2.py` | **Superseded** by v5 |
| `train_hey_magis_v3.py` | **Superseded** by v5 (uses 1.0s input, not 1.5s) |
| `train_hey_magis_v4_1.py` | **Experimental**, lacks validation/early stopping |
| `trim_recordings.py` | **Superseded** by v2/v3 (uses 2.0s clips) |
| `trim_recordings_v2.py` | **Superseded** by v3 |
| `build_negative_manifest_v2.py` | **Redundant** with v1 |
| `download_typeform.py` | **One-time use** (data already downloaded) |
| `download_audio_fixed.py` | **One-time use** (data already downloaded) |
| `quarantine_quiet_positives.py` | **One-time use** (already run) |
| `rank_wavs_by_rms.py` | **One-time use** (analysis complete) |
| `models/Not_Using/` | **Deprecated** metadata |
| `data/negative_manifest_hard.txt` | **Superseded** by `negative_manifest_hard_merged.txt` |

### THIRD-PARTY (Keep As-Is)

| File | Reason |
|---|---|
| `README.md` | Speech Commands dataset (rename or replace with project README) |
| `LICENSE` | Required for legal compliance |
| `validation_list.txt` | Speech Commands partition |
| `testing_list.txt` | Speech Commands partition |

---

## 5. Core Architecture Deep-Dive

### Model Architecture (DS-CNN)

The current model is a **Depthwise Separable CNN** inspired by MobileNet and the ARM DS-CNN keyword spotting architecture. Here is the exact layer structure from `hey_magis_v4_1p5_hard3061.py`:

```
Input: [batch, 148, 40, 1]
│
├── Conv2D(16, 3x3, same, no bias) → BatchNorm → ReLU
│
├── DepthwiseConv2D(3x3, stride=2x2, same, no bias) → BatchNorm → ReLU
├── Conv2D(24, 1x1, same, no bias) → BatchNorm → ReLU
│
├── DepthwiseConv2D(3x3, stride=1x1, same, no bias) → BatchNorm → ReLU
├── Conv2D(32, 1x1, same, no bias) → BatchNorm → ReLU
│
├── DepthwiseConv2D(3x3, stride=2x2, same, no bias) → BatchNorm → ReLU
├── Conv2D(48, 1x1, same, no bias) → BatchNorm → ReLU
│
├── DepthwiseConv2D(3x3, stride=1x1, same, no bias) → BatchNorm → ReLU
├── Conv2D(64, 1x1, same, no bias) → BatchNorm → ReLU
│
├── GlobalAveragePooling2D
├── Dropout(0.25)
└── Dense(1, sigmoid)

Output: [batch, 1]  (probability 0-1)
```

**Parameter count:** ~20K parameters (very lightweight for mobile)

**Spatial dimensions through the network:**
```
Input:  148 × 40
After stride-2 block 1:  74 × 20
After stride-1 block 2:  74 × 20
After stride-2 block 3:  37 × 10
After stride-1 block 4:  37 × 10
GlobalAvgPool:  1 × 1 × 64
```

### Feature Extraction Pipeline

```
Raw Audio (16kHz, mono, 1.5s = 24,000 samples)
    │
    ▼
STFT (frame_length=400 [25ms], frame_step=160 [10ms], fft_length=512, Hann window)
    │
    ▼
Power Spectrogram (|STFT|²)
    │
    ▼
Mel Filter Bank (40 bins, 80Hz - 7600Hz)
    │
    ▼
Log (with floor at 1e-10)
    │
    ▼
Per-Example Normalization: (x - mean) / (std + 1e-6)
    │
    ▼
Trim/Pad to 148 frames × 40 mel bins
    │
    ▼
Add channel dimension → [148, 40, 1]
```

### Augmentation Pipeline (Training Only)

1. **Random Gain:** ±8dB uniform random
2. **Noise Mixing:** 35% probability, random background noise file, SNR between 0-18dB
3. No pitch shifting, speed perturbation, or SpecAugment (these are needed improvements)

### Training Loop

The training loop is custom (not standard Keras `model.fit` across all epochs):

1. For each epoch:
   - Sample `len(train_pos) × NEG_MULTIPLIER` negatives fresh
   - 50% of negatives come from hard negative set, 50% from general negatives
   - Build new `tf.data.Dataset` each epoch
   - Run 1 epoch of `model.fit()` with validation
2. Callbacks: EarlyStopping on `val_auc`, ReduceLROnPlateau, ModelCheckpoint
3. After training: export to .tflite with dynamic range quantization

---

## 6. Model Evolution History

| Version | Script | Input Shape | Clip Length | Features | Architecture | Key Changes |
|---|---|---|---|---|---|---|
| v1 | `train_hey_magis.py` | variable | 2.0s | MFCC | Conv1D 64→128→256 | Original, librosa-based |
| v2 | `train_hey_magis_v2.py` | (98, 40, 1) | 1.0s | Log-mel | DS-CNN | Switched to log-mel + DS-CNN |
| v3 | `train_hey_magis_v3.py` | (98, 40, 1) | 1.0s | Log-mel | DS-CNN | Added hard negatives, per-epoch resampling |
| v4 | `hey_magis_v4_1p5_hard3061.py` | (148, 40, 1) | 1.5s | Log-mel | DS-CNN | Extended to 1.5s, added new_friends, speaker-aware split |
| v4.1 | `train_hey_magis_v4_1.py` | (148, 40, 1) | 1.5s | Log-mel | DS-CNN (different) | Experimental full-neg exposure |
| **v5** | `hey_magis_v4_1p5_hard3061.py` | **(148, 40, 1)** | **1.5s** | **Log-mel** | **DS-CNN** | **Current: hard_neg_merged, HARD_FRAC=0.50** |

### Training Report Analysis

**v3 (1.0s clips, 29 val positives):**
- Perfect scores 0.30-0.70 (recall=0.966, spec=1.000)
- Tiny validation set (29 pos) — **not statistically reliable**

**v4 (1.5s clips, 129 val positives):**
- Perfect scores 0.30-0.80 (recall=1.000, spec=1.000)
- **Suspiciously perfect** — possible overfitting to validation set or data leakage

**v5 (latest, 129 val positives):**
- At t=0.30: recall=0.938, spec=0.995 (2 FP)
- At t=0.90: recall=0.690, spec=1.000 (0 FP)
- At t=0.70: recall=0.829, spec=0.997 (1 FP)
- **More realistic** results — shows actual decision boundary challenges
- **Problem:** Recall drops sharply above 0.80 threshold

---

## 7. Data Inventory

### Positive Samples

| Source | Count | Description |
|---|---|---|
| `positive_new_friends/` | ~120 clips | 6 dedicated speakers, 20 clips each, recorded via Typeform |
| `positive_trimmed_1p5_A/` | ~348 clips | 133 diverse speakers, avg 3-4 clips each, trimmed and padded to 1.5s |
| **Total** | **~468** | **139 unique speakers** |

### Negative Samples

| Source | Count | Description |
|---|---|---|
| Speech Commands dataset | ~100,000 | 35 word categories (backward, bed, bird, cat, ...) |
| TV audio samples | ~5,101 | Sliced TV recording |
| Wrong ways to say "Hey Magis" | ~61 | Phonetically similar phrases |
| Background noise | 6 | Room noise, pink noise, white noise, etc. |
| **Total in manifest** | **105,896** | |

### Hard Negatives

| Source | Count | Description |
|---|---|---|
| `negative_manifest_hard.txt` | ~5,163 | Wrong ways folder + manually identified |
| `hard_negatives_top/` | ~3,000 | Mined from model (score > 0.95 on negatives) |
| **Merged total** | **13,385** | `negative_manifest_hard_merged.txt` |

### Class Imbalance

```
Positives:  468
Negatives:  105,896
Ratio:      1 : 226

Hard negatives: 13,385
Ratio (hard):   1 : 29

Effective training ratio (with NEG_MULTIPLIER=3, HARD_FRAC=0.50):
  Per epoch: 468 pos vs. 1,404 neg (702 hard + 702 general)
  Effective: 1 : 3
```

---

## 8. Feature Pipeline Specification

This section documents the EXACT feature pipeline that must be matched between training and inference (Dart code).

| Parameter | Value | Notes |
|---|---|---|
| Sample rate | 16,000 Hz | All audio must be 16kHz mono |
| Clip duration | 1.5 seconds | 24,000 samples |
| STFT frame length | 400 samples | 25 ms |
| STFT frame step | 160 samples | 10 ms |
| FFT length | 512 | |
| Window function | Hann | |
| `pad_end` | False | Do NOT pad the last frame |
| Mel bins | 40 | |
| Mel lower Hz | 80.0 | |
| Mel upper Hz | 7,600.0 | |
| Mel type | Power spectrogram (|STFT|²) | Not magnitude |
| Log floor | 1e-10 | `tf.maximum(mel, 1e-10)` then `tf.math.log` |
| Normalization | Per-example mean/std | `(x - mean) / (std + 1e-6)` |
| Target frames | 148 | Trim or pad to exactly 148 |
| Channel dimension | 1 | Expand at end: `[148, 40, 1]` |
| **Final input shape** | **[1, 148, 40, 1]** | Batch of 1 for inference |
| **Output shape** | **[1, 1]** | Single sigmoid probability |

### CRITICAL: Feature Mismatch Warning

The older eval script `eval_wakeword_tflite.py` uses `LOWER_HZ=60` and `UPPER_HZ=7800`. This does NOT match training (`80` and `7600`). Always use `eval_wakeword_tflite_buckets.py` which correctly uses `80.0` and `7600.0`.

---

## 9. Known Issues & Technical Debt

### Critical Issues

1. **False Positives from TV Audio:** The model triggers on speech with similar cadence/rhythm to "Hey Magis." The TV negative set (5,101 clips) is not sufficient.

2. **Threshold Sensitivity:** At 0.90, recall drops to 69%. At 0.70, there are still false positives. There is no single threshold that balances FPR and FRR well.

3. **Class Imbalance (1:226):** The raw positive-to-negative ratio biases the decision boundary toward predicting "negative." The current `NEG_MULTIPLIER=3` mitigates this during training, but the evaluation does not reflect real-world class distribution.

4. **Insufficient Positive Augmentation:** Only 468 raw positive samples with minimal augmentation (random gain + noise mixing). No pitch shifting, speed perturbation, or SpecAugment.

5. **Tiny Background Noise Set:** Only 6 noise files for noise injection. This limits robustness to real-world environments.

### Moderate Issues

6. **No Focal Loss:** Using standard binary cross-entropy, which does not down-weight easy examples. This means the model spends most gradient updates on trivially-classified Speech Commands samples.

7. **No AUC-PR Metric:** Using AUC-ROC which can be misleadingly optimistic with extreme class imbalance. AUC-PR (precision-recall) would be more informative.

8. **Validation Set Contamination Risk:** The v4 training report shows perfect scores (recall=1.000, spec=1.000 up to t=0.80), suggesting either overfitting or insufficient validation challenge.

9. **No SNR-Segmented Evaluation:** Cannot assess performance under different noise conditions.

10. **No DET Curve:** No Detection Error Tradeoff curve for proper wake word evaluation.

### Code Quality Issues

11. **Flat file structure:** All 29 Python scripts are in the project root. Should be organized into `scripts/`, `training/`, `evaluation/`, `utils/` directories.

12. **Hardcoded paths:** `debug_tflite_outputs.py` has absolute paths (`C:\Users\miker\...`).

13. **Duplicate scripts:** `train_wakeword_v3.py` is identical to `train_hey_magis_v3.py`.

14. **No `requirements.txt`:** Dependencies are not documented. Must infer from imports.

15. **README is for Speech Commands:** The project `README.md` is the Google Speech Commands dataset readme, not a project-specific readme.

---

## 10. Dependency Map

### Python Dependencies (Inferred from Imports)

| Package | Used By | Purpose |
|---|---|---|
| `tensorflow` (>=2.10) | All training/eval scripts | Model training, TFLite conversion, audio ops |
| `numpy` | All scripts | Numerical operations |
| `librosa` | v1 training, trim_recordings.py | Audio loading, MFCC extraction (legacy) |
| `soundfile` | trim_recordings_v3, slice_tv, pad_to_1p5 | WAV file I/O |
| `resampy` | slice_tv.py | Audio resampling |
| `requests` | download_typeform.py | HTTP downloads |
| `pandas` | download_typeform.py | JSON/CSV handling |
| `scikit-learn` | train_hey_magis.py (v1 only) | train_test_split |
| `tflite_runtime` | eval_wakeword_tflite.py (optional) | Lightweight TFLite inference |

### Standard Library Dependencies

`os`, `sys`, `random`, `pathlib`, `wave`, `csv`, `json`, `math`, `hashlib`, `shutil`, `argparse`, `dataclasses`, `typing`, `datetime`

### Suggested `requirements.txt`

```
tensorflow>=2.10
numpy>=1.21
librosa>=0.10.0
soundfile>=0.12.0
resampy>=0.4.0
```

---

*End of Project Analysis Document*

# MagisAI Wake Word — "Hey Magis"

Custom on-device wake word detection for **MagisAI**, a Catholic AI mobile assistant. The model learns to recognize the phrase **"Hey Magis"** and reject everything else, then ships as a **TensorFlow Lite** (`.tflite`) file for inference in a **Flutter / FlutterFlow** app via `tflite_flutter`.

This repository contains the full pipeline: data collection, preprocessing, training, evaluation, hard-negative mining, and export—not just a pretrained model.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Requirements](#requirements)
4. [Repository Layout](#repository-layout)
5. [Model Contract (Must Not Break)](#model-contract-must-not-break)
6. [Feature Extraction Pipeline](#feature-extraction-pipeline)
7. [Data Layout and Manifests](#data-layout-and-manifests)
8. [Training Pipelines](#training-pipelines)
9. [Recommended Workflow (v6)](#recommended-workflow-v6)
10. [Legacy Workflow (v5)](#legacy-workflow-v5)
11. [Script Reference](#script-reference)
12. [Evaluation and Metrics](#evaluation-and-metrics)
13. [Latest Results (v6, 2026-03-18)](#latest-results-v6-2026-03-18)
14. [Mobile Deployment](#mobile-deployment)
15. [Threshold and Posterior Smoothing](#threshold-and-posterior-smoothing)
16. [Known Issues and Limitations](#known-issues-and-limitations)
17. [Documentation](#documentation)
18. [Speech Commands Dataset](#speech-commands-dataset)
19. [License](#license)

---

## Overview

| Item | Value |
|------|--------|
| Wake phrase | **Hey Magis** |
| Task | Binary classification (wake word vs. not) |
| Sample rate | 16 kHz, mono |
| Clip length | **1.5 s** (24,000 samples) |
| Input tensor | `[1, 148, 40, 1]` — log-mel spectrogram |
| Output tensor | `[1, 1]` — sigmoid probability in `[0, 1]` |
| Architecture | Depthwise-separable CNN (~20K params); v6 adds squeeze-and-excitation |
| Training | Python + TensorFlow on Windows (CPU or GPU) |
| Inference | TensorFlow Lite on iOS / Android |

### What the system does

1. **Listen** — The app captures microphone audio continuously.
2. **Window** — Audio is sliced into 1.5 s windows (often with a sliding hop, e.g. 10 ms in Dart).
3. **Features** — Each window is converted to a 148×40 log-mel map (must match training exactly).
4. **Score** — The `.tflite` model outputs a probability that the clip contains "Hey Magis".
5. **Confirm** — Dart applies a score threshold plus **posterior smoothing** (e.g. 2–3 consecutive hits) before triggering the assistant.

### Pipeline versions

| Version | Location | Status |
|---------|----------|--------|
| **v6** (recommended) | `new/` | Focal loss, SpecAugment, augmented positives, expanded noise, phonetic confusers, comprehensive eval |
| **v5** | `hey_magis_v4_1p5_hard3061.py` | Stable baseline; binary cross-entropy, simpler augmentation |
| v1–v4 | `train_hey_magis*.py` | Historical; 1.0–2.0 s clips, older features |

---

## Quick Start

### 1. Install dependencies (v6 pipeline)

From the project root:

```bash
pip install -r new/requirements.txt
```

### 2. Prepare data

You need WAV data under `data/` (see [Data Layout](#data-layout-and-manifests)). If manifests are missing:

```bash
python build_positive_manifest.py --out data/positive_manifest_v4.txt \
  --dirs data/positive_new_friends data/positive_trimmed_1p5_A

python build_negative_manifest.py --root data/negative \
  --out data/negative_manifest.txt --also-hard --report data/negative_manifest_report.txt
```

### 3. Run the full v6 pipeline

Requires an existing v5 (or other) `.tflite` for hard-negative mining:

```bash
python new/run_pipeline.py --model models/hey_magis_v5_2026-03-15_model.tflite
```

Stages run in order: noise expansion → phonetic negatives → positive augmentation → hard-negative mining → train → evaluate. Each stage **skips automatically** if outputs already exist.

### 4. Train only (if data is ready)

```bash
python new/train.py
python new/evaluate.py --model new/model/hey_magis_v6_<date>_model.tflite --out-dir new/model
```

### 5. Verify TFLite shapes before shipping to the app

```python
import tensorflow as tf
interp = tf.lite.Interpreter(model_path="new/model/hey_magis_v6_2026-03-18_model.tflite")
interp.allocate_tensors()
assert interp.get_input_details()[0]["shape"].tolist() == [1, 148, 40, 1]
assert interp.get_output_details()[0]["shape"].tolist() == [1, 1]
print("OK")
```

---

## Requirements

### Python

- **3.9–3.11** (tested on Windows)
- **TensorFlow** ≥ 2.10, &lt; 2.22 (`new/requirements.txt`)
  - **GPU on Windows:** TensorFlow **2.10** is the last release with native Windows GPU support. For TF 2.11+, use CPU on Windows or train in **WSL2** with GPU.
- See `new/requirements.txt` for full pins: `numpy`, `scipy`, `librosa`, `soundfile`, `scikit-learn`, `matplotlib`, `tqdm`, `pyttsx3` (TTS phonetic negatives)

### Optional / legacy scripts

| Package | Used by |
|---------|---------|
| `resampy` | `slice_tv.py` |
| `requests`, `pandas` | `download_typeform.py` |
| `tflite_runtime` | Optional lightweight inference (eval scripts can use TensorFlow instead) |

### External tools

- **ffmpeg** — Converting downloaded or non-WAV audio (`download_audio_fixed.py`, noise collection)
- **Windows SAPI** — `pyttsx3` for `generate_phonetic_negatives.py` (skip with `--skip-phonetic` on headless servers)

---

## Repository Layout

```
wake_word_project/
│
├── README.md                 ← This file
├── LICENSE                   ← CC BY 4.0 (Speech Commands dataset)
├── validation_list.txt       ← Speech Commands v0.02 validation split
├── testing_list.txt          ← Speech Commands v0.02 test split
│
├── docs/                     ← Deep-dive documentation
│   ├── CONCEPTS_FOR_BEGINNERS.md
│   ├── IMPLEMENTATION_GUIDE.md
│   └── PROJECT_ANALYSIS.md
│
├── new/                      ← **v6 pipeline (recommended)**
│   ├── config.py             ← Single source of truth for paths & hyperparameters
│   ├── run_pipeline.py       ← Orchestrates all v6 stages
│   ├── augment_positives.py
│   ├── generate_noise.py
│   ├── generate_phonetic_negatives.py
│   ├── mine_hard_negatives.py
│   ├── train.py
│   ├── evaluate.py
│   ├── requirements.txt
│   ├── CHANGES.md            ← v6 changelog
│   └── model/                ← v6 .tflite, reports, plots, score CSVs
│
├── data/                     ← All audio + manifests (not always in git)
│   ├── positive_new_friends/
│   ├── positive_trimmed_1p5_A/
│   ├── positive_augmented/       ← v6 offline augmentation output
│   ├── negative/                 ← Speech Commands words, TV, phonetic, noise, etc.
│   ├── hard_negatives_top/       ← Mined high-score negatives
│   ├── positive_manifest_v4.txt
│   ├── negative_manifest.txt
│   └── negative_manifest_hard_merged.txt
│
├── models/                   ← Legacy v3–v5 outputs, eval CSVs, reports
│
├── hey_magis_v4_1p5_hard3061.py   ← v5 training (primary legacy script)
├── eval_wakeword_tflite_buckets.py ← Legacy multi-bucket eval
├── build_*_manifest.py, trim_*, pad_*, slice_tv.py, …
│
├── tv_recording.wav / .m4a   ← Source for TV negative slicing (optional)
└── train_hey_magis*.py       ← Older training versions (reference only)
```

---

## Model Contract (Must Not Break)

The Flutter/Dart inference code is **hardcoded** to this contract. Changing any value without updating the app will break detection or shift score distributions.

| Parameter | Value |
|-----------|--------|
| Input shape | `[1, 148, 40, 1]` |
| Output shape | `[1, 1]` (sigmoid) |
| Sample rate | 16,000 Hz |
| Clip duration | 1.5 s |
| Mel bins | 40 |
| Mel band | 80 Hz – 7,600 Hz |
| Spectrogram | **Power** mel (\|STFT\|²), then `log` with floor `1e-10` |
| Normalization | Per-clip: `(x - mean) / (std + 1e-6)` |
| STFT frame length | 400 samples (25 ms) |
| STFT frame step | 160 samples (10 ms) |
| FFT length | 512 |
| Window | Hann |
| `pad_end` | **False** (do not pad the last STFT frame) |
| Target time frames | 148 (trim or pad mel frames to exactly 148) |

---

## Feature Extraction Pipeline

End-to-end signal processing (training, eval, and Dart must match):

```
Raw WAV (16 kHz mono, 1.5 s = 24,000 samples)
    → STFT (frame=400, step=160, fft=512, Hann, pad_end=False)
    → Power spectrum (|STFT|²)
    → Mel filterbank (40 bins, 80–7600 Hz)
    → log(max(mel, 1e-10))
    → Per-example mean/std normalization
    → Trim/pad to 148 frames × 40 mels
    → Add channel dim → [148, 40, 1]
```

**Warning:** `eval_wakeword_tflite.py` (legacy) uses mel limits **60–7800 Hz**, which **does not** match training. Always use `new/evaluate.py` or `eval_wakeword_tflite_buckets.py` (80–7600 Hz).

---

## Data Layout and Manifests

### Positive samples (~468 raw, ~5,000+ after v6 augmentation)

| Directory | Description |
|-----------|-------------|
| `data/positive_new_friends/` | Crowdsourced recordings (~6 speakers, Typeform) |
| `data/positive_trimmed_1p5_A/` | Trimmed/padded clips (~133 speakers) |
| `data/positive_augmented/` | v6 offline augments (pitch, speed, noise) |

### Negative samples (~105K+ in full manifest)

| Source | Approx. count | Role |
|--------|---------------|------|
| Google Speech Commands (35 word folders) | ~100,000 | General negatives |
| `data/negative/tv_samples/` | ~5,100 | TV audio slices (similar cadence FPs) |
| `Hey__Magis_wrong_ways_to_say/` | ~61 | Mispronunciations |
| `phonetic_confusers/` | v6 TTS phrases | "Hey magic", "Hey Marcus", etc. |
| `_background_noise_` / `_background_noise_expanded_` | 6 + 17 | Mixed during training |

### Hard negatives (~13K+ merged)

| Source | Description |
|--------|-------------|
| `negative_manifest_hard.txt` | Wrong ways + similar folders |
| `hard_negatives_top/` | Top mined confusers (model score ≥ threshold) |
| `negative_manifest_hard_merged.txt` | **Primary hard manifest for training** |

### Manifest format

- One **absolute or project-relative** path per line, UTF-8
- Lines starting with `#` are ignored
- Rebuild after adding audio:

```bash
python build_hard_manifest_from_folder.py   # merges hard_negatives_top into merged manifest
```

### Class imbalance (raw data)

```
Positives:     ~468
Negatives:     ~105,896
Ratio:         ~1 : 226

Per training epoch (NEG_MULTIPLIER=3, HARD_NEG_FRACTION=0.60):
  ~468 pos vs ~1,404 neg (mix of hard + general)
```

---

## Training Pipelines

### v6 (`new/train.py`) — recommended

**Improvements over v5:**

| Area | v6 change |
|------|-----------|
| Loss | Focal loss (γ=2, α=0.75) + label smoothing 0.05 |
| Metric | Early stopping on **val_auc_pr** (not ROC) |
| Positives | Raw + `positive_augmented/` (~5K clips) |
| Hard negs | Mined at score ≥ **0.3**; 60% of batch negatives |
| Online aug | SpecAugment, ±8 dB gain, 40% noise mix, ±50 ms time shift |
| Architecture | DS-CNN + optional SE attention (`--no-se` to disable) |
| Noise pool | Expanded synthetic + original background files |
| Epochs | 60 (default) |

**Model architecture (conceptual):**

```
Input [148, 40, 1]
  → Conv2D(16) + BN + ReLU
  → [DW-Conv stride 2 + Conv 24] × 2 blocks (stride 1 and 2 alternating)
  → SE attention (optional) on last block
  → GlobalAveragePooling → Dropout(0.30) → Dense(1, sigmoid)
```

**Training loop:** Custom per-epoch loop (not one long `fit` on a static dataset):

1. Speaker-aware train/val split on positives
2. Each epoch: resample `len(train_pos) × NEG_MULTIPLIER` negatives (60% hard, 40% general)
3. Build fresh `tf.data` pipeline with online augmentation
4. Callbacks: EarlyStopping (`val_auc_pr`), ReduceLROnPlateau, ModelCheckpoint
5. Export best weights to `.tflite` (dynamic range quantization) + `.h5`

**Outputs:** `new/model/hey_magis_v6_<YYYY-MM-DD>_model.tflite` and matching `*_train_report.txt`

### v5 (`hey_magis_v4_1p5_hard3061.py`) — legacy baseline

- Binary cross-entropy, `val_auc` (ROC)
- Positives: `positive_new_friends` + `positive_trimmed_1p5_A` only
- `HARD_NEG_FRACTION = 0.50`, `NOISE_MIX_PROB = 0.35`
- Output: `models/hey_magis_v5_2026-03-15_model.tflite`

Use v5 as the **mining model** when bootstrapping v6 hard negatives if no v6 model exists yet.

---

## Recommended Workflow (v6)

### Full automated pipeline

```bash
python new/run_pipeline.py --model models/hey_magis_v5_2026-03-15_model.tflite
```

| Flag | Effect |
|------|--------|
| `--skip-noise` | Skip `generate_noise.py` |
| `--skip-phonetic` | Skip TTS phonetic confusers |
| `--skip-augment` | Skip `augment_positives.py` |
| `--skip-mine` | Skip hard-negative mining |
| `--skip-train` | Skip training |
| `--skip-eval` | Skip evaluation |
| `--epochs N` | Override epoch count (default 60) |

### Individual stages

```bash
# 1. Expand background noise (once)
python new/generate_noise.py

# 2. Phonetic confusers via Windows TTS (once)
python new/generate_phonetic_negatives.py

# 3. Offline positive augmentation (~454 → 5000+)
python new/augment_positives.py
python new/augment_positives.py --dry-run   # preview counts only

# 4. Mine hard negatives (needs .tflite)
python new/mine_hard_negatives.py --model models/hey_magis_v5_2026-03-15_model.tflite
python new/mine_hard_negatives.py --model path/to/model.tflite --threshold 0.3

# 5. Train
python new/train.py
python new/train.py --epochs 2 --no-se    # smoke test

# 6. Evaluate
python new/evaluate.py --model new/model/hey_magis_v6_2026-03-18_model.tflite --out-dir new/model
python new/evaluate.py --model path/to/model.tflite --neg-sample 10000 --pos-sample 0
```

### Quick retrain (data already generated)

```bash
python new/run_pipeline.py \
  --skip-noise --skip-phonetic --skip-augment --skip-mine \
  --model models/hey_magis_v5_2026-03-15_model.tflite
```

### Adding new positive recordings

1. Record **16 kHz, mono, WAV**, ~1.5 s per utterance (or trim with `trim_recordings_v3.py`, pad with `pad_to_1p5.py`).
2. Place under `data/positive_new_friends/<speaker_id>/` or `positive_trimmed_1p5_A/`.
3. Rebuild manifest and re-run augmentation + training.

```bash
python trim_recordings_v3.py          # interactive / batch trim to ~1.2s
python pad_to_1p5.py --in_dir data/positive_trimmed_1p2_A --out_dir data/positive_trimmed_1p5_A
python build_positive_manifest.py --out data/positive_manifest_v4.txt \
  --dirs data/positive_new_friends data/positive_trimmed_1p5_A
python new/augment_positives.py
python new/train.py
```

### Adding TV / ambient false-positive audio

`slice_tv.py` cuts a long recording into overlapping 1.5 s clips:

```bash
# Edit INPUT_FILE / OUTPUT_DIR / MANIFEST at top of slice_tv.py
python slice_tv.py
# Produces data/negative/tv_samples/tv_*.wav and appends paths to negative_manifest_hard.txt
```

Default: 50% hop (0.75 s), skips near-silent clips, resamples to 16 kHz.

### Data quality utilities

| Script | Purpose |
|--------|---------|
| `check_clipping.py` | Flag near-fullscale WAVs |
| `quarantine_quiet_positives.py` | Move very quiet positives aside |
| `inspect_wavs.py` / `summarize_wavs.py` | Metadata and RMS stats |
| `rank_wavs_by_rms.py` | Sort by loudness |
| `count_positives.py` / `count_negatives.py` | Quick counts |

---

## Legacy Workflow (v5)

```bash
# Train v5
python hey_magis_v4_1p5_hard3061.py

# Evaluate across buckets (friends, trimmed, 10K neg sample, all hard)
python eval_wakeword_tflite_buckets.py \
  --model models/hey_magis_v5_2026-03-15_model.tflite \
  --pos_newfriends data/positive_new_friends \
  --pos_trimmed data/positive_trimmed_1p5_A \
  --neg_manifest data/negative_manifest.txt \
  --hard_manifest data/negative_manifest_hard_merged.txt

# Mine top false positives (legacy: high threshold)
python mine_hard_negatives.py
python build_hard_manifest_from_folder.py
```

---

## Script Reference

### v6 pipeline (`new/`)

| Script | Description |
|--------|-------------|
| `config.py` | Paths, audio constants, training/eval hyperparameters |
| `run_pipeline.py` | Run all stages with skip flags |
| `augment_positives.py` | Pitch ±2 semitones, speed 0.9/1.1, noise SNR 5/10/15 dB |
| `generate_noise.py` | Synthetic room/HVAC/cafe/outdoor/etc. noise WAVs |
| `generate_phonetic_negatives.py` | TTS "Hey magic", "Hey Marcus", … confusers |
| `mine_hard_negatives.py` | Score negatives; export paths with score ≥ 0.3 |
| `train.py` | Focal loss, SpecAugment, SE block, TFLite export |
| `evaluate.py` | Threshold sweep, FAR/hour, DET/PR/ROC plots, SNR segments |

### Data & manifests (project root)

| Script | Description |
|--------|-------------|
| `build_positive_manifest.py` | List positive WAV paths → manifest |
| `build_negative_manifest.py` | Scan `data/negative`, dedupe, hard manifest |
| `build_negative_manifest_v2.py` | Alternate manifest builder (fast/full dedupe) |
| `build_hard_manifest_from_folder.py` | Merge `hard_negatives_top` into hard merged manifest |

### Audio processing

| Script | Description |
|--------|-------------|
| `trim_recordings_v3.py` | RMS onset/offset, 1.2 s clips, quality buckets |
| `pad_to_1p5.py` | Pad shorter clips to exactly 1.5 s |
| `slice_tv.py` | Slice long TV recording → `tv_samples/` |

### Training (historical)

| Script | Status |
|--------|--------|
| `hey_magis_v4_1p5_hard3061.py` | **v5 — active legacy** |
| `train_hey_magis_v4_1.py` | Experimental (no val/early stop) |
| `train_hey_magis_v3.py` | 1.0 s clips — superseded |
| `train_hey_magis_v2.py` | Early log-mel DS-CNN |
| `train_hey_magis.py` | v1 MFCC + Conv1D — obsolete |
| `train_wakeword_v3.py` | Duplicate of v3 — safe to ignore |

### Evaluation & debug

| Script | Description |
|--------|-------------|
| `eval_wakeword_tflite_buckets.py` | Legacy 4-bucket CSV scores |
| `eval_wakeword_tflite.py` | Legacy single eval (**wrong mel range**) |
| `debug_tflite_outputs.py` | Inspect TFLite I/O dtypes and sample scores |

### Data download (one-time)

| Script | Description |
|--------|-------------|
| `download_typeform.py` | Pull crowdsourced recordings from Typeform API |
| `download_audio_fixed.py` | Download from Mux URLs in JSON → WAV via ffmpeg |

---

## Evaluation and Metrics

### v6 report (`new/evaluate.py`)

Produces under `--out-dir` (default `new/model/`):

| Output | Content |
|--------|---------|
| `eval_v6_<date>_report.txt` | Threshold sweep, FAR/hour, EER, best F1, SNR recall, posterior smoothing |
| `eval_v6_<date>_pr_roc.png` | PR and ROC curves |
| `eval_v6_<date>_det.png` | Detection Error Tradeoff (FRR vs FAR) |
| `eval_v6_*_scores_*.csv` | Per-file scores by bucket |

**Key metrics:**

| Metric | Meaning |
|--------|---------|
| **Recall** | Fraction of positives detected at threshold |
| **FRR** | False reject rate = 1 − recall |
| **FAR** | Fraction of negatives incorrectly accepted |
| **FAR/hour** | Estimated false accepts per hour of negative audio |
| **EER** | Threshold where FAR ≈ FRR |
| **AUC-PR** | Primary training metric (better than ROC under imbalance) |

### Posterior smoothing (deployment)

Offline eval scores **isolated 1.5 s clips**. In the app, detection requires **N consecutive** frames above threshold within a time window—this **lowers real-world FAR** compared to per-clip FAR. The v6 report includes a simulation table for `hits=1..5`.

---

## Latest Results (v6, 2026-03-18)

From `new/model/eval_v6_2026-03-18_report.txt` (454 positives, 10K sampled negatives, 8162 hard negatives scored):

| Threshold | Recall | FAR/hour (neg sample) | Notes |
|-----------|--------|------------------------|--------|
| 0.30 | 99.6% | ~10.3 | Near EER (~0.29) |
| 0.70 | 96.0% | ~0.93 | Strong general negatives |
| 0.80 | 90.1% | ~0.26 | |
| 0.85 | 81.9% | ~0.26 | |
| 0.90 | 64.1% | ~0.13 | High precision, lower recall |
| 0.92 | 49.3% | ~0.13 | Perfect specificity on sample |

**SNR-segmented recall @ 0.80:** clean 90%, 10 dB 73%, 5 dB 57%, 0 dB 35%.

**Suggested deployment band (from eval):** ~**0.29–0.39** for balanced FAR/recall on the held-out eval mix; tighten toward **0.85–0.92** if false triggers must be minimal (with higher miss rate).

Validation during training (augmented positives, different split): `val_auc_pr ≈ 0.999` — treat held-out **real-world** TV/phonetic tests as the source of truth, not validation alone.

---

## Mobile Deployment

1. Copy `hey_magis_v6_<date>_model.tflite` to Firebase Storage (or bundle in app).
2. Update the Flutter model URL / asset path.
3. **Do not change** Dart feature extraction without retraining and re-exporting.
4. Tune **threshold** and `requiredHits` / `hitWindowMs` on device:

| Use case | Threshold | Trade-off |
|----------|-----------|-----------|
| Demo / max recall | 0.80–0.85 | More false triggers |
| Daily use | **0.88–0.92** | Balanced |
| Minimum false triggers | 0.93–0.97 | More missed quiet utterances |

5. Consider `requiredHits=3` with a 500–600 ms window to cut FAR ~50–70% vs `requiredHits=2`.

---

## Threshold and Posterior Smoothing

```
Per-frame score from model (every ~10 ms hop in Dart)
        ↓
Score > threshold ?
        ↓
Count consecutive hits in hitWindowMs
        ↓
hits >= requiredHits → TRIGGER "Hey Magis"
```

Empirically, requiring 3 hits reduces deployment FAR below single-frame eval FAR, at a small recall cost (see eval report posterior table).

---

## Known Issues and Limitations

1. **TV / similar cadence false positives** — TV negatives help but real-world TV speech remains a top FP source; add more targeted hard negatives and re-mine after each model generation.
2. **Threshold vs recall** — At 0.90+ threshold, recall drops sharply on eval; tune for product requirements (FAR vs miss rate).
3. **Small raw positive set** — v6 augmentation mitigates but cannot replace diverse real speakers/environments.
4. **Eval vs deployment** — Clip-level FAR overestimates or underestimates streaming FAR; use posterior smoothing and on-device tests.
5. **Flat repo layout** — Many legacy scripts live in project root; prefer `new/` for new work.
6. **`data/` may be absent in git** — Large audio and manifests are often local-only; clone or restore datasets separately.
7. **Phonetic TTS negatives** — Synthetic; complement with human recordings for best boundary learning.

---

## Documentation

| Document | Audience | Contents |
|----------|----------|----------|
| [docs/CONCEPTS_FOR_BEGINNERS.md](docs/CONCEPTS_FOR_BEGINNERS.md) | New to wake words / ML | Glossary, audio basics, metrics explained |
| [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md) | Implementers | 8 technical requirements, code snippets, verification |
| [docs/PROJECT_ANALYSIS.md](docs/PROJECT_ANALYSIS.md) | Maintainers | File-by-file audit, architecture, data inventory |
| [new/CHANGES.md](new/CHANGES.md) | v6 users | v5→v6 delta, expected improvements, troubleshooting |

---

## Speech Commands Dataset

A large portion of negative training data comes from the Google **[Speech Commands Dataset v0.02](https://arxiv.org/abs/1804.03209)** (~105K one-second utterances, 35 command words + background noise). Original download:

[http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)

Files in this repo from that dataset:

- `validation_list.txt` / `testing_list.txt` — official splits (hash-based, 10% each)
- `data/negative/<word>/` — per-word folders
- `data/negative/_background_noise_/` — long noise clips

**Partitioning logic** (speaker-stable splits): identical filenames sharing the same prefix before `_nohash_` stay in the same split. See the original dataset paper for `which_set()` details.

If you use Speech Commands in published work, cite:

```bibtex
@article{speechcommandsv2,
  author = {{Warden}, P.},
  title = "{Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition}",
  journal = {ArXiv e-prints},
  eprint = {1804.03209},
  year = 2018,
  month = apr,
  url = {https://arxiv.org/abs/1804.03209},
}
```

---

## License

- **Speech Commands** material: [Creative Commons BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see [LICENSE](LICENSE).
- **MagisAI project code and custom "Hey Magis" data**: Use and distribution terms are defined by the project owner/client agreement; do not assume the same license as Speech Commands for proprietary recordings or models.

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| `pyttsx3` fails (no Windows TTS) | `python new/run_pipeline.py --skip-phonetic` |
| CUDA not detected on Windows | Install TF 2.10 + matching CUDA/cuDNN, or train on CPU / WSL2 |
| OOM during training | Lower `BATCH_SIZE` in `new/config.py` (try 16 or 24) |
| Augmentation very slow | First librosa `time_stretch` call JIT-compiles; normal for ~454 files |
| All positive scores low after train | Confirm `positive_augmented/` exists and train log shows thousands of positives |
| TFLite shape mismatch in app | Re-run shape verification snippet above; rebuild Dart mel code |
| Mining finds zero hard negs | Lower `--threshold` (default 0.3); ensure manifest paths exist |

---

*MagisAI Wake Word Project — custom "Hey Magis" detector for on-device Catholic AI assistant.*

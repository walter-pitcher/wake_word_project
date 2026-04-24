# MagisAI Wake Word — Implementation Guide

> **Document Version:** 1.0  
> **Date:** March 17, 2026  
> **Purpose:** Step-by-step guide to implement all 8 requirements from the MagisAI Wake Word Technical Brief. Each section includes what to do, why, where to change code, what the code should look like, and how to verify.

---

## Table of Contents

1. [Before You Start](#1-before-you-start)
2. [Requirement 1: Positive Augmentation Pipeline](#2-requirement-1-positive-augmentation-pipeline)
3. [Requirement 2: Hard Negative Mining](#3-requirement-2-hard-negative-mining)
4. [Requirement 3: Phonetic Negative Set](#4-requirement-3-phonetic-negative-set)
5. [Requirement 4: Loss Function (Focal Loss + Class Weights + AUC-PR)](#5-requirement-4-loss-function)
6. [Requirement 5: Posterior Smoothing (Training-Time Evaluation)](#6-requirement-5-posterior-smoothing)
7. [Requirement 6: Background Noise Expansion](#7-requirement-6-background-noise-expansion)
8. [Requirement 7: Training Configuration Updates](#8-requirement-7-training-configuration-updates)
9. [Requirement 8: Evaluation Requirements (Full Threshold Sweep)](#9-requirement-8-evaluation-requirements)
10. [Deliverables Checklist](#10-deliverables-checklist)
11. [Critical Constraints Reminder](#11-critical-constraints-reminder)
12. [Recommended Execution Order](#12-recommended-execution-order)
13. [Testing & Verification Protocol](#13-testing--verification-protocol)

---

## 1. Before You Start

### Environment Setup

**Required:**
```
Python 3.9-3.11 (Windows)
TensorFlow >= 2.10 (GPU recommended)
NumPy >= 1.21
librosa >= 0.10.0
soundfile >= 0.12.0
scipy >= 1.9.0
audiomentations >= 0.34.0 (NEW — for augmentation)
matplotlib >= 3.5 (NEW — for DET curves)
```

### File You Will Modify

The **only** training script you should modify is:

```
hey_magis_v4_1p5_hard3061.py
```

Create a copy first:
```bash
cp hey_magis_v4_1p5_hard3061.py hey_magis_v6_improved.py
```

All changes described below should be made to `hey_magis_v6_improved.py`.

### Constraints You Must Not Break

| Constraint | Value | Why |
|---|---|---|
| Input shape | `[1, 148, 40, 1]` | Dart inference code is hardcoded to this |
| Output shape | `[1, 1]` (sigmoid) | Flutter app expects single float |
| Sample rate | 16,000 Hz | Mic capture rate |
| Mel bins | 40 | Matches Dart mel filter bank |
| Mel range | 80 - 7600 Hz | Matches Dart extraction |
| Normalization | Per-clip mean/std | Matches Dart normalization |

---

## 2. Requirement 1: Positive Augmentation Pipeline

### Goal
Expand 468 positives to 5,000+ effective samples using diverse augmentation.

### Why This Matters
With only 468 real samples, the model sees very limited variation. In the real world, people say "Hey Magis" with different pitches, speeds, accents, background noise levels, and room acoustics. Augmentation simulates this diversity.

### What to Implement

You need **5 augmentation techniques** applied randomly during training:

#### 1A. Pitch Shifting (±2 semitones)

**Where:** Add as a new function, call it inside `make_example()`.

**How it works:** Shifting pitch changes the perceived tone of the voice without changing duration. ±2 semitones covers the natural variation between male/female speakers.

**Implementation approach:**

Since TensorFlow does not have a native pitch-shift op that works inside `tf.data.map`, you have two options:

**Option A (Recommended): Offline augmentation**
Create augmented WAV files before training. This is simpler and more reliable on Windows.

```python
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

def create_augmented_positives(input_dirs, output_dir, target_count=5000):
    """
    Read all positive WAVs, generate augmented versions until we hit target_count.
    
    Augmentations:
      - Pitch shift: -2, -1, +1, +2 semitones
      - Speed perturbation: 0.9x, 0.95x, 1.05x, 1.1x
      - Noise injection: at SNR 5, 10, 15, 20 dB using background noise files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all source WAVs
    source_wavs = []
    for d in input_dirs:
        source_wavs.extend(sorted(Path(d).rglob("*.wav")))
    
    print(f"Source positives: {len(source_wavs)}")
    
    generated = 0
    
    for wav_path in source_wavs:
        audio, sr = librosa.load(str(wav_path), sr=16000)
        base_name = wav_path.stem
        
        # Original (copy)
        _save_clip(audio, sr, output_dir / f"{base_name}_orig.wav")
        generated += 1
        
        # Pitch shifts
        for semitones in [-2, -1, 1, 2]:
            shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
            _save_clip(shifted, sr, output_dir / f"{base_name}_pitch{semitones:+d}.wav")
            generated += 1
        
        # Speed perturbation
        for rate in [0.9, 0.95, 1.05, 1.1]:
            stretched = librosa.effects.time_stretch(audio, rate=rate)
            _save_clip(stretched, sr, output_dir / f"{base_name}_speed{rate:.2f}.wav")
            generated += 1
        
        if generated >= target_count:
            break
    
    print(f"Generated {generated} augmented positives in {output_dir}")

def _save_clip(audio, sr, path, target_seconds=1.5):
    """Pad/trim to target length and save."""
    target_len = int(sr * target_seconds)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]
    sf.write(str(path), audio, sr)
```

**Option B: Online augmentation in tf.data pipeline**

If you want augmentation applied dynamically during training (different augmentation each epoch), you can use `tf.py_function` to call librosa within the tf.data pipeline. This is slower but provides more diversity.

```python
def augment_wav(wav: tf.Tensor) -> tf.Tensor:
    """Apply random augmentation to a waveform tensor."""
    # Random gain (already implemented)
    wav = random_gain(wav)
    
    # Random noise mixing (already implemented)
    wav = maybe_mix_noise(wav, noise_paths)
    
    # SpecAugment is applied AFTER mel extraction (see below)
    return wav
```

#### 1B. SpecAugment (Time and Frequency Masking)

**Where:** Apply AFTER `wav_to_mel()` but BEFORE returning from `make_example()`.

**What it does:** Randomly masks contiguous blocks of time frames or frequency bins, forcing the model to be robust to partial information. This is one of the most effective augmentation techniques for audio.

```python
def spec_augment(log_mel: tf.Tensor, 
                 num_freq_masks=2, freq_mask_width=5,
                 num_time_masks=2, time_mask_width=15) -> tf.Tensor:
    """
    Apply SpecAugment to a [frames, mels, 1] tensor.
    
    Args:
        log_mel: Input spectrogram [148, 40, 1]
        num_freq_masks: Number of frequency masks to apply
        freq_mask_width: Maximum width of each frequency mask (in mel bins)
        num_time_masks: Number of time masks to apply
        time_mask_width: Maximum width of each time mask (in frames)
    
    Returns:
        Augmented spectrogram [148, 40, 1]
    """
    shape = tf.shape(log_mel)
    T = shape[0]  # 148 frames
    F = shape[1]  # 40 mel bins
    
    augmented = log_mel
    
    # Frequency masking
    for _ in range(num_freq_masks):
        f = tf.random.uniform([], 0, freq_mask_width, dtype=tf.int32)
        f0 = tf.random.uniform([], 0, F - f, dtype=tf.int32)
        mask = tf.concat([
            tf.ones([T, f0, 1]),
            tf.zeros([T, f, 1]),
            tf.ones([T, F - f0 - f, 1])
        ], axis=1)
        augmented = augmented * mask
    
    # Time masking
    for _ in range(num_time_masks):
        t = tf.random.uniform([], 0, time_mask_width, dtype=tf.int32)
        t0 = tf.random.uniform([], 0, T - t, dtype=tf.int32)
        mask = tf.concat([
            tf.ones([t0, F, 1]),
            tf.zeros([t, F, 1]),
            tf.ones([T - t0 - t, F, 1])
        ], axis=0)
        augmented = augmented * mask
    
    return augmented
```

**Where to call it — modify `make_example()`:**

```python
def make_example(path, label, noise_paths):
    wav = load_wav_mono_16k(path)
    if USE_AUGMENT:
        wav = random_gain(wav)
        wav = maybe_mix_noise(wav, noise_paths)
    feat = wav_to_mel(wav)
    if USE_AUGMENT:
        feat = spec_augment(feat)  # ADD THIS LINE
    return feat, label
```

#### 1C. Noise Injection at SNR 0-20dB

**Already implemented** in the current script as `maybe_mix_noise()`. Current config:
- `NOISE_MIX_PROB = 0.35`
- `NOISE_MIX_SNR_DB = (0.0, 18.0)`

**Change needed:** Update to cover full 0-20dB range:
```python
NOISE_MIX_PROB = 0.50       # Increase from 0.35
NOISE_MIX_SNR_DB = (0.0, 20.0)  # Expand from (0.0, 18.0)
```

#### 1D. Room Impulse Response Convolution (Optional/Advanced)

**What it does:** Simulates how audio sounds in different rooms (reverb/echo). This requires RIR (Room Impulse Response) files.

**Where to get RIR files:** MIT Acoustics Lab or the OpenSLR RIR dataset (http://www.openslr.org/28/).

```python
def convolve_rir(wav: tf.Tensor, rir_path: str) -> tf.Tensor:
    """Convolve audio with a room impulse response."""
    rir_bytes = tf.io.read_file(rir_path)
    rir, _ = tf.audio.decode_wav(rir_bytes, desired_channels=1)
    rir = tf.squeeze(rir)
    
    # Normalize RIR
    rir = rir / (tf.reduce_max(tf.abs(rir)) + 1e-8)
    
    # Convolve (1D convolution)
    wav_4d = tf.reshape(wav, [1, -1, 1, 1])
    rir_4d = tf.reshape(rir, [-1, 1, 1, 1])
    conv = tf.nn.conv2d(wav_4d, rir_4d, strides=[1,1,1,1], padding='SAME')
    result = tf.reshape(conv, [-1])
    
    # Trim to original length and normalize
    result = result[:tf.shape(wav)[0]]
    result = result / (tf.reduce_max(tf.abs(result)) + 1e-8)
    return result
```

**This is optional** — implement it if you can source RIR files, but the other augmentations are more critical.

### Verification

After augmentation, verify:
1. Augmented positive count should be 5,000+ (check with `count_positives.py` if using offline augmentation)
2. All augmented files should be exactly 1.5s at 16kHz (24,000 samples)
3. Augmented files should sound like "Hey Magis" with variation (listen to 10-20 random samples)

---

## 3. Requirement 2: Hard Negative Mining

### Goal
Run the current model over the entire negative set, extract samples scoring above 0.3, and oversample them during training.

### Why This Matters
Hard negatives are the samples the model is most confused about — they are the decision boundary. Training on these instead of easy negatives (like "dog" or "bed" which score near 0.0) is far more efficient.

### Step-by-Step

#### Step 1: Run evaluation on ALL negatives

Use the existing evaluation script, but with a lower threshold focus:

```bash
python eval_wakeword_tflite_buckets.py \
  --model models/hey_magis_v5_2026-03-15_model.tflite \
  --pos_newfriends data/positive_new_friends \
  --pos_trimmed data/positive_trimmed_1p5_A \
  --neg_manifest data/negative_manifest.txt \
  --hard_manifest data/negative_manifest_hard_merged.txt \
  --neg_sample 0 \
  --out_dir models
```

Setting `--neg_sample 0` or using all negatives (you may need to modify the script to allow scoring all 105K negatives — this will take time).

#### Step 2: Extract samples scoring above 0.3

Modify `mine_hard_negatives.py`:

```python
MIN_SCORE = 0.3    # Changed from 0.95 to 0.3
TOP_N = 10000      # Increased — we want ALL confusers above 0.3
```

Run it:
```bash
python mine_hard_negatives.py
```

#### Step 3: Rebuild the hard manifest

```bash
python build_hard_manifest_from_folder.py
```

This merges the new hard_negatives_top with existing hard negatives.

#### Step 4: Update training config

In `hey_magis_v6_improved.py`, increase hard negative oversampling:

```python
HARD_NEG_FRACTION = 0.60    # Was 0.50, now 60% of training negatives are hard
```

### Verification

1. Check `data/hard_negatives_top/` has more files than before
2. Check `data/negative_manifest_hard_merged.txt` line count increased
3. Listen to 20 random hard negatives — they should sound vaguely like "Hey Magis" or have similar cadence

---

## 4. Requirement 3: Phonetic Negative Set

### Goal
Record and add phonetically similar phrases as hard negatives.

### Why This Matters
The biggest source of false positives is phrases that sound like "Hey Magis": similar vowel patterns, similar stress patterns, similar phoneme sequences. The model needs to learn the fine-grained difference.

### Phrases to Record

Each phrase should be recorded by at least 5-10 different speakers, 3-5 utterances each:

| Phrase | Why It's Confusing | Priority |
|---|---|---|
| "Hey magic" | Near-identical phonemes | **Critical** |
| "Hey massive" | Similar "ma-" prefix + sibilant | **Critical** |
| "Hey Agnes" | Similar vowel pattern | **High** |
| "Hey Marcus" | Similar "Ma-" + "s" ending | **High** |
| "Okay Magis" | Wrong trigger word, same name | **High** |
| "Hey Maxis" | Near-identical | **High** |
| "Hey Madness" | Similar rhythm | Medium |
| "Hey Mattress" | Similar "Ma-" + "s" | Medium |
| "Hey Magnet" | Similar "Mag-" prefix | Medium |
| "Hey Mavis" | Near-identical to "Magis" | Medium |
| "A Magis" | Incomplete trigger | Medium |
| "Magis" | Missing "Hey" | Medium |
| "Hey" (alone) | Just the trigger word | Medium |
| "Hey Siri" | Common wake word | Low |
| "Hey Google" | Common wake word | Low |

### Recording Specifications

| Parameter | Value |
|---|---|
| Sample rate | 16,000 Hz |
| Format | WAV, 16-bit PCM, mono |
| Duration | 1.5 seconds (pad/trim as needed) |
| Environment | Quiet room + some with background noise |
| Distance | Arm's length (simulating phone use) |

### Where to Put Them

```
data/negative/phonetic_negatives/
├── hey_magic/
│   ├── speaker01_hey_magic_001.wav
│   ├── speaker01_hey_magic_002.wav
│   └── ...
├── hey_massive/
│   └── ...
├── hey_agnes/
│   └── ...
└── ...
```

### After Recording

1. Run the negative manifest builder:
   ```bash
   python build_negative_manifest.py --root data/negative --out data/negative_manifest.txt --also-hard --report data/negative_manifest_report.txt
   ```

2. The phonetic negatives will automatically be included in training as hard negatives (because they'll be in folders detected by the hard-negative naming heuristic, or you can add them manually to the hard manifest).

### Verification

1. Count files in `data/negative/phonetic_negatives/`: aim for 200-500 clips
2. Run eval on them — many should score > 0.3 (if they don't, they're not useful as hard negatives)

---

## 5. Requirement 4: Loss Function

### Goal
Replace binary cross-entropy with focal loss, apply class weights, and switch primary metric to AUC-PR.

### Why This Matters

**Binary Cross-Entropy Problem:** With 105K negatives and 468 positives, most training batches are dominated by easy negatives (Speech Commands words like "dog" that score 0.001). BCE treats every sample equally, so the model optimizes for these easy cases and doesn't focus on the hard decision boundary.

**Focal Loss Solution:** Focal loss down-weights easy examples by a factor of (1 - p)^gamma. A negative sample scoring 0.001 gets a weight of (1-0.001)^2 = 0.998, but a hard negative scoring 0.4 gets (1-0.6)^2 = 0.16. This makes the model focus on the samples it's uncertain about.

**AUC-PR vs AUC-ROC:** AUC-ROC can be misleadingly high with extreme class imbalance because specificity stays high even with many false positives. AUC-PR (precision-recall) directly measures what we care about: when the model says "yes," is it right? (precision) and does it catch all real wake words? (recall).

### Implementation

#### 4A. Focal Loss

Add this function to your training script:

```python
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for binary classification.
    
    Args:
        gamma: Focusing parameter. Higher values = more focus on hard examples.
               gamma=0 is equivalent to standard BCE.
               gamma=2 is the recommended default from the original paper.
        alpha: Weighting factor for the positive class (0-1).
               alpha=0.25 means positives get 0.25 weight, negatives get 0.75.
               For wake word detection with many negatives, use alpha=0.75+ to
               up-weight the rare positive class.
    
    Returns:
        Loss function compatible with model.compile()
    """
    def loss_fn(y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Standard cross-entropy terms
        bce_pos = -y_true * tf.math.log(y_pred)
        bce_neg = -(1 - y_true) * tf.math.log(1 - y_pred)
        
        # Focal weighting
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        
        # Alpha weighting (balance positive/negative importance)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        loss = alpha_t * focal_weight * (bce_pos + bce_neg)
        return tf.reduce_mean(loss)
    
    return loss_fn
```

#### 4B. Class Weights

Calculate weights inversely proportional to class frequency:

```python
def compute_class_weights(n_positive, n_negative):
    """
    Compute class weights inversely proportional to frequency.
    
    Example: 468 positives, 1404 negatives (with NEG_MULTIPLIER=3)
    weight_pos = 1872 / (2 * 468) = 2.0
    weight_neg = 1872 / (2 * 1404) = 0.667
    """
    total = n_positive + n_negative
    weight_pos = total / (2.0 * n_positive)
    weight_neg = total / (2.0 * n_negative)
    return {0: weight_neg, 1: weight_pos}
```

Use `alpha` in focal loss to encode the class weight instead of passing `class_weight` to `model.fit()`:

```python
# Alpha should be higher for the minority (positive) class
# With 468 pos vs 1404 neg: alpha = 1404 / (468 + 1404) = 0.75
alpha = n_negative / (n_positive + n_negative)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=focal_loss(gamma=2.0, alpha=alpha),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="acc"),
        tf.keras.metrics.AUC(name="auc_roc"),
        tf.keras.metrics.AUC(name="auc_pr", curve="PR"),  # NEW
    ]
)
```

#### 4C. AUC-PR Metric

Add AUC-PR as a training metric (shown above). Also update the early stopping callback:

```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc_pr",      # Changed from val_auc to val_auc_pr
        mode="max",
        patience=PATIENCE,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc_pr",      # Changed from val_auc to val_auc_pr
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-5
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ckpt_path),
        monitor="val_auc_pr",      # Changed from val_auc to val_auc_pr
        mode="max",
        save_best_only=True
    ),
]
```

### Verification

1. Training logs should show `auc_pr` metric alongside `auc_roc`
2. `auc_pr` will typically be lower than `auc_roc` — this is expected and more honest
3. The model should show improved precision at high thresholds (0.90+)

---

## 6. Requirement 5: Posterior Smoothing

### Goal
Evaluate the model with multi-frame confirmation logic to match deployment behavior.

### Background

The Dart code already implements hit-window logic:
- `requiredHits = 2` (need 2 consecutive detections)
- `hitWindowMs = 400` (within a 400ms window)

The technical brief asks for 3-5 consecutive frames. During training evaluation, we should simulate this to get more realistic metrics.

### Implementation

Add a smoothing function to the evaluation code:

```python
def apply_posterior_smoothing(scores, required_hits=3, window_size=5):
    """
    Simulate hit-window confirmation on a sequence of frame scores.
    
    In real deployment, the app processes audio in a sliding window and
    requires `required_hits` consecutive scores above threshold within
    `window_size` frames before triggering.
    
    For offline evaluation on isolated clips, we simulate this by:
    1. Breaking each clip's score into what the sliding window would see
    2. Requiring the score to be above threshold for `required_hits`
       consecutive evaluations
    
    For single-clip evaluation (our case), this simply means:
    - Score must exceed threshold (standard behavior for isolated clips)
    - This function is more relevant for continuous audio evaluation
    
    Args:
        scores: Array of per-clip scores
        required_hits: Number of consecutive hits needed
        window_size: Number of frames in the confirmation window
    
    Returns:
        Smoothed detection decisions
    """
    # For isolated clip evaluation, posterior smoothing means we should
    # evaluate at a HIGHER effective threshold to account for the fact
    # that in deployment, multiple consecutive hits are required.
    #
    # Empirically, requiring 3 hits at threshold T is approximately
    # equivalent to requiring 1 hit at threshold T^(1/required_hits)
    # for independent frames. But frames are NOT independent, so
    # the actual mapping depends on the score distribution.
    #
    # Practical recommendation: evaluate at the per-frame threshold
    # and note that deployment FPR will be LOWER than eval FPR.
    return scores
```

**More practically:** Add a note to the evaluation report explaining the relationship between per-clip threshold and deployment behavior:

```python
def estimate_deployment_far(per_clip_far, clips_per_hour=2400, required_hits=3):
    """
    Estimate deployment FAR from per-clip FAR.
    
    If we evaluate 1.5s clips with 10ms hop (sliding window in Dart):
    - Clips per second: ~100 (with 10ms hop)
    - Clips per hour: 360,000
    
    With required_hits=3, the probability of 3 consecutive false positives
    is approximately: per_clip_far^3 (if independent)
    
    But consecutive frames are highly correlated, so the actual FAR
    is higher. A conservative estimate is: per_clip_far^(required_hits/2)
    """
    # Conservative estimate
    deployment_far_per_clip = per_clip_far ** (required_hits / 2.0)
    false_accepts_per_hour = deployment_far_per_clip * clips_per_hour
    return false_accepts_per_hour
```

### What to Include in the Report

Add a section to the evaluation report:
```
POSTERIOR SMOOTHING ANALYSIS
============================
Deployment config: requiredHits=3, hitWindowMs=400
Per-clip FAR at t=0.85: 0.05%
Estimated deployment FAR/hour: ~0.2 false accepts/hour
Note: actual deployment FAR will be lower than per-clip FAR
      due to consecutive-hit requirement.
```

---

## 7. Requirement 6: Background Noise Expansion

### Goal
Expand from 6 background noise files to 15-20 diverse files.

### Why This Matters
Noise injection during training makes the model robust to real-world conditions. With only 6 noise files (all from the Speech Commands dataset), the model only learns to handle those specific noise profiles. Real phones encounter many different noise environments.

### Noise Types Needed

| # | Type | Description | Duration | Source |
|---|---|---|---|---|
| 1-6 | (Existing) | Speech Commands background noise | ~30s each | Already have |
| 7 | Room tone | Empty quiet room ambience | 30-60s | Record yourself |
| 8 | HVAC fan | Air conditioning / heating noise | 30-60s | Record yourself |
| 9 | Cafe ambience | Coffee shop background chatter | 30-60s | Freesound.org |
| 10 | Outdoor urban | City street, traffic | 30-60s | Freesound.org |
| 11 | Outdoor nature | Wind, birds, rustling | 30-60s | Freesound.org |
| 12 | TV murmur | TV playing in background (no clear speech) | 30-60s | Record TV |
| 13 | Kitchen | Dishes, water, cooking sounds | 30-60s | Freesound.org |
| 14 | Office | Keyboard typing, mouse clicks, phone rings | 30-60s | Freesound.org |
| 15 | Car interior | Car engine, road noise | 30-60s | Freesound.org |
| 16 | Music (low) | Quiet background music | 30-60s | Any royalty-free |
| 17 | Church | Church ambience, organ, murmur | 30-60s | Freesound.org |
| 18 | Rain | Rain on windows/roof | 30-60s | Freesound.org |
| 19 | Crowd | General crowd murmur | 30-60s | Freesound.org |
| 20 | Baby/kids | Children playing in background | 30-60s | Freesound.org |

### Where to Get Them

1. **Freesound.org** — Free sound effects library (requires account)
   - Search: "room tone", "cafe ambience", "traffic noise", etc.
   - Download in WAV format
   
2. **Record yourself** — Use your phone in different environments
   - Record 1-2 minutes, then trim to 30-60s

3. **OpenSLR** — Academic noise datasets
   - MUSAN dataset: http://www.openslr.org/17/
   - Contains speech, music, and noise categories

### File Specifications

| Parameter | Value |
|---|---|
| Format | WAV, 16-bit PCM |
| Sample rate | 16,000 Hz |
| Channels | Mono |
| Duration | 30-60 seconds each |
| Naming | `backgroundnoise_<type>.wav` |

### Where to Put Them

```
data/negative/_background_noise_/
├── (existing 6 files)
├── backgroundnoise_room_tone.wav
├── backgroundnoise_hvac_fan.wav
├── backgroundnoise_cafe.wav
├── backgroundnoise_outdoor_urban.wav
├── backgroundnoise_outdoor_nature.wav
├── backgroundnoise_tv_murmur.wav
├── backgroundnoise_kitchen.wav
├── backgroundnoise_office.wav
├── backgroundnoise_car_interior.wav
├── backgroundnoise_music_low.wav
├── backgroundnoise_church.wav
├── backgroundnoise_rain.wav
├── backgroundnoise_crowd.wav
└── backgroundnoise_baby_kids.wav
```

### Conversion Command (if needed)

If files are not 16kHz mono WAV:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 output.wav
```

### Verification

1. Count files: `ls data/negative/_background_noise_/*.wav | wc -l` should be 20+
2. Check all are 16kHz mono: `python inspect_wavs.py` on the folder
3. Listen to each — ensure no clear speech in the "TV murmur" file

---

## 8. Requirement 7: Training Configuration Updates

### Goal
Update training hyperparameters per the technical brief.

### Changes to Make

In `hey_magis_v6_improved.py`, update the CONFIG section:

```python
# =============================
# CONFIG (UPDATED per technical brief)
# =============================
SAMPLE_RATE = 16000
CLIP_SECONDS = 1.5                    # Keep current
NUM_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)  # 24000

FRAME_LENGTH = 400                     # 25ms @ 16k — keep
FRAME_STEP = 160                       # 10ms @ 16k — keep
FFT_LENGTH = 512                       # keep
NUM_MELS = 40                          # keep
TARGET_FRAMES = 148                    # keep
LOWER_HZ = 80.0                        # keep
UPPER_HZ = 7600.0                      # keep

BATCH_SIZE = 32                        # keep
EPOCHS = 60                            # Changed from 50 to 60
LEARNING_RATE = 5e-4                   # keep
PATIENCE = 10                          # keep

NEG_MULTIPLIER = 3                     # keep
HARD_NEG_FRACTION = 0.60              # Changed from 0.50 to 0.60

USE_AUGMENT = True
MAX_GAIN_DB = 8.0                      # keep
NOISE_MIX_PROB = 0.50                  # Changed from 0.35 to 0.50
NOISE_MIX_SNR_DB = (0.0, 20.0)       # Changed from (0.0, 18.0) to (0.0, 20.0)

# SpecAugment config (NEW)
SPEC_AUGMENT = True
NUM_FREQ_MASKS = 2
FREQ_MASK_WIDTH = 5
NUM_TIME_MASKS = 2
TIME_MASK_WIDTH = 15
```

### Summary of All Config Changes

| Parameter | Old Value | New Value | Reason |
|---|---|---|---|
| `EPOCHS` | 50 | 60 | More training time with focal loss |
| `PATIENCE` | 10 | 10 | Keep (already good) |
| `HARD_NEG_FRACTION` | 0.50 | 0.60 | Per technical brief: majority hard negatives |
| `NOISE_MIX_PROB` | 0.35 | 0.50 | More noise augmentation |
| `NOISE_MIX_SNR_DB` | (0.0, 18.0) | (0.0, 20.0) | Slightly wider range |
| `loss` | `binary_crossentropy` | `focal_loss(gamma=2)` | Focus on hard examples |
| `metric` | `auc` (ROC) | `auc_pr` (PR curve) | Better for imbalanced data |
| **NEW** | — | SpecAugment | Time/frequency masking |
| **NEW** | — | Pitch shifting | Positive augmentation |
| **NEW** | — | Speed perturbation | Positive augmentation |

---

## 9. Requirement 8: Evaluation Requirements

### Goal
Provide a comprehensive threshold sweep report with FAR, FRR, DET curve, and SNR-segmented metrics.

### Implementation

Create a new evaluation script `eval_comprehensive.py` (or extend `eval_wakeword_tflite_buckets.py`):

#### 8A. Threshold Sweep with FAR/FRR

```python
import numpy as np
import csv
from pathlib import Path

def comprehensive_threshold_sweep(pos_scores, neg_scores, 
                                    neg_duration_hours=None):
    """
    Compute metrics at many thresholds.
    
    Args:
        pos_scores: Array of scores for positive samples
        neg_scores: Array of scores for negative samples
        neg_duration_hours: Total audio duration of negatives in hours.
                           If provided, computes FAR per hour.
                           If None, estimates from clip count.
    """
    if neg_duration_hours is None:
        # Each clip is 1.5s, so N clips = N * 1.5 / 3600 hours
        neg_duration_hours = len(neg_scores) * 1.5 / 3600.0
    
    thresholds = np.arange(0.01, 1.00, 0.01)
    results = []
    
    P = len(pos_scores)
    N = len(neg_scores)
    
    for t in thresholds:
        tp = int(np.sum(pos_scores >= t))
        fn = P - tp
        fp = int(np.sum(neg_scores >= t))
        tn = N - fp
        
        recall = tp / P if P > 0 else 0.0       # = 1 - FRR
        frr = fn / P if P > 0 else 0.0          # False Reject Rate
        far = fp / N if N > 0 else 0.0          # False Accept Rate
        specificity = tn / N if N > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
        
        # FAR per hour of audio
        far_per_hour = fp / neg_duration_hours if neg_duration_hours > 0 else 0.0
        
        results.append({
            "threshold": float(t),
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "recall": recall,
            "FRR": frr,
            "FAR": far,
            "FAR_per_hour": far_per_hour,
            "specificity": specificity,
            "precision": precision,
            "F1": f1,
        })
    
    return results
```

#### 8B. DET Curve

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_det_curve(pos_scores, neg_scores, output_path="models/det_curve.png"):
    """
    Plot Detection Error Tradeoff (DET) curve.
    
    DET curve plots FRR vs FAR on a normal deviate (probit) scale.
    Better models have curves closer to the origin (lower-left).
    """
    thresholds = np.arange(0.01, 1.00, 0.001)
    
    P = len(pos_scores)
    N = len(neg_scores)
    
    frr_list = []
    far_list = []
    
    for t in thresholds:
        fn = np.sum(pos_scores < t)
        fp = np.sum(neg_scores >= t)
        frr_list.append(fn / P if P > 0 else 0)
        far_list.append(fp / N if N > 0 else 0)
    
    frr_arr = np.array(frr_list)
    far_arr = np.array(far_list)
    
    # Filter out zeros for log scale
    mask = (frr_arr > 0) & (far_arr > 0)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.loglog(far_arr[mask] * 100, frr_arr[mask] * 100, 'b-', linewidth=2)
    ax.set_xlabel("False Accept Rate (%)", fontsize=14)
    ax.set_ylabel("False Reject Rate (%)", fontsize=14)
    ax.set_title("Detection Error Tradeoff (DET) Curve", fontsize=16)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.set_xlim([0.01, 100])
    ax.set_ylim([0.01, 100])
    
    # Mark specific operating points
    for target_far in [0.1, 0.5, 1.0, 5.0]:
        idx = np.argmin(np.abs(far_arr * 100 - target_far))
        if mask[idx]:
            ax.plot(far_arr[idx]*100, frr_arr[idx]*100, 'ro', markersize=8)
            ax.annotate(f"t={thresholds[idx]:.2f}\nFAR={far_arr[idx]*100:.2f}%\nFRR={frr_arr[idx]*100:.2f}%",
                       xy=(far_arr[idx]*100, frr_arr[idx]*100),
                       xytext=(10, 10), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved DET curve: {output_path}")
```

#### 8C. SNR-Segmented Evaluation

To evaluate at different noise levels, create test sets with known SNR:

```python
def create_snr_test_sets(positive_wavs, noise_wavs, snr_levels=[100, 10, 5, 0]):
    """
    Create test copies of positives at different SNR levels.
    
    snr_levels: list of SNR in dB. 100 = clean (no noise added).
    
    Returns dict: {snr_db: [list of augmented wav paths]}
    """
    results = {}
    
    for snr_db in snr_levels:
        results[snr_db] = []
        
        if snr_db >= 50:
            # Clean — use original files
            results[snr_db] = positive_wavs
            continue
        
        # Create noisy versions
        snr_dir = Path(f"data/test_snr_{snr_db}dB")
        snr_dir.mkdir(parents=True, exist_ok=True)
        
        for wav_path in positive_wavs:
            audio, sr = load_audio(wav_path)
            noise = load_random_noise(noise_wavs, sr, len(audio))
            
            # Mix at exact SNR
            sig_rms = np.sqrt(np.mean(audio**2) + 1e-8)
            noi_rms = np.sqrt(np.mean(noise**2) + 1e-8)
            target_noi_rms = sig_rms / (10**(snr_db/20))
            noise_scaled = noise * (target_noi_rms / noi_rms)
            
            mixed = audio + noise_scaled
            mixed = np.clip(mixed, -1.0, 1.0)
            
            out_path = snr_dir / wav_path.name
            sf.write(str(out_path), mixed, sr)
            results[snr_db].append(out_path)
    
    return results
```

Then evaluate the model at each SNR level and report separately.

#### 8D. Full Report Format

The final evaluation report should look like this:

```
================================================================
MAGISAI WAKE WORD EVALUATION REPORT
Model: hey_magis_v6_model.tflite
Date: 2026-03-XX
================================================================

DATASET
  Positives: 468 clips (139 speakers)
  Negatives: 10,000 sampled from 105,896 total
  Hard negatives: 13,385 (all scored)
  Negative audio duration: ~4.17 hours

THRESHOLD SWEEP
  threshold | recall | FRR    | FAR    | FAR/hr | precision | F1
  0.50      | 0.95   | 0.05   | 0.008  | 19.2   | 0.85      | 0.90
  0.60      | 0.93   | 0.07   | 0.004  | 9.6    | 0.90      | 0.91
  0.70      | 0.90   | 0.10   | 0.002  | 4.8    | 0.93      | 0.91
  0.80      | 0.87   | 0.13   | 0.001  | 2.4    | 0.96      | 0.91
  0.85      | 0.84   | 0.16   | 0.0005 | 1.2    | 0.97      | 0.90
  0.90      | 0.80   | 0.20   | 0.0002 | 0.5    | 0.99      | 0.88
  0.95      | 0.72   | 0.28   | 0.0000 | 0.0    | 1.00      | 0.84

RECOMMENDED THRESHOLD: 0.82
  Justification: Best F1 score with FAR < 1.0/hour
  Expected deployment: ~1-2 false accepts per hour
  With posterior smoothing (3 hits): <0.5 false accepts per hour

SNR-SEGMENTED RECALL
  SNR    | recall @ t=0.82
  Clean  | 0.95
  10 dB  | 0.88
  5 dB   | 0.80
  0 dB   | 0.65

DET CURVE: see models/det_curve.png

TOP FALSE POSITIVES (inspect these):
  0.987  data/negative/backward/speaker123_nohash_0.wav
  0.954  data/negative/tv_samples/tv_01234.wav
  ... (top 50 listed)

TOP FALSE NEGATIVES (missed wake words):
  0.312  data/positive_trimmed_1p5_A/hey_magis_speaker45.wav
  0.287  data/positive_new_friends/.../clip_003.wav
  ... (all missed at recommended threshold)
```

### Verification

1. Report file is generated automatically
2. DET curve PNG is saved
3. FAR/hour is computed
4. SNR segments are evaluated separately
5. Top false positives and false negatives are listed

---

## 10. Deliverables Checklist

| # | Deliverable | File | Status |
|---|---|---|---|
| 1 | Updated training script | `hey_magis_v6_improved.py` | |
| 2 | .tflite model file | `models/hey_magis_v6_model.tflite` | |
| 3 | .h5 Keras model | `models/hey_magis_v6_model.h5` | |
| 4 | Comprehensive evaluation script | `eval_comprehensive.py` | |
| 5 | Threshold sweep report | `models/hey_magis_v6_eval_report.txt` | |
| 6 | DET curve plot | `models/det_curve.png` | |
| 7 | Score CSVs (per bucket) | `models/scores_*.csv` | |
| 8 | SNR-segmented eval results | In eval report | |
| 9 | This implementation guide | `docs/IMPLEMENTATION_GUIDE.md` | |
| 10 | Changeolog / writeup | `docs/CHANGELOG_V6.md` | |

### Critical Verification Before Delivery

```python
# Verify the .tflite model has correct shapes
import tensorflow as tf
interp = tf.lite.Interpreter(model_path="models/hey_magis_v6_model.tflite")
interp.allocate_tensors()

inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

assert inp["shape"].tolist() == [1, 148, 40, 1], f"BAD INPUT SHAPE: {inp['shape']}"
assert out["shape"].tolist() == [1, 1], f"BAD OUTPUT SHAPE: {out['shape']}"
print("Model shapes verified OK")
```

---

## 11. Critical Constraints Reminder

These constraints are NON-NEGOTIABLE. Breaking any of them means the Flutter app breaks.

| Constraint | Value | Consequence of Breaking |
|---|---|---|
| Input shape | `[1, 148, 40, 1]` | Dart code crashes / wrong inference |
| Output shape | `[1, 1]` sigmoid | App cannot interpret output |
| Sample rate | 16kHz | Feature extraction mismatch |
| Mel bins | 40 | Shape mismatch |
| Mel range | 80-7600 Hz | Different features than training |
| Normalization | Per-clip mean/std | Score distribution shift |
| TFLite plugin | `tflite_flutter` compatible | App cannot load model |
| Platform | Windows training | Script must run on Windows |
| Folder structure | Keep existing | Manifests break |
| Manifest format | One path per line | Loader breaks |

---

## 12. Recommended Execution Order

Follow this order to avoid dependency issues:

```
Phase 1: Data Preparation (Days 1-2)
├── 1. Record phonetic negatives (Req 3)
├── 2. Collect background noise files (Req 6)
├── 3. Create augmented positives offline (Req 1A, 1B if offline)
└── 4. Rebuild manifests

Phase 2: Hard Negative Mining (Day 3)
├── 5. Run eval on all negatives with current model
├── 6. Extract hard negatives above 0.3
└── 7. Rebuild hard negative manifest

Phase 3: Training Script Updates (Days 4-5)
├── 8. Copy training script → hey_magis_v6_improved.py
├── 9. Implement focal loss (Req 4A)
├── 10. Add AUC-PR metric (Req 4C)
├── 11. Add SpecAugment (Req 1B)
├── 12. Update config values (Req 7)
└── 13. Test training runs (small epochs first)

Phase 4: Full Training (Days 5-6)
├── 14. Run full training (60 epochs)
├── 15. Verify .tflite output shape
└── 16. Quick sanity check on scores

Phase 5: Evaluation (Days 6-7)
├── 17. Implement comprehensive eval script (Req 8)
├── 18. Run full threshold sweep
├── 19. Generate DET curve
├── 20. Run SNR-segmented evaluation
├── 21. Analyze posterior smoothing impact (Req 5)
└── 22. Write final report

Phase 6: Iteration (Days 7-10)
├── 23. Review false positives / false negatives
├── 24. Re-mine hard negatives with new model
├── 25. Retrain if needed
└── 26. Final delivery
```

---

## 13. Testing & Verification Protocol

### Quick Smoke Test (After Every Change)

```bash
# Train for 2 epochs to verify no crashes
python hey_magis_v6_improved.py  # (modify EPOCHS=2 temporarily)
```

### Shape Verification (After Training)

```python
import tensorflow as tf
interp = tf.lite.Interpreter(model_path="models/hey_magis_v6_model.tflite")
interp.allocate_tensors()
print("Input:", interp.get_input_details()[0]["shape"])   # Must be [1, 148, 40, 1]
print("Output:", interp.get_output_details()[0]["shape"])  # Must be [1, 1]
```

### Score Sanity Check (After Training)

```bash
python eval_wakeword_tflite_buckets.py \
  --model models/hey_magis_v6_model.tflite \
  --pos_newfriends data/positive_new_friends \
  --pos_trimmed data/positive_trimmed_1p5_A \
  --neg_manifest data/negative_manifest.txt \
  --hard_manifest data/negative_manifest_hard_merged.txt
```

Expected:
- Positive scores: mostly > 0.5 (mean > 0.7)
- General negative scores: mostly < 0.1 (mean < 0.05)
- Hard negative scores: more spread (mean < 0.3)

### Audio Augmentation Verification

Listen to 10 random augmented positives:
1. Can you still understand "Hey Magis"?
2. Does the pitch variation sound natural?
3. Is the noise level realistic?

### Final Acceptance Criteria

| Metric | Target | How to Measure |
|---|---|---|
| Recall at recommended threshold | > 85% | Threshold sweep report |
| FAR per hour at recommended threshold | < 2.0 | Threshold sweep report |
| Recall at 0dB SNR | > 60% | SNR-segmented eval |
| Model input shape | `[1, 148, 40, 1]` | TFLite interpreter check |
| Model output shape | `[1, 1]` | TFLite interpreter check |
| Model loads in tflite_flutter | Yes | Deploy to test device |

---

*End of Implementation Guide*

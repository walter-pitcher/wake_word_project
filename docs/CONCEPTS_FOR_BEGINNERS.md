# MagisAI Wake Word — Concepts for Beginners

> **Who is this for?** You are a junior developer who has been asked to work on or understand this wake word detection project. This document explains every concept from scratch, assuming no prior knowledge of audio processing, machine learning, or wake word detection.
>
> **Date:** March 17, 2026

---

## Table of Contents

1. [What is a Wake Word?](#1-what-is-a-wake-word)
2. [How Our System Works (The Big Picture)](#2-how-our-system-works-the-big-picture)
3. [Understanding Sound & Audio](#3-understanding-sound--audio)
4. [From Sound to Numbers: Feature Extraction](#4-from-sound-to-numbers-feature-extraction)
5. [What is a Neural Network?](#5-what-is-a-neural-network)
6. [Our Model Architecture: DS-CNN](#6-our-model-architecture-ds-cnn)
7. [Training a Model: How It Learns](#7-training-a-model-how-it-learns)
8. [The Dataset: Positives and Negatives](#8-the-dataset-positives-and-negatives)
9. [Class Imbalance: The Core Challenge](#9-class-imbalance-the-core-challenge)
10. [Data Augmentation: Making More From Less](#10-data-augmentation-making-more-from-less)
11. [Loss Functions: How the Model Measures Its Mistakes](#11-loss-functions-how-the-model-measures-its-mistakes)
12. [Evaluation Metrics: Measuring Success](#12-evaluation-metrics-measuring-success)
13. [TensorFlow Lite: From Training to Phone](#13-tensorflow-lite-from-training-to-phone)
14. [The Threshold Decision](#14-the-threshold-decision)
15. [Hard Negative Mining: Teaching the Model the Tricky Stuff](#15-hard-negative-mining-teaching-the-model-the-tricky-stuff)
16. [SpecAugment: A Clever Training Trick](#16-specaugment-a-clever-training-trick)
17. [Focal Loss: Focusing on What Matters](#17-focal-loss-focusing-on-what-matters)
18. [The Full Pipeline: End to End](#18-the-full-pipeline-end-to-end)
19. [Key Terms Glossary](#19-key-terms-glossary)
20. [Common Mistakes to Avoid](#20-common-mistakes-to-avoid)

---

## 1. What is a Wake Word?

A **wake word** (also called a "hot word" or "keyword") is a specific phrase that activates a voice assistant. You already know some:

- **"Hey Siri"** — activates Apple's Siri
- **"OK Google"** — activates Google Assistant
- **"Alexa"** — activates Amazon's Alexa

Our wake word is **"Hey Magis"** (pronounced "Hey MAH-jis"). It activates the MagisAI Catholic assistant app.

### Why is this hard?

The phone is always listening. Every second, it's processing audio and asking: "Did someone just say Hey Magis?" The challenge is:

1. **Say "yes" when someone actually says it** — this is called **recall** (catching the real thing)
2. **Say "no" when they say anything else** — this is called **specificity** (rejecting everything else)

The tricky part: phrases like "Hey magic," "Hey Marcus," or even random TV audio can sound similar enough to fool the model.

---

## 2. How Our System Works (The Big Picture)

Here's what happens from the moment someone speaks to the moment the app responds:

```
Person says "Hey Magis"
        │
        ▼
Phone microphone captures audio (16,000 times per second)
        │
        ▼
Audio is collected in a 1.5-second sliding window
        │
        ▼
Audio is converted to a "spectrogram" (visual picture of sound)
        │
        ▼
Spectrogram is fed into a tiny neural network (the model)
        │
        ▼
Model outputs a probability: 0.0 = "definitely not" ... 1.0 = "definitely yes"
        │
        ▼
If probability > threshold (e.g., 0.85), trigger the assistant
        │
        ▼
App starts speech-to-text to hear the actual command
```

There are two separate systems working together:

### Training System (Python, on your computer)
- Uses TensorFlow to train the neural network
- Processes thousands of audio files
- Produces a `.tflite` model file
- Runs once (or whenever you retrain)

### Inference System (Dart/Flutter, on the phone)
- Loads the `.tflite` model
- Processes live microphone audio in real-time
- Makes predictions 100 times per second
- Runs continuously while the app is open

**You only need to modify the training system.** The phone app stays the same — it just loads whatever model you give it.

---

## 3. Understanding Sound & Audio

### What is Sound?

Sound is vibrations in the air. When you speak, your vocal cords vibrate, creating waves of pressure. A microphone converts these pressure waves into electrical signals, which a computer stores as numbers.

### Digital Audio Basics

| Concept | Explanation | Our Value |
|---|---|---|
| **Sample Rate** | How many times per second we measure the sound wave | 16,000 Hz (16,000 measurements per second) |
| **Sample** | A single measurement (a number between -1.0 and 1.0) | Float32 |
| **Mono** | One channel (vs. stereo which has two) | We use mono |
| **WAV** | A file format that stores raw audio without compression | All our data is WAV |

### Why 16,000 Hz?

Human speech mostly lives in frequencies between 100 Hz and 8,000 Hz. By the **Nyquist theorem**, you need to sample at least 2× the highest frequency you care about. So 16,000 Hz captures everything up to 8,000 Hz, which is perfect for speech.

Music uses 44,100 Hz because instruments have higher frequencies, but for wake word detection, 16,000 Hz saves computation and memory.

### What Does Our Audio Look Like as Numbers?

A 1.5-second clip at 16,000 Hz = **24,000 numbers**. Each number is between -1.0 and 1.0:

```
[0.003, -0.012, 0.045, 0.089, -0.067, 0.123, -0.234, ...]
 ←— 24,000 values total —→
```

Most of those numbers are near zero (silence). When someone speaks, the numbers get bigger (positive and negative).

---

## 4. From Sound to Numbers: Feature Extraction

We can't feed 24,000 raw numbers directly into a neural network (well, we could, but it would be very slow and not work well). Instead, we transform the audio into a more useful representation called a **spectrogram**.

### Step 1: STFT (Short-Time Fourier Transform)

This is the key magic. The idea is:

> Instead of looking at the raw audio wave, look at **which frequencies are present** at each moment in time.

Think of it like a music equalizer display — it shows you which frequencies (bass, mid, treble) are active at any moment.

**How it works:**
1. Take a small window of audio (25 milliseconds = 400 samples)
2. Apply the Fourier Transform to find which frequencies are present
3. Slide the window forward by 10 milliseconds (160 samples)
4. Repeat

This gives us a grid:
- Horizontal axis: **time** (148 windows for 1.5 seconds of audio)
- Vertical axis: **frequency** (257 frequency bins from the FFT)
- Each cell: **how loud** that frequency is at that time

### Step 2: Mel Filter Bank

Humans don't perceive frequencies linearly. We can easily tell the difference between 100 Hz and 200 Hz, but 5,000 Hz and 5,100 Hz sound almost the same. The **Mel scale** mimics human perception.

The Mel filter bank:
1. Takes the 257 raw frequency bins
2. Groups them into **40 Mel bins** (fewer bins at high frequencies, more at low)
3. Covers the range 80 Hz to 7,600 Hz

### Step 3: Logarithm

We take the log of the Mel energies because:
- Human loudness perception is logarithmic (doubling the energy sounds like a small increase)
- It compresses the dynamic range (very loud and very quiet sounds become more similar in scale)

### Step 4: Normalization

For each audio clip, we compute:
```
normalized = (value - mean) / standard_deviation
```

This ensures all clips have similar scale, regardless of how loudly someone spoke or how close they were to the microphone.

### The Final Result

After all these steps, our 24,000-sample audio clip becomes a **148 × 40** matrix (plus a channel dimension, making it 148 × 40 × 1):

```
148 time frames × 40 mel frequency bins × 1 channel

Think of it as a small grayscale image:
- Width: 40 pixels (frequency)
- Height: 148 pixels (time)
- Channels: 1 (like a black-and-white photo)
```

This is called a **log-mel spectrogram**, and it's the input to our neural network.

### Visual Analogy

Imagine you're looking at sheet music:
- The horizontal axis is time (left to right)
- The vertical axis is pitch (low notes at bottom, high notes at top)
- Dark areas are where notes are being played

A log-mel spectrogram is like automated sheet music for any sound, not just musical notes.

---

## 5. What is a Neural Network?

A neural network is a mathematical function that learns patterns from data. Think of it as a very complicated equation with thousands of adjustable knobs (called **parameters** or **weights**).

### The Basic Idea

```
Input (spectrogram)  →  [Neural Network]  →  Output (probability)
                              ↑
                    Thousands of adjustable knobs
```

During **training**:
1. Show the network a spectrogram
2. It produces a prediction (e.g., 0.3 = "probably not Hey Magis")
3. Compare to the correct answer (1.0 = "it IS Hey Magis")
4. Calculate the error (0.3 vs 1.0 = big error)
5. Adjust the knobs slightly to reduce the error
6. Repeat millions of times

After enough training, the knobs settle into values that make the network good at distinguishing "Hey Magis" from everything else.

### Types of Layers We Use

#### Convolutional Layer (Conv2D)

Imagine sliding a small magnifying glass (3×3 pixels) across the spectrogram. At each position, the layer computes a weighted sum of the pixels under the glass. Different "filters" look for different patterns (one might detect rising pitch, another might detect consonant sounds).

- **Conv2D(16)** = 16 different filters, each looking for a different pattern
- The output is 16 "feature maps" (like 16 different views of the spectrogram)

#### Depthwise Separable Convolution (DepthwiseConv2D + Conv2D 1×1)

This is a cheaper version of a regular convolution. Instead of one big operation, it splits into two smaller ones:
1. **Depthwise**: Apply a separate filter to each input channel independently
2. **Pointwise (1×1 Conv)**: Combine the results across channels

This uses 8-9× fewer computations than a regular Conv2D, which is crucial for running on a phone.

#### Batch Normalization

Makes training more stable by normalizing the values between layers. Think of it as keeping all the numbers in a reasonable range so the network doesn't get confused by very large or very small values.

#### ReLU (Rectified Linear Unit)

A simple function: if the number is negative, make it zero. If positive, keep it.
```
ReLU(x) = max(0, x)
```
This adds "nonlinearity" — without it, the entire network would just be one big linear equation, which can't learn complex patterns.

#### Global Average Pooling

Takes all the spatial information (the 37×10 grid at the end of convolutions) and averages it all into a single vector of 64 numbers. This makes the model's final decision independent of exact timing.

#### Dropout

During training, randomly sets some values to zero (25% of them). This prevents **overfitting** — the model memorizing the training data instead of learning general patterns.

#### Dense (Fully Connected)

A traditional neural network layer where every input connects to every output. Our final Dense layer has just 1 output with a **sigmoid** activation.

#### Sigmoid

Squashes any number into the range [0, 1]:
```
sigmoid(x) = 1 / (1 + e^(-x))
```
- Very negative numbers → close to 0 ("definitely not Hey Magis")
- Very positive numbers → close to 1 ("definitely Hey Magis")
- Zero → 0.5 ("not sure")

---

## 6. Our Model Architecture: DS-CNN

DS-CNN stands for **Depthwise Separable Convolutional Neural Network**. It's designed to be small and fast enough to run on a phone.

### The Layer-by-Layer Journey

```
Input: [148, 40, 1] — our spectrogram

1. Conv2D(16) → BatchNorm → ReLU
   Shape: [148, 40, 16]
   "Look for 16 basic patterns in the spectrogram"

2. DepthwiseConv2D(stride=2) → BatchNorm → ReLU → Conv2D(24) → BatchNorm → ReLU
   Shape: [74, 20, 24]
   "Shrink the image by half, look for 24 patterns"

3. DepthwiseConv2D → BatchNorm → ReLU → Conv2D(32) → BatchNorm → ReLU
   Shape: [74, 20, 32]
   "Same size, look for 32 more complex patterns"

4. DepthwiseConv2D(stride=2) → BatchNorm → ReLU → Conv2D(48) → BatchNorm → ReLU
   Shape: [37, 10, 48]
   "Shrink by half again, 48 patterns"

5. DepthwiseConv2D → BatchNorm → ReLU → Conv2D(64) → BatchNorm → ReLU
   Shape: [37, 10, 64]
   "64 high-level patterns"

6. GlobalAveragePooling2D
   Shape: [64]
   "Summarize everything into 64 numbers"

7. Dropout(0.25)
   "Randomly zero 25% of values (training only)"

8. Dense(1, sigmoid)
   Shape: [1]
   "Make final decision: probability from 0 to 1"
```

### Why This Architecture?

- **Small:** ~20,000 parameters (compare to ChatGPT's 175 billion). Fits in a few kilobytes.
- **Fast:** Depthwise separable convolutions use ~8× fewer operations than regular ones.
- **Proven:** Based on ARM's research for keyword spotting on microcontrollers.
- **Quantization-friendly:** Simple operations (Conv, BN, ReLU) convert well to TFLite.

---

## 7. Training a Model: How It Learns

### The Training Loop

Training is like teaching a student by giving them practice tests:

```
For each epoch (1 to 60):
    For each batch (32 samples at a time):
        1. Show 32 spectrograms to the model
        2. Model predicts 32 probabilities
        3. Compare predictions to correct answers
        4. Calculate the "loss" (how wrong the model is)
        5. Adjust weights to reduce the loss (backpropagation)
    
    After all batches:
        6. Test on validation set (data the model hasn't seen)
        7. If validation performance improved, save the model
        8. If not improved for 10 epochs, stop early (patience)
```

### Key Training Concepts

#### Epoch
One complete pass through all training data. We train for up to 60 epochs.

#### Batch
A group of 32 samples processed together. This is more efficient than processing one at a time (GPUs can do parallel math).

#### Learning Rate
How big the weight adjustments are. Too high = overshoots the optimal weights. Too low = takes forever. We start at 0.0005 and reduce it when progress stalls.

#### Validation Set
About 20% of our data that we never train on. We use it to check if the model is actually learning useful patterns (vs. just memorizing the training data).

#### Early Stopping
If the model hasn't improved on the validation set for 10 epochs, we stop training and use the best model from earlier. This prevents overfitting.

#### Overfitting
When the model memorizes the training data but fails on new data. Like a student who memorizes answers instead of understanding concepts. Signs: training accuracy is 99% but validation accuracy is 70%.

### Our Special Training Trick: Per-Epoch Negative Resampling

We have 468 positives but 105,896 negatives. If we used all negatives, the model would be overwhelmed. Instead:

Each epoch, we randomly sample a new set of negatives:
- 3× the number of positives = 1,404 negatives
- 50-60% from the hard negative list (the tricky ones)
- 40-50% from the general negative list (easy variety)

This means every epoch sees different negatives, giving the model broad exposure over time while keeping each epoch balanced.

---

## 8. The Dataset: Positives and Negatives

### Positives ("Hey Magis" recordings)

We have **468 recordings** of people saying "Hey Magis":

| Source | Clips | Speakers | Description |
|---|---|---|---|
| New friends | ~120 | 6 | Dedicated speakers, 20 clips each |
| Trimmed collection | ~348 | 133 | Diverse speakers, 3-4 clips each |
| **Total** | **468** | **139** | |

These were collected via a Typeform survey and manually trimmed to 1.5 seconds each.

### Negatives (everything else)

We have **105,896 audio clips** that are NOT "Hey Magis":

| Source | Clips | Description |
|---|---|---|
| Speech Commands | ~100,000 | 35 words: "yes", "no", "up", "dog", etc. |
| TV audio | ~5,101 | Sliced TV recording |
| Wrong ways | ~61 | "Hey magic", "Hey Marcus", etc. |
| Background noise | 6 | Room noise, pink noise |
| Hard negatives | ~3,000 | Highest-scoring false positives from previous model |

### Manifests

A **manifest** is a text file listing all the audio files, one per line:

```
data\positive_new_friends\heymagis_cristina_...\cristina_iphone_quiet_..._001.wav
data\positive_new_friends\heymagis_cristina_...\cristina_iphone_quiet_..._002.wav
data\positive_trimmed_1p5_A\hey_magis_speaker45.wav
...
```

The training script reads these manifests to know which files to load.

---

## 9. Class Imbalance: The Core Challenge

This is perhaps the most important concept in this project.

### The Problem

We have:
- 468 positives (0.4% of data)
- 105,896 negatives (99.6% of data)

That's a **1:226 ratio**.

If a model just always says "no" to everything, it would be right 99.6% of the time! But it would be useless — it would never detect the wake word.

### Why Standard Accuracy is Misleading

| Model | Accuracy | Recall | Useful? |
|---|---|---|---|
| "Always say no" | 99.6% | 0% | No! Never detects wake word |
| "Always say yes" | 0.4% | 100% | No! Phone rings constantly |
| Good model | ~95% | ~85% | Yes! |

This is why we use metrics like **recall**, **precision**, **FAR**, and **FRR** instead of just accuracy.

### How We Deal With It

1. **Resample negatives each epoch** — Only use 1,404 negatives per epoch (3× positives), not all 105K
2. **Hard negative emphasis** — 50-60% of those negatives are hard (confusing) ones
3. **Focal loss** — Down-weight easy negatives that the model already classifies correctly
4. **Class weights** — Give positive samples more importance in the loss function
5. **AUC-PR metric** — Use a metric that's honest about performance with imbalanced data

---

## 10. Data Augmentation: Making More From Less

### The Problem

468 recordings is not enough for a good model. The model needs to hear "Hey Magis" said in many different ways to generalize well.

### The Solution: Augmentation

Data augmentation creates modified copies of existing recordings. Each modification simulates a real-world variation:

#### Pitch Shifting
Move the voice up or down in pitch (like adjusting a guitar tuning).
- **Why:** Different people have different voice pitches
- **How much:** ±2 semitones (small, natural-sounding change)

#### Speed Perturbation
Speed up or slow down the audio.
- **Why:** People say "Hey Magis" at different speeds
- **How much:** 0.9× to 1.1× (10% variation)

#### Noise Injection
Add background noise to clean recordings.
- **Why:** Real phones are used in noisy environments
- **How much:** SNR 0-20 dB (from very noisy to barely noticeable)

**What is SNR?** Signal-to-Noise Ratio measures how loud the speech is relative to the noise:
- **20 dB** = Speech is 10× louder than noise (quiet room)
- **10 dB** = Speech is 3× louder than noise (cafe)
- **0 dB** = Speech and noise are equally loud (very noisy)

#### SpecAugment
After converting to a spectrogram, randomly block out rectangles:
- **Time masking:** Black out some time frames (forces model to work with partial words)
- **Frequency masking:** Black out some frequency bands (forces model to not rely on one specific frequency)

#### Random Gain
Randomly make the audio louder or quieter (±8 dB).
- **Why:** People hold phones at different distances, speak at different volumes

### Result

Starting from 468 real recordings, augmentation can create 5,000+ effective training samples, each slightly different from the original.

---

## 11. Loss Functions: How the Model Measures Its Mistakes

### What is a Loss Function?

The loss function tells the model **how wrong** its prediction is. The goal of training is to minimize this number.

### Binary Cross-Entropy (BCE) — Current

The standard loss function for yes/no classification:

```
For a positive sample (y=1), prediction p:
    loss = -log(p)
    If p=0.99 (confident correct), loss = 0.01 (tiny)
    If p=0.01 (confident wrong), loss = 4.6 (huge)

For a negative sample (y=0), prediction p:
    loss = -log(1-p)
    If p=0.01 (confident correct), loss = 0.01 (tiny)
    If p=0.99 (confident wrong), loss = 4.6 (huge)
```

**Problem with BCE:** It treats all samples equally. An easy negative (score 0.001) contributes the same gradient as a hard negative (score 0.4). Since 99% of negatives are easy, the model spends most of its training effort on samples it already handles perfectly.

### Focal Loss — What We Need

Focal loss adds a **focusing factor** that down-weights easy samples:

```
focal_loss = -(1 - p_correct)^gamma * log(p_correct)

Where gamma = 2 (the "focusing parameter")

For an easy negative (model scores 0.001):
    p_correct = 0.999 (model is very correct)
    focal_weight = (1 - 0.999)^2 = 0.000001
    loss = 0.000001 * (-log(0.999)) = basically zero

For a hard negative (model scores 0.4):
    p_correct = 0.6 (model is somewhat correct)
    focal_weight = (1 - 0.6)^2 = 0.16
    loss = 0.16 * (-log(0.6)) = significant
```

So the model focuses its learning on the hard cases and ignores the easy ones. This is exactly what we need!

---

## 12. Evaluation Metrics: Measuring Success

### Confusion Matrix

Every prediction falls into one of four categories:

|  | Model says "Yes" | Model says "No" |
|---|---|---|
| **Actually "Hey Magis"** | True Positive (TP) | False Negative (FN) |
| **Not "Hey Magis"** | False Positive (FP) | True Negative (TN) |

- **TP:** Correctly detected wake word (good!)
- **TN:** Correctly rejected non-wake-word (good!)
- **FP:** False alarm — model triggered when it shouldn't (bad!)
- **FN:** Missed detection — model didn't trigger when it should (bad!)

### Key Metrics

#### Recall (Sensitivity, True Positive Rate)
"Of all the real Hey Magis utterances, how many did we catch?"
```
Recall = TP / (TP + FN)
```
- Recall = 0.90 means we catch 90% of wake words
- Higher is better, but comes at the cost of more false alarms

#### Precision
"When we trigger, how often is it a real Hey Magis?"
```
Precision = TP / (TP + FP)
```
- Precision = 0.95 means 95% of triggers are legitimate
- Higher means fewer false alarms

#### False Accept Rate (FAR)
"How often do we falsely trigger?"
```
FAR = FP / (FP + TN)
```
Usually expressed per hour of audio (e.g., "2 false accepts per hour").

#### False Reject Rate (FRR)
"How often do we miss a real wake word?"
```
FRR = FN / (TP + FN) = 1 - Recall
```

#### F1 Score
Balances precision and recall:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

#### AUC-ROC vs AUC-PR

**AUC** = Area Under the Curve (higher is better, max is 1.0)

- **AUC-ROC:** Area under the Receiver Operating Characteristic curve (plots recall vs. false positive rate). Can be misleadingly high with imbalanced data.
- **AUC-PR:** Area under the Precision-Recall curve. More honest for imbalanced data because it doesn't get inflated by large numbers of true negatives.

#### DET Curve

The **Detection Error Tradeoff** curve plots FRR vs FAR on a logarithmic scale. It's the standard evaluation tool in the speech/speaker recognition community. Better models have curves closer to the bottom-left corner.

---

## 13. TensorFlow Lite: From Training to Phone

### What is TensorFlow?

**TensorFlow** is Google's open-source library for building and training neural networks. It's what our Python training scripts use.

### What is TensorFlow Lite?

**TensorFlow Lite** is a lightweight version designed for mobile devices and embedded systems. It:
- Loads much faster than full TensorFlow
- Uses less memory
- Supports hardware acceleration (GPU, NPU on phones)
- Can't train — only runs pre-trained models (inference)

### The Conversion Process

```
Training (Python)                    Deployment (Phone)
                                    
TensorFlow Model (.h5)              
    │                               
    ▼                               
TFLite Converter                    
    │                               
    ├── Quantization (optional)     
    │   Makes model smaller/faster  
    │                               
    ▼                               
.tflite file (few KB)  ─────────→  tflite_flutter plugin (Dart)
                                        │
                                        ▼
                                    Real-time inference on phone
```

### Quantization

The model uses `tf.lite.Optimize.DEFAULT` which applies **dynamic range quantization**:
- Weights are stored as 8-bit integers (instead of 32-bit floats)
- Activations are computed in float32 at runtime
- Model size reduces by ~4× with minimal accuracy loss

### Critical: Input/Output Shape

The phone app (Dart code) is hardcoded to expect:
- **Input:** `[1, 148, 40, 1]` — one spectrogram at a time
- **Output:** `[1, 1]` — one probability value

If the model has a different shape, the app crashes. This is why **you must never change the input/output shape**.

---

## 14. The Threshold Decision

### What is the Threshold?

The model outputs a number between 0 and 1. The **threshold** is the cutoff: above it, we trigger; below it, we don't.

```
Model output: 0.87
Threshold: 0.85
→ 0.87 > 0.85 → TRIGGER! 🎤

Model output: 0.72
Threshold: 0.85
→ 0.72 < 0.85 → Don't trigger ✗
```

### The Trade-Off

| Lower Threshold (e.g., 0.70) | Higher Threshold (e.g., 0.95) |
|---|---|
| Catches more wake words (high recall) | Catches fewer wake words (low recall) |
| More false alarms (high FAR) | Very few false alarms (low FAR) |
| Annoying: triggers on similar phrases | Annoying: users have to repeat themselves |

### The Sweet Spot

We need to find a threshold where:
- FAR < 1-2 per hour (not too many false alarms)
- Recall > 85% (catches most real wake words)

This is typically around **0.80-0.90**, but the exact number depends on the model quality.

### Posterior Smoothing (Hit Window)

To reduce false alarms further, the phone app doesn't trigger on a single detection. It requires **multiple consecutive detections** within a time window:

```
Frame 1: 0.88 → above threshold ✓ (hit 1)
Frame 2: 0.91 → above threshold ✓ (hit 2)  → TRIGGER! (2 consecutive hits)
```

vs.

```
Frame 1: 0.88 → above threshold ✓ (hit 1)
Frame 2: 0.45 → below threshold ✗ (reset counter)
Frame 3: 0.87 → above threshold ✓ (hit 1 again — not enough)
```

This means a single noise spike won't cause a false trigger — the model needs to be confident for multiple frames in a row.

---

## 15. Hard Negative Mining: Teaching the Model the Tricky Stuff

### The Problem

Most negatives are easy. The word "dog" sounds nothing like "Hey Magis." The model learns to reject it instantly.

But some audio is tricky:
- TV dialogue with similar rhythm
- "Hey magic" (very similar phonemes)
- Background music with vowel-like patterns

### The Solution

1. Run the current model on all 105K negatives
2. Find the ones it scores highest (e.g., > 0.3)
3. These are the **hard negatives** — the samples closest to the decision boundary
4. Feed these to the model more often during training

This is called **hard negative mining**, and it's one of the most effective ways to improve a classifier.

### In Our Project

We already have this pipeline:
1. `eval_wakeword_tflite_buckets.py` → scores all negatives
2. `mine_hard_negatives.py` → copies the top scorers to `hard_negatives_top/`
3. `build_hard_manifest_from_folder.py` → creates the merged manifest
4. Training script uses `HARD_NEG_FRACTION=0.60` → 60% of each epoch's negatives are hard

### The Iterative Loop

```
Train model v1 → Mine hard negatives → Train model v2 (with hard negs) →
Mine new hard negatives → Train model v3 → ... → Each version gets better
```

---

## 16. SpecAugment: A Clever Training Trick

### The Idea

SpecAugment (from Google, 2019) is one of the most impactful augmentation techniques for audio models. It works on the spectrogram (after feature extraction, before the model):

1. **Frequency masking:** Randomly zero out a horizontal strip
   - Forces the model to not rely on one specific frequency range
   
2. **Time masking:** Randomly zero out a vertical strip
   - Forces the model to recognize the wake word even if part of it is obscured

### Visual Example

```
Original spectrogram:          After SpecAugment:
┌────────────────┐             ┌────────────────┐
│▓▓▓░░░▓▓▓░░░▓▓│             │▓▓▓░░░▓▓▓░░░▓▓│
│▓▓▓░░░▓▓▓░░░▓▓│             │▓▓▓░░░▓▓▓░░░▓▓│
│▓▓▓░░░▓▓▓░░░▓▓│             │████████████████│ ← Frequency mask
│▓▓▓░░░▓▓▓░░░▓▓│             │████████████████│ ← (2 mel bins zeroed)
│▓▓▓░░░▓▓▓░░░▓▓│             │▓▓▓░░░▓▓▓░░░▓▓│
│▓▓▓░░░▓▓▓░░░▓▓│             │▓▓▓░░░▓▓▓░░░▓▓│
└────────────────┘             └──█───────█─────┘
                                   ↑         ↑
                              Time masks (some frames zeroed)
```

### Why It Works

Without SpecAugment, the model might learn shortcuts like "Hey Magis always has energy at 1000 Hz." With masking, sometimes that frequency is blocked, so the model must learn more robust features.

---

## 17. Focal Loss: Focusing on What Matters

### The Restaurant Analogy

Imagine you're a food critic reviewing 1000 meals:
- 990 are obviously terrible fast food (easy to reject)
- 7 are clearly excellent gourmet food (easy to accept)
- 3 are tricky — they look gourmet but have hidden problems

With standard BCE, you'd spend equal time writing reviews for all 1000. Most of your effort is wasted on obvious cases.

With focal loss, you'd spend almost all your time on the 3 tricky cases, barely glancing at the obvious ones.

### The Math (Simplified)

```
BCE loss:       loss = -log(p)
Focal loss:     loss = -(1-p)^gamma × log(p)

gamma = 2 (our setting)
```

The `(1-p)^gamma` factor is the key:
- When p is high (model is confident and correct): `(1-0.99)^2 = 0.0001` → almost no loss
- When p is low (model is uncertain or wrong): `(1-0.5)^2 = 0.25` → significant loss

So the model's learning signal comes almost entirely from the cases it struggles with.

---

## 18. The Full Pipeline: End to End

### Complete Data Flow

```
1. DATA COLLECTION
   ├── Record "Hey Magis" from various speakers
   ├── Download Speech Commands dataset (negatives)
   ├── Slice TV recordings (negatives)
   └── Record phonetic negatives ("Hey magic", etc.)

2. DATA PREPARATION
   ├── Trim silence from recordings
   ├── Pad to exactly 1.5 seconds
   ├── Build positive manifest (list of all positive WAVs)
   ├── Build negative manifest (list of all negative WAVs)
   └── Build hard negative manifest (confusing negatives)

3. AUGMENTATION (during training)
   ├── Random gain (±8 dB)
   ├── Noise mixing (SNR 0-20 dB)
   ├── SpecAugment (time + frequency masking)
   └── (Optional: pitch shift, speed perturbation — offline)

4. FEATURE EXTRACTION (during training)
   ├── Load 16kHz mono WAV
   ├── STFT (25ms window, 10ms hop)
   ├── Mel filter bank (40 bins, 80-7600 Hz)
   ├── Log transform
   ├── Per-clip normalization
   └── Output: [148, 40, 1] spectrogram

5. TRAINING
   ├── Build DS-CNN model
   ├── Compile with focal loss + AUC-PR metric
   ├── For each epoch:
   │   ├── Sample fresh negatives (60% hard)
   │   ├── Build tf.data pipeline with augmentation
   │   └── Train one epoch, evaluate on validation
   ├── Early stopping (patience=10 on val_auc_pr)
   └── Save best model

6. EXPORT
   ├── Save Keras model (.h5)
   ├── Convert to TFLite (.tflite)
   └── Verify input shape = [1, 148, 40, 1]

7. EVALUATION
   ├── Score all positives and negatives with .tflite model
   ├── Threshold sweep (FAR, FRR, recall, precision at every threshold)
   ├── DET curve
   ├── SNR-segmented evaluation
   └── Recommend deployment threshold

8. HARD NEGATIVE MINING (iterate)
   ├── Find negatives scoring > 0.3
   ├── Add to hard negative set
   └── Retrain (go to step 3)

9. DEPLOYMENT
   ├── Upload .tflite to Firebase Storage
   ├── Flutter app downloads model
   ├── tflite_flutter loads model
   └── Real-time inference on phone microphone
```

### Files Involved at Each Stage

| Stage | Key Files |
|---|---|
| Data prep | `trim_recordings_v3.py`, `pad_to_1p5.py`, `slice_tv.py` |
| Manifest building | `build_positive_manifest.py`, `build_negative_manifest.py`, `build_hard_manifest_from_folder.py` |
| Training | `hey_magis_v4_1p5_hard3061.py` (the main script) |
| Evaluation | `eval_wakeword_tflite_buckets.py` |
| Mining | `mine_hard_negatives.py` |
| Utilities | `count_positives.py`, `count_negatives.py`, `inspect_wavs.py` |

---

## 19. Key Terms Glossary

| Term | Definition |
|---|---|
| **Augmentation** | Creating modified copies of data to increase diversity |
| **AUC** | Area Under the Curve — a single number summarizing model performance (0-1, higher is better) |
| **Backpropagation** | Algorithm for computing how to adjust model weights based on the loss |
| **Batch** | Group of samples processed together (our batch size = 32) |
| **Binary classification** | Predicting one of two classes (wake word vs. not wake word) |
| **Class imbalance** | When one class has far more samples than another |
| **CNN** | Convolutional Neural Network — uses sliding filters to detect patterns |
| **DET curve** | Detection Error Tradeoff — standard evaluation plot for keyword detection |
| **Depthwise separable convolution** | Efficient convolution that splits into two steps |
| **DS-CNN** | Depthwise Separable CNN — our model architecture |
| **Epoch** | One complete pass through all training data |
| **FAR** | False Accept Rate — how often the model falsely triggers |
| **Feature extraction** | Converting raw audio into a format the model can process |
| **FFT** | Fast Fourier Transform — algorithm to find frequency content of audio |
| **Focal loss** | Loss function that focuses on hard examples |
| **FRR** | False Reject Rate — how often the model misses real wake words |
| **Hard negative** | A non-wake-word sample that the model finds confusing |
| **Inference** | Running a trained model to make predictions |
| **Log-mel spectrogram** | Our feature representation: log of mel-filtered spectrogram |
| **Loss function** | Mathematical formula measuring how wrong the model's predictions are |
| **Manifest** | Text file listing paths to audio files, one per line |
| **Mel scale** | Frequency scale matching human perception (more resolution at low frequencies) |
| **Overfitting** | Model memorizes training data instead of learning general patterns |
| **Posterior smoothing** | Requiring multiple consecutive detections before triggering |
| **Precision** | Of all triggers, what fraction were correct |
| **Quantization** | Converting model from float32 to smaller data types for efficiency |
| **Recall** | Of all real wake words, what fraction were detected |
| **ReLU** | Activation function: max(0, x) |
| **Sample rate** | Number of audio measurements per second (16,000 Hz) |
| **Sigmoid** | Function that squashes values to [0, 1] range |
| **SNR** | Signal-to-Noise Ratio — how loud speech is relative to noise |
| **SpecAugment** | Augmentation technique that masks parts of the spectrogram |
| **Spectrogram** | 2D representation of audio showing frequency content over time |
| **STFT** | Short-Time Fourier Transform — breaks audio into overlapping windows and analyzes frequencies |
| **TFLite** | TensorFlow Lite — lightweight runtime for mobile inference |
| **Threshold** | Cutoff value above which the model triggers a detection |
| **Training** | Process of adjusting model weights to minimize loss on data |
| **Validation** | Data held back from training to check generalization |

---

## 20. Common Mistakes to Avoid

### Mistake 1: Changing the Input Shape
**Never** change the model input from `[1, 148, 40, 1]`. The Dart code on the phone is hardcoded to this shape. If you change it, the app breaks.

### Mistake 2: Using Different Feature Parameters in Training vs. Evaluation
The mel range (80-7600 Hz), FFT length (512), hop size (160), etc. must be IDENTICAL in training and evaluation. If they differ even slightly, the scores will be wrong.

### Mistake 3: Evaluating on Training Data
Never report metrics on data the model trained on. Always use a held-out validation or test set. The v4 training report shows suspiciously perfect scores — this is a warning sign.

### Mistake 4: Ignoring Hard Negatives
If you only train on Speech Commands (easy negatives), the model will seem perfect in evaluation but fail badly on real-world audio like TV dialogue.

### Mistake 5: Using Only Accuracy as a Metric
With 468 positives and 105K negatives, a model that always says "no" gets 99.6% accuracy. Use recall, FAR, FRR, and AUC-PR instead.

### Mistake 6: Not Listening to False Positives
After evaluation, always listen to the top false positives (highest-scoring negatives). They tell you exactly what the model is confused about and guide your next improvement.

### Mistake 7: Training Too Long
More epochs is not always better. After the model converges, additional training can cause overfitting. Use early stopping (patience=10) and monitor validation AUC-PR.

### Mistake 8: Forgetting to Set Seeds
Always call `set_seeds(42)` at the start. This ensures your results are reproducible. Without it, random initialization means every training run gives different results.

### Mistake 9: Not Running on Windows
The training machine is Windows. Some Python packages behave differently on Linux/Mac. Always test on the actual training machine.

### Mistake 10: Modifying the Dart Code
The Flutter inference pipeline is already written and working. Do not touch it. Only deliver a compatible `.tflite` file.

---

*End of Concepts Document*

**Next steps for you:**
1. Read `PROJECT_ANALYSIS.md` to understand every file in the project
2. Read `IMPLEMENTATION_GUIDE.md` to understand what changes to make
3. Re-read this document whenever you encounter a concept you don't understand
4. Start with the simplest changes (config updates) and work up to complex ones (focal loss, augmentation)

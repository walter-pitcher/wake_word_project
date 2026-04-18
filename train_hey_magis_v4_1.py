#!/usr/bin/env python3
"""
Wake Word Trainer v4.1 — HEY MAGIS (Full-Negative Exposure)
 
- 1.5s clips everywhere
- ALL negatives used (randomly sampled per epoch)
- Hard negatives always included
- Strong noise robustness
- Designed for far-field + accents + noise
"""
 
import os, random
from pathlib import Path
from typing import List
import numpy as np
import tensorflow as tf
 
# =============================
# CONFIG
# =============================
SAMPLE_RATE = 16000
CLIP_SECONDS = 1.5
NUM_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
 
FRAME_LENGTH = 400
FRAME_STEP = 160
FFT_LENGTH = 512
NUM_MELS = 40
TARGET_FRAMES = 148
 
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-3
PATIENCE = 6
 
NEG_MULTIPLIER = 8           # VERY IMPORTANT
HARD_NEG_WEIGHT = 2          # oversample hard negatives
 
USE_AUGMENT = True
MAX_GAIN_DB = 8.0
NOISE_MIX_PROB = 0.4
NOISE_SNR_DB = (6.0, 18.0)
 
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
 
POS_DIRS = [
    DATA_DIR / "positive_new_friends",
    DATA_DIR / "positive_trimmed_1p5_A",
]
 
NEG_MANIFEST = DATA_DIR / "negative_manifest.txt"
HARD_NEG_MANIFEST = DATA_DIR / "negative_manifest_hard.txt"
NOISE_DIR = DATA_DIR / "negative" / "_background_noise_"
 
RUN_NAME = "hey_magis_v4_1"
 
# =============================
# UTIL
# =============================
def list_wavs_multi(folders):
    out = []
    for f in folders:
        out.extend(f.rglob("*.wav"))
    return out
 
def read_manifest(path: Path):
    return [Path(l.strip()) for l in path.read_text().splitlines() if l.strip()]
 
def load_wav(path):
    wav, sr = tf.audio.decode_wav(tf.io.read_file(path), desired_channels=1)
    wav = tf.squeeze(wav)
    tf.debugging.assert_equal(sr, SAMPLE_RATE)
    wav = wav[:NUM_SAMPLES]
    pad = NUM_SAMPLES - tf.shape(wav)[0]
    wav = tf.cond(pad > 0, lambda: tf.pad(wav, [[0, pad]]), lambda: wav)
    return wav
 
def augment(wav, noise_paths_tensor):
    """Apply augmentation with gain and optional noise mixing."""
    if not USE_AUGMENT:
        return wav
 
    # Random gain
    gain_db = tf.random.uniform([], -MAX_GAIN_DB, MAX_GAIN_DB)
    wav *= tf.pow(10.0, gain_db / 20.0)
    wav = tf.clip_by_value(wav, -1, 1)
 
    # Noise mixing (only if noise paths provided)
    if noise_paths_tensor is not None and tf.size(noise_paths_tensor) > 0:
        def mix_noise():
            idx = tf.random.uniform([], 0, tf.shape(noise_paths_tensor)[0], dtype=tf.int32)
            noise_path = tf.gather(noise_paths_tensor, idx)
            noise = load_wav(noise_path)
            # Random roll to vary noise position
            shift = tf.random.uniform([], 0, NUM_SAMPLES, dtype=tf.int32)
            noise = tf.roll(noise, shift=shift, axis=0)
            # Apply SNR
            snr = tf.random.uniform([], NOISE_SNR_DB[0], NOISE_SNR_DB[1])
            wav_rms = tf.sqrt(tf.reduce_mean(tf.square(wav)) + 1e-8)
            noise_rms = tf.sqrt(tf.reduce_mean(tf.square(noise)) + 1e-8)
            noise_scaled = noise * (wav_rms / (noise_rms * tf.pow(10.0, snr / 20.0) + 1e-8))
            return tf.clip_by_value(wav + noise_scaled, -1, 1)
 
        wav = tf.cond(
            tf.random.uniform([]) < NOISE_MIX_PROB,
            mix_noise,
            lambda: wav
        )
    return wav
 
def wav_to_mel(wav):
    stft = tf.signal.stft(wav, FRAME_LENGTH, FRAME_STEP, FFT_LENGTH)
    spec = tf.square(tf.abs(stft))
    mel_w = tf.signal.linear_to_mel_weight_matrix(
        NUM_MELS, FFT_LENGTH // 2 + 1, SAMPLE_RATE, 80, 7600
    )
    mel = tf.matmul(spec, mel_w)
    mel = tf.math.log(tf.maximum(mel, 1e-6))
    mel = (mel - tf.reduce_mean(mel)) / (tf.math.reduce_std(mel) + 1e-6)
    mel = mel[:TARGET_FRAMES]
    mel = tf.pad(mel, [[0, TARGET_FRAMES - tf.shape(mel)[0]], [0, 0]])
    return tf.expand_dims(mel, -1)
 
def make_example(path, label, noise_paths_tensor):
    """Load audio, apply augmentation, and convert to mel spectrogram."""
    wav = load_wav(path)
    wav = augment(wav, noise_paths_tensor)
    return wav_to_mel(wav), label
 
# =============================
# MODEL
# =============================
def build_model():
    inp = tf.keras.Input((TARGET_FRAMES, 40, 1))
    x = inp
    for ch in [16, 24, 32, 48, 64]:
        x = tf.keras.layers.Conv2D(ch, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, out)
 
# =============================
# MAIN
# =============================
def main():
    MODELS_DIR.mkdir(exist_ok=True)
 
    pos = list_wavs_multi(POS_DIRS)
    neg = read_manifest(NEG_MANIFEST)
    hard = read_manifest(HARD_NEG_MANIFEST)
 
    noise_list = list(NOISE_DIR.glob("*.wav")) if NOISE_DIR.exists() else []
    # Convert to tensor for use in tf.data pipeline
    noise_tensor = tf.constant([str(p) for p in noise_list]) if noise_list else None
 
    print(f"Positives: {len(pos)}")
    print(f"Negatives: {len(neg)}")
    print(f"Hard negs: {len(hard)}")
 
    random.shuffle(pos)
    split = int(0.8 * len(pos))
    train_pos, val_pos = pos[:split], pos[split:]
 
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
 
    for epoch in range(EPOCHS):
        neg_sample = random.sample(neg, min(len(neg), len(train_pos) * NEG_MULTIPLIER))
        hard_sample = random.choices(hard, k=len(hard) * HARD_NEG_WEIGHT)
 
        train_paths = (
            [str(p) for p in train_pos] +
            [str(p) for p in neg_sample] +
            [str(p) for p in hard_sample]
        )
 
        train_labels = (
            [1] * len(train_pos) +
            [0] * (len(neg_sample) + len(hard_sample))
        )
 
        ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        ds = ds.shuffle(4000).map(
            lambda p, y: make_example(p, tf.cast(y, tf.float32), noise_tensor),
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
 
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        model.fit(ds, epochs=1, verbose=1)
 
    model.save(MODELS_DIR / f"{RUN_NAME}.keras")
 
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite = converter.convert()
    (MODELS_DIR / f"{RUN_NAME}.tflite").write_bytes(tflite)
 
    print("\n✅ FINAL MODEL SAVED")
 
if __name__ == "__main__":
    main()
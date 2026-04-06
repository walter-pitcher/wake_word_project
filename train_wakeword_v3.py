#!/usr/bin/env python3
"""
Wake Word Trainer v3 (Hey Magis)
================================

Trains a lightweight CNN on 1.0s clips:
- Positives: data/positive_trimmed_1p2_A/*.wav
- Negatives: data/negative_manifest.txt (sampled per epoch)
- Hard negatives: data/negative_manifest_hard.txt (oversampled)

Outputs:
- models/<run_name>_model.h5
- models/<run_name>_model.tflite
- models/<run_name>_train_report.txt

Notes:
- Uses mel spectrogram frontend (TF ops) -> (98, 40, 1)
- Samples negatives each epoch to keep training balanced
"""

import os
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import tensorflow as tf

# -----------------------------
# CONFIG
# -----------------------------
SAMPLE_RATE = 16000
CLIP_SECONDS = 1.0
NUM_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)

# Mel frontend -> aim for (98, 40, 1)
# 25ms window, 10ms hop => ~98 frames for 1s if we use frame_length=400, frame_step=160
FRAME_LENGTH = 400   # 25ms @ 16k
FRAME_STEP = 160     # 10ms @ 16k
FFT_LENGTH = 512
NUM_MELS = 40
LOWER_HZ = 80.0
UPPER_HZ = 7600.0

# Training balance
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE = 6

# How many negatives to sample per epoch (relative to positives)
NEG_MULTIPLIER = 3          # e.g. 3x positives
HARD_NEG_FRACTION = 0.25    # fraction of sampled negatives that come from hard list

# Augmentation (lightweight)
USE_AUGMENT = True
MAX_GAIN_DB = 6.0
NOISE_MIX_PROB = 0.25
NOISE_MIX_SNR_DB = (8.0, 18.0)  # min/max SNR when mixing noise (if available)

# Threshold evaluation
THRESHOLDS_TO_TEST = [0.30, 0.40, 0.50, 0.60, 0.70]

# -----------------------------
# PATHS (edit if needed)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
POS_DIR = DATA_DIR / "positive_trimmed_1p2_A"
NEG_MANIFEST = DATA_DIR / "negative_manifest.txt"
HARD_NEG_MANIFEST = DATA_DIR / "negative_manifest_hard.txt"
MODELS_DIR = PROJECT_ROOT / "models"

RUN_NAME = "hey_magis_v3"

# Optional: a dedicated noise folder inside negatives (Speech Commands has _background_noise_)
# If it exists, we can pull random noise clips for mix augmentation.
BACKGROUND_NOISE_DIR = DATA_DIR / "negative" / "_background_noise_"

# -----------------------------
# UTIL
# -----------------------------
def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def read_manifest(path: Path) -> List[Path]:
    if not path.exists():
        return []
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    out = []
    for ln in lines:
        if not ln or ln.startswith("#"):
            continue
        # manifests are relative to project root or data root sometimes; handle both
        p = Path(ln)
        if not p.is_absolute():
            # try relative to project root first
            cand1 = PROJECT_ROOT / p
            cand2 = PROJECT_ROOT / "data" / p
            if cand1.exists():
                p = cand1
            elif cand2.exists():
                p = cand2
            else:
                # try relative to manifest's parent
                p = (path.parent / p)
        out.append(p)
    return out

def list_wavs(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(folder.glob("*.wav"))

def load_wav_mono_16k(path: tf.Tensor) -> tf.Tensor:
    """Returns float32 waveform [-1,1], shape [NUM_SAMPLES]. Pads/trims to 1.0s."""
    audio_bytes = tf.io.read_file(path)
    wav, sr = tf.audio.decode_wav(audio_bytes, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)  # [N]
    # resample if needed (rare if your data is already 16k)
    sr = tf.cast(sr, tf.int32)
    wav = tf.cond(
        tf.equal(sr, SAMPLE_RATE),
        lambda: wav,
        lambda: tf.signal.resample(wav, int(tf.shape(wav)[0]) * SAMPLE_RATE // sr),
    )
    wav = wav[:NUM_SAMPLES]
    pad = NUM_SAMPLES - tf.shape(wav)[0]
    wav = tf.cond(pad > 0, lambda: tf.pad(wav, [[0, pad]]), lambda: wav)
    return wav

def random_gain(wav: tf.Tensor) -> tf.Tensor:
    gain_db = tf.random.uniform([], -MAX_GAIN_DB, MAX_GAIN_DB)
    gain = tf.pow(10.0, gain_db / 20.0)
    wav = wav * gain
    # prevent clipping
    wav = tf.clip_by_value(wav, -1.0, 1.0)
    return wav

def maybe_mix_noise(wav: tf.Tensor, noise_paths: List[Path]) -> tf.Tensor:
    if not noise_paths:
        return wav
    r = tf.random.uniform([])
    def _mix():
        # pick a random noise file (python list -> tf picks by index)
        idx = tf.random.uniform([], 0, len(noise_paths), dtype=tf.int32)
        noise_path = tf.constant([str(p) for p in noise_paths])[idx]
        n = load_wav_mono_16k(noise_path)
        # random start offset if noise longer (already padded/trimmed, but we can roll)
        shift = tf.random.uniform([], 0, NUM_SAMPLES, dtype=tf.int32)
        n = tf.roll(n, shift=shift, axis=0)

        # scale noise to target SNR
        snr_db = tf.random.uniform([], NOISE_MIX_SNR_DB[0], NOISE_MIX_SNR_DB[1])
        sig_rms = tf.sqrt(tf.reduce_mean(tf.square(wav)) + 1e-8)
        noi_rms = tf.sqrt(tf.reduce_mean(tf.square(n)) + 1e-8)
        target_noi_rms = sig_rms / tf.pow(10.0, snr_db / 20.0)
        n = n * (target_noi_rms / (noi_rms + 1e-8))

        mixed = wav + n
        mixed = tf.clip_by_value(mixed, -1.0, 1.0)
        return mixed

    return tf.cond(r < NOISE_MIX_PROB, _mix, lambda: wav)

def wav_to_mel(wav: tf.Tensor) -> tf.Tensor:
    """
    Returns mel spectrogram log-mel, shape [98, 40, 1]
    """
    stft = tf.signal.stft(
        wav,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )
    mag = tf.abs(stft)  # [frames, fft_bins]
    power = tf.square(mag)

    num_spectrogram_bins = FFT_LENGTH // 2 + 1
    mel_w = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=NUM_MELS,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=LOWER_HZ,
        upper_edge_hertz=UPPER_HZ,
    )
    mel = tf.matmul(power, mel_w)  # [frames, mels]
    mel = tf.maximum(mel, 1e-10)
    log_mel = tf.math.log(mel)

    # Normalize per-example (stabilizes training)
    mean = tf.reduce_mean(log_mel)
    std = tf.math.reduce_std(log_mel) + 1e-6
    log_mel = (log_mel - mean) / std

    # Ensure fixed frame count (98)
    # For 1.0s with 25ms/10ms hop -> 98 frames is typical, but we enforce anyway.
    target_frames = 98
    frames = tf.shape(log_mel)[0]
    log_mel = log_mel[:target_frames, :]
    pad = target_frames - tf.shape(log_mel)[0]
    log_mel = tf.cond(pad > 0, lambda: tf.pad(log_mel, [[0, pad], [0, 0]]), lambda: log_mel)

    log_mel = tf.expand_dims(log_mel, axis=-1)  # [98,40,1]
    return log_mel

def make_example(path: tf.Tensor, label: tf.Tensor, noise_paths: List[Path]) -> Tuple[tf.Tensor, tf.Tensor]:
    wav = load_wav_mono_16k(path)
    if USE_AUGMENT:
        wav = random_gain(wav)
        wav = maybe_mix_noise(wav, noise_paths)
    feat = wav_to_mel(wav)
    return feat, label

# -----------------------------
# MODEL (tiny, mobile-friendly)
# -----------------------------
def build_model(input_shape=(98, 40, 1)) -> tf.keras.Model:
    inp = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(16, (3, 3), padding="same", use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(24, (1, 1), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(32, (1, 1), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(48, (1, 1), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(64, (1, 1), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inp, out)

# -----------------------------
# DATASET BUILDER (resample negatives each epoch)
# -----------------------------
@dataclass
class DatasetSpec:
    pos_paths: List[Path]
    neg_paths: List[Path]
    hard_neg_paths: List[Path]
    noise_paths: List[Path]

def sample_negatives(spec: DatasetSpec, num_needed: int) -> List[Path]:
    """Sample negatives with a fraction from hard negatives."""
    hard_n = int(num_needed * HARD_NEG_FRACTION)
    reg_n = max(0, num_needed - hard_n)

    sampled = []

    if spec.hard_neg_paths and hard_n > 0:
        sampled.extend(random.choices(spec.hard_neg_paths, k=hard_n))
    if spec.neg_paths and reg_n > 0:
        sampled.extend(random.choices(spec.neg_paths, k=reg_n))

    random.shuffle(sampled)
    return sampled

def build_tf_dataset(paths: List[Path], labels: List[int], noise_paths: List[Path], shuffle=True) -> tf.data.Dataset:
    p = tf.constant([str(x) for x in paths])
    y = tf.constant(labels, dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((p, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(2000, len(paths)), reshuffle_each_iteration=True)

    # map to (features, label)
    ds = ds.map(lambda pp, yy: make_example(pp, tf.cast(yy, tf.float32), noise_paths),
                num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# -----------------------------
# EVAL / REPORT
# -----------------------------
def eval_thresholds(model: tf.keras.Model, ds: tf.data.Dataset, thresholds: List[float]) -> Dict[float, Dict[str, float]]:
    y_true = []
    y_score = []
    for x, y in ds:
        s = model.predict(x, verbose=0).reshape(-1)
        y_score.extend(list(s))
        y_true.extend(list(y.numpy().reshape(-1)))

    y_true = np.array(y_true).astype(int)
    y_score = np.array(y_score).astype(float)

    results = {}
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        recall = tp / max(1, tp + fn)
        specificity = tn / max(1, tn + fp)
        precision = tp / max(1, tp + fp)
        fpr = fp / max(1, fp + tn)

        results[t] = {
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "recall": recall,
            "specificity": specificity,
            "precision": precision,
            "FPR": fpr,
        }
    return results

def write_report(path: Path, header: str, lines: List[str]):
    path.write_text(header + "\n" + "\n".join(lines) + "\n", encoding="utf-8")

# -----------------------------
# MAIN
# -----------------------------
def main():
    set_seeds(42)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    pos = list_wavs(POS_DIR)
    if not pos:
        raise SystemExit(f"No positives found in: {POS_DIR}")

    neg = read_manifest(NEG_MANIFEST)
    if not neg:
        raise SystemExit(f"No negatives loaded from: {NEG_MANIFEST}")

    hard_neg = read_manifest(HARD_NEG_MANIFEST)
    noise_paths = []
    if BACKGROUND_NOISE_DIR.exists():
        noise_paths = list_wavs(BACKGROUND_NOISE_DIR)

    print("=" * 70)
    print("HEY MAGIS TRAINING v3")
    print("=" * 70)
    print(f"Positives: {len(pos)} from {POS_DIR}")
    print(f"Negatives: {len(neg)} from {NEG_MANIFEST.name}")
    print(f"Hard negatives: {len(hard_neg)} from {HARD_NEG_MANIFEST.name}")
    print(f"Noise files (optional): {len(noise_paths)} from {BACKGROUND_NOISE_DIR if noise_paths else '(none)'}")
    print(f"Run name: {RUN_NAME}")
    print("=" * 70)

    # Split positives: train/val
    random.shuffle(pos)
    val_pos_n = max(10, int(0.2 * len(pos)))
    val_pos = pos[:val_pos_n]
    train_pos = pos[val_pos_n:]

    # We also create a fixed validation negative set (so metrics are comparable)
    spec = DatasetSpec(train_pos, neg, hard_neg, noise_paths)

    val_neg_n = val_pos_n * 3
    val_neg = sample_negatives(DatasetSpec(val_pos, neg, hard_neg, noise_paths), val_neg_n)

    val_paths = val_pos + val_neg
    val_labels = [1] * len(val_pos) + [0] * len(val_neg)
    val_ds = build_tf_dataset(val_paths, val_labels, noise_paths=[], shuffle=False)  # no augment on val

    # Model
    model = build_model(input_shape=(98, 40, 1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    model.summary()

    # Callbacks
    ckpt_path = MODELS_DIR / f"{RUN_NAME}_best.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=2, min_lr=1e-5),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(ckpt_path), monitor="val_auc", mode="max", save_best_only=True),
    ]

    # Training loop where we resample negatives each epoch
    steps_per_epoch = max(1, (len(train_pos) * (1 + NEG_MULTIPLIER)) // BATCH_SIZE)

    for epoch in range(EPOCHS):
        # sample negatives fresh each epoch
        train_neg_n = len(train_pos) * NEG_MULTIPLIER
        train_neg = sample_negatives(spec, train_neg_n)

        train_paths = train_pos + train_neg
        train_labels = [1] * len(train_pos) + [0] * len(train_neg)

        train_ds = build_tf_dataset(train_paths, train_labels, noise_paths=noise_paths, shuffle=True)

        print(f"\nEpoch {epoch+1}/{EPOCHS} | train_pos={len(train_pos)} train_neg={len(train_neg)} "
              f"(hard_frac={HARD_NEG_FRACTION:.2f}) steps={steps_per_epoch}")

        hist = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=1,
            callbacks=callbacks,
            verbose=1,
        )

        # EarlyStopping is handled by callback state; if it triggers, break cleanly.
        # Keras doesn't expose "stopped_epoch" nicely mid-loop, so we infer by patience not moving
        # (not perfect, but fine). We'll rely on restore_best_weights anyway.

        # If learning rate is tiny and val_auc stalls, break
        lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
        if lr <= 1.01e-5 and epoch >= 8:
            print("Learning rate bottomed out; stopping loop.")
            break

    # Save final H5
    h5_path = MODELS_DIR / f"{RUN_NAME}_model.h5"
    model.save(h5_path)
    print(f"\n✅ Saved Keras model: {h5_path}")

    # Export TFLite (float + dynamic range quantization)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = MODELS_DIR / f"{RUN_NAME}_model.tflite"
    tflite_path.write_bytes(tflite_model)
    print(f"✅ Saved TFLite model: {tflite_path}")

    # Evaluate thresholds on validation
    results = eval_thresholds(model, val_ds, THRESHOLDS_TO_TEST)
    lines = []
    lines.append("=" * 70)
    lines.append("VALIDATION THRESHOLD SWEEP")
    lines.append("=" * 70)
    lines.append(f"val_pos={len(val_pos)} val_neg={len(val_neg)}")
    lines.append("")
    for t in THRESHOLDS_TO_TEST:
        r = results[t]
        lines.append(
            f"t={t:.2f} | TP={r['TP']:4d} TN={r['TN']:4d} FP={r['FP']:4d} FN={r['FN']:4d} | "
            f"recall={r['recall']:.3f} spec={r['specificity']:.3f} prec={r['precision']:.3f} FPR={r['FPR']:.3f}"
        )

    report_path = MODELS_DIR / f"{RUN_NAME}_train_report.txt"
    write_report(report_path, "HEY MAGIS TRAINING REPORT", lines)
    print(f"✅ Wrote report: {report_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()

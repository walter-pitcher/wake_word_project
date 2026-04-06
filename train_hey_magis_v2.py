#!/usr/bin/env python3
"""
Hey Magis Wake Word Trainer v2 (Log-Mel + DS-CNN, TFLite export)
===============================================================

- Input: 16kHz, mono WAV clips (expected 1.0s, but we pad/trim to 16000 samples)
- Features: log-mel spectrogram (NO MFCC/DCT step)
- Model: small depthwise-separable CNN (DS-CNN-ish) for TFLite
- Outputs: Keras .h5, float32 .tflite, int8 .tflite, metadata .json

Recommended folders (your setup):
  positives: data/positive_trimmed_1p2_A
  negatives: data/negative
"""

import os
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import tensorflow as tf


# ----------------------------
# CONFIG (edit if needed)
# ----------------------------
SAMPLE_RATE = 16000
CLIP_SAMPLES = 16000  # 1.0s

# Log-mel params (matches your trimmer hop/frame nicely)
FRAME_LENGTH = 400   # 25ms @ 16k
FRAME_STEP   = 160   # 10ms @ 16k
FFT_LENGTH   = 512
NUM_MEL_BINS = 40
LOWER_HZ     = 20.0
UPPER_HZ     = 7600.0

# Training
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15
SEED = 1337

# Class balance
NEG_POS_RATIO = 2.0  # target negatives ~= 2x positives (we'll sample)
# If you have tons of negatives, we sub-sample them.

# Augmentations
AUG_PROB = 0.85
GAIN_MIN = 0.7
GAIN_MAX = 1.3
TIME_SHIFT_MAX_MS = 80  # small random shift
MIX_NOISE_PROB = 0.50
SNR_DB_MIN = 5.0
SNR_DB_MAX = 20.0

# TFLite
EXPORT_INT8 = True


@dataclass
class Paths:
    pos_dir: Path
    neg_dir: Path
    out_dir: Path
    run_name: str


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def list_wavs(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted([p for p in folder.glob("*.wav") if p.is_file()])


def decode_wav(file_path: tf.Tensor) -> tf.Tensor:
    """Decode wav -> float32 [-1,1], shape [samples]."""
    audio_bytes = tf.io.read_file(file_path)
    wav, sr = tf.audio.decode_wav(audio_bytes, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)  # [samples]
    # decode_wav returns float32
    return wav


def pad_or_trim(wav: tf.Tensor, target_len: int = CLIP_SAMPLES) -> tf.Tensor:
    wav = wav[:target_len]
    pad = target_len - tf.shape(wav)[0]
    wav = tf.cond(
        pad > 0,
        lambda: tf.pad(wav, [[0, pad]]),
        lambda: wav
    )
    return wav


def rms(x: tf.Tensor) -> tf.Tensor:
    return tf.sqrt(tf.reduce_mean(tf.square(x)) + 1e-9)


def random_gain(wav: tf.Tensor) -> tf.Tensor:
    g = tf.random.uniform([], GAIN_MIN, GAIN_MAX)
    return tf.clip_by_value(wav * g, -1.0, 1.0)


def time_shift(wav: tf.Tensor) -> tf.Tensor:
    max_shift = int(SAMPLE_RATE * (TIME_SHIFT_MAX_MS / 1000.0))
    shift = tf.random.uniform([], -max_shift, max_shift + 1, dtype=tf.int32)
    return tf.roll(wav, shift=shift, axis=0)


def mix_with_noise(clean: tf.Tensor, noise: tf.Tensor) -> tf.Tensor:
    """
    Mix clean + noise at a random SNR (in dB).
    Assumes both already 1s.
    """
    snr_db = tf.random.uniform([], SNR_DB_MIN, SNR_DB_MAX)
    clean_rms = rms(clean)
    noise_rms = rms(noise)

    # scale noise to achieve desired SNR:
    # snr_db = 20 log10(clean_rms / noise_scaled_rms)
    # => noise_scaled_rms = clean_rms / 10^(snr_db/20)
    target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    scale = target_noise_rms / (noise_rms + 1e-9)

    mixed = clean + noise * scale
    return tf.clip_by_value(mixed, -1.0, 1.0)


def wav_to_logmel(wav: tf.Tensor) -> tf.Tensor:
    """
    Convert waveform [16000] -> log-mel spectrogram [time, mel, 1]
    """
    stft = tf.signal.stft(
        wav,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH,
        window_fn=tf.signal.hann_window,
        pad_end=False
    )  # [frames, fft_bins]
    mag = tf.abs(stft)

    num_spectrogram_bins = FFT_LENGTH // 2 + 1
    mel_wts = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=NUM_MEL_BINS,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=LOWER_HZ,
        upper_edge_hertz=UPPER_HZ
    )
    mel = tf.matmul(tf.square(mag), mel_wts)  # power mel
    logmel = tf.math.log(mel + 1e-6)

    # Per-utterance normalization (helps stability)
    mean = tf.reduce_mean(logmel)
    std = tf.math.reduce_std(logmel) + 1e-6
    logmel = (logmel - mean) / std

    # add channel dim
    logmel = tf.expand_dims(logmel, axis=-1)  # [frames, mel, 1]
    return logmel


def build_model(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    """
    Small DS-CNN-ish model designed to quantize well.
    """
    inp = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding="same", use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    def ds_block(x, pw_filters, stride=(1, 1)):
        x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=stride, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(pw_filters, (1, 1), padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return x

    x = ds_block(x, 24, stride=(2, 2))
    x = ds_block(x, 32, stride=(1, 1))
    x = ds_block(x, 48, stride=(2, 2))
    x = ds_block(x, 64, stride=(1, 1))

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inp, out)


def make_dataset(
    pos_files: List[Path],
    neg_files: List[Path],
    batch_size: int,
    training: bool
) -> tf.data.Dataset:
    """
    Creates tf.data dataset of (logmel, label)
    label: 1 = wake word, 0 = non-wake
    """

    # Build a balanced-ish list
    pos = pos_files
    target_neg = int(len(pos) * NEG_POS_RATIO)
    if len(neg_files) >= target_neg:
        neg = random.sample(neg_files, target_neg)
    else:
        neg = neg_files[:]  # take all we have

    files = [(str(p), 1) for p in pos] + [(str(n), 0) for n in neg]
    random.shuffle(files)

    file_paths = tf.constant([f[0] for f in files])
    labels = tf.constant([f[1] for f in files], dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if training:
        ds = ds.shuffle(buffer_size=len(files), reshuffle_each_iteration=True)

    # We'll need negatives as noise sources for mixing
    noise_pool = tf.constant([str(n) for n in neg_files]) if len(neg_files) else None

    def _map_fn(fp, y):
        wav = decode_wav(fp)
        wav = pad_or_trim(wav)

        if training:
            # augment
            do_aug = tf.random.uniform([]) < AUG_PROB

            def aug_branch():
                w = wav
                w = random_gain(w)
                w = time_shift(w)

                # optionally mix with a random negative as background noise
                if noise_pool is not None:
                    do_mix = tf.random.uniform([]) < MIX_NOISE_PROB

                    def mix_branch():
                        idx = tf.random.uniform([], 0, tf.shape(noise_pool)[0], dtype=tf.int32)
                        noise_fp = noise_pool[idx]
                        noise = decode_wav(noise_fp)
                        noise = pad_or_trim(noise)
                        return mix_with_noise(w, noise)

                    w2 = tf.cond(do_mix, mix_branch, lambda: w)
                    return w2
                return w

            wav2 = tf.cond(do_aug, aug_branch, lambda: wav)
            wav = wav2

        feat = wav_to_logmel(wav)
        return feat, y

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def confusion_counts(y_true: np.ndarray, y_prob: np.ndarray, thresh: float = 0.5):
    y_pred = (y_prob >= thresh).astype(np.int32)
    y_true = y_true.astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def export_tflite(model: tf.keras.Model, out_float: Path, out_int8: Path, rep_ds: tf.data.Dataset):
    # Float32
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite = converter.convert()
    out_float.write_bytes(tflite)

    if not EXPORT_INT8:
        return

    # Int8 quant
    def rep_gen():
        for x, _ in rep_ds.take(200):
            # converter expects [1, ...] batches typically; we already have batches
            yield [tf.cast(x, tf.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_int8 = converter.convert()
    out_int8.write_bytes(tflite_int8)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pos", required=True, help="Positive folder (wakeword wavs)")
    parser.add_argument("--neg", required=True, help="Negative folder (non-wake wavs)")
    parser.add_argument("--out", default="models", help="Output folder for models")
    parser.add_argument("--name", default="hey_magis_v2", help="Run name/prefix")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    set_seeds(SEED)

    paths = Paths(
        pos_dir=Path(args.pos),
        neg_dir=Path(args.neg),
        out_dir=Path(args.out),
        run_name=args.name
    )
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    pos_files = list_wavs(paths.pos_dir)
    neg_files = list_wavs(paths.neg_dir)

    if len(pos_files) < 20:
        raise RuntimeError(f"Too few positives found: {len(pos_files)} in {paths.pos_dir}")
    if len(neg_files) < 20:
        raise RuntimeError(f"Too few negatives found: {len(neg_files)} in {paths.neg_dir}")

    print("\n" + "=" * 70)
    print("HEY MAGIS TRAINING v2")
    print("=" * 70)
    print(f"Positives: {len(pos_files)} from {paths.pos_dir}")
    print(f"Negatives: {len(neg_files)} from {paths.neg_dir}")
    print(f"Output:    {paths.out_dir.resolve()}")
    print(f"Run name:  {paths.run_name}")
    print("=" * 70 + "\n")

    # Split
    random.shuffle(pos_files)
    random.shuffle(neg_files)

    n_pos_val = int(len(pos_files) * VAL_SPLIT)
    n_neg_val = int(len(neg_files) * VAL_SPLIT)

    pos_val = pos_files[:n_pos_val]
    pos_train = pos_files[n_pos_val:]

    neg_val = neg_files[:n_neg_val]
    neg_train = neg_files[n_neg_val:]

    train_ds = make_dataset(pos_train, neg_train, args.batch, training=True)
    val_ds   = make_dataset(pos_val,   neg_val,   args.batch, training=False)

    # Determine input shape from one batch
    for xb, yb in train_ds.take(1):
        input_shape = tuple(xb.shape[1:])
    print(f"Feature shape: {input_shape}  (frames, mel, channels)\n")

    model = build_model(input_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=3, min_lr=1e-5),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Eval + confusion
    y_true = []
    y_prob = []
    for xb, yb in val_ds:
        p = model.predict(xb, verbose=0).reshape(-1)
        y_true.extend(yb.numpy().reshape(-1).tolist())
        y_prob.extend(p.tolist())

    y_true = np.array(y_true, dtype=np.float32)
    y_prob = np.array(y_prob, dtype=np.float32)

    tp, tn, fp, fn = confusion_counts(y_true, y_prob, thresh=0.5)
    print("\n" + "=" * 70)
    print("VALIDATION @ threshold=0.50")
    print("=" * 70)
    print(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    if (tp + fn) > 0:
        print(f"Recall (TPR): {tp/(tp+fn):.3f}")
    if (tn + fp) > 0:
        print(f"Specificity (TNR): {tn/(tn+fp):.3f}")
    if (tp + fp) > 0:
        print(f"Precision: {tp/(tp+fp):.3f}")
    print("=" * 70 + "\n")

    # Save Keras
    out_h5 = paths.out_dir / f"{paths.run_name}_keras.h5"
    model.save(out_h5)
    print(f"Saved Keras model: {out_h5}")

    # Export TFLite
    out_float = paths.out_dir / f"{paths.run_name}_float32.tflite"
    out_int8  = paths.out_dir / f"{paths.run_name}_int8.tflite"
    export_tflite(model, out_float, out_int8, rep_ds=train_ds)
    print(f"Saved TFLite float32: {out_float}")
    if EXPORT_INT8:
        print(f"Saved TFLite int8:    {out_int8}")

    # Metadata
    meta = {
        "name": paths.run_name,
        "sample_rate": SAMPLE_RATE,
        "clip_samples": CLIP_SAMPLES,
        "feature": {
            "type": "log_mel_spectrogram",
            "frame_length": FRAME_LENGTH,
            "frame_step": FRAME_STEP,
            "fft_length": FFT_LENGTH,
            "num_mel_bins": NUM_MEL_BINS,
            "lower_hz": LOWER_HZ,
            "upper_hz": UPPER_HZ,
            "normalization": "per_utterance_mean_std"
        },
        "labels": {"0": "not_wakeword", "1": "wakeword"},
        "recommended_thresholds": {
            "start": 0.50,
            "notes": "Tune threshold using on-device false positives/negatives."
        },
        "train": {
            "pos_dir": str(paths.pos_dir),
            "neg_dir": str(paths.neg_dir),
            "val_split": VAL_SPLIT,
            "batch_size": args.batch
        }
    }
    out_meta = paths.out_dir / f"{paths.run_name}_metadata.json"
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"Saved metadata: {out_meta}")

    print("\n✅ Training + export complete.")


if __name__ == "__main__":
    # Reduce TF noise
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()

#!/usr/bin/env python3
"""
Wake-Word Trainer v6 — MagisAI "Hey Magis"
============================================
Key improvements over v4/v5:
  1. Augmented positives  (454 -> 5 000+)
  2. Focal loss           (gamma=2, alpha=0.75)
  3. Label smoothing      (eps=0.05)
  4. SpecAugment          (time + freq masking during training)
  5. AUC-PR primary metric
  6. Hard-neg fraction    raised to 60 %
  7. Phonetic confusers   in hard-negative pool
  8. Expanded noise pool  (6 -> 23 files)
  9. SE attention         optional squeeze-and-excitation
 10. 60 training epochs

Constraints preserved:
  Input shape [1, 148, 40, 1], output single sigmoid [0-1], TFLite-compatible.

Usage:
    python new/train.py
    python new/train.py --no-se --epochs 80
"""

import sys, os, random, hashlib, argparse, datetime
from pathlib import Path
from typing import List, Dict

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PROJECT_ROOT, DATA_DIR, MODELS_DIR, NEW_MODEL_DIR,
    POS_TRIMMED_DIR, POS_FRIENDS_DIR, POS_AUGMENTED_DIR,
    NEG_MANIFEST, HARD_NEG_MANIFEST, HARD_NEG_TOP_DIR,
    HARD_NEG_MINED_MANIFEST, PHONETIC_NEG_DIR,
    BACKGROUND_NOISE_DIR, EXPANDED_NOISE_DIR,
    SAMPLE_RATE, NUM_SAMPLES, CLIP_SECONDS,
    FRAME_LENGTH, FRAME_STEP, FFT_LENGTH,
    NUM_MELS, LOWER_HZ, UPPER_HZ, TARGET_FRAMES, INPUT_SHAPE,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, MIN_LR, PATIENCE,
    LR_PATIENCE, LR_FACTOR, NEG_MULTIPLIER, HARD_NEG_FRACTION,
    FOCAL_GAMMA, FOCAL_ALPHA, LABEL_SMOOTHING,
    USE_SE_ATTENTION, DROPOUT_RATE,
    MAX_GAIN_DB, NOISE_MIX_PROB, NOISE_MIX_SNR_DB, TIME_SHIFT_MAX,
    SPEC_AUGMENT_PROB, TIME_MASK_PARAM, FREQ_MASK_PARAM,
    NUM_TIME_MASKS, NUM_FREQ_MASKS,
    THRESHOLDS_TO_TEST, RUN_NAME,
)

# ═══════════════════════════════════════════════════════════════════
# SEEDING
# ═══════════════════════════════════════════════════════════════════

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ═══════════════════════════════════════════════════════════════════
# PATH / MANIFEST HELPERS
# ═══════════════════════════════════════════════════════════════════

def read_manifest(path: Path) -> List[Path]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        p = Path(ln)
        if not p.is_absolute():
            c1 = PROJECT_ROOT / p
            c2 = PROJECT_ROOT / "data" / p
            p = c1 if c1.exists() else (c2 if c2.exists() else path.parent / p)
        out.append(p)
    return out


def list_wavs(folder: Path) -> List[Path]:
    return sorted(folder.glob("*.wav")) if folder.exists() else []


def list_wavs_recursive(folder: Path) -> List[Path]:
    return sorted(folder.rglob("*.wav")) if folder.exists() else []

# ═══════════════════════════════════════════════════════════════════
# AUDIO LOADING  (TF ops — runs inside tf.data pipeline)
# ═══════════════════════════════════════════════════════════════════

def load_wav_mono_16k(path: tf.Tensor) -> tf.Tensor:
    raw = tf.io.read_file(path)
    wav, sr = tf.audio.decode_wav(raw, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    wav = wav[:NUM_SAMPLES]
    pad = NUM_SAMPLES - tf.shape(wav)[0]
    return tf.cond(pad > 0, lambda: tf.pad(wav, [[0, pad]]), lambda: wav)

# ═══════════════════════════════════════════════════════════════════
# ONLINE AUGMENTATION  (TF ops)
# ═══════════════════════════════════════════════════════════════════

def random_gain(wav: tf.Tensor) -> tf.Tensor:
    gain_db = tf.random.uniform([], -MAX_GAIN_DB, MAX_GAIN_DB)
    return tf.clip_by_value(wav * tf.pow(10.0, gain_db / 20.0), -1.0, 1.0)


def random_time_shift(wav: tf.Tensor) -> tf.Tensor:
    shift = tf.random.uniform([], -TIME_SHIFT_MAX, TIME_SHIFT_MAX + 1, dtype=tf.int32)
    return tf.roll(wav, shift=shift, axis=0)


def maybe_mix_noise(wav: tf.Tensor, noise_paths: List[Path]) -> tf.Tensor:
    if not noise_paths:
        return wav
    r = tf.random.uniform([])

    def _mix():
        idx = tf.random.uniform([], 0, len(noise_paths), dtype=tf.int32)
        npath = tf.constant([str(p) for p in noise_paths])[idx]
        n = load_wav_mono_16k(npath)
        n = tf.roll(n, shift=tf.random.uniform([], 0, NUM_SAMPLES, dtype=tf.int32), axis=0)
        snr = tf.random.uniform([], NOISE_MIX_SNR_DB[0], NOISE_MIX_SNR_DB[1])
        sig_rms = tf.sqrt(tf.reduce_mean(tf.square(wav)) + 1e-8)
        noi_rms = tf.sqrt(tf.reduce_mean(tf.square(n)) + 1e-8)
        n = n * (sig_rms / tf.pow(10.0, snr / 20.0) / (noi_rms + 1e-8))
        return tf.clip_by_value(wav + n, -1.0, 1.0)

    return tf.cond(r < NOISE_MIX_PROB, _mix, lambda: wav)

# ═══════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION  (must match Dart pipeline exactly)
# ═══════════════════════════════════════════════════════════════════

def wav_to_mel(wav: tf.Tensor) -> tf.Tensor:
    stft = tf.signal.stft(wav, frame_length=FRAME_LENGTH,
                          frame_step=FRAME_STEP, fft_length=FFT_LENGTH,
                          window_fn=tf.signal.hann_window, pad_end=False)
    power = tf.square(tf.abs(stft))
    mel_w = tf.signal.linear_to_mel_weight_matrix(
        NUM_MELS, FFT_LENGTH // 2 + 1, SAMPLE_RATE, LOWER_HZ, UPPER_HZ)
    log_mel = tf.math.log(tf.maximum(tf.matmul(power, mel_w), 1e-10))
    mean = tf.reduce_mean(log_mel)
    std  = tf.math.reduce_std(log_mel) + 1e-6
    log_mel = (log_mel - mean) / std
    log_mel = log_mel[:TARGET_FRAMES, :]
    pad = TARGET_FRAMES - tf.shape(log_mel)[0]
    log_mel = tf.cond(pad > 0,
                      lambda: tf.pad(log_mel, [[0, pad], [0, 0]]),
                      lambda: log_mel)
    return tf.expand_dims(log_mel, -1)                    # [148, 40, 1]

# ═══════════════════════════════════════════════════════════════════
# SPEC-AUGMENT  (time + frequency masking)
# ═══════════════════════════════════════════════════════════════════

def _apply_masks(mel: tf.Tensor) -> tf.Tensor:
    for _ in range(NUM_TIME_MASKS):
        t  = tf.random.uniform([], 1, TIME_MASK_PARAM + 1, dtype=tf.int32)
        t0 = tf.random.uniform([], 0, TARGET_FRAMES - TIME_MASK_PARAM, dtype=tf.int32)
        mask = 1.0 - tf.cast(
            tf.logical_and(tf.range(TARGET_FRAMES) >= t0,
                           tf.range(TARGET_FRAMES) < t0 + t), tf.float32)
        mel = mel * tf.reshape(mask, [TARGET_FRAMES, 1, 1])

    for _ in range(NUM_FREQ_MASKS):
        f  = tf.random.uniform([], 1, FREQ_MASK_PARAM + 1, dtype=tf.int32)
        f0 = tf.random.uniform([], 0, NUM_MELS - FREQ_MASK_PARAM, dtype=tf.int32)
        mask = 1.0 - tf.cast(
            tf.logical_and(tf.range(NUM_MELS) >= f0,
                           tf.range(NUM_MELS) < f0 + f), tf.float32)
        mel = mel * tf.reshape(mask, [1, NUM_MELS, 1])
    return mel


def spec_augment(mel: tf.Tensor) -> tf.Tensor:
    return tf.cond(tf.random.uniform([]) < SPEC_AUGMENT_PROB,
                   lambda: _apply_masks(mel), lambda: mel)

# ═══════════════════════════════════════════════════════════════════
# DATASET CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════

def make_train_example(path, label, noise_paths):
    wav = load_wav_mono_16k(path)
    wav = random_gain(wav)
    wav = random_time_shift(wav)
    wav = maybe_mix_noise(wav, noise_paths)
    feat = wav_to_mel(wav)
    feat = spec_augment(feat)
    return feat, label


def make_val_example(path, label):
    wav = load_wav_mono_16k(path)
    feat = wav_to_mel(wav)
    return feat, label


def build_train_dataset(paths, labels, noise_paths):
    p = tf.constant([str(x) for x in paths])
    y = tf.constant(labels, dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((p, y))
    ds = ds.shuffle(min(8000, len(paths)), reshuffle_each_iteration=True)
    ds = ds.map(
        lambda pp, yy: make_train_example(pp, yy, noise_paths),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_val_dataset(paths, labels):
    p = tf.constant([str(x) for x in paths])
    y = tf.constant(labels, dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((p, y))
    ds = ds.map(
        lambda pp, yy: make_val_example(pp, yy),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def sample_negatives(neg_paths, hard_neg_paths, num_needed):
    hard_n = int(num_needed * HARD_NEG_FRACTION)
    reg_n  = max(0, num_needed - hard_n)
    sampled = []
    if hard_neg_paths and hard_n > 0:
        sampled.extend(random.choices(hard_neg_paths, k=hard_n))
    if neg_paths and reg_n > 0:
        sampled.extend(random.choices(neg_paths, k=reg_n))
    random.shuffle(sampled)
    return sampled

# ═══════════════════════════════════════════════════════════════════
# FOCAL LOSS
# ═══════════════════════════════════════════════════════════════════

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA,
                 label_smoothing=LABEL_SMOOTHING, **kw):
        super().__init__(**kw)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1, 1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1, 1]), tf.float32)
        if self.label_smoothing > 0:
            y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -(y_true * tf.math.log(y_pred) +
                (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, self.gamma) * bce)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(gamma=self.gamma, alpha=self.alpha,
                   label_smoothing=self.label_smoothing)
        return cfg

# ═══════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE  (DS-CNN with optional SE attention)
# ═══════════════════════════════════════════════════════════════════

def _ds_block(x, filters, stride):
    x = tf.keras.layers.DepthwiseConv2D(
        (3, 3), strides=(stride, stride), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, (1, 1), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def _se_block(x, ratio=4):
    ch = x.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling2D()(x)
    se = tf.keras.layers.Dense(ch // ratio, activation="relu")(se)
    se = tf.keras.layers.Dense(ch, activation="sigmoid")(se)
    se = tf.keras.layers.Reshape((1, 1, ch))(se)
    return tf.keras.layers.Multiply()([x, se])


def build_model(use_se: bool = USE_SE_ATTENTION) -> tf.keras.Model:
    inp = tf.keras.layers.Input(shape=INPUT_SHAPE)

    x = tf.keras.layers.Conv2D(16, (3, 3), padding="same", use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = _ds_block(x, 24, stride=2)
    x = _ds_block(x, 32, stride=1)
    x = _ds_block(x, 48, stride=2)
    x = _ds_block(x, 64, stride=1)

    if use_se:
        x = _se_block(x, ratio=4)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inp, out, name="hey_magis_v6")

# ═══════════════════════════════════════════════════════════════════
# VALIDATION HELPERS
# ═══════════════════════════════════════════════════════════════════

def evaluate_on_val(model, val_ds):
    y_true, y_score = [], []
    for x, y in val_ds:
        y_score.extend(model.predict(x, verbose=0).flatten().tolist())
        y_true.extend(y.numpy().flatten().tolist())
    y_true  = np.array(y_true)
    y_score = np.array(y_score)
    return {"auc_roc": _auc_roc(y_true, y_score),
            "auc_pr": _auc_pr(y_true, y_score),
            "y_true": y_true, "y_score": y_score}


def _auc_roc(yt, ys):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(yt, ys))
    except Exception:
        return 0.0


def _auc_pr(yt, ys):
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(yt, ys))
    except Exception:
        return 0.0


def threshold_sweep(y_true, y_score, thresholds):
    results = {}
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = int(np.sum((y_true == 1) & (pred == 1)))
        tn = int(np.sum((y_true == 0) & (pred == 0)))
        fp = int(np.sum((y_true == 0) & (pred == 1)))
        fn = int(np.sum((y_true == 1) & (pred == 0)))
        results[t] = dict(
            TP=tp, TN=tn, FP=fp, FN=fn,
            recall=tp / max(1, tp + fn),
            specificity=tn / max(1, tn + fp),
            precision=tp / max(1, tp + fp),
            FPR=fp / max(1, fp + tn),
        )
    return results

# ═══════════════════════════════════════════════════════════════════
# SPEAKER-AWARE SPLIT
# ═══════════════════════════════════════════════════════════════════

def _friend_speaker(p: Path) -> str:
    try:
        return p.relative_to(POS_FRIENDS_DIR).parts[0]
    except Exception:
        return "unknown"


def _is_val_by_hash(stem: str, frac: float = 0.20) -> bool:
    h = int(hashlib.md5(stem.encode()).hexdigest(), 16)
    return (h % 1000) < int(frac * 1000)


def split_positives(friends, trimmed, augmented):
    speakers = sorted({_friend_speaker(p) for p in friends})
    random.shuffle(speakers)
    val_spk = set(speakers[:1]) if speakers else set()

    train_f = [p for p in friends if _friend_speaker(p) not in val_spk]
    val_f   = [p for p in friends if _friend_speaker(p) in val_spk]

    val_trim   = [p for p in trimmed if _is_val_by_hash(p.stem)]
    train_trim = [p for p in trimmed if not _is_val_by_hash(p.stem)]

    train_stems = {p.stem for p in train_f + train_trim}
    val_stems   = {p.stem for p in val_f   + val_trim}

    train_aug, val_aug = [], []
    for p in augmented:
        parts = p.stem.split("__")
        orig_stem = parts[2] if len(parts) >= 4 else p.stem
        if orig_stem in val_stems:
            val_aug.append(p)
        else:
            train_aug.append(p)

    train = train_f + train_trim + train_aug
    val   = val_f   + val_trim   + val_aug

    if len(val) < 10:
        all_orig = friends + trimmed
        val = all_orig[:max(10, int(0.2 * len(all_orig)))]

    return train, val, {
        "val_speakers": sorted(val_spk),
        "train_orig": len(train_f) + len(train_trim),
        "train_aug": len(train_aug),
        "val_orig": len(val_f) + len(val_trim),
        "val_aug": len(val_aug),
    }

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--no-se", action="store_true", help="Disable SE attention")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seeds(args.seed)
    output_dir = NEW_MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    run = f"{RUN_NAME}_{timestamp}"

    # ────── collect data ──────
    friends   = list_wavs_recursive(POS_FRIENDS_DIR)
    trimmed   = list_wavs(POS_TRIMMED_DIR)
    augmented = list_wavs(POS_AUGMENTED_DIR)

    if not friends and not trimmed:
        raise SystemExit("No positives found. Run augment_positives.py first?")

    neg_paths  = read_manifest(NEG_MANIFEST)
    hard_paths = read_manifest(HARD_NEG_MANIFEST)
    hard_top   = list_wavs_recursive(HARD_NEG_TOP_DIR)
    hard_mined = read_manifest(HARD_NEG_MINED_MANIFEST)
    phonetic   = list_wavs_recursive(PHONETIC_NEG_DIR)

    hard_all = list({str(p): p for p in hard_paths + hard_top + hard_mined + phonetic}.values())

    noise_paths = list_wavs(BACKGROUND_NOISE_DIR) + list_wavs(EXPANDED_NOISE_DIR)

    # ────── split ──────
    train_pos, val_pos, split_info = split_positives(friends, trimmed, augmented)

    print("=" * 72)
    print(f"  HEY MAGIS TRAINING  v6   ({run})")
    print("=" * 72)
    print(f"  Positives  friends={len(friends)}  trimmed={len(trimmed)}  "
          f"augmented={len(augmented)}")
    print(f"  Split      train={len(train_pos)} "
          f"(orig {split_info['train_orig']} + aug {split_info['train_aug']})  "
          f"val={len(val_pos)} "
          f"(orig {split_info['val_orig']} + aug {split_info['val_aug']})")
    print(f"  Val speakers held out: {split_info['val_speakers']}")
    print(f"  Negatives  general={len(neg_paths)}  hard={len(hard_all)}  "
          f"phonetic={len(phonetic)}")
    print(f"  Noise files: {len(noise_paths)}")
    print(f"  Focal loss:  gamma={FOCAL_GAMMA}  alpha={FOCAL_ALPHA}  "
          f"smooth={LABEL_SMOOTHING}")
    print(f"  SE attention: {not args.no_se}")
    print("=" * 72)

    # ────── fixed validation set ──────
    val_neg_n = len(val_pos) * NEG_MULTIPLIER
    val_neg   = sample_negatives(neg_paths, hard_all, val_neg_n)
    val_paths  = val_pos + val_neg
    val_labels = [1] * len(val_pos) + [0] * len(val_neg)
    val_ds = build_val_dataset(val_paths, val_labels)

    # ────── model ──────
    use_se = USE_SE_ATTENTION and (not args.no_se)
    model = build_model(use_se=use_se)
    loss_fn = FocalLoss()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.AUC(name="auc_roc"),
            tf.keras.metrics.AUC(curve="PR", name="auc_pr"),
        ],
    )
    model.summary()

    # ────── training loop (per-epoch negative resampling) ──────
    best_weights = output_dir / f"{run}_best.weights.h5"
    best_metric  = 0.0
    wait = 0
    lr_wait = 0
    current_lr = LEARNING_RATE
    history_lines = []

    for epoch in range(args.epochs):
        train_neg_n = len(train_pos) * NEG_MULTIPLIER
        train_neg   = sample_negatives(neg_paths, hard_all, train_neg_n)
        train_paths  = train_pos + train_neg
        train_labels = [1] * len(train_pos) + [0] * len(train_neg)

        train_ds = build_train_dataset(train_paths, train_labels, noise_paths)

        print(f"\n-- Epoch {epoch + 1}/{args.epochs}  "
              f"pos={len(train_pos)} neg={len(train_neg)} "
              f"(hard {HARD_NEG_FRACTION:.0%})  lr={current_lr:.2e} --")

        model.fit(train_ds, epochs=1, verbose=1)

        val_res = evaluate_on_val(model, val_ds)
        val_ap = val_res["auc_pr"]
        val_ar = val_res["auc_roc"]
        line = (f"  val_auc_pr={val_ap:.4f}  val_auc_roc={val_ar:.4f}  "
                f"best_auc_pr={best_metric:.4f}")
        print(line)
        history_lines.append(f"epoch={epoch+1:3d}  {line.strip()}")

        if val_ap > best_metric:
            best_metric = val_ap
            model.save_weights(str(best_weights))
            wait = 0
            lr_wait = 0
            print(f"  ** New best AUC-PR: {best_metric:.4f}  (weights saved)")
        else:
            wait += 1
            lr_wait += 1
            if lr_wait >= LR_PATIENCE:
                current_lr = max(current_lr * LR_FACTOR, MIN_LR)
                model.optimizer.learning_rate.assign(current_lr)
                lr_wait = 0
                print(f"  LR reduced -> {current_lr:.2e}")

        if wait >= PATIENCE:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

        if current_lr <= MIN_LR * 1.01 and epoch >= 10:
            print("  LR bottomed out — stopping.")
            break

    # ── restore best ──
    if best_weights.exists():
        model.load_weights(str(best_weights))
        print(f"\nRestored best weights (AUC-PR={best_metric:.4f})")

    # ────── export ──────
    h5_path = output_dir / f"{run}_model.h5"
    model.save(h5_path)
    print(f"Saved Keras model: {h5_path}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    tflite_path = output_dir / f"{run}_model.tflite"
    tflite_path.write_bytes(tflite_bytes)
    print(f"Saved TFLite model: {tflite_path}")

    # ────── threshold sweep on val ──────
    final = evaluate_on_val(model, val_ds)
    sweep = threshold_sweep(final["y_true"], final["y_score"], THRESHOLDS_TO_TEST)

    report = []
    report.append("=" * 72)
    report.append(f"HEY MAGIS TRAINING REPORT  v6  ({run})")
    report.append("=" * 72)
    report.append(f"AUC-PR  = {final['auc_pr']:.4f}")
    report.append(f"AUC-ROC = {final['auc_roc']:.4f}")
    report.append(f"val_pos = {len(val_pos)}   val_neg = {len(val_neg)}")
    report.append("")
    report.append("THRESHOLD SWEEP")
    report.append("-" * 72)
    report.append(f"{'thresh':>7s}  {'TP':>5s} {'TN':>5s} {'FP':>5s} {'FN':>5s}  "
                  f"{'recall':>7s} {'specif':>7s} {'prec':>7s} {'FPR':>7s}")
    for t in THRESHOLDS_TO_TEST:
        r = sweep[t]
        report.append(
            f"  {t:.2f}   {r['TP']:5d} {r['TN']:5d} {r['FP']:5d} {r['FN']:5d}  "
            f"{r['recall']:.4f}  {r['specificity']:.4f}  "
            f"{r['precision']:.4f}  {r['FPR']:.4f}")
    report.append("")
    report.append("TRAINING HISTORY (per-epoch)")
    report.append("-" * 72)
    report.extend(history_lines)

    report_path = output_dir / f"{run}_train_report.txt"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report: {report_path}")

    best_f1, best_t = 0, 0.5
    for t, r in sweep.items():
        f1 = 2 * r["precision"] * r["recall"] / max(1e-8, r["precision"] + r["recall"])
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"\nRecommended threshold (best F1={best_f1:.3f}): {best_t:.2f}")
    print("Done.")


if __name__ == "__main__":
    main()

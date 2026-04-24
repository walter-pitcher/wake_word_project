#!/usr/bin/env python3
"""
Hard-Negative Mining — MagisAI Wake Word v6
============================================
Run the current .tflite model over the *entire* negative set and
extract every sample whose score exceeds a configurable threshold
(default 0.3).  The resulting manifest is used during training with
3-5x oversampling so that the model focuses on decision-boundary samples.

Usage:
    python new/mine_hard_negatives.py --model models/hey_magis_v5_2026-03-15_model.tflite
    python new/mine_hard_negatives.py --model models/hey_magis_v5_2026-03-15_model.tflite --threshold 0.2
"""

import sys, os, argparse, csv, random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    SAMPLE_RATE, NUM_SAMPLES, FRAME_LENGTH, FRAME_STEP,
    FFT_LENGTH, NUM_MELS, LOWER_HZ, UPPER_HZ, TARGET_FRAMES,
    PROJECT_ROOT, DATA_DIR, MODELS_DIR,
    NEG_MANIFEST, HARD_NEG_MANIFEST, HARD_NEG_TOP_DIR,
    HARD_NEG_MINED_MANIFEST, HARD_NEG_SCORE_THRESHOLD,
)

# ─────────────────── feature extraction (TF ops) ─────────────────

def load_wav_mono_16k(path_str: str) -> np.ndarray:
    raw = tf.io.read_file(path_str)
    wav, sr = tf.audio.decode_wav(raw, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    wav = wav[:NUM_SAMPLES]
    pad = NUM_SAMPLES - tf.shape(wav)[0]
    wav = tf.cond(pad > 0, lambda: tf.pad(wav, [[0, pad]]), lambda: wav)
    return wav.numpy()


def wav_to_mel_np(wav: np.ndarray) -> np.ndarray:
    wav_t = tf.constant(wav, dtype=tf.float32)
    stft = tf.signal.stft(wav_t, frame_length=FRAME_LENGTH,
                          frame_step=FRAME_STEP, fft_length=FFT_LENGTH,
                          window_fn=tf.signal.hann_window, pad_end=False)
    power = tf.square(tf.abs(stft))
    mel_w = tf.signal.linear_to_mel_weight_matrix(
        NUM_MELS, FFT_LENGTH // 2 + 1, SAMPLE_RATE, LOWER_HZ, UPPER_HZ)
    mel = tf.matmul(power, mel_w)
    log_mel = tf.math.log(tf.maximum(mel, 1e-10))
    mean = tf.reduce_mean(log_mel)
    std  = tf.math.reduce_std(log_mel) + 1e-6
    log_mel = (log_mel - mean) / std
    log_mel = log_mel[:TARGET_FRAMES, :]
    pad = TARGET_FRAMES - tf.shape(log_mel)[0]
    log_mel = tf.cond(pad > 0,
                      lambda: tf.pad(log_mel, [[0, pad], [0, 0]]),
                      lambda: log_mel)
    return tf.expand_dims(log_mel, -1).numpy()          # [148, 40, 1]


# ─────────────────── manifest / path helpers ─────────────────────

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


def list_wavs_recursive(folder: Path) -> List[Path]:
    return sorted(folder.rglob("*.wav")) if folder.exists() else []


# ─────────────────────── scoring engine ──────────────────────────

def score_file(interp, in_idx, out_idx, path: Path) -> float:
    try:
        wav = load_wav_mono_16k(str(path))
        feat = wav_to_mel_np(wav)
        x = np.expand_dims(feat, 0).astype(np.float32)   # [1,148,40,1]
        interp.set_tensor(in_idx, x)
        interp.invoke()
        return float(interp.get_tensor(out_idx).flatten()[0])
    except Exception:
        return -1.0


# ─────────────────────────── main ────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=".tflite model path")
    ap.add_argument("--threshold", type=float, default=HARD_NEG_SCORE_THRESHOLD)
    ap.add_argument("--out", type=str, default=str(HARD_NEG_MINED_MANIFEST))
    ap.add_argument("--scores-csv", type=str, default="",
                    help="Optional: write all scores to CSV")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = (PROJECT_ROOT / model_path).resolve()

    print("=" * 70)
    print("HARD-NEGATIVE MINING  v6")
    print("=" * 70)
    print(f"Model:      {model_path}")
    print(f"Threshold:  {args.threshold}")

    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    in_idx  = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]
    print(f"Input shape:  {interp.get_input_details()[0]['shape']}")

    neg_paths  = read_manifest(NEG_MANIFEST)
    hard_paths = read_manifest(HARD_NEG_MANIFEST)
    hard_top   = list_wavs_recursive(HARD_NEG_TOP_DIR)
    all_neg = list({str(p): p for p in neg_paths + hard_paths + hard_top}.values())
    print(f"Total negative files to score: {len(all_neg)}")

    above = []
    all_scores: List[Tuple[str, float]] = []

    for p in tqdm(all_neg, desc="Scoring", unit="file"):
        if not p.exists():
            continue
        s = score_file(interp, in_idx, out_idx, p)
        if s < 0:
            continue
        all_scores.append((str(p), s))
        if s >= args.threshold:
            above.append((str(p), s))

    above.sort(key=lambda x: x[1], reverse=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# Hard negatives mined with threshold={args.threshold}\n")
        f.write(f"# Model: {model_path.name}\n")
        f.write(f"# Total mined: {len(above)}\n")
        for p_str, _ in above:
            f.write(p_str + "\n")

    print(f"\nMined {len(above)} hard negatives (score >= {args.threshold})")
    print(f"Manifest written: {out_path}")

    if all_scores:
        scores = np.array([s for _, s in all_scores])
        for t in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            count = int(np.sum(scores >= t))
            print(f"  score >= {t:.1f}: {count:>6d}  ({100*count/len(scores):.2f}%)")

    if args.scores_csv:
        csv_path = Path(args.scores_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "score"])
            for p_str, s in sorted(all_scores, key=lambda x: x[1], reverse=True):
                w.writerow([p_str, f"{s:.6f}"])
        print(f"Full scores CSV: {csv_path}")

    print("Done.")


if __name__ == "__main__":
    main()

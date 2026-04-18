#!/usr/bin/env python3
"""
Evaluate wake word TFLite model on multiple buckets and write per-bucket CSVs.

Buckets:
- Positives (new friends): data/positive_new_friends (recursive)
- Positives (trimmed):     data/positive_trimmed_1p5_A (non-recursive or recursive; both ok)
- Negatives (random 10k):  sampled from data/negative_manifest.txt
- Hard negatives (all):    all rows from combined hard manifest (e.g. 3061)

Outputs:
- scores_pos_newfriends.csv
- scores_pos_trimmed.csv
- scores_neg_10k.csv
- scores_hard.csv

CSV format:
path,score
<abs_or_rel_path>,<float>

Notes:
- Uses TF ops for feature extraction to match training.
- Adapts to model input shape: [1, FRAMES, MELS, 1]
"""

import argparse
import csv
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf


# -------------------------
# Feature config (match training)
# -------------------------
SAMPLE_RATE = 16000
FRAME_LENGTH = 400   # 25ms @ 16k
FRAME_STEP   = 160   # 10ms @ 16k
FFT_LENGTH   = 512
LOWER_HZ     = 80.0
UPPER_HZ     = 7600.0


# -------------------------
# Utilities
# -------------------------
def read_manifest_lines(manifest_path: Path) -> List[Path]:
    """Reads a manifest of wav paths (one per line). Returns Paths (as written)."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    lines = manifest_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: List[Path] = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        out.append(Path(ln))
    return out


def resolve_paths(paths: List[Path], project_root: Path, manifest_parent: Path) -> List[Path]:
    """
    Resolve possibly-relative paths in a robust way:
    - absolute -> keep
    - relative -> try:
        1) project_root / rel
        2) project_root / "data" / rel
        3) manifest_parent / rel
    """
    resolved: List[Path] = []
    for p in paths:
        if p.is_absolute():
            resolved.append(p)
            continue

        cand1 = project_root / p
        cand2 = project_root / "data" / p
        cand3 = manifest_parent / p

        if cand1.exists():
            resolved.append(cand1)
        elif cand2.exists():
            resolved.append(cand2)
        elif cand3.exists():
            resolved.append(cand3)
        else:
            # keep something intelligible; we’ll skip if missing later
            resolved.append(cand1)
    return resolved


def list_wavs_recursive(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*.wav") if p.is_file()])


def write_scores_csv(out_csv: Path, rows: List[Tuple[str, float]]):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "score"])
        for path_str, score in rows:
            w.writerow([path_str, f"{score:.6f}"])


# -------------------------
# Audio + features (match training)
# -------------------------
def load_wav_mono_16k(path: tf.Tensor, num_samples: int) -> tf.Tensor:
    """Load wav, enforce 16kHz mono, pad/trim to num_samples, float32 [-1,1]."""
    audio_bytes = tf.io.read_file(path)
    wav, sr = tf.audio.decode_wav(audio_bytes, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sr = tf.cast(sr, tf.int32)

    tf.debugging.assert_equal(sr, SAMPLE_RATE, message="Expected 16kHz WAV.")

    wav = wav[:num_samples]
    pad = num_samples - tf.shape(wav)[0]
    wav = tf.cond(pad > 0, lambda: tf.pad(wav, [[0, pad]]), lambda: wav)
    return wav


def wav_to_logmel(wav: tf.Tensor, num_mels: int, target_frames: int) -> tf.Tensor:
    """
    Compute log-mel with per-example normalization.
    Output shape: [target_frames, num_mels, 1]
    """
    stft = tf.signal.stft(
        wav,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )
    mag = tf.abs(stft)
    power = tf.square(mag)

    num_spectrogram_bins = FFT_LENGTH // 2 + 1
    mel_w = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mels,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=LOWER_HZ,
        upper_edge_hertz=UPPER_HZ,
    )
    mel = tf.matmul(power, mel_w)
    mel = tf.maximum(mel, 1e-10)
    log_mel = tf.math.log(mel)

    # per-example normalize
    mean = tf.reduce_mean(log_mel)
    std = tf.math.reduce_std(log_mel) + 1e-6
    log_mel = (log_mel - mean) / std

    # enforce frames
    log_mel = log_mel[:target_frames, :]
    pad = target_frames - tf.shape(log_mel)[0]
    log_mel = tf.cond(pad > 0, lambda: tf.pad(log_mel, [[0, pad], [0, 0]]), lambda: log_mel)

    log_mel = tf.expand_dims(log_mel, axis=-1)
    return log_mel


# -------------------------
# TFLite scoring
# -------------------------
def load_tflite_interpreter(model_path: Path):
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    return interp


def get_model_io_shapes(interp) -> Tuple[int, int]:
    """
    Returns (frames, mels) from input tensor shape [1, frames, mels, 1]
    """
    in_det = interp.get_input_details()[0]
    shape = in_det["shape"]
    # expect [1, F, M, 1]
    if len(shape) != 4:
        raise ValueError(f"Unexpected model input shape: {shape}")
    frames = int(shape[1])
    mels = int(shape[2])
    return frames, mels


def required_num_samples_from_frames(frames: int) -> int:
    """
    Smallest N that yields `frames` STFT frames with pad_end=False:
    frames = floor((N - frame_length)/frame_step) + 1
    => N_min = (frames - 1)*frame_step + frame_length
    """
    return (frames - 1) * FRAME_STEP + FRAME_LENGTH


@tf.function
def make_feature_tensor(path_str: tf.Tensor, num_samples: int, num_mels: int, frames: int) -> tf.Tensor:
    wav = load_wav_mono_16k(path_str, num_samples)
    feat = wav_to_logmel(wav, num_mels=num_mels, target_frames=frames)
    feat = tf.expand_dims(feat, axis=0)  # [1, F, M, 1]
    return feat


def score_files(interp, files: List[Path], num_samples: int, frames: int, mels: int, progress_every: int = 500):
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    rows: List[Tuple[str, float]] = []
    total = len(files)

    for i, p in enumerate(files, start=1):
        if not p.exists():
            continue

        x = make_feature_tensor(tf.constant(str(p)), num_samples, mels, frames).numpy().astype(np.float32)
        interp.set_tensor(in_det["index"], x)
        interp.invoke()
        y = interp.get_tensor(out_det["index"])
        score = float(y.reshape(-1)[0])
        rows.append((str(p), score))

        if progress_every and (i % progress_every == 0 or i == total):
            print(f"  {i}/{total}")

    return rows


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .tflite model")
    ap.add_argument("--pos_newfriends", required=True, help="Folder: data/positive_new_friends (recursive)")
    ap.add_argument("--pos_trimmed", required=True, help="Folder: data/positive_trimmed_1p5_A")
    ap.add_argument("--neg_manifest", required=True, help="Manifest: data/negative_manifest.txt")
    ap.add_argument("--hard_manifest", required=True, help="Manifest: combined hard negatives (e.g. 3061 lines)")
    ap.add_argument("--neg_sample", type=int, default=10000, help="How many random negatives to sample")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for negative sampling")
    ap.add_argument("--out_dir", default="models", help="Output folder for CSVs")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent

    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = (project_root / model_path).resolve()

    pos_newfriends_dir = (project_root / args.pos_newfriends).resolve()
    pos_trimmed_dir = (project_root / args.pos_trimmed).resolve()

    neg_manifest_path = (project_root / args.neg_manifest).resolve()
    hard_manifest_path = (project_root / args.hard_manifest).resolve()

    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("WAKE WORD MULTI-BUCKET EVAL (TFLITE)")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Pos (new friends): {pos_newfriends_dir}")
    print(f"Pos (trimmed):     {pos_trimmed_dir}")
    print(f"Neg manifest:      {neg_manifest_path}")
    print(f"Hard manifest:     {hard_manifest_path}")
    print(f"Neg sample:        {args.neg_sample}")
    print(f"Out dir:           {out_dir}")
    print("=" * 70)

    interp = load_tflite_interpreter(model_path)
    frames, mels = get_model_io_shapes(interp)
    num_samples = required_num_samples_from_frames(frames)

    print(f"Model expects input: [1, {frames}, {mels}, 1]")
    print(f"Derived num_samples: {num_samples} (~{num_samples/SAMPLE_RATE:.3f}s @16k)")
    print("")

    # Gather positives
    pos_new = list_wavs_recursive(pos_newfriends_dir)
    pos_trim = list_wavs_recursive(pos_trimmed_dir)  # safe even if flat

    # Load and resolve manifest paths
    neg_lines = read_manifest_lines(neg_manifest_path)
    neg_paths = resolve_paths(neg_lines, project_root, neg_manifest_path.parent)
    neg_paths = [p for p in neg_paths if p.exists()]

    hard_lines = read_manifest_lines(hard_manifest_path)
    hard_paths = resolve_paths(hard_lines, project_root, hard_manifest_path.parent)
    hard_paths = [p for p in hard_paths if p.exists()]

    # Sample negatives
    random.seed(args.seed)
    if len(neg_paths) < args.neg_sample:
        print(f"⚠️ Requested {args.neg_sample} negatives, but only {len(neg_paths)} exist. Using all.")
        neg_sampled = neg_paths
    else:
        neg_sampled = random.sample(neg_paths, args.neg_sample)

    print(f"Counts:")
    print(f"  pos_newfriends: {len(pos_new)}")
    print(f"  pos_trimmed:    {len(pos_trim)}")
    print(f"  neg_available:  {len(neg_paths)}")
    print(f"  neg_sampled:    {len(neg_sampled)}")
    print(f"  hard_all:       {len(hard_paths)}")
    print("")

    # Score each bucket
    print("[1/4] Scoring positives (new friends)...")
    rows_pos_new = score_files(interp, pos_new, num_samples, frames, mels, progress_every=200)

    print("[2/4] Scoring positives (trimmed 1p5)...")
    rows_pos_trim = score_files(interp, pos_trim, num_samples, frames, mels, progress_every=200)

    print("[3/4] Scoring negatives (random 10k)...")
    rows_neg_10k = score_files(interp, neg_sampled, num_samples, frames, mels, progress_every=500)

    print("[4/4] Scoring hard negatives (all)...")
    rows_hard = score_files(interp, hard_paths, num_samples, frames, mels, progress_every=500)

    # Write CSVs
    out_pos_new = out_dir / "scores_pos_newfriends.csv"
    out_pos_trim = out_dir / "scores_pos_trimmed.csv"
    out_neg_10k = out_dir / "scores_neg_10k.csv"
    out_hard = out_dir / "scores_hard.csv"

    write_scores_csv(out_pos_new, rows_pos_new)
    write_scores_csv(out_pos_trim, rows_pos_trim)
    write_scores_csv(out_neg_10k, rows_neg_10k)
    write_scores_csv(out_hard, rows_hard)

    print("\n✅ Wrote CSVs:")
    print(f"  {out_pos_new}")
    print(f"  {out_pos_trim}")
    print(f"  {out_neg_10k}")
    print(f"  {out_hard}")
    print("\nDone.")


if __name__ == "__main__":
    main()

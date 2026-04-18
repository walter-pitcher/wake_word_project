import argparse
import csv
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# Try tflite runtime first; fall back to TF if needed.
try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    Interpreter = None

try:
    import tensorflow as tf
except Exception:
    tf = None


# -----------------------------
# Audio / Feature Params
# -----------------------------
SR = 16000
WIN_SECONDS = 1.0  # we feed ~1s into the model
FRAME_LENGTH = 400  # 25 ms @ 16k
FRAME_STEP = 160    # 10 ms @ 16k
FFT_LENGTH = 512
NUM_MELS = 40
TARGET_FRAMES = 98  # your training printed (98,40,1)

EPS = 1e-6


@dataclass
class ScoreRow:
    path: str
    score: float


def die(msg: str, code: int = 1):
    print(f"\nERROR: {msg}\n", file=sys.stderr)
    raise SystemExit(code)


def ensure_tf():
    if tf is None:
        die("TensorFlow not found. Install it or use tflite_runtime. (pip install tensorflow)")
    return tf


def find_wavs(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.wav") if p.is_file()]


def load_wav_mono_16k(path: Path, strict_16k: bool = True) -> np.ndarray:
    """
    Loads WAV and returns float32 mono [-1,1].
    Strict mode rejects files not at 16k instead of resampling (keeps behavior consistent).
    """
    ensure_tf()
    audio_bin = tf.io.read_file(str(path))
    wav, sr = tf.audio.decode_wav(audio_bin, desired_channels=1)
    sr = int(sr.numpy())

    if strict_16k and sr != SR:
        raise ValueError(f"{path} sample rate is {sr}, expected {SR} (strict_16k=True)")

    wav = tf.squeeze(wav, axis=-1).numpy().astype(np.float32)

    # Force exact 1.0s window (pad/trim)
    target_len = int(SR * WIN_SECONDS)
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)), mode="constant")
    elif len(wav) > target_len:
        wav = wav[:target_len]

    return wav


def wav_to_logmel(wav: np.ndarray) -> np.ndarray:
    """
    Produces shape (98,40,1) log-mel spectrogram.
    Uses typical KWS setup: 25ms window, 10ms hop.
    """
    ensure_tf()
    x = tf.convert_to_tensor(wav, dtype=tf.float32)

    stft = tf.signal.stft(
        x,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )
    spec = tf.abs(stft) ** 2  # power spectrogram

    num_spec_bins = spec.shape[-1]
    mel_w = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=NUM_MELS,
        num_spectrogram_bins=num_spec_bins,
        sample_rate=SR,
        lower_edge_hertz=60.0,
        upper_edge_hertz=7800.0,
    )
    mel = tf.matmul(spec, mel_w)
    logmel = tf.math.log(mel + EPS)

    # Ensure exactly 98 frames
    frames = tf.shape(logmel)[0]
    if logmel.shape[0] is None:
        logmel = logmel[:TARGET_FRAMES, :]
    else:
        logmel = logmel[:TARGET_FRAMES, :]

    # If short (rare), pad to 98
    pad_needed = TARGET_FRAMES - tf.shape(logmel)[0]
    logmel = tf.cond(
        pad_needed > 0,
        lambda: tf.pad(logmel, [[0, pad_needed], [0, 0]]),
        lambda: logmel
    )

    # Add channel dim: (98,40,1)
    feat = tf.expand_dims(logmel, axis=-1).numpy().astype(np.float32)
    return feat


def load_tflite_interpreter(model_path: Path):
    if Interpreter is None:
        ensure_tf()
        return tf.lite.Interpreter(model_path=str(model_path))
    return Interpreter(model_path=str(model_path))


def tflite_predict(interp, feat: np.ndarray) -> float:
    """
    feat: (98,40,1) float32
    returns scalar score in [0,1] (typically sigmoid output)
    """
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    inp = feat[np.newaxis, ...]  # (1,98,40,1)

    # Handle quantized models if needed
    in_dtype = input_details[0]["dtype"]
    if in_dtype == np.uint8 or in_dtype == np.int8:
        scale, zero = input_details[0]["quantization"]
        if scale == 0:
            die("Quantized input has scale=0; cannot quantize.")
        inp_q = np.round(inp / scale + zero).astype(in_dtype)
        inp = inp_q
    else:
        inp = inp.astype(in_dtype)

    interp.set_tensor(input_details[0]["index"], inp)
    interp.invoke()

    out = interp.get_tensor(output_details[0]["index"])
    score = float(np.squeeze(out))

    # If quantized output, dequantize
    out_dtype = output_details[0]["dtype"]
    if out_dtype == np.uint8 or out_dtype == np.int8:
        scale, zero = output_details[0]["quantization"]
        score = (score - zero) * scale

    return score


def write_scores_csv(rows: List[ScoreRow], out_csv: Path):
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "score"])
        for r in rows:
            w.writerow([r.path, f"{r.score:.6f}"])


def sweep_thresholds(pos_scores: np.ndarray, neg_scores: np.ndarray,
                     far_cap: float = 0.001) -> Tuple[List[dict], dict]:
    """
    far_cap: max FAR allowed (e.g. 0.001 = 0.1%)
    Returns: (list of rows for csv), (best row)
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    results = []
    best = None

    P = len(pos_scores)
    N = len(neg_scores)

    for t in thresholds:
        tp = int(np.sum(pos_scores >= t))
        fn = P - tp
        fp = int(np.sum(neg_scores >= t))
        tn = N - fp

        recall = tp / P if P else 0.0
        far = fp / N if N else 0.0  # false accept rate on negatives
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        frr = fn / P if P else 0.0

        row = {
            "threshold": float(t),
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "recall": recall,
            "precision": precision,
            "F1": f1,
            "FAR": far,
            "FRR": frr,
        }
        results.append(row)

        # Best = highest F1 subject to FAR cap; tie-breaker lower FRR then higher threshold
        if far <= far_cap:
            if best is None:
                best = row
            else:
                if row["F1"] > best["F1"] + 1e-9:
                    best = row
                elif abs(row["F1"] - best["F1"]) < 1e-9:
                    if row["FRR"] < best["FRR"] - 1e-9:
                        best = row
                    elif abs(row["FRR"] - best["FRR"]) < 1e-9 and row["threshold"] > best["threshold"]:
                        best = row

    # If nothing meets FAR cap, pick lowest FAR achievable with best recall
    if best is None and results:
        # sort by FAR asc, recall desc
        best = sorted(results, key=lambda r: (r["FAR"], -r["recall"], -r["threshold"]))[0]

    return results, best


def write_thresholds_csv(rows: List[dict], out_csv: Path):
    keys = ["threshold", "TP", "FN", "FP", "TN", "recall", "precision", "F1", "FAR", "FRR"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow([r[k] if not isinstance(r[k], float) else f"{r[k]:.6f}" for k in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"models\hey_magis_v3_model.tflite")
    ap.add_argument("--pos_dir", default=r"data\positive_trimmed_1p2_A")
    ap.add_argument("--neg_dir", default=r"data\negative")
    ap.add_argument("--neg_sample", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--far_cap", type=float, default=0.001, help="Max FAR allowed for choosing threshold (0.001=0.1%)")
    ap.add_argument("--strict_16k", action="store_true", help="Reject non-16k wavs instead of resampling")
    ap.add_argument("--top_fp", type=int, default=50, help="List top-N highest scoring negatives")
    ap.add_argument("--out_prefix", default=r"models\eval_v3")
    args = ap.parse_args()

    project_root = Path.cwd()
    model_path = (project_root / args.model).resolve()
    pos_dir = (project_root / args.pos_dir).resolve()
    neg_dir = (project_root / args.neg_dir).resolve()

    if not model_path.exists():
        die(f"Model not found: {model_path}")
    if not pos_dir.exists():
        die(f"Positive dir not found: {pos_dir}")
    if not neg_dir.exists():
        die(f"Negative dir not found: {neg_dir}")

    # Collect files
    pos_files = sorted(find_wavs(pos_dir))
    neg_files_all = find_wavs(neg_dir)

    if len(pos_files) == 0:
        die(f"No positive wavs found in {pos_dir}")
    if len(neg_files_all) == 0:
        die(f"No negative wavs found in {neg_dir}")

    random.seed(args.seed)
    if args.neg_sample > 0 and len(neg_files_all) > args.neg_sample:
        neg_files = random.sample(neg_files_all, args.neg_sample)
    else:
        neg_files = neg_files_all

    print("======================================================================")
    print("WAKE WORD TFLITE EVAL")
    print("======================================================================")
    print(f"Model:      {model_path}")
    print(f"Positives:  {len(pos_files)} from {pos_dir}")
    print(f"Negatives:  {len(neg_files)} sampled from {neg_dir} (total available: {len(neg_files_all)})")
    print(f"Strict 16k: {bool(args.strict_16k)}")
    print(f"FAR cap:    {args.far_cap} (e.g. 0.001 = 0.1%)")
    print("======================================================================")

    # Interpreter
    interp = load_tflite_interpreter(model_path)
    interp.allocate_tensors()

    # Score positives
    pos_rows: List[ScoreRow] = []
    bad_pos = 0
    for i, p in enumerate(pos_files, 1):
        try:
            wav = load_wav_mono_16k(p, strict_16k=args.strict_16k)
            feat = wav_to_logmel(wav)
            s = tflite_predict(interp, feat)
            pos_rows.append(ScoreRow(str(p), s))
        except Exception as e:
            bad_pos += 1
        if i % 25 == 0 or i == len(pos_files):
            print(f"[POS] {i}/{len(pos_files)}")

    # Score negatives
    neg_rows: List[ScoreRow] = []
    bad_neg = 0
    for i, p in enumerate(neg_files, 1):
        try:
            wav = load_wav_mono_16k(p, strict_16k=args.strict_16k)
            feat = wav_to_logmel(wav)
            s = tflite_predict(interp, feat)
            neg_rows.append(ScoreRow(str(p), s))
        except Exception:
            bad_neg += 1
        if i % 500 == 0 or i == len(neg_files):
            print(f"[NEG] {i}/{len(neg_files)}")

    if len(pos_rows) == 0:
        die("All positives failed to load/score. Check audio format / strict_16k.")
    if len(neg_rows) == 0:
        die("All negatives failed to load/score. Check audio format / strict_16k.")

    pos_scores = np.array([r.score for r in pos_rows], dtype=np.float32)
    neg_scores = np.array([r.score for r in neg_rows], dtype=np.float32)

    # Sweep thresholds
    thresh_rows, best = sweep_thresholds(pos_scores, neg_scores, far_cap=args.far_cap)

    out_prefix = (project_root / args.out_prefix).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    scores_pos_csv = Path(str(out_prefix) + "_scores_pos.csv")
    scores_neg_csv = Path(str(out_prefix) + "_scores_neg.csv")
    thresholds_csv = Path(str(out_prefix) + "_thresholds.csv")
    summary_txt = Path(str(out_prefix) + "_summary.txt")

    write_scores_csv(pos_rows, scores_pos_csv)
    write_scores_csv(neg_rows, scores_neg_csv)
    write_thresholds_csv(thresh_rows, thresholds_csv)

    # Top false positives (highest scoring negatives)
    top_fp = sorted(neg_rows, key=lambda r: r.score, reverse=True)[:max(1, args.top_fp)]

    def pct(x): return f"{x*100:.3f}%"

    with summary_txt.open("w", encoding="utf-8") as f:
        f.write("======================================================================\n")
        f.write("WAKE WORD TFLITE EVAL SUMMARY\n")
        f.write("======================================================================\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Positives scored: {len(pos_rows)} (failed: {bad_pos})\n")
        f.write(f"Negatives scored: {len(neg_rows)} (failed: {bad_neg})\n")
        f.write(f"Negatives available total: {len(neg_files_all)}\n")
        f.write(f"Negatives sampled: {len(neg_files)}\n")
        f.write(f"FAR cap used for selection: {args.far_cap}\n\n")

        f.write("Score distribution:\n")
        f.write(f"  Pos score min/mean/p95/max: {pos_scores.min():.4f} / {pos_scores.mean():.4f} / {np.percentile(pos_scores,95):.4f} / {pos_scores.max():.4f}\n")
        f.write(f"  Neg score min/mean/p99/max: {neg_scores.min():.4f} / {neg_scores.mean():.4f} / {np.percentile(neg_scores,99):.4f} / {neg_scores.max():.4f}\n\n")

        f.write("Recommended threshold (best F1 under FAR cap; tie-breaks lower FRR, then higher threshold):\n")
        f.write(f"  threshold={best['threshold']:.2f}\n")
        f.write(f"  FAR={pct(best['FAR'])}  FRR={pct(best['FRR'])}\n")
        f.write(f"  precision={best['precision']:.4f}  recall={best['recall']:.4f}  F1={best['F1']:.4f}\n")
        f.write(f"  TP={best['TP']} FN={best['FN']} FP={best['FP']} TN={best['TN']}\n\n")

        f.write("Top scoring negatives (inspect these first; these are your most dangerous false triggers):\n")
        for r in top_fp:
            f.write(f"  {r.score:.6f}  {r.path}\n")

        f.write("\nFiles written:\n")
        f.write(f"  {scores_pos_csv}\n")
        f.write(f"  {scores_neg_csv}\n")
        f.write(f"  {thresholds_csv}\n")
        f.write(f"  {summary_txt}\n")

    print("\n======================================================================")
    print("DONE")
    print("======================================================================")
    print(f"✅ Wrote: {scores_pos_csv}")
    print(f"✅ Wrote: {scores_neg_csv}")
    print(f"✅ Wrote: {thresholds_csv}")
    print(f"✅ Wrote: {summary_txt}")
    print(f"\nRecommended threshold: {best['threshold']:.2f}  (FAR={best['FAR']*100:.3f}%, FRR={best['FRR']*100:.3f}%)")
    print("Next: open the summary and listen to the top-scoring negatives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

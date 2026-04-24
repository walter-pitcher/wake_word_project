#!/usr/bin/env python3
"""
Comprehensive Evaluation — MagisAI Wake Word v6
=================================================
Produces a full validation report for a .tflite model:
  - Threshold sweep  (FAR, FRR, Recall, Specificity, Precision)
  - FAR per hour of audio
  - DET curve  (saved as PNG)
  - Precision-Recall and ROC curves  (saved as PNG)
  - SNR-segmented recall  (clean / 10 dB / 5 dB / 0 dB)
  - Posterior-smoothing simulation  (1-5 consecutive hits)
  - Recommended deployment threshold  (EER-based and F1-based)

Usage:
    python new/evaluate.py --model models/hey_magis_v6_2026-03-18_model.tflite
    python new/evaluate.py --model models/hey_magis_v6_2026-03-18_model.tflite --neg-sample 20000
"""

import sys, os, argparse, csv, random, datetime
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PROJECT_ROOT, DATA_DIR, MODELS_DIR,
    SAMPLE_RATE, NUM_SAMPLES, CLIP_SECONDS,
    FRAME_LENGTH, FRAME_STEP, FFT_LENGTH,
    NUM_MELS, LOWER_HZ, UPPER_HZ, TARGET_FRAMES,
    POS_FRIENDS_DIR, POS_TRIMMED_DIR,
    NEG_MANIFEST, HARD_NEG_MANIFEST, HARD_NEG_TOP_DIR,
    HARD_NEG_MINED_MANIFEST,
    BACKGROUND_NOISE_DIR, EXPANDED_NOISE_DIR,
    THRESHOLDS_TO_TEST, EVAL_SNR_LEVELS,
    POSTERIOR_SMOOTHING_COUNTS,
)

# ═══════════════════════════════════════════════════════════════════
# AUDIO and FEATURES
# ═══════════════════════════════════════════════════════════════════

def load_wav_np(path: str) -> np.ndarray:
    raw = tf.io.read_file(path)
    wav, _ = tf.audio.decode_wav(raw, desired_channels=1)
    wav = tf.squeeze(wav, -1).numpy()
    if len(wav) > NUM_SAMPLES:
        wav = wav[:NUM_SAMPLES]
    elif len(wav) < NUM_SAMPLES:
        wav = np.pad(wav, (0, NUM_SAMPLES - len(wav)))
    return wav.astype(np.float32)


def wav_to_feature(wav: np.ndarray) -> np.ndarray:
    t = tf.constant(wav, dtype=tf.float32)
    stft = tf.signal.stft(t, frame_length=FRAME_LENGTH,
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
    if pad > 0:
        log_mel = tf.pad(log_mel, [[0, pad], [0, 0]])
    feat = tf.expand_dims(tf.expand_dims(log_mel, 0), -1)
    return feat.numpy().astype(np.float32)


def add_noise_np(audio, noise, snr_db):
    sig_rms = np.sqrt(np.mean(audio ** 2) + 1e-8)
    noi_rms = np.sqrt(np.mean(noise ** 2) + 1e-8)
    target  = sig_rms / (10.0 ** (snr_db / 20.0))
    return np.clip(audio + noise * (target / (noi_rms + 1e-8)), -1.0, 1.0)

# ═══════════════════════════════════════════════════════════════════
# MANIFEST / PATH HELPERS
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


def list_wavs_recursive(folder: Path) -> List[Path]:
    return sorted(folder.rglob("*.wav")) if folder.exists() else []

# ═══════════════════════════════════════════════════════════════════
# TFLITE SCORING
# ═══════════════════════════════════════════════════════════════════

def load_interp(model_path):
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    return interp


def score_wav(interp, in_idx, out_idx, wav):
    x = wav_to_feature(wav)
    interp.set_tensor(in_idx, x)
    interp.invoke()
    return float(interp.get_tensor(out_idx).flatten()[0])


def score_file(interp, in_idx, out_idx, path):
    try:
        wav = load_wav_np(str(path))
        return score_wav(interp, in_idx, out_idx, wav)
    except Exception:
        return -1.0

# ═══════════════════════════════════════════════════════════════════
# METRIC COMPUTATIONS
# ═══════════════════════════════════════════════════════════════════

def threshold_sweep(y_true, y_score, thresholds):
    results = {}
    n_neg = int(np.sum(y_true == 0))
    neg_hours = max(1e-8, n_neg * CLIP_SECONDS / 3600.0)
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = int(np.sum((y_true == 1) & (pred == 1)))
        tn = int(np.sum((y_true == 0) & (pred == 0)))
        fp = int(np.sum((y_true == 0) & (pred == 1)))
        fn = int(np.sum((y_true == 1) & (pred == 0)))
        results[t] = dict(
            TP=tp, TN=tn, FP=fp, FN=fn,
            recall=tp / max(1, tp + fn),
            FRR=fn / max(1, tp + fn),
            specificity=tn / max(1, tn + fp),
            precision=tp / max(1, tp + fp),
            FAR=fp / max(1, fp + tn),
            FAR_per_hour=fp / neg_hours,
        )
    return results


def find_eer(y_true, y_score):
    thresholds = np.linspace(0, 1, 1001)
    best_diff, eer_t = float("inf"), 0.5
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = np.sum((y_true == 1) & (pred == 1))
        fn = np.sum((y_true == 1) & (pred == 0))
        fp = np.sum((y_true == 0) & (pred == 1))
        tn = np.sum((y_true == 0) & (pred == 0))
        frr = fn / max(1, tp + fn)
        far = fp / max(1, fp + tn)
        diff = abs(frr - far)
        if diff < best_diff:
            best_diff = diff
            eer_t = t
    return eer_t

# ═══════════════════════════════════════════════════════════════════
# SNR-SEGMENTED EVALUATION
# ═══════════════════════════════════════════════════════════════════

def load_noise_pool():
    pool = []
    for d in [BACKGROUND_NOISE_DIR, EXPANDED_NOISE_DIR]:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.wav")):
            try:
                pool.append(load_wav_np(str(f)))
            except Exception:
                pass
    return pool


def noise_segment(pool, length):
    if not pool:
        return np.random.randn(length).astype(np.float32) * 0.005
    n = pool[np.random.randint(len(pool))]
    if len(n) <= length:
        n = np.tile(n, (length // len(n)) + 1)
    start = np.random.randint(0, len(n) - length)
    return n[start:start + length]


def eval_snr_levels(interp, in_idx, out_idx, pos_files, threshold, noise_pool):
    results = {}
    for snr in EVAL_SNR_LEVELS:
        label = "clean" if snr == float("inf") else f"{snr:.0f}dB"
        hits, total = 0, 0
        for p in pos_files:
            try:
                wav = load_wav_np(str(p))
            except Exception:
                continue
            if snr != float("inf"):
                seg = noise_segment(noise_pool, NUM_SAMPLES)
                wav = add_noise_np(wav, seg, snr)
            s = score_wav(interp, in_idx, out_idx, wav)
            if s >= threshold:
                hits += 1
            total += 1
        results[label] = hits / max(1, total)
    return results

# ═══════════════════════════════════════════════════════════════════
# POSTERIOR-SMOOTHING SIMULATION
# ═══════════════════════════════════════════════════════════════════

def eval_posterior_smoothing(interp, in_idx, out_idx,
                             pos_files, threshold, noise_pool, n_shifts=5):
    results = {}
    for req_hits in POSTERIOR_SMOOTHING_COUNTS:
        passed, total = 0, 0
        for p in pos_files:
            try:
                wav = load_wav_np(str(p))
            except Exception:
                continue
            total += 1
            consec = 0
            triggered = False
            for shift_i in range(n_shifts):
                shifted = np.roll(wav, shift_i * FRAME_STEP)
                s = score_wav(interp, in_idx, out_idx, shifted)
                if s >= threshold:
                    consec += 1
                    if consec >= req_hits:
                        triggered = True
                        break
                else:
                    consec = 0
            if triggered:
                passed += 1
        results[req_hits] = passed / max(1, total)
    return results

# ═══════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════

def save_det_curve(y_true, y_score, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping DET curve.")
        return

    thresholds = np.linspace(0, 1, 501)
    fars, frrs = [], []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = np.sum((y_true == 1) & (pred == 1))
        fn = np.sum((y_true == 1) & (pred == 0))
        fp = np.sum((y_true == 0) & (pred == 1))
        tn = np.sum((y_true == 0) & (pred == 0))
        fars.append(fp / max(1, fp + tn))
        frrs.append(fn / max(1, tp + fn))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fars, frrs, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Accept Rate (FAR)")
    ax.set_ylabel("False Reject Rate (FRR)")
    ax.set_title("DET Curve — Hey Magis v6")
    ax.set_xlim(0, 0.15)
    ax.set_ylim(0, 0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  DET curve -> {out_path}")


def save_pr_roc_curves(y_true, y_score, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)
    total_pos = np.sum(y_true)
    total_neg = np.sum(1 - y_true)

    prec = tp_cum / (tp_cum + fp_cum + 1e-8)
    rec  = tp_cum / (total_pos + 1e-8)
    fpr  = fp_cum / (total_neg + 1e-8)
    tpr  = rec

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(rec, prec, linewidth=2, color="tab:blue")
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision-Recall Curve")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.02)
    ax1.grid(True, alpha=0.3)

    ax2.plot(fpr, tpr, linewidth=2, color="tab:orange")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.02)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  PR / ROC curves -> {out_path}")

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=".tflite model path")
    ap.add_argument("--neg-sample", type=int, default=10000)
    ap.add_argument("--pos-sample", type=int, default=0,
                    help="Subsample positives for SNR/smoothing eval (0=all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="models")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = (PROJECT_ROOT / model_path).resolve()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  COMPREHENSIVE EVALUATION — Hey Magis v6")
    print("=" * 72)
    print(f"  Model: {model_path}")

    interp = load_interp(model_path)
    in_det  = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    in_idx, out_idx = in_det["index"], out_det["index"]
    print(f"  Input shape: {in_det['shape']}")

    # ── gather files ──
    pos_friends = list_wavs_recursive(POS_FRIENDS_DIR)
    pos_trimmed = list_wavs_recursive(POS_TRIMMED_DIR)
    pos_all = pos_friends + pos_trimmed

    neg_all = read_manifest(NEG_MANIFEST)
    neg_all = [p for p in neg_all if p.exists()]
    neg_sampled = random.sample(neg_all, min(args.neg_sample, len(neg_all)))

    hard_paths = read_manifest(HARD_NEG_MANIFEST)
    hard_top   = list_wavs_recursive(HARD_NEG_TOP_DIR)
    hard_mined = read_manifest(HARD_NEG_MINED_MANIFEST)
    hard_all   = [p for p in set(hard_paths + hard_top + hard_mined) if p.exists()]

    print(f"  Positives: {len(pos_all)}  "
          f"(friends={len(pos_friends)}, trimmed={len(pos_trimmed)})")
    print(f"  Negatives sampled: {len(neg_sampled)} / {len(neg_all)}")
    print(f"  Hard negatives: {len(hard_all)}")
    print("=" * 72)

    # ── 1  Score all buckets ──
    print("\n[1/6] Scoring positives ...")
    pos_scores = []
    for p in pos_all:
        s = score_file(interp, in_idx, out_idx, p)
        if s >= 0:
            pos_scores.append(s)
    pos_scores = np.array(pos_scores)

    print(f"[2/6] Scoring negatives ({len(neg_sampled)}) ...")
    neg_scores = []
    for i, p in enumerate(neg_sampled):
        s = score_file(interp, in_idx, out_idx, p)
        if s >= 0:
            neg_scores.append(s)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(neg_sampled)}")
    neg_scores = np.array(neg_scores)

    print(f"[3/6] Scoring hard negatives ({len(hard_all)}) ...")
    hard_scores = []
    for i, p in enumerate(hard_all):
        s = score_file(interp, in_idx, out_idx, p)
        if s >= 0:
            hard_scores.append(s)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(hard_all)}")
    hard_scores = np.array(hard_scores)

    # ── 2  Threshold sweep ──
    y_true = np.concatenate([
        np.ones(len(pos_scores)),
        np.zeros(len(neg_scores)),
        np.zeros(len(hard_scores)),
    ])
    y_score = np.concatenate([pos_scores, neg_scores, hard_scores])
    sweep = threshold_sweep(y_true, y_score, THRESHOLDS_TO_TEST)
    eer_t = find_eer(y_true, y_score)

    # ── 3  SNR-segmented evaluation ──
    print("[4/6] SNR-segmented evaluation ...")
    noise_pool = load_noise_pool()
    snr_files = pos_all
    if args.pos_sample > 0 and len(snr_files) > args.pos_sample:
        snr_files = random.sample(snr_files, args.pos_sample)

    snr_results = {}
    for eval_t in [0.80, 0.85, 0.90, 0.95]:
        snr_results[eval_t] = eval_snr_levels(
            interp, in_idx, out_idx, snr_files, eval_t, noise_pool)

    # ── 4  Posterior-smoothing simulation ──
    print("[5/6] Posterior-smoothing simulation ...")
    smooth_files = snr_files[:min(100, len(snr_files))]
    smooth_results = {}
    for eval_t in [0.85, 0.90, 0.95]:
        smooth_results[eval_t] = eval_posterior_smoothing(
            interp, in_idx, out_idx, smooth_files, eval_t, noise_pool)

    # ── 5  Plots ──
    print("[6/6] Generating plots ...")
    ts = datetime.datetime.now().strftime("%Y-%m-%d")
    save_det_curve(y_true, y_score, out_dir / f"eval_v6_{ts}_det.png")
    save_pr_roc_curves(y_true, y_score, out_dir / f"eval_v6_{ts}_pr_roc.png")

    # ── 6  Report ──
    lines = []
    lines.append("=" * 72)
    lines.append(f"EVALUATION REPORT — Hey Magis v6  ({ts})")
    lines.append(f"Model: {model_path.name}")
    lines.append("=" * 72)
    lines.append(f"Positives scored:   {len(pos_scores)}")
    lines.append(f"Negatives scored:   {len(neg_scores)}")
    lines.append(f"Hard-neg scored:    {len(hard_scores)}")
    lines.append(f"EER threshold:      {eer_t:.3f}")
    lines.append("")

    lines.append("THRESHOLD SWEEP")
    lines.append("-" * 72)
    lines.append(f"{'thresh':>7s} {'TP':>5s} {'TN':>6s} {'FP':>5s} {'FN':>4s}  "
                 f"{'Recall':>7s} {'FRR':>7s} {'Specif':>7s} {'FAR':>7s} {'FAR/h':>8s}")
    for t in THRESHOLDS_TO_TEST:
        r = sweep[t]
        lines.append(
            f"  {t:.2f}  {r['TP']:5d} {r['TN']:6d} {r['FP']:5d} {r['FN']:4d}  "
            f"{r['recall']:.4f}  {r['FRR']:.4f}  {r['specificity']:.4f}  "
            f"{r['FAR']:.5f}  {r['FAR_per_hour']:.2f}")

    lines.append("")
    lines.append("SNR-SEGMENTED RECALL")
    lines.append("-" * 72)
    lines.append(f"{'threshold':>10s}  {'clean':>7s}  {'10dB':>7s}  {'5dB':>7s}  {'0dB':>7s}")
    for t_val, snr_dict in sorted(snr_results.items()):
        vals = [f"{snr_dict.get(k, 0):.3f}" for k in ["clean", "10dB", "5dB", "0dB"]]
        lines.append(f"  {t_val:.2f}      {'  '.join(vals)}")

    lines.append("")
    lines.append("POSTERIOR-SMOOTHING SIMULATION (recall with N consecutive hits)")
    lines.append("-" * 72)
    header = f"{'threshold':>10s}  " + "  ".join(f"hits={h}" for h in POSTERIOR_SMOOTHING_COUNTS)
    lines.append(header)
    for t_val, sm_dict in sorted(smooth_results.items()):
        vals = [f"{sm_dict.get(h, 0):.3f}" for h in POSTERIOR_SMOOTHING_COUNTS]
        lines.append(f"  {t_val:.2f}      {'   '.join(vals)}")

    lines.append("")
    lines.append("SCORE DISTRIBUTION")
    lines.append("-" * 72)
    for label, scores in [("Positives", pos_scores),
                          ("Negatives", neg_scores),
                          ("Hard-neg", hard_scores)]:
        if len(scores) == 0:
            continue
        lines.append(f"  {label:12s}  n={len(scores):>6d}  "
                     f"mean={np.mean(scores):.4f}  "
                     f"median={np.median(scores):.4f}  "
                     f"min={np.min(scores):.4f}  "
                     f"max={np.max(scores):.4f}")

    best_f1, best_f1_t = 0, 0.5
    for t, r in sweep.items():
        f1 = 2 * r["precision"] * r["recall"] / max(1e-8, r["precision"] + r["recall"])
        if f1 > best_f1:
            best_f1, best_f1_t = f1, t

    lines.append("")
    lines.append("RECOMMENDED THRESHOLDS")
    lines.append("-" * 72)
    lines.append(f"  EER point:  {eer_t:.3f}")
    lines.append(f"  Best F1:    {best_f1_t:.2f}  (F1={best_f1:.3f})")
    lines.append(f"  Suggested deployment range:  {eer_t:.2f} - {min(eer_t + 0.10, 0.99):.2f}")

    report_path = out_dir / f"eval_v6_{ts}_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written: {report_path}")

    for name, scores, files in [
        ("pos_friends",  pos_scores[:len(pos_friends)], pos_friends),
        ("pos_trimmed",  pos_scores[len(pos_friends):], pos_trimmed),
        ("neg_sampled",  neg_scores, neg_sampled),
        ("hard_neg",     hard_scores, hard_all),
    ]:
        csv_path = out_dir / f"eval_v6_{ts}_scores_{name}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "score"])
            for fp, sc in zip(files, scores):
                w.writerow([str(fp), f"{sc:.6f}"])
        print(f"  CSV -> {csv_path.name}")

    print(f"\nEER threshold:  {eer_t:.3f}")
    print(f"Best F1 threshold: {best_f1_t:.2f}  (F1={best_f1:.3f})")
    print("Done.")


if __name__ == "__main__":
    main()

import csv
import numpy as np
import tensorflow as tf
from pathlib import Path

MODEL = r"C:\Users\miker\wake_word_project\models\hey_magis_v3_model.tflite"
POS_CSV = r"C:\Users\miker\wake_word_project\models\eval_v3_scores_pos.csv"
NEG_CSV = r"C:\Users\miker\wake_word_project\models\eval_v3_scores_neg.csv"

def read_first_n(csv_path, n=10):
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if i >= n:
                break
            rows.append((row["path"], float(row["score"])))
    return rows

def read_bottom_n(csv_path, n=10):
    # cheap: read all (these files are only 10k rows for neg)
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((row["path"], float(row["score"])))
    rows.sort(key=lambda x: x[1])
    return rows[:n]

def load_wav_16k_mono(path):
    audio = tf.io.read_file(path)
    wav, sr = tf.audio.decode_wav(audio, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sr = tf.cast(sr, tf.int32)
    return wav, sr

def feature_logmel_98x40(wav, sr):
    # IMPORTANT: this must match your TRAINING feature pipeline.
    # These are “reasonable defaults” but we’ll compare stats first.
    # If stats look off, we will copy your training settings exactly.
    if sr != 16000:
        # enforce (don’t resample here, per your choice)
        raise ValueError(f"Expected 16kHz but got {int(sr)}")

    # 98 frames target implies ~1s audio with 10ms hop and 25ms window-ish.
    frame_length = int(0.025 * 16000)  # 400
    frame_step   = int(0.010 * 16000)  # 160
    stft = tf.signal.stft(wav, frame_length=frame_length, frame_step=frame_step,
                          fft_length=512, window_fn=tf.signal.hann_window, pad_end=True)
    spec = tf.abs(stft) ** 2

    num_mel_bins = 40
    mel_mat = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=spec.shape[-1],
        sample_rate=16000,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0,
    )
    mel = tf.tensordot(spec, mel_mat, 1)
    mel.set_shape(spec.shape[:-1].concatenate(mel_mat.shape[-1:]))

    logmel = tf.math.log(mel + 1e-6)

    # force to 98 frames (pad/trim)
    T = tf.shape(logmel)[0]
    logmel = tf.cond(
        T < 98,
        lambda: tf.pad(logmel, [[0, 98 - T], [0, 0]]),
        lambda: logmel[:98, :]
    )

    x = tf.expand_dims(logmel, axis=-1)   # (98,40,1)
    return x

def run_one(interp, path):
    wav, sr = load_wav_16k_mono(path)
    x = feature_logmel_98x40(wav, sr)
    x_np = np.expand_dims(x.numpy().astype(np.float32), axis=0)  # (1,98,40,1)

    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    print("Input details:", interp.get_input_details()[0]["shape"], interp.get_input_details()[0]["dtype"])
    print("Output details:", interp.get_output_details()[0]["shape"], interp.get_output_details()[0]["dtype"])

    # If model input is quantized, quantize input
    if in_det["dtype"] != np.float32:
        scale, zp = in_det["quantization"]
        if scale == 0:
            raise RuntimeError("Input quantization scale is 0.")
        x_q = np.round(x_np / scale + zp).astype(in_det["dtype"])
        interp.set_tensor(in_det["index"], x_q)
    else:
        interp.set_tensor(in_det["index"], x_np)

    interp.invoke()
    y = interp.get_tensor(out_det["index"])

    # If output is quantized, DE-quantize output (this is commonly missed)
    if out_det["dtype"] != np.float32:
        scale, zp = out_det["quantization"]
        y = (y.astype(np.float32) - zp) * scale

    score = float(np.squeeze(y))
    stats = {
        "min": float(np.min(x_np)),
        "max": float(np.max(x_np)),
        "mean": float(np.mean(x_np)),
        "std": float(np.std(x_np)),
    }
    return score, stats, in_det, out_det

def main():
    interp = tf.lite.Interpreter(model_path=MODEL)
    interp.allocate_tensors()

    top_pos = read_first_n(POS_CSV, 10)           # first 10 positives
    top_neg = read_first_n(NEG_CSV, 10)           # first 10 negatives (these are your top-scoring negs)
    low_pos = read_bottom_n(POS_CSV, 5)           # lowest-scoring positives

    picks = [
        ("TOP_POS", top_pos[0][0], top_pos[0][1]),
        ("LOW_POS", low_pos[0][0], low_pos[0][1]),
        ("TOP_NEG", top_neg[0][0], top_neg[0][1]),
    ]

    print("MODEL:", MODEL)
    for label, p, csv_score in picks:
        p = str(Path(p))
        score, stats, in_det, out_det = run_one(interp, p)
        print("\n---", label, "---")
        print("file:", p)
        print("csv_score:", csv_score)
        print("recomputed_score:", score)
        print("feat_stats:", stats)
        print("input_dtype:", in_det["dtype"], "input_quant:", in_det["quantization"])
        print("output_dtype:", out_det["dtype"], "output_quant:", out_det["quantization"])

if __name__ == "__main__":
    main()

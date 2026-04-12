import argparse
from pathlib import Path
import wave
import numpy as np

TARGET_SEC = 1.5
TARGET_SR = 16000

def read_wav_pcm16_mono(path: Path):
    with wave.open(str(path), "rb") as wf:
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        n = wf.getnframes()
        if sw != 2:
            raise ValueError(f"not PCM16 (sampwidth={sw})")
        raw = wf.readframes(n)

    x = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1).astype(np.int16)
    return x, sr

def write_wav_pcm16_mono(path: Path, x: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(x.astype(np.int16).tobytes())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", choices=["pad_end", "pad_both"], default="pad_end",
                    help="pad_end: add silence at end; pad_both: split pad start/end")
    ap.add_argument("--report", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    wavs = sorted(in_dir.rglob("*.wav"))
    tgt_n = int(TARGET_SEC * TARGET_SR)

    stats = {
        "total": 0,
        "written": 0,
        "already_target": 0,
        "padded": 0,
        "too_long_skipped": 0,
        "wrong_sr_skipped": 0,
        "errors": 0,
    }

    lines = []
    lines.append("============================================================")
    lines.append("PAD TO 1.5s REPORT")
    lines.append("============================================================")
    lines.append(f"In : {in_dir}")
    lines.append(f"Out: {out_dir}")
    lines.append(f"Target: {TARGET_SEC:.3f}s @ {TARGET_SR} Hz (samples={tgt_n})")
    lines.append(f"Mode: {args.mode}")
    lines.append("")

    for p in wavs:
        stats["total"] += 1
        rel = p.relative_to(in_dir)
        out_p = out_dir / rel

        try:
            x, sr = read_wav_pcm16_mono(p)

            if sr != TARGET_SR:
                stats["wrong_sr_skipped"] += 1
                lines.append(f"[SKIP wrong_sr] {p} sr={sr}")
                continue

            n = len(x)
            if n == tgt_n:
                stats["already_target"] += 1
                if not args.dry_run:
                    write_wav_pcm16_mono(out_p, x, sr)
                stats["written"] += 1
                continue

            if n > tgt_n:
                # Do NOT crop here, because you said these are already trimmed.
                stats["too_long_skipped"] += 1
                lines.append(f"[SKIP too_long] {p} dur={n/sr:.3f}s")
                continue

            # pad
            pad = tgt_n - n
            if args.mode == "pad_end":
                x2 = np.pad(x, (0, pad), mode="constant", constant_values=0)
            else:
                pre = pad // 2
                post = pad - pre
                x2 = np.pad(x, (pre, post), mode="constant", constant_values=0)

            stats["padded"] += 1
            if not args.dry_run:
                write_wav_pcm16_mono(out_p, x2, sr)
            stats["written"] += 1

        except Exception as e:
            stats["errors"] += 1
            lines.append(f"[ERROR] {p}  {e}")

    lines.append("")
    lines.append("------------------------------------------------------------")
    lines.append("SUMMARY")
    lines.append("------------------------------------------------------------")
    for k in ["total","written","already_target","padded","too_long_skipped","wrong_sr_skipped","errors"]:
        lines.append(f"{k}: {stats[k]}")

    out_text = "\n".join(lines) + "\n"

    if args.report:
        Path(args.report).write_text(out_text, encoding="utf-8")
        print(f"✅ Wrote report: {args.report}")
    print(out_text)

if __name__ == "__main__":
    main()

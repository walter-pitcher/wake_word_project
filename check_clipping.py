import argparse
from pathlib import Path
import wave
import numpy as np

def read_wav_pcm16(path: Path):
    with wave.open(str(path), "rb") as wf:
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        n = wf.getnframes()
        if sw != 2:
            raise ValueError(f"{path} not PCM16 (sampwidth={sw})")
        raw = wf.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1).astype(np.int16)
    return x, sr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--clip-count", type=int, default=10, help=">= this many near-fullscale samples flags file")
    ap.add_argument("--near", type=int, default=32760, help="near full-scale int16 threshold")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.dir)
    wavs = sorted(root.rglob("*.wav"))

    lines = []
    bad = 0
    for p in wavs:
        try:
            x, sr = read_wav_pcm16(p)
            # count samples very close to max/min => clipping indicator
            c = int(np.sum((x >= args.near) | (x <= -args.near)))
            if c >= args.clip_count:
                bad += 1
                lines.append(f"{c:6d}  {p}")
        except Exception as e:
            bad += 1
            lines.append(f"ERROR  {p}  {e}")

    header = [
        "============================================================",
        "CLIPPING REPORT",
        "============================================================",
        f"Root: {root}",
        f"Total wavs scanned: {len(wavs)}",
        f"Near-fullscale threshold: {args.near}",
        f"Clip sample-count threshold: {args.clip_count}",
        f"Flagged files: {bad}",
        "",
        "Flagged list (clip_count then path):"
    ]
    out_text = "\n".join(header + lines) + "\n"

    if args.out:
        Path(args.out).write_text(out_text, encoding="utf-8")
        print(f"✅ Wrote: {args.out}")
    print(out_text)

if __name__ == "__main__":
    main()

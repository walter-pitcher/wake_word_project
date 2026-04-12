import argparse
from pathlib import Path
import wave
import numpy as np

def read_wav(path: Path):
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        audio = wf.readframes(n)
    if sw != 2:
        raise ValueError(f"{path} sample width {sw} not PCM16")
    x = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    x /= 32768.0
    return x, sr

def rms_dbfs_first_sec(x: np.ndarray, sr: int, sec: float = 1.0):
    n = int(sr * sec)
    x1 = x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))
    rms = np.sqrt(np.mean(np.square(x1)) + 1e-12)
    dbfs = 20.0 * np.log10(rms + 1e-12)
    return float(dbfs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--top", type=int, default=20, help="how many to show per list")
    ap.add_argument("--out", default=None, help="optional output txt path")
    args = ap.parse_args()

    root = Path(args.dir)
    wavs = sorted(root.rglob("*.wav"))
    rows = []
    for p in wavs:
        try:
            x, sr = read_wav(p)
            db = rms_dbfs_first_sec(x, sr, 1.0)
            rows.append((db, str(p)))
        except Exception as e:
            rows.append((999.0, f"{p}  ERROR: {e}"))

    rows_ok = [r for r in rows if r[0] < 100]
    rows_ok.sort(key=lambda t: t[0])  # quietest first

    quiet = rows_ok[:args.top]
    loud = list(reversed(rows_ok[-args.top:]))

    lines = []
    lines.append("QUIETEST (lowest RMS dBFS first 1s)")
    for db, p in quiet:
        lines.append(f"{db:8.3f}  {p}")
    lines.append("")
    lines.append("LOUDEST (highest RMS dBFS first 1s)")
    for db, p in loud:
        lines.append(f"{db:8.3f}  {p}")

    text = "\n".join(lines)
    print(text)

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\n✅ wrote: {args.out}")

if __name__ == "__main__":
    main()

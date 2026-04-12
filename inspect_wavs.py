import argparse
import os
from pathlib import Path
import wave
import contextlib
import math
import struct

def rms_dbfs_pcm16(frames: bytes) -> float:
    if not frames:
        return -float("inf")
    n = len(frames) // 2
    if n == 0:
        return -float("inf")
    samples = struct.unpack("<" + "h"*n, frames[:n*2])
    acc = 0.0
    for s in samples:
        acc += float(s) * float(s)
    rms = math.sqrt(acc / n)
    if rms <= 0:
        return -float("inf")
    return 20.0 * math.log10(rms / 32768.0)

def inspect_one(path: Path):
    try:
        with contextlib.closing(wave.open(str(path), "rb")) as wf:
            ch = wf.getnchannels()
            sr = wf.getframerate()
            sw = wf.getsampwidth()
            nframes = wf.getnframes()
            dur = nframes / float(sr) if sr else 0.0

            # Read first ~1 sec or full if shorter (for RMS quick check)
            to_read = min(nframes, sr) if sr else nframes
            frames = wf.readframes(to_read)

            subtype = {1: "PCM8?", 2: "PCM16", 3: "PCM24?", 4: "PCM32/FLOAT?"}.get(sw, f"{sw*8}-bit?")
            rms = None
            if sw == 2:
                rms = rms_dbfs_pcm16(frames)

            return {
                "path": str(path),
                "sr": sr,
                "ch": ch,
                "width": sw,
                "subtype_guess": subtype,
                "frames": nframes,
                "dur_s": dur,
                "rms_dbfs_1s": rms
            }
    except wave.Error as e:
        return {"path": str(path), "error": f"wave.Error: {e}"}
    except Exception as e:
        return {"path": str(path), "error": f"{type(e).__name__}: {e}"}

def iter_wavs(root: Path):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".wav"):
                yield Path(dirpath) / fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory to scan")
    ap.add_argument("--limit", type=int, default=20, help="Max files to print")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.exists():
        print(f"bad Not found: {root}")
        return 2

    rows = []
    for i, p in enumerate(iter_wavs(root), start=1):
        rows.append(inspect_one(p))
        if i >= args.limit:
            break

    print(f"DIR: {root}")
    for r in rows:
        if "error" in r:
            print(f"  bad {r['path']}  | {r['error']}")
        else:
            print(
                f"  ok {Path(r['path']).name:35s} | sr={r['sr']} ch={r['ch']} "
                f"w={r['width']}({r['subtype_guess']}) dur={r['dur_s']:.3f}s "
                f"rms1s={r['rms_dbfs_1s'] if r['rms_dbfs_1s'] is not None else 'n/a'}"
            )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

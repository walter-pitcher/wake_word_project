import argparse
import os
import wave
from pathlib import Path
from collections import Counter
import math

def rms_dbfs_first_n_seconds(wav_path: Path, seconds: float = 1.0):
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        nframes = wf.getnframes()

        # Read first N seconds (or all if shorter)
        frames_to_read = min(int(sr * seconds), nframes)
        raw = wf.readframes(frames_to_read)

    # Only support PCM16 safely (2 bytes)
    if sw != 2:
        return None

    import struct
    samples = struct.unpack("<" + "h" * (len(raw) // 2), raw)

    # If stereo, downmix by taking every ch-th sample average
    if ch > 1:
        # quick downmix: take left channel only (simple, consistent)
        samples = samples[0::ch]

    if len(samples) == 0:
        return -180.0

    # RMS
    s2 = 0.0
    for s in samples:
        s2 += float(s) * float(s)
    rms = math.sqrt(s2 / len(samples))

    # Convert to dBFS (full-scale 32768)
    if rms <= 1e-12:
        return -180.0
    dbfs = 20.0 * math.log10(rms / 32768.0)
    return dbfs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--rms_seconds", type=float, default=1.0)
    ap.add_argument("--out", default="wav_summary.txt")
    args = ap.parse_args()

    root = Path(args.dir)
    wavs = sorted(root.rglob("*.wav"))

    sr_counts = Counter()
    ch_counts = Counter()
    sw_counts = Counter()

    durations = []
    rms_vals = []
    bad = []

    for p in wavs:
        try:
            with wave.open(str(p), "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                sw = wf.getsampwidth()
                nframes = wf.getnframes()
                dur = nframes / float(sr) if sr else 0.0

            sr_counts[sr] += 1
            ch_counts[ch] += 1
            sw_counts[sw] += 1
            durations.append(dur)

            rms = rms_dbfs_first_n_seconds(p, args.rms_seconds)
            if rms is not None:
                rms_vals.append(rms)
        except Exception as e:
            bad.append((str(p), repr(e)))

    def stats(arr):
        if not arr:
            return None
        arr2 = sorted(arr)
        n = len(arr2)
        def pct(q):
            i = int(q * (n-1))
            return arr2[i]
        return {
            "n": n,
            "min": arr2[0],
            "p10": pct(0.10),
            "p50": pct(0.50),
            "p90": pct(0.90),
            "max": arr2[-1],
        }

    dur_stats = stats(durations)
    rms_stats = stats(rms_vals)

    out_lines = []
    out_lines.append("============================================================")
    out_lines.append("WAV SUMMARY")
    out_lines.append("============================================================")
    out_lines.append(f"Root: {root}")
    out_lines.append(f"Total wav files: {len(wavs)}")
    out_lines.append("")
    out_lines.append("Sample rate counts:")
    for k,v in sorted(sr_counts.items()):
        out_lines.append(f"  {k} Hz: {v}")
    out_lines.append("Channel counts:")
    for k,v in sorted(ch_counts.items()):
        out_lines.append(f"  {k} ch: {v}")
    out_lines.append("Sample width counts:")
    for k,v in sorted(sw_counts.items()):
        out_lines.append(f"  {k} bytes: {v}")
    out_lines.append("")
    if dur_stats:
        out_lines.append("Duration stats (seconds):")
        for k in ["min","p10","p50","p90","max"]:
            out_lines.append(f"  {k}: {dur_stats[k]:.3f}")
    if rms_stats:
        out_lines.append("")
        out_lines.append(f"RMS dBFS stats (first {args.rms_seconds:.1f}s):")
        for k in ["min","p10","p50","p90","max"]:
            out_lines.append(f"  {k}: {rms_stats[k]:.3f}")
    out_lines.append("")
    out_lines.append(f"Bad files: {len(bad)}")
    for p, e in bad[:50]:
        out_lines.append(f"  {p} -> {e}")
    if len(bad) > 50:
        out_lines.append(f"  ... {len(bad)-50} more")

    Path(args.out).write_text("\n".join(out_lines), encoding="utf-8")
    print(f"✅ Wrote: {args.out}")

if __name__ == "__main__":
    main()

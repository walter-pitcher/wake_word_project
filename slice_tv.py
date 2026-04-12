#!/usr/bin/env python3
"""
Slice a long audio file into 1.5s negative samples
and append them to negative_manifest_hard.txt
"""

from pathlib import Path
import numpy as np
import soundfile as sf

# =============================
# CONFIG — adjust these paths
# =============================
INPUT_FILE = Path("tv_recording.wav")  # your 1 hour recording
OUTPUT_DIR = Path("data/negative/tv_samples")
MANIFEST = Path("data/negative_manifest_hard.txt")

SAMPLE_RATE = 16000
CLIP_SECONDS = 1.5
HOP_SECONDS = 0.75  # 50% overlap for more samples
# =============================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {INPUT_FILE}...")
    audio, sr = sf.read(INPUT_FILE)

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        import resampy
        print(f"Resampling from {sr}Hz to {SAMPLE_RATE}Hz...")
        audio = resampy.resample(audio, sr, SAMPLE_RATE)

    clip_len = int(SAMPLE_RATE * CLIP_SECONDS)
    hop_len = int(SAMPLE_RATE * HOP_SECONDS)

    clips_written = 0
    new_paths = []

    for i, start in enumerate(range(0, len(audio) - clip_len, hop_len)):
        chunk = audio[start:start + clip_len].astype(np.float32)

        # Skip near-silent clips — they add little value
        rms = np.sqrt(np.mean(chunk ** 2))
        if rms < 0.001:
            continue

        out_path = OUTPUT_DIR / f"tv_{i:05d}.wav"
        sf.write(out_path, chunk, SAMPLE_RATE)
        new_paths.append(str(out_path))
        clips_written += 1

    # Append to manifest
    with open(MANIFEST, "a") as f:
        for p in new_paths:
            f.write(p + "\n")

    print(f"Done. Written {clips_written} clips to {OUTPUT_DIR}")
    print(f"Manifest updated: {MANIFEST}")

if __name__ == "__main__":
    main()
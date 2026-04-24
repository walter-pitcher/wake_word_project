#!/usr/bin/env python3
"""
Positive Augmentation Pipeline — MagisAI Wake Word v6
=====================================================
Expands ~454 raw positive samples to 5 000+ effective training clips using:

  1. Pitch shifting   (+-1, +-2 semitones)
  2. Speed perturbation (0.9x, 1.1x)
  3. Noise injection   (SNR 5 / 10 / 15 dB)
  4. Combined:  pitch + noise, speed + noise

Every augmented file is exactly 1.5 s (24 000 samples) @ 16 kHz mono,
matching the Dart inference pipeline.

Usage:
    python new/augment_positives.py            # uses defaults from config
    python new/augment_positives.py --dry-run  # preview without writing
"""

import sys, os, argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    SAMPLE_RATE, NUM_SAMPLES,
    POS_FRIENDS_DIR, POS_TRIMMED_DIR, POS_AUGMENTED_DIR,
    BACKGROUND_NOISE_DIR, EXPANDED_NOISE_DIR,
    PITCH_SHIFTS, SPEED_FACTORS, NOISE_SNRS_AUG,
    TARGET_AUGMENTED_COUNT,
)

# ───────────────────────── audio helpers ─────────────────────────

def load_wav(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return _fix_length(audio)


def _fix_length(audio: np.ndarray) -> np.ndarray:
    if len(audio) >= NUM_SAMPLES:
        return audio[:NUM_SAMPLES]
    return np.pad(audio, (0, NUM_SAMPLES - len(audio)))


def save_wav(audio: np.ndarray, path: Path) -> None:
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")


# ─────────────────────── augmentation ops ────────────────────────

def pitch_shift(audio: np.ndarray, semitones: float) -> np.ndarray:
    return _fix_length(
        librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=semitones)
    )


def speed_change(audio: np.ndarray, factor: float) -> np.ndarray:
    return _fix_length(librosa.effects.time_stretch(audio, rate=factor))


def add_noise(audio: np.ndarray, noise_segment: np.ndarray,
              snr_db: float) -> np.ndarray:
    sig_rms = np.sqrt(np.mean(audio ** 2) + 1e-8)
    noi_rms = np.sqrt(np.mean(noise_segment ** 2) + 1e-8)
    target = sig_rms / (10.0 ** (snr_db / 20.0))
    noise_segment = noise_segment * (target / (noi_rms + 1e-8))
    return np.clip(audio + noise_segment, -1.0, 1.0)


# ────────────────────── noise file loading ───────────────────────

def load_noise_pool(dirs):
    pool = []
    for d in dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.wav")):
            try:
                a, sr = sf.read(str(f), dtype="float32")
                if sr != SAMPLE_RATE:
                    a = librosa.resample(a, orig_sr=sr, target_sr=SAMPLE_RATE)
                if a.ndim > 1:
                    a = a.mean(axis=1)
                pool.append(a)
            except Exception:
                pass
    return pool


def random_noise_segment(pool, length):
    if not pool:
        return np.random.randn(length).astype(np.float32) * 0.005
    noise = pool[np.random.randint(len(pool))]
    if len(noise) <= length:
        noise = np.tile(noise, (length // len(noise)) + 1)
    start = np.random.randint(0, len(noise) - length)
    return noise[start : start + length]


# ─────────────────────── main pipeline ───────────────────────────

def collect_originals():
    files = []
    if POS_FRIENDS_DIR.exists():
        files.extend(sorted(POS_FRIENDS_DIR.rglob("*.wav")))
    if POS_TRIMMED_DIR.exists():
        files.extend(sorted(POS_TRIMMED_DIR.glob("*.wav")))
    return files


def make_augmentations(noise_pool):
    augs = []
    for s in PITCH_SHIFTS:
        tag = f"pitch_{'+' if s > 0 else ''}{s}st"
        augs.append((tag, lambda a, s=s: pitch_shift(a, s)))
    for f in SPEED_FACTORS:
        augs.append((f"speed_{f:.2f}", lambda a, f=f: speed_change(a, f)))
    for snr in NOISE_SNRS_AUG:
        augs.append((
            f"noise_{snr}dB",
            lambda a, snr=snr: add_noise(
                a, random_noise_segment(noise_pool, NUM_SAMPLES), snr),
        ))
    augs.append((
        "pitch+1_noise10",
        lambda a: add_noise(pitch_shift(a, 1),
                            random_noise_segment(noise_pool, NUM_SAMPLES), 10),
    ))
    augs.append((
        "speed09_noise5",
        lambda a: add_noise(speed_change(a, 0.9),
                            random_noise_segment(noise_pool, NUM_SAMPLES), 5),
    ))
    return augs


def source_tag(path: Path) -> tuple:
    try:
        path.relative_to(POS_FRIENDS_DIR)
        return "friends", path.parent.name
    except ValueError:
        return "trimmed", "pool"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan without writing files")
    args = ap.parse_args()

    print("=" * 70)
    print("POSITIVE AUGMENTATION PIPELINE  v6")
    print("=" * 70)

    originals = collect_originals()
    print(f"Original positives found: {len(originals)}")
    if not originals:
        raise SystemExit("No positive WAV files found — check config paths.")

    noise_pool = load_noise_pool([BACKGROUND_NOISE_DIR, EXPANDED_NOISE_DIR])
    print(f"Background-noise files loaded: {len(noise_pool)}")

    augs = make_augmentations(noise_pool)
    expected = len(originals) * len(augs)
    print(f"Augmentations per sample: {len(augs)}  ->  expected total: {expected}")

    if args.dry_run:
        print("\n[DRY RUN] — no files written.")
        return

    POS_AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)
    written = 0

    for wav_path in tqdm(originals, desc="Augmenting", unit="file"):
        try:
            audio = load_wav(wav_path)
        except Exception as e:
            tqdm.write(f"  SKIP {wav_path.name}: {e}")
            continue

        src, spk = source_tag(wav_path)

        for suffix, fn in augs:
            try:
                aug = fn(audio)
                out_name = f"{src}__{spk}__{wav_path.stem}__{suffix}.wav"
                save_wav(aug, POS_AUGMENTED_DIR / out_name)
                written += 1
            except Exception as e:
                tqdm.write(f"  AUG-FAIL ({suffix}) {wav_path.name}: {e}")

    print(f"\n{'=' * 70}")
    print(f"DONE — wrote {written} augmented files to {POS_AUGMENTED_DIR}")
    print(f"Originals: {len(originals)}  |  Target was >= {TARGET_AUGMENTED_COUNT}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phonetic-Confuser Generator — MagisAI Wake Word v6
====================================================
Uses the Windows SAPI5 TTS engine (via pyttsx3) to synthesise phrases
that are phonetically close to "Hey Magis" and could cause false
positives.  Multiple rate / voice variations are generated per phrase.

All output clips are resampled to 16 kHz mono, trimmed/padded to 1.5 s,
and saved as PCM-16 WAV files in data/negative/phonetic_confusers/.

Usage:
    python new/generate_phonetic_negatives.py
    python new/generate_phonetic_negatives.py --voices 0 1
"""

import sys, os, argparse, tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SAMPLE_RATE, NUM_SAMPLES, PHONETIC_NEG_DIR

PHRASES = [
    "Hey magic",
    "Hey massive",
    "Hey Agnes",
    "Hey Marcus",
    "Okay Magis",
    "Hey Maxis",
    "Hey mattress",
    "Hey Mavis",
    "Hey Magnus",
    "Hey Maddox",
    "Hey marriages",
    "Hey Margaret",
    "Hey manage",
    "Hey majesty",
    "Hey malice",
    "Hey my guess",
    "Hey mad just",
]

RATE_MULTIPLIERS = [0.85, 1.0, 1.15]


def _fix_length(audio: np.ndarray) -> np.ndarray:
    if len(audio) >= NUM_SAMPLES:
        return audio[:NUM_SAMPLES]
    return np.pad(audio, (0, NUM_SAMPLES - len(audio)))


def save_wav(audio: np.ndarray, path: Path):
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")


def generate_with_pyttsx3(voice_indices=None):
    try:
        import pyttsx3
    except ImportError:
        print("pyttsx3 not installed. Run:  pip install pyttsx3")
        return 0

    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    print(f"Available TTS voices: {len(voices)}")
    for i, v in enumerate(voices):
        print(f"  [{i}] {v.name}")

    if not voices:
        print("No TTS voices found from SAPI5. Cannot generate phonetic negatives.")
        return 0

    if voice_indices is None:
        voice_indices = list(range(min(3, len(voices))))

    PHONETIC_NEG_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    total = len([vi for vi in voice_indices if vi < len(voices)]) * len(RATE_MULTIPLIERS) * len(PHRASES)
    done = 0

    for vi in voice_indices:
        if vi >= len(voices):
            continue
        engine.setProperty("voice", voices[vi].id)
        v_tag = f"v{vi}"

        for rate_mult in RATE_MULTIPLIERS:
            engine.setProperty("rate", int(150 * rate_mult))
            r_tag = f"r{rate_mult:.2f}"

            for phrase in PHRASES:
                safe_name = phrase.lower().replace(" ", "_")
                out_name = f"phonetic__{safe_name}__{v_tag}_{r_tag}.wav"
                out_path = PHONETIC_NEG_DIR / out_name
                done += 1
                print(f"[{done}/{total}] {phrase} ({v_tag}, {r_tag})")

                if out_path.exists():
                    count += 1
                    print("  exists, skipping")
                    continue

                # Windows-specific: NamedTemporaryFile keeps the handle open and can block
                # SAPI from writing, which makes pyttsx3 appear to hang after listing voices.
                fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)

                try:
                    engine.save_to_file(phrase, tmp_path)
                    engine.runAndWait()

                    audio, sr = sf.read(tmp_path, dtype="float32")
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    if sr != SAMPLE_RATE:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                    audio = _fix_length(audio)
                    save_wav(audio, out_path)
                    count += 1
                    print(f"  saved -> {out_name}")
                except Exception as e:
                    print(f"  FAIL: {phrase} ({v_tag}, {r_tag}): {e}")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    engine.stop()
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", nargs="*", type=int, default=None,
                    help="Voice indices to use (default: first 3)")
    args = ap.parse_args()

    print("=" * 70)
    print("PHONETIC-CONFUSER GENERATOR  v6")
    print("=" * 70)

    n = generate_with_pyttsx3(args.voices)
    print(f"\nGenerated {n} phonetic-confuser clips in {PHONETIC_NEG_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()

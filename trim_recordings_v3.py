#!/usr/bin/env python3
"""
Hey Magis Wake Word Audio Trimmer v2.3 (1.2s window)
===================================================

Goal:
- Consistent 1.2s clips aligned to speech onset for wake-word training

Rules:
- Detect speech onset using RMS dBFS frames (25ms frame, 10ms hop)
- Search onset only within first MAX_ONSET_SEARCH_SEC seconds (prevents bogus late onsets)
- Extract [-150ms, +1050ms] around onset (1.2s total)
- Hard-trim or pad to exactly 1.2s
- Normalize peak to 0.95
- Output PCM16 WAV

Quality bucketing:
- A: coverage >= 0.45 and segments == 1
- B: coverage >= 0.20
- C: coverage >= 0.12
- Reject: silent / no onset / processing error

Notes:
- "Segments" is computed within the onset-search window (not the whole file)
- You can tighten segment rules later once trimming is stable
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import shutil
from dataclasses import dataclass
from typing import Tuple, List, Optional


# =============================================================================
# CONFIG
# =============================================================================

SAMPLE_RATE = 16000

TARGET_DURATION = 1.2
PADDING_BEFORE = 0.150
PADDING_AFTER  = 1.050  # 0.150 + 1.050 = 1.200s

FRAME_LENGTH_MS = 25
HOP_LENGTH_MS = 10
MIN_SPEECH_FRAMES = 3  # consecutive frames to confirm onset

SPEECH_THRESHOLD_DBFS = -38.0
SILENCE_THRESHOLD_DBFS = -55.0  # make this stricter if you're still seeing "silent -60 dBFS" pass

MAX_ONSET_SEARCH_SEC = 2.5       # <--- critical: prevents insane late onsets
MIN_SILENCE_GAP = 0.100          # seconds gap to count new segment
OFFSET_TAIL_MS = 50              # pad offset a bit to avoid clipped consonants

# "coverage" rejects (barely audible)
MIN_COVERAGE_REJECT = 0.12

# quality buckets (tune later)
A_COVERAGE = 0.45
B_COVERAGE = 0.20
C_COVERAGE = 0.12


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AudioAnalysis:
    onset_time: Optional[float]
    offset_time: Optional[float]
    speech_duration: float
    trailing_silence: float
    num_segments: int
    speech_coverage: float
    is_silent: bool
    peak_dbfs: float
    rms_dbfs: float
    rejection_reason: Optional[str] = None
    bucket: Optional[str] = None


@dataclass
class ProcessingResult:
    success: bool
    filename: str
    bucket: Optional[str] = None
    rejection_reason: Optional[str] = None
    onset_delay: float = 0.0
    speech_duration: float = 0.0
    speech_coverage: float = 0.0
    num_segments: int = 0


# =============================================================================
# CORE DSP
# =============================================================================

def rms_dbfs_frames(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    frame_length = int(FRAME_LENGTH_MS * sr / 1000)
    hop_length = int(HOP_LENGTH_MS * sr / 1000)

    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms_dbfs = 20 * np.log10(rms + 1e-9)

    times = librosa.frames_to_time(np.arange(len(rms_dbfs)), sr=sr, hop_length=hop_length)
    return rms_dbfs, times


def clamp_onset_search(rms_dbfs: np.ndarray, times: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    hop_sec = HOP_LENGTH_MS / 1000.0
    max_frames = int(MAX_ONSET_SEARCH_SEC / hop_sec)
    max_frames = min(max_frames, len(rms_dbfs))
    return rms_dbfs[:max_frames], times[:max_frames]


def detect_onset_time(audio: np.ndarray, sr: int) -> Optional[float]:
    rms_dbfs, times = rms_dbfs_frames(audio, sr)
    max_frames = int(MAX_ONSET_SEARCH_SEC / (HOP_LENGTH_MS / 1000.0))

    rms_dbfs = rms_dbfs[:max_frames]
    times = times[:max_frames]

    is_speech = rms_dbfs > SPEECH_THRESHOLD_DBFS

    consecutive = 0
    for i, s in enumerate(is_speech):
        if s:
            consecutive += 1
            if consecutive >= MIN_SPEECH_FRAMES:
                onset_frame = i - MIN_SPEECH_FRAMES + 1
                return float(times[onset_frame])
        else:
            consecutive = 0

    return None


def detect_offset_time(audio: np.ndarray, sr: int, onset_time: float) -> float:
    rms_dbfs, times = rms_dbfs_frames(audio, sr)
    # still clamp offset search to same early region (keeps it sane)
    rms_dbfs, times = clamp_onset_search(rms_dbfs, times, sr)

    is_speech = rms_dbfs > SPEECH_THRESHOLD_DBFS

    # find last run end of speech (>= MIN_SPEECH_FRAMES)
    consecutive = 0
    last_run_end = None

    for i, s in enumerate(is_speech):
        if s:
            consecutive += 1
        else:
            if consecutive >= MIN_SPEECH_FRAMES:
                last_run_end = i - 1
            consecutive = 0

    if consecutive >= MIN_SPEECH_FRAMES:
        last_run_end = len(is_speech) - 1

    if last_run_end is None:
        # fallback: offset = onset + small slice
        return min(onset_time + 0.30, len(audio) / sr)

    offset = float(times[last_run_end]) + (OFFSET_TAIL_MS / 1000.0)
    return min(offset, len(audio) / sr)


def compute_speech_coverage(audio: np.ndarray, sr: int, onset_time: float, offset_time: float) -> float:
    rms_dbfs, times = rms_dbfs_frames(audio, sr)
    rms_dbfs, times = clamp_onset_search(rms_dbfs, times, sr)
    is_speech = rms_dbfs > SPEECH_THRESHOLD_DBFS

    onset_idx = int(np.searchsorted(times, onset_time))
    offset_idx = int(np.searchsorted(times, offset_time))

    onset_idx = max(0, min(onset_idx, len(is_speech) - 1))
    offset_idx = max(onset_idx + 1, min(offset_idx, len(is_speech)))

    window = is_speech[onset_idx:offset_idx]
    if len(window) == 0:
        return 0.0
    return float(np.sum(window) / len(window))


def count_speech_segments(audio: np.ndarray, sr: int) -> int:
    rms_dbfs, times = rms_dbfs_frames(audio, sr)
    rms_dbfs, times = clamp_onset_search(rms_dbfs, times, sr)
    is_speech = rms_dbfs > SPEECH_THRESHOLD_DBFS

    if not np.any(is_speech):
        return 0

    hop_sec = HOP_LENGTH_MS / 1000.0
    min_gap_frames = int(MIN_SILENCE_GAP / hop_sec)

    segments = 0
    in_speech = False
    silence_count = 0

    for s in is_speech:
        if s:
            if not in_speech:
                segments += 1
                in_speech = True
            silence_count = 0
        else:
            silence_count += 1
            if in_speech and silence_count >= min_gap_frames:
                in_speech = False

    return segments


def is_silent(audio: np.ndarray) -> Tuple[bool, float]:
    rms = float(np.sqrt(np.mean(audio**2)))
    rms_dbfs = 20 * np.log10(rms + 1e-9)
    if rms_dbfs < SILENCE_THRESHOLD_DBFS:
        return True, rms_dbfs
    if np.max(np.abs(audio)) < 0.001:
        return True, rms_dbfs
    return False, rms_dbfs


def extract_window(audio: np.ndarray, sr: int, onset_time: float) -> np.ndarray:
    target_samples = int(TARGET_DURATION * sr)
    onset_idx = int(onset_time * sr)

    start = onset_idx - int(PADDING_BEFORE * sr)
    end   = onset_idx + int(PADDING_AFTER  * sr)

    pad_left = max(0, -start)
    pad_right = max(0, end - len(audio))

    start = max(0, start)
    end = min(len(audio), end)

    chunk = audio[start:end]
    if pad_left or pad_right:
        chunk = np.pad(chunk, (pad_left, pad_right), mode="constant")

    if len(chunk) > target_samples:
        chunk = chunk[:target_samples]
    elif len(chunk) < target_samples:
        chunk = np.pad(chunk, (0, target_samples - len(chunk)), mode="constant")

    return chunk


def normalize_peak(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    m = np.max(np.abs(audio))
    if m > 0:
        return audio * (target_peak / m)
    return audio


def assign_bucket(coverage: float, segments: int) -> Optional[str]:
    if coverage >= A_COVERAGE and segments == 1:
        return "A"
    if coverage >= B_COVERAGE:
        return "B"
    if coverage >= C_COVERAGE:
        return "C"
    return None


def analyze(audio: np.ndarray, sr: int) -> AudioAnalysis:
    peak = float(np.max(np.abs(audio)))
    peak_dbfs = 20 * np.log10(peak + 1e-9)

    silent, rms_dbfs = is_silent(audio)
    if silent:
        return AudioAnalysis(
            onset_time=None, offset_time=None,
            speech_duration=0.0, trailing_silence=0.0,
            num_segments=0, speech_coverage=0.0,
            is_silent=True, peak_dbfs=peak_dbfs, rms_dbfs=rms_dbfs,
            rejection_reason=f"silent (rms {rms_dbfs:.1f} dBFS)"
        )

    onset = detect_onset_time(audio, sr)
    if onset is None:
        return AudioAnalysis(
            onset_time=None, offset_time=None,
            speech_duration=0.0, trailing_silence=0.0,
            num_segments=0, speech_coverage=0.0,
            is_silent=False, peak_dbfs=peak_dbfs, rms_dbfs=rms_dbfs,
            rejection_reason="no_onset"
        )

    offset = detect_offset_time(audio, sr, onset)

    speech_duration = max(0.0, offset - onset)
    audio_duration = len(audio) / sr
    trailing = max(0.0, audio_duration - offset)

    segments = count_speech_segments(audio, sr)
    coverage = compute_speech_coverage(audio, sr, onset, offset)

    if coverage < MIN_COVERAGE_REJECT:
        return AudioAnalysis(
            onset_time=onset, offset_time=offset,
            speech_duration=speech_duration, trailing_silence=trailing,
            num_segments=segments, speech_coverage=coverage,
            is_silent=False, peak_dbfs=peak_dbfs, rms_dbfs=rms_dbfs,
            rejection_reason=f"low_coverage ({coverage*100:.0f}%)"
        )

    bucket = assign_bucket(coverage, segments)
    if bucket is None:
        return AudioAnalysis(
            onset_time=onset, offset_time=offset,
            speech_duration=speech_duration, trailing_silence=trailing,
            num_segments=segments, speech_coverage=coverage,
            is_silent=False, peak_dbfs=peak_dbfs, rms_dbfs=rms_dbfs,
            rejection_reason=f"low_quality (cov {coverage*100:.0f}%, seg {segments})"
        )

    return AudioAnalysis(
        onset_time=onset, offset_time=offset,
        speech_duration=speech_duration, trailing_silence=trailing,
        num_segments=segments, speech_coverage=coverage,
        is_silent=False, peak_dbfs=peak_dbfs, rms_dbfs=rms_dbfs,
        bucket=bucket
    )


def process_file(in_path: Path, out_dir_A: Path, out_dir_B: Path, out_dir_C: Path) -> ProcessingResult:
    try:
        audio, sr = librosa.load(in_path, sr=SAMPLE_RATE)
        a = analyze(audio, sr)

        if a.rejection_reason:
            return ProcessingResult(False, in_path.name, rejection_reason=a.rejection_reason,
                                    onset_delay=float(a.onset_time or 0.0),
                                    speech_duration=a.speech_duration,
                                    speech_coverage=a.speech_coverage,
                                    num_segments=a.num_segments)

        chunk = extract_window(audio, sr, float(a.onset_time))
        chunk = normalize_peak(chunk, 0.95)

        if a.bucket == "A":
            out_path = out_dir_A / in_path.name
        elif a.bucket == "B":
            out_path = out_dir_B / in_path.name
        else:
            out_path = out_dir_C / in_path.name

        sf.write(out_path, chunk, sr, subtype="PCM_16")
        return ProcessingResult(True, in_path.name, bucket=a.bucket,
                                onset_delay=float(a.onset_time),
                                speech_duration=a.speech_duration,
                                speech_coverage=a.speech_coverage,
                                num_segments=a.num_segments)

    except Exception as e:
        return ProcessingResult(False, in_path.name, rejection_reason=f"error: {e}")


def process_dir(input_dir: str,
                out_base: str,
                rejected_dir: Optional[str] = None,
                backup_dir: Optional[str] = None) -> None:

    inp = Path(input_dir)
    wavs = sorted(inp.glob("*.wav"))

    outA = Path(out_base + "_A"); outA.mkdir(parents=True, exist_ok=True)
    outB = Path(out_base + "_B"); outB.mkdir(parents=True, exist_ok=True)
    outC = Path(out_base + "_C"); outC.mkdir(parents=True, exist_ok=True)

    rej = Path(rejected_dir) if rejected_dir else None
    if rej: rej.mkdir(parents=True, exist_ok=True)

    bak = Path(backup_dir) if backup_dir else None
    if bak: bak.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {len(wavs)} wavs from: {input_dir}")
    print(f"Target duration: {TARGET_DURATION:.1f}s  | window: -{int(PADDING_BEFORE*1000)}ms / +{int(PADDING_AFTER*1000)}ms")
    print(f"Speech threshold: {SPEECH_THRESHOLD_DBFS:.1f} dBFS  | onset search: first {MAX_ONSET_SEARCH_SEC:.1f}s")
    print("")

    stats = {"total": len(wavs), "A": 0, "B": 0, "C": 0, "rejected": 0, "silent": 0, "no_onset": 0, "low_cov": 0, "errors": 0}

    for i, w in enumerate(wavs, 1):
        if bak:
            shutil.copy2(w, bak / w.name)

        r = process_file(w, outA, outB, outC)

        if r.success:
            stats[r.bucket] += 1
            print(f"[{i:4d}/{len(wavs)}] ✓ {w.name:30s} | {r.bucket} | onset {r.onset_delay*1000:4.0f}ms | cov {r.speech_coverage*100:2.0f}% | seg {r.num_segments}")
        else:
            stats["rejected"] += 1
            reason = r.rejection_reason or "reject"
            if "silent" in reason:
                stats["silent"] += 1
            elif "no_onset" in reason:
                stats["no_onset"] += 1
            elif "coverage" in reason:
                stats["low_cov"] += 1
            elif "error" in reason:
                stats["errors"] += 1

            if rej:
                shutil.copy2(w, rej / w.name)

            print(f"[{i:4d}/{len(wavs)}] ✗ {w.name:30s} | REJECT {reason}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Total: {stats['total']}")
    print(f"A: {stats['A']} | B: {stats['B']} | C: {stats['C']}")
    print(f"Rejected: {stats['rejected']} (silent {stats['silent']}, no_onset {stats['no_onset']}, low_cov {stats['low_cov']}, errors {stats['errors']})")
    print("\nOutputs:")
    print(f"  {outA}")
    print(f"  {outB}")
    print(f"  {outC}")
    if rej:
        print(f"  {rej}")


if __name__ == "__main__":
    # Windows defaults
    if os.name == "nt":
        default_input = r"data\positive"
        default_out = r"data\positive_trimmed_1p2"
        default_rej = r"data\rejected"
        default_bak = r"data\positive_backup"
    else:
        default_input = "data/positive"
        default_out = "data/positive_trimmed_1p2"
        default_rej = "data/rejected"
        default_bak = "data/positive_backup"

    print("=" * 70)
    print("🎤 HEY MAGIS WAKE WORD TRIMMER v2.3 (1.2s)")
    print("=" * 70)
    print("1) Process all recordings (A/B/C buckets + rejected)")
    print("2) Process with backup")
    print("")

    choice = input("Choose option (1-2): ").strip()

    if choice == "2":
        inp = input(f"Input directory [{default_input}]: ").strip() or default_input
        outb = input(f"Output base [{default_out}]: ").strip() or default_out
        rej = input(f"Rejected directory [{default_rej}]: ").strip() or default_rej
        bak = input(f"Backup directory [{default_bak}]: ").strip() or default_bak
        process_dir(inp, outb, rejected_dir=rej, backup_dir=bak)
    else:
        inp = input(f"Input directory [{default_input}]: ").strip() or default_input
        outb = input(f"Output base [{default_out}]: ").strip() or default_out
        rej = input(f"Rejected directory [{default_rej}]: ").strip() or default_rej
        process_dir(inp, outb, rejected_dir=rej, backup_dir=None)

    print("\n✅ Done!")

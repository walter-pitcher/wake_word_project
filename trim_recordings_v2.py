#!/usr/bin/env python3
"""
Hey Magis Wake Word Audio Trimmer v2.3
======================================

Goal:
- Produce clean, consistent 1.0s positive clips for wake-word training.
- Robust onset/offset detection using frame RMS in dBFS (absolute scale).
- Avoid per-sample spike triggering; avoid relative-to-max dB.

Trimming rules:
- Detect speech onset using RMS frame analysis (25ms frames, 10ms hop)
- Onset requires MIN_SPEECH_FRAMES consecutive "speech" frames
- Extract [-150ms, +850ms] window around onset (1.0s total)
- Hard-trim or pad to exactly 1.0s
- Reject clips with:
  - Silent/empty recording
  - No speech detected (onset not found)
  - More than MAX_SPEECH_SEGMENTS speech segments
  - Phrase longer than MAX_SPEECH_DURATION
  - Trailing silence > MAX_TRAILING_SILENCE (only if original clip > 1.0s)
  - Low speech coverage (< MIN_SPEECH_COVERAGE)

v2.3 Improvements (per your request):
- Fix offset detection using a reverse scan (stable, no offset_frame=None weirdness)
- Segment counting requires a minimum speech run length (filters tiny breath/click runs)
- Compute RMS/is_speech ONCE per file and reuse (consistent + faster)
- Clamp trailing silence and handle all edge cases cleanly
- PCM16 output enforced
"""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import librosa
import numpy as np
import soundfile as sf


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000

TARGET_DURATION = 1.0
PADDING_BEFORE = 0.150
PADDING_AFTER = 0.850  # PADDING_BEFORE + PADDING_AFTER == 1.0s

FRAME_LENGTH_MS = 25
HOP_LENGTH_MS = 10

# Detection thresholds (dBFS, where 0 dBFS == full-scale 1.0 float)
SPEECH_THRESHOLD_DBFS = -35.0
SILENCE_THRESHOLD_DBFS = -45.0

# Onset/offset/segment robustness
MIN_SPEECH_FRAMES = 3            # onset requires 3 consecutive speech frames (~30ms)
OFFSET_TAIL_MS = 50              # add 50ms tail after offset (then clamp)
MIN_SILENCE_GAP = 0.100          # 100ms silence gap => new segment
MIN_SPEECH_RUN_MS = 50           # a "segment" must have at least 50ms of speech
MIN_SPEECH_COVERAGE = 0.15       # at least 15% of frames between onset & offset are speech

# Rejection thresholds
MAX_SPEECH_DURATION = 0.850
MAX_TRAILING_SILENCE = 0.300
MAX_SPEECH_SEGMENTS = 1


# ──────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────────────────────────

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


@dataclass
class ProcessingResult:
    success: bool
    filename: str
    rejection_reason: Optional[str] = None
    onset_delay: float = 0.0
    speech_duration: float = 0.0
    speech_coverage: float = 0.0
    num_segments: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _frame_params(sr: int) -> Tuple[int, int]:
    frame_length = int(FRAME_LENGTH_MS * sr / 1000)
    hop_length = int(HOP_LENGTH_MS * sr / 1000)
    return frame_length, hop_length


def compute_frame_rms_dbfs(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      rms_dbfs: shape [num_frames]
      frame_times: seconds for each frame index
    """
    frame_length, hop_length = _frame_params(sr)
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms_dbfs = 20.0 * np.log10(rms + 1e-9)
    frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    return rms_dbfs, frame_times


def compute_levels_dbfs(audio: np.ndarray) -> Tuple[float, float]:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
    peak_dbfs = 20.0 * np.log10(peak + 1e-9)
    rms_dbfs = 20.0 * np.log10(rms + 1e-9)
    return peak_dbfs, rms_dbfs


def is_silent_recording(audio: np.ndarray, silence_threshold_dbfs: float) -> bool:
    """
    "Silent" means overall RMS is below threshold OR audio is effectively empty/corrupt.
    """
    if audio is None or len(audio) == 0:
        return True

    peak = float(np.max(np.abs(audio)))
    if peak < 0.001:
        return True

    rms = float(np.sqrt(np.mean(audio**2)))
    rms_dbfs = 20.0 * np.log10(rms + 1e-9)
    return rms_dbfs < silence_threshold_dbfs


def find_onset_frame(is_speech: np.ndarray, min_consecutive: int) -> Optional[int]:
    """
    First frame index where we have min_consecutive consecutive True values.
    """
    consec = 0
    for i, s in enumerate(is_speech):
        if s:
            consec += 1
            if consec >= min_consecutive:
                return i - min_consecutive + 1
        else:
            consec = 0
    return None


def find_offset_frame_reverse(is_speech: np.ndarray, min_consecutive: int) -> Optional[int]:
    """
    Stable offset detection:
    Find the last run of speech frames that is at least min_consecutive long,
    returning the END index of that run.
    """
    consec = 0
    for i in range(len(is_speech) - 1, -1, -1):
        if is_speech[i]:
            consec += 1
            if consec >= min_consecutive:
                # i is the START of the found run; the END is i + consec - 1
                return i + consec - 1
        else:
            consec = 0
    return None


def compute_speech_coverage(is_speech: np.ndarray, onset_frame: int, offset_frame: int) -> float:
    """
    Speech coverage ratio between onset and offset frames inclusive.
    """
    onset_frame = max(0, min(onset_frame, len(is_speech) - 1))
    offset_frame = max(onset_frame, min(offset_frame, len(is_speech) - 1))
    window = is_speech[onset_frame: offset_frame + 1]
    if window.size == 0:
        return 0.0
    return float(np.sum(window)) / float(window.size)


def count_speech_segments(
    is_speech: np.ndarray,
    sr: int,
    min_silence_gap_s: float,
    min_speech_run_ms: float
) -> int:
    """
    Count distinct speech segments separated by >= min_silence_gap_s of silence.
    Also require each speech run to be >= min_speech_run_ms to count as a segment
    (filters tiny breath/click runs).
    """
    if not np.any(is_speech):
        return 0

    _, hop_length = _frame_params(sr)
    min_gap_frames = int(min_silence_gap_s * sr / hop_length)
    min_run_frames = int((min_speech_run_ms / 1000.0) * sr / hop_length)
    min_run_frames = max(1, min_run_frames)

    segments = 0
    in_speech = False
    silence_count = 0
    speech_run = 0

    for s in is_speech:
        if s:
            speech_run += 1
            silence_count = 0
            if not in_speech:
                in_speech = True
        else:
            if in_speech:
                silence_count += 1
                # end segment if gap is long enough
                if silence_count >= min_gap_frames:
                    # finalize the speech run as a segment only if long enough
                    if speech_run >= min_run_frames:
                        segments += 1
                    in_speech = False
                    silence_count = 0
                    speech_run = 0
            else:
                # already in silence
                pass

    # finalize if ended while still in speech
    if in_speech and speech_run >= min_run_frames:
        segments += 1

    return segments


def extract_window(audio: np.ndarray, onset_time: float, sr: int) -> np.ndarray:
    """
    Extract [onset - PADDING_BEFORE, onset + PADDING_AFTER] and force exact 1.0s.
    """
    target_samples = int((PADDING_BEFORE + PADDING_AFTER) * sr)
    onset_idx = int(onset_time * sr)

    start_idx = onset_idx - int(PADDING_BEFORE * sr)
    end_idx = onset_idx + int(PADDING_AFTER * sr)

    pad_before = 0
    pad_after = 0

    if start_idx < 0:
        pad_before = -start_idx
        start_idx = 0

    if end_idx > len(audio):
        pad_after = end_idx - len(audio)
        end_idx = len(audio)

    extracted = audio[start_idx:end_idx]

    if pad_before > 0 or pad_after > 0:
        extracted = np.pad(extracted, (pad_before, pad_after), mode="constant")

    # exact length
    if len(extracted) > target_samples:
        extracted = extracted[:target_samples]
    elif len(extracted) < target_samples:
        extracted = np.pad(extracted, (0, target_samples - len(extracted)), mode="constant")

    return extracted


def normalize_peak(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak <= 0:
        return audio
    return audio * (target_peak / peak)


# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def analyze_audio(audio: np.ndarray, sr: int) -> AudioAnalysis:
    peak_dbfs, rms_dbfs = compute_levels_dbfs(audio)

    if is_silent_recording(audio, SILENCE_THRESHOLD_DBFS):
        return AudioAnalysis(
            onset_time=None,
            offset_time=None,
            speech_duration=0.0,
            trailing_silence=0.0,
            num_segments=0,
            speech_coverage=0.0,
            is_silent=True,
            peak_dbfs=peak_dbfs,
            rms_dbfs=rms_dbfs,
            rejection_reason="Silent/empty recording"
        )

    rms_frames_dbfs, frame_times = compute_frame_rms_dbfs(audio, sr)
    is_speech = rms_frames_dbfs > SPEECH_THRESHOLD_DBFS

    onset_frame = find_onset_frame(is_speech, MIN_SPEECH_FRAMES)
    if onset_frame is None:
        return AudioAnalysis(
            onset_time=None,
            offset_time=None,
            speech_duration=0.0,
            trailing_silence=0.0,
            num_segments=0,
            speech_coverage=0.0,
            is_silent=False,
            peak_dbfs=peak_dbfs,
            rms_dbfs=rms_dbfs,
            rejection_reason="No speech detected (onset not found)"
        )

    offset_frame = find_offset_frame_reverse(is_speech, MIN_SPEECH_FRAMES)
    if offset_frame is None:
        return AudioAnalysis(
            onset_time=None,
            offset_time=None,
            speech_duration=0.0,
            trailing_silence=0.0,
            num_segments=0,
            speech_coverage=0.0,
            is_silent=False,
            peak_dbfs=peak_dbfs,
            rms_dbfs=rms_dbfs,
            rejection_reason="No speech detected (offset not found)"
        )

    # Convert frames to times
    onset_time = float(frame_times[onset_frame]) if onset_frame < len(frame_times) else 0.0
    offset_time = float(frame_times[offset_frame]) if offset_frame < len(frame_times) else (len(audio) / sr)

    # add tail and clamp
    offset_time = min(offset_time + (OFFSET_TAIL_MS / 1000.0), len(audio) / sr)

    speech_duration = float(offset_time - onset_time)
    audio_duration = float(len(audio) / sr)
    trailing_silence = max(0.0, audio_duration - offset_time)

    # segments (requires minimum run length)
    num_segments = count_speech_segments(
        is_speech=is_speech,
        sr=sr,
        min_silence_gap_s=MIN_SILENCE_GAP,
        min_speech_run_ms=MIN_SPEECH_RUN_MS
    )

    # coverage computed from frame indices; ensure offset >= onset
    if offset_frame < onset_frame:
        offset_frame = onset_frame
    speech_coverage = compute_speech_coverage(is_speech, onset_frame, offset_frame)

    # rejection logic
    rejection_reason = None
    if num_segments == 0:
        rejection_reason = "No speech detected (no segments)"
    elif num_segments > MAX_SPEECH_SEGMENTS:
        rejection_reason = f"Multiple speech segments ({num_segments} detected, max {MAX_SPEECH_SEGMENTS})"
    elif speech_duration > MAX_SPEECH_DURATION:
        rejection_reason = f"Phrase too long ({speech_duration*1000:.0f}ms > {MAX_SPEECH_DURATION*1000:.0f}ms)"
    elif (audio_duration > TARGET_DURATION) and (trailing_silence > MAX_TRAILING_SILENCE):
        rejection_reason = f"Trailing silence too long ({trailing_silence*1000:.0f}ms > {MAX_TRAILING_SILENCE*1000:.0f}ms)"
    elif speech_coverage < MIN_SPEECH_COVERAGE:
        rejection_reason = f"Low speech coverage ({speech_coverage*100:.0f}% < {MIN_SPEECH_COVERAGE*100:.0f}%)"

    return AudioAnalysis(
        onset_time=onset_time,
        offset_time=offset_time,
        speech_duration=speech_duration,
        trailing_silence=trailing_silence,
        num_segments=num_segments,
        speech_coverage=speech_coverage,
        is_silent=False,
        peak_dbfs=peak_dbfs,
        rms_dbfs=rms_dbfs,
        rejection_reason=rejection_reason
    )


# ──────────────────────────────────────────────────────────────────────────────
# PROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def process_audio_file(input_path: Path, output_path: Path) -> ProcessingResult:
    try:
        audio, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)

        analysis = analyze_audio(audio, sr)

        if analysis.is_silent or analysis.onset_time is None or analysis.rejection_reason:
            return ProcessingResult(
                success=False,
                filename=input_path.name,
                rejection_reason=analysis.rejection_reason or "Rejected",
                onset_delay=float(analysis.onset_time or 0.0),
                speech_duration=float(analysis.speech_duration),
                speech_coverage=float(analysis.speech_coverage),
                num_segments=int(analysis.num_segments),
            )

        processed = extract_window(audio, analysis.onset_time, sr)
        processed = normalize_peak(processed, target_peak=0.95)

        # Force PCM16 output
        sf.write(output_path, processed, sr, subtype="PCM_16")

        return ProcessingResult(
            success=True,
            filename=input_path.name,
            onset_delay=float(analysis.onset_time),
            speech_duration=float(analysis.speech_duration),
            speech_coverage=float(analysis.speech_coverage),
            num_segments=int(analysis.num_segments),
        )

    except Exception as e:
        return ProcessingResult(
            success=False,
            filename=input_path.name,
            rejection_reason=f"Processing error: {e}"
        )


def _default_paths() -> Tuple[str, str, str, str]:
    if os.name == "nt":
        return (r"data\positive", r"data\positive_trimmed", r"data\rejected", r"data\positive_backup")
    return ("data/positive", "data/positive_trimmed", "data/rejected", "data/positive_backup")


def process_all_recordings(
    input_dir: str,
    output_dir: str,
    rejected_dir: Optional[str] = None,
    backup_dir: Optional[str] = None
) -> Dict[str, float]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rejected_path = None
    if rejected_dir:
        rejected_path = Path(rejected_dir)
        rejected_path.mkdir(parents=True, exist_ok=True)

    backup_path = None
    if backup_dir:
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(list(input_path.glob("*.wav")))

    print("\n" + "═" * 78)
    print(f"Processing {len(wav_files)} WAV files")
    print("═" * 78)
    print(f"Target duration: {TARGET_DURATION:.1f}s  |  Window: [-{PADDING_BEFORE*1000:.0f}ms, +{PADDING_AFTER*1000:.0f}ms]")
    print(f"Speech threshold: {SPEECH_THRESHOLD_DBFS:.0f} dBFS  |  Silence threshold: {SILENCE_THRESHOLD_DBFS:.0f} dBFS")
    print(f"Max speech duration: {MAX_SPEECH_DURATION*1000:.0f}ms  |  Max trailing: {MAX_TRAILING_SILENCE*1000:.0f}ms")
    print(f"Min speech coverage: {MIN_SPEECH_COVERAGE*100:.0f}%  |  Max segments: {MAX_SPEECH_SEGMENTS}")
    print("═" * 78)

    stats = {
        "total": len(wav_files),
        "accepted": 0,
        "rejected": 0,
        "silent": 0,
        "no_speech": 0,
        "multi_segment": 0,
        "too_long": 0,
        "trailing_silence": 0,
        "low_coverage": 0,
        "errors": 0,
        "total_onset_delay": 0.0,
        "max_onset_delay": 0.0,
        "total_coverage": 0.0,
    }

    for i, wav_file in enumerate(wav_files, 1):
        if backup_path:
            shutil.copy2(wav_file, backup_path / wav_file.name)

        out_file = output_path / wav_file.name
        result = process_audio_file(wav_file, out_file)

        if result.success:
            stats["accepted"] += 1
            stats["total_onset_delay"] += result.onset_delay
            stats["max_onset_delay"] = max(stats["max_onset_delay"], result.onset_delay)
            stats["total_coverage"] += result.speech_coverage
            status = "✓"
            print(
                f"[{i:4d}/{len(wav_files)}] {status} {wav_file.name[:32]:32s}"
                f" | onset: {result.onset_delay*1000:4.0f}ms"
                f" | speech: {result.speech_duration*1000:4.0f}ms"
                f" | cov: {result.speech_coverage*100:2.0f}%"
            )
        else:
            stats["rejected"] += 1
            reason = (result.rejection_reason or "").lower()
            status = "✗"

            if "silent" in reason or "empty" in reason:
                stats["silent"] += 1
            elif "onset not found" in reason or "no speech" in reason:
                stats["no_speech"] += 1
            elif "multiple speech segments" in reason or "segments" in reason:
                stats["multi_segment"] += 1
            elif "phrase too long" in reason:
                stats["too_long"] += 1
            elif "trailing silence" in reason:
                stats["trailing_silence"] += 1
            elif "coverage" in reason:
                stats["low_coverage"] += 1
            elif "processing error" in reason or "error" in reason:
                stats["errors"] += 1

            if rejected_path:
                shutil.copy2(wav_file, rejected_path / wav_file.name)

            if out_file.exists():
                out_file.unlink()

            print(f"[{i:4d}/{len(wav_files)}] {status} {wav_file.name[:32]:32s} | REJECTED: {result.rejection_reason}")

    print("\n" + "═" * 78)
    print("PROCESSING COMPLETE")
    print("═" * 78)
    total = max(1, stats["total"])
    print(f"✓ Accepted: {stats['accepted']}/{stats['total']} ({100*stats['accepted']/total:.1f}%)")
    print(f"✗ Rejected: {stats['rejected']}/{stats['total']} ({100*stats['rejected']/total:.1f}%)")

    if stats["rejected"] > 0:
        print("\nRejection breakdown:")
        print(f"  - Silent/empty recordings:    {stats['silent']}")
        print(f"  - No speech detected:         {stats['no_speech']}")
        print(f"  - Multiple speech segments:   {stats['multi_segment']}")
        print(f"  - Phrase too long:            {stats['too_long']}")
        print(f"  - Trailing silence:           {stats['trailing_silence']}")
        print(f"  - Low speech coverage:        {stats['low_coverage']}")
        print(f"  - Processing errors:          {stats['errors']}")

    if stats["accepted"] > 0:
        avg_delay = stats["total_onset_delay"] / stats["accepted"]
        avg_cov = stats["total_coverage"] / stats["accepted"]
        print("\nAccepted file stats:")
        print(f"  - Average onset delay trimmed: {avg_delay*1000:.0f}ms")
        print(f"  - Maximum onset delay trimmed: {stats['max_onset_delay']*1000:.0f}ms")
        print(f"  - Average speech coverage:     {avg_cov*100:.0f}%")

    print(f"\n📁 Processed files saved to: {output_path}")
    if rejected_path:
        print(f"📁 Rejected files copied to: {rejected_path}")
    if backup_path:
        print(f"📁 Original files backed up to: {backup_path}")

    print(f"\n📊 USABLE RECORDINGS: {stats['accepted']} files")
    return stats


def analyze_recordings(input_dir: str) -> List[dict]:
    input_path = Path(input_dir)
    wav_files = sorted(list(input_path.glob("*.wav")))

    print("\n" + "═" * 78)
    print(f"Analyzing {len(wav_files)} recordings")
    print("═" * 78)
    print(f"Speech threshold: {SPEECH_THRESHOLD_DBFS:.0f} dBFS | Silence threshold: {SILENCE_THRESHOLD_DBFS:.0f} dBFS")
    print(f"Min speech coverage: {MIN_SPEECH_COVERAGE*100:.0f}% | Min speech run: {MIN_SPEECH_RUN_MS}ms")
    print("═" * 78)

    results: List[dict] = []

    for i, wav_file in enumerate(wav_files, 1):
        try:
            audio, sr = librosa.load(wav_file, sr=SAMPLE_RATE, mono=True)
            a = analyze_audio(audio, sr)
            would_accept = (not a.is_silent) and (a.rejection_reason is None) and (a.onset_time is not None)

            row = {
                "file": wav_file.name,
                "would_accept": would_accept,
                "rejection_reason": a.rejection_reason,
                "onset_delay": float(a.onset_time or 0.0),
                "speech_duration": float(a.speech_duration),
                "trailing_silence": float(a.trailing_silence),
                "num_segments": int(a.num_segments),
                "speech_coverage": float(a.speech_coverage),
                "peak_dbfs": float(a.peak_dbfs),
                "rms_dbfs": float(a.rms_dbfs),
            }
            results.append(row)

            status = "✓" if would_accept else "✗"
            print(
                f"[{i:4d}/{len(wav_files)}] {status} {wav_file.name[:30]:30s}"
                f" | RMS: {a.rms_dbfs:5.1f}dB"
                f" | seg: {a.num_segments}"
                f" | cov: {a.speech_coverage*100:2.0f}%"
                + (f" | onset: {(a.onset_time or 0)*1000:4.0f}ms | speech: {a.speech_duration*1000:4.0f}ms"
                   if would_accept else f" | {a.rejection_reason}")
            )
        except Exception as e:
            results.append({"file": wav_file.name, "would_accept": False, "rejection_reason": f"Error: {e}"})
            print(f"[{i:4d}/{len(wav_files)}] ✗ {wav_file.name[:30]:30s} | Error: {e}")

    accepted = sum(1 for r in results if r.get("would_accept"))
    total = max(1, len(results))
    rejected = total - accepted

    print("\n" + "═" * 78)
    print("ANALYSIS SUMMARY")
    print("═" * 78)
    print(f"Would accept: {accepted}/{total} ({100*accepted/total:.1f}%)")
    print(f"Would reject: {rejected}/{total} ({100*rejected/total:.1f}%)")

    # Breakdown
    if rejected > 0:
        buckets: Dict[str, int] = {}
        for r in results:
            if r.get("would_accept"):
                continue
            reason = (r.get("rejection_reason") or "Unknown")
            rl = reason.lower()
            if "silent" in rl or "empty" in rl:
                key = "Silent/empty"
            elif "onset not found" in rl or "no speech" in rl:
                key = "No speech detected"
            elif "multiple speech segments" in rl or "segments" in rl:
                key = "Multiple segments"
            elif "phrase too long" in rl:
                key = "Phrase too long"
            elif "trailing silence" in rl:
                key = "Trailing silence"
            elif "coverage" in rl:
                key = "Low speech coverage"
            elif "error" in rl:
                key = "Other/Error"
            else:
                key = "Other/Error"
            buckets[key] = buckets.get(key, 0) + 1

        print("\nRejection breakdown:")
        for k, v in sorted(buckets.items(), key=lambda kv: -kv[1]):
            print(f"  - {k}: {v}")

    # Basic tuning hint
    acceptance_rate = accepted / total
    print("\n" + "═" * 78)
    print("TUNING SUGGESTIONS")
    print("═" * 78)
    if acceptance_rate < 0.70:
        print("⚠️  Acceptance < 70%:")
        print(f"   - Try lowering SPEECH_THRESHOLD_DBFS (e.g., -38 or -40). Current: {SPEECH_THRESHOLD_DBFS}")
        print(f"   - If 'Low speech coverage' dominates, consider lowering MIN_SPEECH_COVERAGE (e.g., 0.12). Current: {MIN_SPEECH_COVERAGE}")
        print(f"   - If 'Phrase too long' dominates, increase MAX_SPEECH_DURATION (e.g., 0.95). Current: {MAX_SPEECH_DURATION}")
    elif acceptance_rate > 0.95:
        print("⚠️  Acceptance > 95%: thresholds may be too loose (risk of keeping junk).")
    else:
        print(f"✓ Acceptance looks reasonable ({acceptance_rate*100:.0f}%).")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    default_input, default_output, default_rejected, default_backup = _default_paths()

    print("=" * 78)
    print("🎤 HEY MAGIS WAKE WORD TRIMMER v2.3")
    print("=" * 78)
    print("\nTrimming rules:")
    print(f"  • Onset: RMS frames, {MIN_SPEECH_FRAMES} consecutive speech frames")
    print(f"  • Window: [-{PADDING_BEFORE*1000:.0f}ms, +{PADDING_AFTER*1000:.0f}ms] => {TARGET_DURATION:.1f}s")
    print(f"  • Reject: silent/no-speech, >{MAX_SPEECH_SEGMENTS} seg, >{MAX_SPEECH_DURATION*1000:.0f}ms, "
          f"> {MAX_TRAILING_SILENCE*1000:.0f}ms trailing, <{MIN_SPEECH_COVERAGE*100:.0f}% coverage")
    print("\nThresholds (dBFS):")
    print(f"  • Speech:  {SPEECH_THRESHOLD_DBFS} dBFS")
    print(f"  • Silence: {SILENCE_THRESHOLD_DBFS} dBFS")
    print("\nOptions:")
    print("1. Analyze recordings (preview accept/reject + tuning suggestions)")
    print("2. Process all recordings (trim accepted, copy rejected)")
    print("3. Process with backup (keeps originals)")
    print("4. Custom paths\n")

    choice = input("Choose option (1-4): ").strip()

    if choice == "1":
        input_dir = input(f"Input directory [{default_input}]: ").strip() or default_input
        analyze_recordings(input_dir)

    elif choice == "2":
        input_dir = input(f"Input directory [{default_input}]: ").strip() or default_input
        output_dir = input(f"Output directory [{default_output}]: ").strip() or default_output
        rejected_dir = input(f"Rejected files directory [{default_rejected}]: ").strip() or default_rejected
        process_all_recordings(input_dir, output_dir, rejected_dir, backup_dir=None)

    elif choice == "3":
        input_dir = input(f"Input directory [{default_input}]: ").strip() or default_input
        output_dir = input(f"Output directory [{default_output}]: ").strip() or default_output
        rejected_dir = input(f"Rejected files directory [{default_rejected}]: ").strip() or default_rejected
        backup_dir = input(f"Backup directory [{default_backup}]: ").strip() or default_backup
        process_all_recordings(input_dir, output_dir, rejected_dir, backup_dir=backup_dir)

    elif choice == "4":
        input_dir = input("Input directory: ").strip()
        output_dir = input("Output directory: ").strip()
        rejected_dir = input("Rejected directory (or Enter to skip): ").strip() or None
        backup_dir = input("Backup directory (or Enter to skip): ").strip() or None
        process_all_recordings(input_dir, output_dir, rejected_dir=rejected_dir, backup_dir=backup_dir)

    print("\n✅ Done!")

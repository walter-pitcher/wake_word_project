#!/usr/bin/env python3
"""
Background-Noise Expansion — MagisAI Wake Word v6
===================================================
Generates 17 synthetic noise files (60 s each, 16 kHz mono) covering:

  - Room tone, HVAC / fan hum, cafe / crowd babble
  - Outdoor ambience (wind, traffic, park)
  - TV murmur, appliance hum, rain

These complement the 6 existing files in _background_noise_/ and are
written to _background_noise_expanded_/.

Usage:
    python new/generate_noise.py
"""

import sys, os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal as sig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SAMPLE_RATE, EXPANDED_NOISE_DIR

DURATION_SEC = 60
NUM_SAMPLES  = SAMPLE_RATE * DURATION_SEC


def _white(n):
    return np.random.randn(n).astype(np.float32)


def _pink(n):
    uneven = n % 2
    x = np.random.randn(n // 2 + 1 + uneven) + 1j * np.random.randn(n // 2 + 1 + uneven)
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
    freqs[0] = 1.0
    x = x / np.sqrt(freqs)
    out = np.fft.irfft(x, n=n).astype(np.float32)
    return out / (np.max(np.abs(out)) + 1e-8)


def _brown(n):
    return np.cumsum(_white(n)).astype(np.float32)


def _bandpass(audio, low, high, order=4):
    sos = sig.butter(order, [low, high], btype="band", fs=SAMPLE_RATE, output="sos")
    return sig.sosfilt(sos, audio).astype(np.float32)


def _lowpass(audio, cutoff, order=4):
    sos = sig.butter(order, cutoff, btype="low", fs=SAMPLE_RATE, output="sos")
    return sig.sosfilt(sos, audio).astype(np.float32)


def _normalize(audio, peak=0.7):
    mx = np.max(np.abs(audio)) + 1e-8
    return (audio / mx * peak).astype(np.float32)


# ───────────────────── noise generators ──────────────────────────

def room_tone():
    return _normalize(_pink(NUM_SAMPLES), peak=0.05)

def room_tone_2():
    return _normalize(_white(NUM_SAMPLES) * 0.3 + _pink(NUM_SAMPLES) * 0.7, peak=0.06)

def hvac_fan_1():
    n = _brown(NUM_SAMPLES)
    n = _lowpass(n, 500)
    t = np.arange(NUM_SAMPLES) / SAMPLE_RATE
    n += 0.15 * np.sin(2 * np.pi * 120 * t).astype(np.float32)
    return _normalize(n, peak=0.25)

def hvac_fan_2():
    n = _brown(NUM_SAMPLES)
    n = _lowpass(n, 350)
    t = np.arange(NUM_SAMPLES) / SAMPLE_RATE
    n += 0.1 * np.sin(2 * np.pi * 60 * t).astype(np.float32)
    n += 0.05 * np.sin(2 * np.pi * 180 * t).astype(np.float32)
    return _normalize(n, peak=0.20)

def cafe_babble_1():
    out = np.zeros(NUM_SAMPLES, dtype=np.float32)
    for _ in range(8):
        voice = _bandpass(_pink(NUM_SAMPLES), 200, 4000)
        env = np.abs(_lowpass(_white(NUM_SAMPLES), 3)) + 0.3
        out += voice * env
    return _normalize(out, peak=0.35)

def cafe_babble_2():
    out = np.zeros(NUM_SAMPLES, dtype=np.float32)
    for _ in range(12):
        voice = _bandpass(_pink(NUM_SAMPLES), 150, 3500)
        env = np.abs(_lowpass(_white(NUM_SAMPLES), 2)) + 0.2
        out += voice * env
    clink = _white(NUM_SAMPLES) * 0.02
    return _normalize(out + clink, peak=0.30)

def outdoor_wind():
    w = _brown(NUM_SAMPLES)
    w = _lowpass(w, 800)
    gusts = np.abs(_lowpass(_white(NUM_SAMPLES), 0.5))
    w = w * (0.3 + 0.7 * gusts)
    return _normalize(w, peak=0.30)

def outdoor_traffic():
    base = _brown(NUM_SAMPLES)
    base = _bandpass(base, 50, 2000)
    t = np.arange(NUM_SAMPLES) / SAMPLE_RATE
    rumble = 0.2 * np.sin(2 * np.pi * 40 * t).astype(np.float32)
    return _normalize(base + rumble, peak=0.25)

def outdoor_park():
    wind = _lowpass(_brown(NUM_SAMPLES), 600) * 0.3
    birds = _bandpass(_pink(NUM_SAMPLES), 2000, 6000) * 0.15
    env = np.abs(_lowpass(_white(NUM_SAMPLES), 1.5))
    birds = birds * env
    return _normalize(wind + birds, peak=0.20)

def tv_murmur_1():
    voice = _bandpass(_pink(NUM_SAMPLES), 200, 5000)
    env = np.abs(_lowpass(_white(NUM_SAMPLES), 4)) + 0.2
    return _normalize(voice * env, peak=0.25)

def tv_murmur_2():
    voice = _bandpass(_pink(NUM_SAMPLES), 300, 4500)
    music = _bandpass(_pink(NUM_SAMPLES), 100, 7000) * 0.15
    env = np.abs(_lowpass(_white(NUM_SAMPLES), 3)) + 0.25
    return _normalize(voice * env + music, peak=0.25)

def appliance_hum_60():
    t = np.arange(NUM_SAMPLES, dtype=np.float32) / SAMPLE_RATE
    hum = np.zeros(NUM_SAMPLES, dtype=np.float32)
    for h in [60, 120, 180, 240]:
        hum += (0.5 / (h / 60)) * np.sin(2 * np.pi * h * t + np.random.uniform(0, 2 * np.pi))
    hum += _white(NUM_SAMPLES) * 0.02
    return _normalize(hum, peak=0.15)

def appliance_hum_50():
    t = np.arange(NUM_SAMPLES, dtype=np.float32) / SAMPLE_RATE
    hum = np.zeros(NUM_SAMPLES, dtype=np.float32)
    for h in [50, 100, 150, 200]:
        hum += (0.5 / (h / 50)) * np.sin(2 * np.pi * h * t + np.random.uniform(0, 2 * np.pi))
    hum += _white(NUM_SAMPLES) * 0.02
    return _normalize(hum, peak=0.15)

def white_noise_light():
    return _normalize(_white(NUM_SAMPLES), peak=0.10)

def mixed_indoor():
    hvac = _lowpass(_brown(NUM_SAMPLES), 400) * 0.4
    t = np.arange(NUM_SAMPLES, dtype=np.float32) / SAMPLE_RATE
    hum  = 0.08 * np.sin(2 * np.pi * 60 * t).astype(np.float32)
    babble = _bandpass(_pink(NUM_SAMPLES), 200, 3000) * 0.1
    return _normalize(hvac + hum + babble, peak=0.22)

def rain_light():
    r = _white(NUM_SAMPLES)
    r = _bandpass(r, 500, 6000)
    env = np.abs(_lowpass(_white(NUM_SAMPLES), 2)) * 0.5 + 0.5
    return _normalize(r * env, peak=0.25)

def rain_heavy():
    r = _white(NUM_SAMPLES)
    r = _bandpass(r, 200, 7000)
    return _normalize(r, peak=0.35)


GENERATORS = [
    ("room_tone_1",       room_tone),
    ("room_tone_2",       room_tone_2),
    ("hvac_fan_1",        hvac_fan_1),
    ("hvac_fan_2",        hvac_fan_2),
    ("cafe_babble_1",     cafe_babble_1),
    ("cafe_babble_2",     cafe_babble_2),
    ("outdoor_wind",      outdoor_wind),
    ("outdoor_traffic",   outdoor_traffic),
    ("outdoor_park",      outdoor_park),
    ("tv_murmur_1",       tv_murmur_1),
    ("tv_murmur_2",       tv_murmur_2),
    ("appliance_hum_60",  appliance_hum_60),
    ("appliance_hum_50",  appliance_hum_50),
    ("white_noise_light", white_noise_light),
    ("mixed_indoor",      mixed_indoor),
    ("rain_light",        rain_light),
    ("rain_heavy",        rain_heavy),
]


def main():
    print("=" * 70)
    print("BACKGROUND-NOISE EXPANSION  v6")
    print("=" * 70)

    EXPANDED_NOISE_DIR.mkdir(parents=True, exist_ok=True)

    for name, gen_fn in GENERATORS:
        out_path = EXPANDED_NOISE_DIR / f"{name}.wav"
        audio = gen_fn()
        sf.write(str(out_path), audio.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")
        rms = np.sqrt(np.mean(audio ** 2))
        print(f"  {name:25s}  RMS={rms:.4f}  -> {out_path.name}")

    print(f"\nWrote {len(GENERATORS)} noise files to {EXPANDED_NOISE_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()

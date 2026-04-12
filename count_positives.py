#!/usr/bin/env python3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

folders = {
    "positive_new_friends": DATA_DIR / "positive_new_friends",
    "positive_trimmed_1p5_A": DATA_DIR / "positive_trimmed_1p5_A",
}

total = 0
for name, folder in folders.items():
    if not folder.exists():
        print(f"{name}: FOLDER NOT FOUND")
        continue
    
    wavs = list(folder.rglob("*.wav"))
    print(f"\n{name}: {len(wavs)} total wav files")
    
    # Show subfolder breakdown
    subfolders = [f for f in folder.iterdir() if f.is_dir()]
    for sub in sorted(subfolders):
        count = len(list(sub.rglob("*.wav")))
        print(f"  {sub.name}: {count} wavs")
    
    total += len(wavs)

print(f"\nGRAND TOTAL positives: {total}")
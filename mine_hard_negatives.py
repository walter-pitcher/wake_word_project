#!/usr/bin/env python3
import csv
import os
import shutil
from pathlib import Path

CSV_PATH = Path(r"models\eval_v3_scores_neg.csv")
OUT_DIR  = Path(r"data\hard_negatives_top")
TOP_N    = 3000   # start here; we can bump later
MIN_SCORE = 0.95  # optional filter; keep high-confidence false alarms

def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing: {CSV_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                score = float(r["score"])
                path = r["path"]
                rows.append((score, path))
            except Exception:
                continue

    rows.sort(key=lambda x: x[0], reverse=True)

    picked = []
    for score, path in rows:
        if score < MIN_SCORE:
            break
        picked.append((score, path))
        if len(picked) >= TOP_N:
            break

    print(f"Found {len(rows)} neg rows. Copying {len(picked)} hard negatives to {OUT_DIR}")

    copied = 0
    for i, (score, p) in enumerate(picked, 1):
        src = Path(p)
        if not src.exists():
            continue
        # keep filename unique-ish
        dst = OUT_DIR / f"{score:.6f}__{src.parent.name}__{src.name}"
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
        if i % 250 == 0:
            print(f"  {i}/{len(picked)}")

    print(f"✅ Copied: {copied}")
    print(f"📁 {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()

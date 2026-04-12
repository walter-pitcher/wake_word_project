#!/usr/bin/env python3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA = PROJECT_ROOT / "data"

HARD_MANIFEST_IN = DATA / "negative_manifest_hard.txt"
HARD_FOLDER = DATA / "hard_negatives_top"   # adjust if needed
HARD_MANIFEST_OUT = DATA / "negative_manifest_hard_merged.txt"

def normalize(p: Path) -> str:
    # store as relative paths from project root for portability
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(p)

def main():
    items = []

    # Existing hard manifest (if present)
    if HARD_MANIFEST_IN.exists():
        for ln in HARD_MANIFEST_IN.read_text(encoding="utf-8", errors="ignore").splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                items.append(ln)

    # Add all wavs from mined hard folder
    if HARD_FOLDER.exists():
        for wav in HARD_FOLDER.rglob("*.wav"):
            items.append(normalize(wav))

    # Deduplicate while preserving order
    seen = set()
    merged = []
    for x in items:
        if x not in seen:
            merged.append(x)
            seen.add(x)

    HARD_MANIFEST_OUT.write_text("\n".join(merged) + "\n", encoding="utf-8")
    print(f"✅ Wrote: {HARD_MANIFEST_OUT}")
    print(f"   total hard entries: {len(merged)}")
    print(f"   from old hard: {HARD_MANIFEST_IN.exists()}")
    print(f"   from folder: {HARD_FOLDER} (exists={HARD_FOLDER.exists()})")

if __name__ == "__main__":
    main()

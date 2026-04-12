#!/usr/bin/env python3
"""
Build negative manifests (recursive) + junk report.

What it does:
- Recursively finds .wav files under data/negative (and all subfolders)
- Writes a manifest file: one path per line (relative by default)
- Optionally writes a "hard negatives" manifest for folders like:
    - Hey_Magis_wrong_ways_to_say
    - wrong_ways
    - hard_negative
- Generates a report of "unnecessary items":
    - non-audio files (txt/json/md/log/etc.)
    - hidden/system files (Thumbs.db, .DS_Store, desktop.ini)
    - zero-byte files
    - very small wavs (likely corrupt/empty)
    - duplicate wavs (same file content hash)

Usage (Windows PowerShell):
  python build_negative_manifest.py --root "C:\\Users\\miker\\wake_word_project\\data\\negative" --out "data\\negative_manifest.txt"

Recommended:
  python build_negative_manifest.py --root "data\\negative" --out "data\\negative_manifest.txt" --also-hard --report "data\\negative_manifest_report.txt"

"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Tuple


SYSTEM_JUNK_NAMES = {
    "thumbs.db",
    "desktop.ini",
    ".ds_store",
}
AUDIO_EXTS = {".wav"}  # keep strict, since your pipeline is wav-based

# Heuristic: "hard negative" folders (near-miss phrases, etc.)
HARD_NEGATIVE_HINTS = {
    "hey_magis_wrong_ways_to_say",
    "wrong_ways",
    "hard_negative",
    "hard_negatives",
    "near_miss",
    "confusables",
}

DEFAULT_MIN_WAV_BYTES = 2048  # tiny wavs are often corrupt/empty; tweak as needed


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def is_hard_negative_path(path: Path) -> bool:
    parts = [p.lower() for p in path.parts]
    return any(hint in parts for hint in HARD_NEGATIVE_HINTS)


def normalize_output_path(
    path: Path,
    *,
    root: Path,
    style: str = "relative",
) -> str:
    """
    Return path formatted for manifest.
    style:
      - relative: relative to root's parent (keeps 'negative/...'), or to root itself if you prefer
      - absolute: full absolute path
      - posix: forward slashes
      - windows: backslashes
    """
    if style == "absolute":
        out = str(path.resolve())
    else:
        # relative to the project folder that contains /data (common case)
        # If root = .../data/negative, then parent = .../data, parent.parent = project root
        # We’ll prefer relative to project root if possible.
        project_root = root
        # Try: negative is likely root itself; project root might be two levels up (data/negative)
        # Safe approach: if ".../data/negative", use ".../" above "data"
        if root.name.lower() == "negative" and root.parent.name.lower() == "data":
            project_root = root.parent.parent
        else:
            # fallback: relative to root itself
            project_root = root

        try:
            out = str(path.resolve().relative_to(project_root.resolve()))
        except Exception:
            out = str(path)

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to negative folder (e.g. data\\negative)")
    ap.add_argument("--out", required=True, help="Output manifest path (txt)")
    ap.add_argument("--report", default="", help="Optional report output path (txt)")
    ap.add_argument("--also-hard", action="store_true", help="Also write a hard-negatives manifest")
    ap.add_argument("--hard-out", default="", help="Hard negatives manifest path (txt). If empty, auto-names.")
    ap.add_argument("--path-style", choices=["relative", "absolute"], default="relative", help="Manifest path style")
    ap.add_argument("--posix-slashes", action="store_true", help="Use forward slashes in manifest paths")
    ap.add_argument("--min-wav-bytes", type=int, default=DEFAULT_MIN_WAV_BYTES, help="Flag wavs smaller than this")
    ap.add_argument("--dedupe", action="store_true", help="Detect duplicates by file content hash")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Root folder not found or not a directory: {root}")

    out_manifest = Path(args.out).expanduser()
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    hard_manifest = None
    if args.also_hard:
        if args.hard_out:
            hard_manifest = Path(args.hard_out).expanduser()
        else:
            # auto-name next to main manifest
            hard_manifest = out_manifest.with_name(out_manifest.stem + "_hard" + out_manifest.suffix)
        hard_manifest.parent.mkdir(parents=True, exist_ok=True)

    report_path = Path(args.report).expanduser() if args.report else None
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)

    wav_paths: List[Path] = []
    hard_wav_paths: List[Path] = []
    junk_files: List[Tuple[str, Path]] = []
    suspicious_wavs: List[Tuple[str, Path]] = []

    # For duplicate detection
    hash_to_paths: Dict[str, List[Path]] = {}

    # Walk everything
    for p in root.rglob("*"):
        if p.is_dir():
            continue

        name_lower = p.name.lower()
        ext_lower = p.suffix.lower()

        # System junk
        if name_lower in SYSTEM_JUNK_NAMES:
            junk_files.append(("system/junk file", p))
            continue

        # Hidden-ish files (starts with dot)
        if name_lower.startswith(".") and ext_lower not in AUDIO_EXTS:
            junk_files.append(("hidden non-audio file", p))
            continue

        # Only accept wav
        if ext_lower in AUDIO_EXTS:
            # Basic sanity checks
            try:
                size = p.stat().st_size
            except Exception:
                size = -1

            if size == 0:
                suspicious_wavs.append(("zero-byte wav", p))
                continue

            if size > 0 and size < args.min_wav_bytes:
                suspicious_wavs.append((f"very small wav (<{args.min_wav_bytes} bytes)", p))
                # Keep it out of manifest by default—usually these are broken/empty.
                continue

            wav_paths.append(p)
            if args.also_hard and is_hard_negative_path(p):
                hard_wav_paths.append(p)

            if args.dedupe:
                try:
                    h = sha1_file(p)
                    hash_to_paths.setdefault(h, []).append(p)
                except Exception:
                    suspicious_wavs.append(("hashing failed", p))

        else:
            # non-wav files
            junk_files.append((f"non-wav file ({ext_lower or 'no extension'})", p))

    # Duplicate report
    duplicates: List[List[Path]] = []
    if args.dedupe:
        for h, paths in hash_to_paths.items():
            if len(paths) > 1:
                duplicates.append(paths)

    # Sort for stable manifests
    wav_paths_sorted = sorted(wav_paths, key=lambda x: str(x).lower())
    hard_wav_paths_sorted = sorted(hard_wav_paths, key=lambda x: str(x).lower())

    # Write manifests
    def write_manifest(paths: List[Path], dest: Path) -> None:
        lines = []
        for p in paths:
            s = normalize_output_path(p, root=root, style=args.path_style)
            if args.posix_slashes:
                s = s.replace("\\", "/")
            lines.append(s)
        dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    write_manifest(wav_paths_sorted, out_manifest)
    if hard_manifest is not None:
        write_manifest(hard_wav_paths_sorted, hard_manifest)

    # Build report
    report_lines: List[str] = []
    report_lines.append("=" * 78)
    report_lines.append("NEGATIVE MANIFEST BUILD REPORT")
    report_lines.append("=" * 78)
    report_lines.append(f"Root: {root.resolve()}")
    report_lines.append(f"Manifest: {out_manifest.resolve()}")
    if hard_manifest:
        report_lines.append(f"Hard-manifest: {hard_manifest.resolve()}")
    report_lines.append("")
    report_lines.append(f"Total wavs included: {len(wav_paths_sorted)}")
    if hard_manifest:
        report_lines.append(f"Hard-negative wavs included: {len(hard_wav_paths_sorted)}")
    report_lines.append(f"Suspicious wavs excluded: {len(suspicious_wavs)}")
    report_lines.append(f"Non-wav/junk files found: {len(junk_files)}")
    if args.dedupe:
        report_lines.append(f"Duplicate groups (by content hash): {len(duplicates)}")

    # Print key findings
    report_lines.append("")
    report_lines.append("-" * 78)
    report_lines.append("SUSPICIOUS WAVS (EXCLUDED)")
    report_lines.append("-" * 78)
    if suspicious_wavs:
        for reason, p in suspicious_wavs[:200]:
            report_lines.append(f"[{reason}] {p}")
        if len(suspicious_wavs) > 200:
            report_lines.append(f"... ({len(suspicious_wavs)-200} more)")
    else:
        report_lines.append("None")

    report_lines.append("")
    report_lines.append("-" * 78)
    report_lines.append("NON-WAV / JUNK FILES (REVIEW / REMOVE IF NOT NEEDED)")
    report_lines.append("-" * 78)
    if junk_files:
        for reason, p in junk_files[:200]:
            report_lines.append(f"[{reason}] {p}")
        if len(junk_files) > 200:
            report_lines.append(f"... ({len(junk_files)-200} more)")
    else:
        report_lines.append("None")

    if args.dedupe:
        report_lines.append("")
        report_lines.append("-" * 78)
        report_lines.append("DUPLICATE WAV GROUPS (SAME AUDIO CONTENT)")
        report_lines.append("-" * 78)
        if duplicates:
            for group in duplicates[:50]:
                report_lines.append("GROUP:")
                for p in group:
                    report_lines.append(f"  - {p}")
            if len(duplicates) > 50:
                report_lines.append(f"... ({len(duplicates)-50} more groups)")
        else:
            report_lines.append("None")

    # Output report to file or console
    report_text = "\n".join(report_lines) + "\n"
    if report_path:
        report_path.write_text(report_text, encoding="utf-8")
        print(f"\n✅ Wrote report: {report_path.resolve()}")
    else:
        print("\n" + report_text)

    print(f"✅ Wrote manifest: {out_manifest.resolve()}")
    if hard_manifest:
        print(f"✅ Wrote hard manifest: {hard_manifest.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

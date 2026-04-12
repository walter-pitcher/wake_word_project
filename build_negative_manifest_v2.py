#!/usr/bin/env python3
"""
Build negative manifests (recursive) + junk report.

Key features:
- Recursively finds .wav files under data/negative (and all subfolders)
- Writes a manifest file: one path per line (relative to project root by default)
- Optionally writes a "hard negatives" manifest using folder-name hints
- Generates a report of "unnecessary items" (non-audio, archives, system files, tiny wavs)
- Optional duplicate detection:
    --dedupe fast   (default): cheap fingerprint (size + partial hash)
    --dedupe full               full sha1 of entire file (slow for 100k+ files)

Windows example (run from project root):
  python build_negative_manifest_v2.py --root data\\negative --out data\\negative_manifest.txt --report data\\negative_manifest_report.txt --also-hard

Add your custom hard-negative folder:
  python build_negative_manifest_v2.py --root data\\negative --out data\\negative_manifest.txt --also-hard --hard-hints Hey_Magis_wrong_ways_to_say

"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# --- config ---
AUDIO_EXTS = {".wav"}

SYSTEM_JUNK_NAMES = {
    "thumbs.db",
    "desktop.ini",
    ".ds_store",
}

ARCHIVE_EXTS = {
    ".zip", ".tar", ".gz", ".tgz", ".rar", ".7z"
}

DEFAULT_MIN_WAV_BYTES = 2048  # tiny wavs are often corrupt/empty; tweak if needed

# defaults: your “near miss” / hard negative bucket
DEFAULT_HARD_HINTS = [
    "hey_magis_wrong_ways_to_say",
    "wrong_ways",
    "hard_negative",
    "hard_negatives",
    "near_miss",
    "confusables",
]


# --- helpers ---
def sha1_full(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha1_partial(path: Path, head_bytes: int = 65536, tail_bytes: int = 65536) -> str:
    """
    Fast-ish fingerprint:
    - file size
    - sha1(head + tail)
    Works well for catching obvious duplicates without hashing 100k full files.
    """
    size = path.stat().st_size
    h = hashlib.sha1()
    h.update(str(size).encode("utf-8"))
    with path.open("rb") as f:
        head = f.read(head_bytes)
        h.update(head)
        if size > head_bytes + tail_bytes:
            f.seek(max(0, size - tail_bytes))
        tail = f.read(tail_bytes)
        h.update(tail)
    return h.hexdigest()


def normalize_manifest_path(p: Path, root: Path, style: str) -> str:
    """
    style:
      - relative_project: relative to project root (parent of 'data' if root is data/negative)
      - relative_root: relative to --root
      - absolute: full absolute path
    """
    p = p.resolve()
    root = root.resolve()

    if style == "absolute":
        return str(p)

    if style == "relative_root":
        try:
            return str(p.relative_to(root))
        except Exception:
            return str(p)

    # relative_project (default)
    # If root is .../data/negative, project root is .../ (parent of data)
    project_root = root
    if root.name.lower() == "negative" and root.parent.name.lower() == "data":
        project_root = root.parent.parent
    try:
        return str(p.relative_to(project_root.resolve()))
    except Exception:
        return str(p)


def is_hard_negative_path(p: Path, hard_hints: List[str]) -> bool:
    """
    Substring match against each folder name in the path.
    This is more forgiving than “exact folder part equals hint”.
    """
    hints = [h.lower() for h in hard_hints]
    parts = [part.lower() for part in p.parts]
    for part in parts:
        for h in hints:
            if h in part:
                return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to negative folder (e.g. data\\negative)")
    ap.add_argument("--out", required=True, help="Output manifest path (txt)")

    ap.add_argument("--report", default="", help="Optional report output path (txt)")

    ap.add_argument("--also-hard", action="store_true", help="Also write a hard-negatives manifest")
    ap.add_argument("--hard-out", default="", help="Hard negatives manifest path (txt). If empty, auto-names.")
    ap.add_argument(
        "--hard-hints",
        nargs="*",
        default=None,
        help="Override/extend hard-negative folder hints (space-separated).",
    )

    ap.add_argument(
        "--path-style",
        choices=["relative_project", "relative_root", "absolute"],
        default="relative_project",
        help="Manifest path style",
    )
    ap.add_argument("--posix-slashes", action="store_true", help="Use forward slashes in manifest paths")
    ap.add_argument("--min-wav-bytes", type=int, default=DEFAULT_MIN_WAV_BYTES, help="Flag wavs smaller than this")

    ap.add_argument(
        "--dedupe",
        choices=["off", "fast", "full"],
        default="off",
        help="Duplicate detection mode",
    )
    ap.add_argument("--max-dup-groups", type=int, default=50, help="How many duplicate groups to print in report")

    args = ap.parse_args()

    root = Path(args.root).expanduser()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Root folder not found or not a directory: {root}")

    out_manifest = Path(args.out).expanduser()
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    report_path = Path(args.report).expanduser() if args.report else None
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)

    hard_hints = DEFAULT_HARD_HINTS if args.hard_hints is None else args.hard_hints

    hard_manifest: Optional[Path] = None
    if args.also_hard:
        if args.hard_out:
            hard_manifest = Path(args.hard_out).expanduser()
        else:
            hard_manifest = out_manifest.with_name(out_manifest.stem + "_hard" + out_manifest.suffix)
        hard_manifest.parent.mkdir(parents=True, exist_ok=True)

    wav_paths: List[Path] = []
    hard_wav_paths: List[Path] = []
    junk_files: List[Tuple[str, Path]] = []
    suspicious_wavs: List[Tuple[str, Path]] = []

    # duplicate detection
    fp_to_paths: Dict[str, List[Path]] = {}

    # walk files
    for p in root.rglob("*"):
        if p.is_dir():
            continue

        name_lower = p.name.lower()
        ext_lower = p.suffix.lower()

        # system junk
        if name_lower in SYSTEM_JUNK_NAMES:
            junk_files.append(("system/junk file", p))
            continue

        # archives inside negative folder (not “bad”, but usually unnecessary here)
        if ext_lower in ARCHIVE_EXTS or name_lower.endswith(".tar.gz"):
            junk_files.append(("archive file (consider moving out of negative)", p))
            continue

        # allow only wavs in manifest
        if ext_lower in AUDIO_EXTS:
            try:
                size = p.stat().st_size
            except Exception:
                size = -1

            if size == 0:
                suspicious_wavs.append(("zero-byte wav", p))
                continue

            if 0 < size < args.min_wav_bytes:
                suspicious_wavs.append((f"very small wav (<{args.min_wav_bytes} bytes)", p))
                continue

            wav_paths.append(p)

            if hard_manifest is not None and is_hard_negative_path(p, hard_hints):
                hard_wav_paths.append(p)

            if args.dedupe != "off":
                try:
                    fp = sha1_partial(p) if args.dedupe == "fast" else sha1_full(p)
                    fp_to_paths.setdefault(fp, []).append(p)
                except Exception:
                    suspicious_wavs.append(("hashing failed", p))

        else:
            # non-wav
            junk_files.append((f"non-wav file ({ext_lower or 'no extension'})", p))

    wav_paths_sorted = sorted(wav_paths, key=lambda x: str(x).lower())
    hard_wav_paths_sorted = sorted(hard_wav_paths, key=lambda x: str(x).lower())

    # write manifests
    def write_manifest(paths: List[Path], dest: Path) -> None:
        lines: List[str] = []
        for p in paths:
            s = normalize_manifest_path(p, root, args.path_style)
            if args.posix_slashes:
                s = s.replace("\\", "/")
            lines.append(s)
        dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    write_manifest(wav_paths_sorted, out_manifest)
    if hard_manifest is not None:
        write_manifest(hard_wav_paths_sorted, hard_manifest)

    # duplicates
    duplicates: List[List[Path]] = []
    if args.dedupe != "off":
        for fp, paths in fp_to_paths.items():
            if len(paths) > 1:
                duplicates.append(paths)
        duplicates.sort(key=lambda g: -len(g))

    # report
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
        report_lines.append(f"Hard hints: {', '.join(hard_hints)}")
    report_lines.append(f"Suspicious wavs excluded: {len(suspicious_wavs)}")
    report_lines.append(f"Non-wav/junk files found: {len(junk_files)}")
    if args.dedupe != "off":
        report_lines.append(f"Duplicate groups ({args.dedupe}): {len(duplicates)}")

    report_lines.append("")
    report_lines.append("-" * 78)
    report_lines.append("SUSPICIOUS WAVS (EXCLUDED)")
    report_lines.append("-" * 78)
    report_lines.extend([f"[{reason}] {p}" for reason, p in suspicious_wavs[:200]] or ["None"])
    if len(suspicious_wavs) > 200:
        report_lines.append(f"... ({len(suspicious_wavs)-200} more)")

    report_lines.append("")
    report_lines.append("-" * 78)
    report_lines.append("NON-WAV / JUNK FILES (REVIEW / REMOVE IF NOT NEEDED)")
    report_lines.append("-" * 78)
    report_lines.extend([f"[{reason}] {p}" for reason, p in junk_files[:200]] or ["None"])
    if len(junk_files) > 200:
        report_lines.append(f"... ({len(junk_files)-200} more)")

    if args.dedupe != "off":
        report_lines.append("")
        report_lines.append("-" * 78)
        report_lines.append("DUPLICATE WAV GROUPS (SAME AUDIO CONTENT)")
        report_lines.append("-" * 78)
        if duplicates:
            for group in duplicates[: args.max_dup_groups]:
                report_lines.append(f"GROUP (n={len(group)}):")
                for p in group:
                    report_lines.append(f"  - {p}")
            if len(duplicates) > args.max_dup_groups:
                report_lines.append(f"... ({len(duplicates)-args.max_dup_groups} more groups)")
        else:
            report_lines.append("None")

    report_text = "\n".join(report_lines) + "\n"
    if report_path:
        report_path.write_text(report_text, encoding="utf-8")
        print(f"✅ Wrote report: {report_path.resolve()}")
    else:
        print(report_text)

    print(f"✅ Wrote manifest: {out_manifest.resolve()}")
    if hard_manifest:
        print(f"✅ Wrote hard manifest: {hard_manifest.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import os
import re
import shutil
from pathlib import Path

LINE_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s+(.+?\.wav)\s*$", re.IGNORECASE)

def parse_rank_file(rank_path: Path):
    rows = []
    bad_lines = 0
    for i, line in enumerate(rank_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        if not m:
            bad_lines += 1
            continue
        score = float(m.group(1))
        wav_path = m.group(2).strip().strip('"')
        rows.append((score, wav_path, i))
    return rows, bad_lines

def resolve_path(p: str, base_dir: Path) -> Path:
    # If already absolute, keep it; otherwise resolve relative to base_dir
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (base_dir / pp).resolve()

def safe_move(src: Path, dst_dir: Path, keep_tree_from: Path | None, collisions: dict):
    """
    Move src into dst_dir. If keep_tree_from is provided and src is under that root,
    recreate subfolders. Otherwise drop into root of dst_dir.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    if keep_tree_from and keep_tree_from in src.parents:
        rel = src.relative_to(keep_tree_from)
        target = dst_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
    else:
        target = dst_dir / src.name

    # Handle name collisions
    if target.exists():
        key = str(target).lower()
        collisions[key] = collisions.get(key, 0) + 1
        stem = target.stem
        suffix = target.suffix
        target = target.with_name(f"{stem}__dup{collisions[key]}{suffix}")

    shutil.move(str(src), str(target))
    return target

def main():
    ap = argparse.ArgumentParser(
        description="Move quiet positives (below threshold dBFS) to a quarantine folder based on a rank file."
    )
    ap.add_argument("--rank", required=True, help="Path to rank file (lines: <dbfs> <path.wav>)")
    ap.add_argument("--threshold", type=float, default=-70.0, help="Move files with score < threshold (default: -70)")
    ap.add_argument("--base-dir", default=".", help="Base directory for resolving relative paths in rank file")
    ap.add_argument("--out-dir", default="data/positive_quarantine", help="Quarantine output directory")
    ap.add_argument("--keep-tree-root", default="", help="If set, preserve relative folder structure from this root")
    ap.add_argument("--dry-run", action="store_true", help="Print what would move without moving")
    ap.add_argument("--report", default="data/positive_quarantine_report.txt", help="Where to write a report")
    args = ap.parse_args()

    rank_path = Path(args.rank).resolve()
    base_dir = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    keep_tree_root = Path(args.keep_tree_root).resolve() if args.keep_tree_root else None
    report_path = Path(args.report).resolve()

    if not rank_path.exists():
        raise SystemExit(f"Rank file not found: {rank_path}")

    rows, bad_lines = parse_rank_file(rank_path)
    if not rows:
        raise SystemExit(f"No parsable rows found in rank file: {rank_path} (bad lines: {bad_lines})")

    # Determine moves
    to_move = []
    missing = []
    for score, wav_str, lineno in rows:
        if score < args.threshold:
            src = resolve_path(wav_str, base_dir)
            if not src.exists():
                missing.append((score, wav_str, lineno))
                continue
            to_move.append((score, src, lineno))

    # Sort quietest first
    to_move.sort(key=lambda x: x[0])

    collisions = {}
    moved = []
    if args.dry_run:
        print(f"[DRY RUN] Would move {len(to_move)} file(s) with dbfs < {args.threshold} to: {out_dir}")
        for score, src, _ in to_move[:50]:
            print(f"  {score:8.3f}  {src}")
        if len(to_move) > 50:
            print(f"  ... ({len(to_move)-50} more)")
    else:
        print(f"Moving {len(to_move)} file(s) with dbfs < {args.threshold} to: {out_dir}")
        for idx, (score, src, lineno) in enumerate(to_move, start=1):
            target = safe_move(src, out_dir, keep_tree_root, collisions)
            moved.append((score, str(src), str(target), lineno))
            if idx % 200 == 0:
                print(f"  {idx}/{len(to_move)}")

    # Write report
    report_lines = []
    report_lines.append("=" * 78)
    report_lines.append("QUIET POSITIVE QUARANTINE REPORT")
    report_lines.append("=" * 78)
    report_lines.append(f"Rank file:   {rank_path}")
    report_lines.append(f"Base dir:    {base_dir}")
    report_lines.append(f"Threshold:   dbfs < {args.threshold}")
    report_lines.append(f"Out dir:     {out_dir}")
    report_lines.append(f"Keep tree:   {keep_tree_root if keep_tree_root else '(none)'}")
    report_lines.append(f"Dry run:     {args.dry_run}")
    report_lines.append(f"Parsed rows: {len(rows)} (bad/unparsed lines: {bad_lines})")
    report_lines.append(f"Moved:       {len(moved) if not args.dry_run else 0}")
    report_lines.append(f"Would move:  {len(to_move) if args.dry_run else 0}")
    report_lines.append(f"Missing:     {len(missing)}")
    report_lines.append("")

    if missing:
        report_lines.append("-" * 78)
        report_lines.append("MISSING FILES REFERENCED IN RANK FILE (NOT MOVED)")
        report_lines.append("-" * 78)
        for score, wav_str, lineno in missing[:200]:
            report_lines.append(f"line {lineno:>5}: {score:8.3f}  {wav_str}")
        if len(missing) > 200:
            report_lines.append(f"... ({len(missing)-200} more)")
        report_lines.append("")

    if not args.dry_run and moved:
        report_lines.append("-" * 78)
        report_lines.append("MOVED FILES")
        report_lines.append("-" * 78)
        for score, src, dst, lineno in moved[:500]:
            report_lines.append(f"line {lineno:>5}: {score:8.3f}  {src}  ->  {dst}")
        if len(moved) > 500:
            report_lines.append(f"... ({len(moved)-500} more)")
        report_lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"✅ Report written: {report_path}")

if __name__ == "__main__":
    main()

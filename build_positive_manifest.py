from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dirs", nargs="+", required=True)
    args = ap.parse_args()

    paths = []
    for d in args.dirs:
        d = Path(d)
        paths.extend(sorted(d.rglob("*.wav")))

    out = Path(args.out)
    out.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
    print(f"✅ Wrote {out} ({len(paths)} wavs)")

if __name__ == "__main__":
    main()

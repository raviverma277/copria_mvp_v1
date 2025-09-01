#!/usr/bin/env python3

import argparse, sys, os, json
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.parsing.dispatch import parse_files


def main():
    parser = argparse.ArgumentParser(
        description="Classify sample documents into functional buckets."
    )
    parser.add_argument(
        "--samples-dir",
        type=str,
        default=str(Path(ROOT) / "data"),
        help="Folder with sample docs (default: ./data)",
    )
    args = parser.parse_args()

    samples = Path(args.samples_dir)
    if not samples.exists():
        print(f"[ERROR] Samples dir not found: {samples}")
        sys.exit(1)

    # Open files
    uploads = []
    for p in samples.iterdir():
        if p.is_file() and p.suffix.lower() in {
            ".pdf",
            ".xlsx",
            ".xls",
            ".eml",
            ".msg",
        }:
            uploads.append(open(p, "rb"))

    bundle = parse_files(uploads)

    # Close file handles
    for fh in uploads:
        try:
            fh.close()
        except Exception:
            pass

    # Summary
    def count(k):
        return len(bundle.get(k, []))

    print("=== Classification Summary ===")
    for key in [
        "submission",
        "sov",
        "loss_run",
        "questionnaire",
        "email_body",
        "other",
    ]:
        print(f"  {key}: {count(key)}")

    # Details
    print("\n=== Details ===")
    for key in [
        "submission",
        "sov",
        "loss_run",
        "questionnaire",
        "email_body",
        "other",
    ]:
        print(f"\n-- {key.upper()} --")
        for i, item in enumerate(bundle.get(key, []), 1):
            name = item.get("filename", "(no name)")
            meta = item.get("meta", {})
            page_tags = item.get("page_tags", [])
            print(f"[{i}] {name} | meta={meta}")
            if page_tags:
                tops = [
                    f"p{t.get('page')}:{t.get('type')}({round(float(t.get('confidence',0)),2)})"
                    for t in page_tags[:8]
                ]
                more = " ..." if len(page_tags) > 8 else ""
                print("   page_tags:", ", ".join(tops) + more)

    # Save JSON for inspection
    out = Path(samples) / "classification_output.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, default=str)
    print(f"\nSaved details to: {out}")


if __name__ == "__main__":
    main()

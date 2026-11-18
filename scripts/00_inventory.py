#!/usr/bin/env python3
"""
Step 0 - Inventory.

Scans <papers_dir>/<year>/**.pdf (works whether each paper is a loose PDF or sits
in its own sub-folder), converts each PDF to Markdown ONCE, caches the text, and
records deterministic features. Every later step reads the cached text, so the
LLM sees exactly the text that quote verification later checks against.

Output
  data/interim/manifest.csv         one row per paper
  data/interim/text/<paper_id>.md   cached Markdown text
  data/interim/features/<paper_id>.json

Usage
  python scripts/00_inventory.py --papers-dir data/papers
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from icwsm_datahub.pdftext import deterministic_features, make_paper_id, pdf_to_markdown  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--papers-dir", default="data/papers", help="Folder containing 2020/ ... 2026/")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--years", default="2020-2026")
    ap.add_argument("--force", action="store_true", help="Re-convert PDFs even if text is cached")
    args = ap.parse_args()

    y0, y1 = (int(x) for x in args.years.split("-"))
    papers_dir, data_dir = Path(args.papers_dir), Path(args.data_dir)
    text_dir = data_dir / "interim" / "text"
    feat_dir = data_dir / "interim" / "features"
    text_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    pdfs = []
    for year in range(y0, y1 + 1):
        folder = papers_dir / str(year)
        if not folder.exists():
            print(f"[warn] missing folder {folder}")
            continue
        found = sorted(p for p in folder.rglob("*.pdf") if not p.name.startswith("._"))
        print(f"{year}: {len(found)} PDFs")
        pdfs += [(year, p) for p in found]

    rows, seen_hashes = [], {}
    for year, pdf in tqdm(pdfs, desc="Converting PDFs"):
        pid = make_paper_id(year, pdf)
        text_path = text_dir / f"{pid}.md"
        feat_path = feat_dir / f"{pid}.json"
        try:
            if text_path.exists() and feat_path.exists() and not args.force:
                feats = json.loads(feat_path.read_text())
            else:
                md, pages = pdf_to_markdown(pdf)
                text_path.write_text(md, encoding="utf-8")
                feats = deterministic_features(pdf, md, pages)
                feat_path.write_text(json.dumps(feats, indent=2))
        except Exception as exc:  # corrupted PDF, password, etc.
            print(f"[error] {pdf}: {exc}")
            continue

        dup_of = seen_hashes.get(feats["sha256"])
        seen_hashes.setdefault(feats["sha256"], pid)
        rows.append({
            "paper_id": pid,
            "year": year,
            "pdf_path": str(pdf),
            "pdf_filename": pdf.name,
            "text_path": str(text_path),
            "page_count": feats["page_count"],
            "n_chars": feats["n_chars"],
            "approx_tokens": feats["n_chars"] // 4,
            "checklist_detected": feats["checklist_detected"],
            "n_urls_in_body": len(feats["urls_in_body"]),
            "duplicate_of": dup_of,
        })

    df = pd.DataFrame(rows)
    out = data_dir / "interim" / "manifest.csv"
    df.to_csv(out, index=False)
    print(f"\nManifest: {out} ({len(df)} papers)")
    if not df.empty:
        print(df.groupby("year").agg(papers=("paper_id", "count"), median_pages=("page_count", "median"),
                                     checklist=("checklist_detected", "mean")).round(2))
        dups = df[df["duplicate_of"].notna()]
        if len(dups):
            print(f"\n[warn] {len(dups)} duplicate PDFs (same SHA-256). They will be skipped in extraction:")
            print(dups[["paper_id", "duplicate_of"]].to_string(index=False))
        short = df[df["n_chars"] < 8000]
        if len(short):
            print(f"\n[warn] {len(short)} PDFs produced very little text (scanned or broken?):")
            print(short[["paper_id", "n_chars"]].to_string(index=False))


if __name__ == "__main__":
    main()

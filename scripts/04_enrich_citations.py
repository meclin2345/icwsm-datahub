#!/usr/bin/env python3
"""
Step 4 - Bibliometric enrichment from OpenAlex (free, no key).

Matches each paper title to OpenAlex and records the paper's DOI (10.1609/icwsm...),
citation count and citations per year. Citation counts are the outcome variable for
"which dataset properties predict reuse?".

Output: data/enrich/citations.csv
Usage : python scripts/04_enrich_citations.py --mailto you@university.edu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from icwsm_datahub.curation import load_overrides  # noqa: E402

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover
    fuzz = None


def best_match(title: str, year: int, mailto: str) -> dict:
    params = {"search": title[:250], "filter": f"publication_year:{year - 1}-{year + 1}", "per-page": 5,
              "mailto": mailto,
              "select": "id,doi,display_name,publication_year,cited_by_count,counts_by_year,primary_location"}
    try:
        r = requests.get("https://api.openalex.org/works", params=params, timeout=30)
        results = r.json().get("results", []) if r.status_code == 200 else []
    except Exception as exc:
        return {"openalex_error": str(exc)[:120]}
    best, best_score = None, 0.0
    for w in results:
        cand = w.get("display_name") or ""
        score = fuzz.token_set_ratio(title.lower(), cand.lower()) if fuzz else float(title.lower() == cand.lower()) * 100
        venue = ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
        if "icwsm" in venue.lower() or "weblogs and social media" in venue.lower():
            score += 5  # prefer the proceedings version over arXiv preprints
        if score > best_score:
            best, best_score = w, score
    if not best or best_score < 88:
        return {"openalex_match_score": best_score}
    by_year = {str(c["year"]): c["cited_by_count"] for c in best.get("counts_by_year", [])}
    return {
        "openalex_id": best["id"], "paper_doi": (best.get("doi") or "").replace("https://doi.org/", "") or None,
        "openalex_title": best.get("display_name"), "openalex_match_score": round(best_score, 1),
        "citations": best.get("cited_by_count"), "citations_by_year": json.dumps(by_year),
        "venue": ((best.get("primary_location") or {}).get("source") or {}).get("display_name"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--mailto", required=True, help="Your email (OpenAlex polite pool)")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((data_dir / "extracted").glob("*.json"))]
    if not recs:
        sys.exit("No extracted records.")
    overrides = load_overrides(data_dir / "curation" / "overrides.csv")
    rows = []
    for rec in tqdm(recs, desc="OpenAlex"):
        title = rec["extraction"]["paper"]["title"]
        for o in overrides.get(rec["paper_id"], []):
            if o["field"] == "paper.title" and o["value"]:
                title = o["value"]
        filled = any(u.get("filled_fields") for u in rec.get("provenance", {}).get("usage", []))
        rows.append({"paper_id": rec["paper_id"], "year": rec["year"], "title": title, "title_from_pdf": filled,
                     **best_match(title, int(rec["year"]), args.mailto)})
        time.sleep(0.12)
    out = data_dir / "enrich"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "citations.csv", index=False)
    matched = df["openalex_id"].notna().mean() if "openalex_id" in df else 0
    print(f"Matched {100 * matched:.1f}% of papers. Unmatched titles are listed with their best score; "
          f"fix them by hand in {out / 'citations.csv'} if needed.")


if __name__ == "__main__":
    main()

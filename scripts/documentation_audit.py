#!/usr/bin/env python3
"""
Manual audit of documentation claims (datasheets, data cards...) after the paper checklist.

  python scripts/documentation_audit.py list
      Every paper from the checklist year on that claims documentation, with its current category.
  python scripts/documentation_audit.py set 41077 answered_no "Checklist 5(g) answered No" --curator MJ
      Record what you found. The first argument is any unique part of the paper_id.

Categories
  in_paper         documentation (e.g. a full datasheet) is included in the paper or its appendix
  external_file    the paper points to a documentation file elsewhere (e.g. DATASHEET.md on GitHub)
  self_declared    the paper says the paper itself serves as the documentation
  bare_yes         checklist answered "Yes", with no pointer and nothing in the paper
  answered_no      checklist answered "No"            -> also corrects documentation_framework
  other_framework  points to a non-dataset framework, e.g. a model card -> also corrects it

answered_no and other_framework are extraction errors: the tool writes an override setting
distribution.documentation_framework to "None mentioned", so the catalog is corrected too.
Then rebuild:  python scripts/05_build_catalog.py && python scripts/06_analysis.py
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import pandas as pd

CATEGORIES = {
    "in_paper": "Documentation included in the paper or its appendix",
    "external_file": "Points to a documentation file outside the paper",
    "self_declared": "States that the paper itself is the documentation",
    "bare_yes": "Checklist 'Yes' with no pointer or evidence",
    "answered_no": "Checklist answered 'No' (extraction error, corrected)",
    "other_framework": "Non-dataset framework, e.g. model card (extraction error, corrected)",
}
CORRECTS = {"answered_no", "other_framework"}
AUDIT = Path("data/curation/documentation_audit.csv")
OVERRIDES = Path("data/curation/overrides.csv")
CATALOG = Path("catalog/icwsm_datahub_catalog.csv")
# Same definition as the FAIR F2 indicator used in Finding 4: standard frameworks only (not a plain README).
STANDARD = {"Datasheets for Datasets", "Data Statements", "Data Card or dataset card", "Dataset Nutrition Label",
            "Croissant or other machine-readable metadata"}


def candidates() -> pd.DataFrame:
    if not CATALOG.exists():
        sys.exit("Build the catalog first: python scripts/05_build_catalog.py")
    d = pd.read_csv(CATALOG)
    by_year = d.groupby("year")["checklist_detected"].apply(lambda s: s.astype(bool).mean())
    start = next((int(y) for y, v in by_year.items() if v > 0.5), None)
    claimed = d["documentation_framework"].isin(STANDARD)
    # also keep papers already audited (their claim may since have been corrected to "None mentioned")
    audited = set(pd.read_csv(AUDIT)["paper_id"]) if AUDIT.exists() else set()
    return d[(d["year"] >= (start or 0)) & (claimed | d["paper_id"].isin(audited))].copy()


def cmd_list() -> None:
    c = candidates()
    a = pd.read_csv(AUDIT, dtype=str).fillna("") if AUDIT.exists() else pd.DataFrame(columns=["paper_id", "category"])
    cat = dict(zip(a["paper_id"], a["category"]))
    c["category"] = c["paper_id"].map(cat).astype(object).where(c["paper_id"].isin(cat), "(not yet)")
    c["main_text_mention"] = c["body_mentions_datasheet"]
    pd.set_option("display.width", 200)
    print(c[["paper_id", "documentation_framework", "main_text_mention", "category"]].to_string(index=False))
    done = (c["category"] != "(not yet)").sum()
    print(f"\n{done} of {len(c)} classified.")


def cmd_set(part: str, category: str, note: str, curator: str) -> None:
    if category not in CATEGORIES:
        sys.exit(f"Unknown category. Use one of: {', '.join(CATEGORIES)}")
    ids = pd.read_csv(CATALOG)["paper_id"]
    hits = [p for p in ids if part.lower() in p.lower()]
    if len(hits) != 1:
        sys.exit(f"'{part}' matches {len(hits)} papers: {hits[:5]}. Use a longer part of the paper_id.")
    pid = hits[0]
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(AUDIT, dtype=str).fillna("") if AUDIT.exists() else pd.DataFrame(
        columns=["paper_id", "category", "note", "curator", "checked_on"])
    a = a[a["paper_id"] != pid]
    a = pd.concat([a, pd.DataFrame([{"paper_id": pid, "category": category, "note": note, "curator": curator,
                                     "checked_on": datetime.date.today().isoformat()}])], ignore_index=True)
    a.to_csv(AUDIT, index=False)
    print(f"{pid}: {category}")
    if category in CORRECTS:
        o = pd.read_csv(OVERRIDES, dtype=str).fillna("") if OVERRIDES.exists() else pd.DataFrame(
            columns=["paper_id", "field", "value", "reason", "curator"])
        o = o[~((o["paper_id"] == pid) & (o["field"] == "distribution.documentation_framework"))]
        o = pd.concat([o, pd.DataFrame([{"paper_id": pid, "field": "distribution.documentation_framework",
                                         "value": "None mentioned", "reason": f"documentation audit: {category}; {note}",
                                         "curator": curator}])], ignore_index=True)
        o.to_csv(OVERRIDES, index=False)
        print("   also corrected documentation_framework -> None mentioned (override written)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    s = sub.add_parser("set")
    s.add_argument("paper")
    s.add_argument("category")
    s.add_argument("note", nargs="?", default="")
    s.add_argument("--curator", required=True)
    args = ap.parse_args()
    if args.cmd == "list":
        cmd_list()
    else:
        cmd_set(args.paper, args.category, args.note, args.curator)


if __name__ == "__main__":
    main()

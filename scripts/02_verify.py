#!/usr/bin/env python3
"""
Step 2 - Verify extractions. Three jobs:

(a) Grounding (default). For every record:
      * is each evidence quote really in the paper?   (fuzzy match >= 90)
      * is each dataset URL / DOI really in the paper? (catches hallucinated links)
      * do fields contradict each other or the deterministic features?
    -> data/interim/verification.csv  + a summary you can report as
       "x% of quotes and y% of URLs were verbatim-grounded".

(b) --make-sample N  Draw a year-stratified sample for blind double-coding by
    humans. Writes a blank coding sheet (no LLM values, so coders are not anchored).

(c) --score FILE     Compare a completed human sheet to the LLM output: per-field
    accuracy, Cohen's kappa (categorical), Jaccard (multi-label).
    -> results/tables/extraction_validation.{csv,tex}

Usage
  python scripts/02_verify.py
  python scripts/02_verify.py --make-sample 60
  python scripts/02_verify.py --score validation/human_coding_sheet_filled.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from icwsm_datahub.normalize import clean_doi, normalize_extraction  # noqa: E402
from icwsm_datahub.pdftext import normalize_for_matching  # noqa: E402
from icwsm_datahub.schema import leaf_fields  # noqa: E402

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover
    fuzz = None

QUOTE_FIELDS = ["availability_quote", "scale_quote", "annotation_quote", "ethics_quote", "license_quote"]

# Fields humans double-code. Chosen to cover every section and every figure in the paper.
VALIDATION_FIELDS = {
    "collection.platforms": "multi",
    "collection.collection_methods": "multi",
    "collection.modalities": "multi",
    "collection.total_records": "number",
    "collection.temporal_coverage_start": "text",
    "collection.temporal_coverage_end": "text",
    "collection.languages_iso": "multi",
    "collection.data_geography": "multi",
    "collection.event_category": "single",
    "annotation.annotation_paradigms": "multi",
    "annotation.iaa_reported": "bool",
    "distribution.release_status": "single",
    "distribution.hosting_repositories": "multi",
    "distribution.persistent_doi": "text",
    "distribution.distribution_form": "single",
    "distribution.access_mechanism": "single",
    "distribution.license_category": "single",
    "distribution.documentation_framework": "single",
    "ethics.irb_status": "single",
    "ethics.pii_anonymization_applied": "bool",
    "ethics.discusses_platform_tos": "bool",
}


def load_records(data_dir: Path) -> List[Dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((data_dir / "extracted").glob("*.json"))]


def get_path(obj: Dict, dotted: str) -> Any:
    for part in dotted.split("."):
        obj = obj.get(part) if isinstance(obj, dict) else None
    return obj


# ----------------------------------------------------------------------------- (a) grounding
def quote_score(quote: str, norm_text: str) -> float:
    q = normalize_for_matching(quote)
    if not q:
        return np.nan
    if q in norm_text:
        return 100.0
    if fuzz is None:
        return 0.0
    return float(fuzz.partial_ratio(q, norm_text))


def url_in_text(url: str, squashed_text: str) -> bool:
    key = re.sub(r"^https?://(www\.)?", "", url.lower()).rstrip("/")
    key = re.sub(r"\s+", "", key)
    return key in squashed_text or key.split("?")[0] in squashed_text


def consistency_flags(ex: Dict[str, Any], det: Dict[str, Any], year: int) -> List[str]:
    c, a, d, e = ex["collection"], ex["annotation"], ex["distribution"], ex["ethics"]
    v_ = ex["evidence"]
    flags = []
    if d["release_status"] == "Released at publication" and not d["dataset_urls"] and not d["persistent_doi"]:
        flags.append("released_but_no_link")
    known_llm = re.compile(r"gpt|chatgpt|llama|claude|gemini|palm|bard|mistral|mixtral|falcon|vicuna|qwen|deepseek|"
                           r"gemma|phi-?\d|olmo|command[- ]?r|flan|bloom|opt-|large language model|\bllm", re.I)
    if any(p_.startswith(("LLM", "Hybrid")) for p_ in a["annotation_paradigms"]) and a["llms_used"] \
            and not any(known_llm.search(m) for m in a["llms_used"]):
        flags.append("llm_paradigm_but_no_llm_named")
    if a["iaa_reported"] and not a["iaa_metrics"]:
        flags.append("iaa_without_metric")
    if any("LLM" in p for p in a["annotation_paradigms"]) and not a["llms_used"]:
        flags.append("llm_paradigm_without_model_name")
    if e["includes_paper_checklist"] != det.get("checklist_detected"):
        flags.append("checklist_llm_vs_regex_disagree")
    if ex["distribution"]["explicitly_mentions_fair"] and not det.get("mentions_fair_regex"):
        flags.append("fair_claim_not_in_text")
    s, t = c.get("temporal_coverage_start"), c.get("temporal_coverage_end")
    if s and t and s[:7] > t[:7]:
        flags.append("coverage_start_after_end")
    if t and t[:4].isdigit() and int(t[:4]) > year:
        flags.append("coverage_ends_after_publication")
    if d["persistent_doi"] and "zenodo" in d["persistent_doi"] and "Zenodo" not in d["hosting_repositories"]:
        flags.append("zenodo_doi_but_host_missing")
    if d["distribution_form"] == "Dehydrated IDs only" and "Text" in c["modalities"] and not c["platforms"]:
        flags.append("dehydrated_without_platform")
    if not c["platforms"]:
        flags.append("no_platform")
    quotes = " ".join(filter(None, [v_.get("availability_quote"), v_.get("ethics_quote"), e.get("anonymization_details")]))
    ids_only = re.search(r"\bonly\b[^.]{0,60}\bIDs?\b", quotes, re.I)
    if d["distribution_form"].startswith("Mixed") and ids_only:
        flags.append("mixed_form_but_text_says_ids_only")
    if e["pii_anonymization_applied"] and re.search(r"\bonly\b[^.]{0,40}\bIDs?\b", e.get("anonymization_details") or "", re.I):
        flags.append("pii_true_for_ids_only_release")
    if d["distribution_form"].startswith("Anonymized") and not e["pii_anonymization_applied"]:
        flags.append("anonymized_form_but_no_pii_step")
    if re.search(r"\binfer", v_.get("ambiguity_notes") or "", re.I):
        flags.append("notes_mention_inference")
    aq = v_.get("availability_quote")
    if aq and (d["dataset_urls"] or d["persistent_doi"]) and not re.search(
            r"https?://|doi\.org|10\.\d{4}|zenodo|github|dataverse|figshare|osf\.io|hugging|kaggle|drive\.google|dropbox", aq, re.I):
        flags.append("availability_quote_names_no_location")
    if "GitHub or GitLab" in d["hosting_repositories"] and not any(("github.com" in u or "gitlab" in u) for u in d["dataset_urls"]):
        flags.append("code_host_listed_as_dataset_host")
    if a["num_annotated_records"] and a["num_annotated_records"] == c["total_records"]:
        flags.append("annotated_count_equals_total_records")
    return flags


def run_grounding(data_dir: Path) -> None:
    rows = []
    for rec in load_records(data_dir):
        pid = rec["paper_id"]
        text_path = data_dir / "interim" / "text" / f"{pid}.md"
        raw = text_path.read_text(encoding="utf-8") if text_path.exists() else ""
        norm = normalize_for_matching(raw)
        squashed = re.sub(r"\s+", "", raw.lower())
        ex = normalize_extraction(rec["extraction"])
        row = {"paper_id": pid, "year": rec["year"]}

        for qf in QUOTE_FIELDS:
            q = ex["evidence"].get(qf)
            row[f"{qf}_score"] = quote_score(q, norm) if q else np.nan

        urls = ex["distribution"]["dataset_urls"]
        grounded = [url_in_text(u, squashed) for u in urls]
        row["n_dataset_urls"] = len(urls)
        row["n_urls_grounded"] = int(sum(grounded))
        row["ungrounded_urls"] = "; ".join(u for u, g in zip(urls, grounded) if not g)
        doi = ex["distribution"]["persistent_doi"]
        row["doi_grounded"] = (doi.lower() in squashed) if doi else np.nan
        flags = consistency_flags(ex, rec["deterministic"], rec["year"])
        usage = rec.get("provenance", {}).get("usage", [])
        if any(u.get("filled_fields") for u in usage):
            flags.append("title_taken_from_pdf_check_it")
        if any(u.get("json_repaired") for u in usage):
            flags.append("json_was_repaired_check_quotes")
        row["flags"] = "; ".join(flags)
        row["n_flags"] = len(flags)
        row["confidence"] = ex["evidence"]["extraction_confidence"]
        rows.append(row)

    df = pd.DataFrame(rows)
    out = data_dir / "interim" / "verification.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} records)\n")
    if df.empty:
        return
    print("Evidence quotes found verbatim or near-verbatim (score >= 90):")
    for qf in QUOTE_FIELDS:
        s = df[f"{qf}_score"].dropna()
        if len(s):
            print(f"  {qf:<22} {100 * (s >= 90).mean():5.1f}%  (n={len(s)})")
    tot = df["n_dataset_urls"].sum()
    if tot:
        print(f"\nDataset URLs present in paper text: {100 * df['n_urls_grounded'].sum() / tot:.1f}% of {tot}")
    d = df["doi_grounded"].dropna()
    if len(d):
        print(f"Dataset DOIs present in paper text: {100 * d.astype(bool).mean():.1f}% of {len(d)}")
    flags = df["flags"].str.split("; ").explode().replace("", np.nan).dropna().value_counts()
    if len(flags):
        print("\nConsistency flags (review these records by hand):")
        print(flags.to_string())
    print(f"\nConfidence: {df['confidence'].value_counts().to_dict()}")


# ----------------------------------------------------------------------------- (b) sampling
def make_sample(data_dir: Path, n: int, seed: int, out_dir: Path) -> None:
    recs = load_records(data_dir)
    if not recs:
        sys.exit("No extracted records yet.")
    meta = pd.DataFrame({"paper_id": [r["paper_id"] for r in recs], "year": [r["year"] for r in recs],
                         "title": [r["extraction"]["paper"]["title"] for r in recs],
                         "pdf_filename": [r["pdf_filename"] for r in recs]})
    frac = min(1.0, n / len(meta))
    sample = (meta.groupby("year", group_keys=False)
              .apply(lambda g: g.sample(max(1, round(len(g) * frac)), random_state=seed)))
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = sample.copy()
    for f in VALIDATION_FIELDS:
        sheet[f] = ""
    sheet["coder_id"] = ""
    sheet["notes"] = ""
    sheet.to_csv(out_dir / "human_coding_sheet.csv", index=False)

    book = ["# Human coding codebook", "",
            "Code each paper from the PDF only. Do not look at the catalog or LLM output.",
            "Multi-valued fields: separate values with `; ` and use the exact labels below.",
            "Unknown: leave the cell empty. Numbers: plain integers (14200000).", ""]
    wanted = {f.split(".")[1] for f in VALIDATION_FIELDS}
    for r in leaf_fields():
        if r["field"] in wanted:
            book += [f"## {r['section']}.{r['field']}  ({r['type']})", r["description"], ""]
            if r["allowed_values"]:
                book += ["Allowed values:", ""] + [f"- {v}" for v in r["allowed_values"].split(" | ")] + [""]
    (out_dir / "CODEBOOK.md").write_text("\n".join(book), encoding="utf-8")
    print(f"Sampled {len(sample)} papers (stratified by year) -> {out_dir / 'human_coding_sheet.csv'}")
    print("Have two coders fill it independently (copy per coder) to also report human-human agreement.")


# ----------------------------------------------------------------------------- (c) scoring
def _norm_set(v: Any) -> frozenset:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return frozenset()
    if isinstance(v, list):
        items = v
    else:
        items = str(v).split(";")
    return frozenset(str(x).strip().lower() for x in items if str(x).strip())


def _norm_scalar(v: Any, kind: str) -> Any:
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
        return None
    if kind == "bool":
        return str(v).strip().lower() in ("true", "1", "yes", "y")
    if kind == "number":
        try:
            return float(str(v).replace(",", ""))
        except ValueError:
            return None
    if kind == "text":
        s = str(v).strip().lower()
        return clean_doi(s) or s
    return str(v).strip().lower()


def cohen_kappa(a: List[Any], b: List[Any]) -> float:
    labels = sorted({str(x) for x in a + b})
    if len(labels) < 2:
        return np.nan
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        m[idx[str(x)], idx[str(y)]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = (m.sum(0) * m.sum(1)).sum() / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def score(data_dir: Path, human_file: Path, results_dir: Path) -> None:
    human = pd.read_csv(human_file)
    recs = {r["paper_id"]: normalize_extraction(r["extraction"]) for r in load_records(data_dir)}
    rows = []
    for field, kind in VALIDATION_FIELDS.items():
        llm_vals, hum_vals = [], []
        for _, h in human.iterrows():
            if h["paper_id"] not in recs:
                continue
            lv = get_path(recs[h["paper_id"]], field)
            hv = h.get(field)
            if kind == "multi":
                llm_vals.append(_norm_set(lv))
                hum_vals.append(_norm_set(hv))
            else:
                llm_vals.append(_norm_scalar(lv, kind))
                hum_vals.append(_norm_scalar(hv, kind))
        n = len(llm_vals)
        if n == 0:
            continue
        if kind == "multi":
            exact = np.mean([x == y for x, y in zip(llm_vals, hum_vals)])
            jacc = np.mean([len(x & y) / len(x | y) if (x | y) else 1.0 for x, y in zip(llm_vals, hum_vals)])
            rows.append({"field": field, "type": kind, "n": n, "accuracy": exact, "jaccard": jacc, "kappa": np.nan})
        elif kind == "number":
            ok = [(x is None and y is None) or (x is not None and y is not None and abs(x - y) <= 0.05 * max(abs(y), 1))
                  for x, y in zip(llm_vals, hum_vals)]
            rows.append({"field": field, "type": kind, "n": n, "accuracy": np.mean(ok), "jaccard": np.nan, "kappa": np.nan})
        else:
            acc = np.mean([x == y for x, y in zip(llm_vals, hum_vals)])
            kap = cohen_kappa([str(x) for x in llm_vals], [str(y) for y in hum_vals])
            rows.append({"field": field, "type": kind, "n": n, "accuracy": acc, "jaccard": np.nan, "kappa": kap})

    df = pd.DataFrame(rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "tables").mkdir(exist_ok=True)
    df.to_csv(results_dir / "tables" / "extraction_validation.csv", index=False)
    tex = df.assign(field=df["field"].str.replace("_", r"\_", regex=False)).to_latex(
        index=False, float_format="%.2f", na_rep="--",
        caption="Agreement between LLM extraction and blind human coding.", label="tab:validation")
    (results_dir / "tables" / "extraction_validation.tex").write_text(tex)
    print(df.round(3).to_string(index=False))
    print(f"\nMacro accuracy: {df['accuracy'].mean():.3f}   median kappa: {df['kappa'].median():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--make-sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--score", type=Path, default=None)
    ap.add_argument("--validation-dir", default="validation")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    if args.make_sample:
        make_sample(data_dir, args.make_sample, args.seed, Path(args.validation_dir))
    elif args.score:
        score(data_dir, args.score, Path(args.results_dir))
    else:
        run_grounding(data_dir)


if __name__ == "__main__":
    main()

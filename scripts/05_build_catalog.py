#!/usr/bin/env python3
"""
Step 5 - Build the public catalog.

Merges LLM extraction + deterministic features + verification + 2026 audit +
citations + manual curation overrides, derives FAIR indicators and each dataset's
2026 "fate", and writes:

  catalog/icwsm_datahub_catalog.json      nested, one object per paper (richest form)
  catalog/icwsm_datahub_catalog.csv       flat, lists joined with "; "
  catalog/icwsm_datahub_catalog.parquet   flat, lists kept as lists
  catalog/DATA_DICTIONARY.md              every column, its type and meaning
  catalog/croissant.json                  MLCommons Croissant metadata (FAIR, machine-readable)
  docs/data/catalog.json                  compact file loaded by the web explorer

Manual fixes go in data/curation/overrides.csv (paper_id, field, value, reason, curator),
e.g. "2021_x_ab12cd, distribution.persistent_doi, 10.5281/zenodo.123, DOI in footnote, NS".
They are applied after normalization and counted per record, so curation is auditable.

Usage
  python scripts/05_build_catalog.py
  python scripts/05_build_catalog.py --demo        # synthetic data to test the stack
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from icwsm_datahub.fair import (FAIR_INDICATORS, FATE_COLORS, FATE_ORDER, dataset_fate, fair_indicators,  # noqa: E402
                                fair_score, funnel_flags, needs_closed_api)
from icwsm_datahub.normalize import (API_EVENTS, FAMILY_COLORS, FAMILY_ORDER, HOST_SHORT, north_south,  # noqa: E402
                                     normalize_extraction, region_of)
from icwsm_datahub.schema import SCHEMA_VERSION, leaf_fields  # noqa: E402
from icwsm_datahub.curation import apply_overrides, load_overrides  # noqa: E402

CATALOG_VERSION = "1.0.0"
TEXT_DIR = None  # set in main(); cached paper text, used for body-only mention checks
# Metadata, networks and timestamps accompany almost every dataset, so "multimodal"
# means more than one CONTENT modality.
CONTENT_MODALITIES = {"Text", "Image", "Video", "Audio"}
CONFERENCE_MONTH = 6  # ICWSM is held in June; used for dataset exposure age

DERIVED_DOCS = {
    "paper_id": "Stable identifier: <year>_<slug of PDF name>_<hash>.",
    "year": "ICWSM proceedings year (from the folder the PDF came from).",
    "pdf_filename": "Original PDF file name.",
    "page_count": "Number of PDF pages.",
    "checklist_detected": "True if the AAAI/ICWSM paper checklist text is found by regex.",
    "flags": "Automatic consistency flags raised by 02_verify.py.",
    "n_authors": "Number of authors.",
    "author_regions": "World-region(s) of author affiliations (World Bank-style regions).",
    "author_global_south": "True if any author affiliation is in the Global South (Brandt-line-style approximation, see normalize.py).",
    "platform_families": "Platforms grouped into 10 families used in figures and the explorer.",
    "n_platforms": "Number of distinct platforms.",
    "closed_access_dependencies": "Data-access routes used for collection that have since closed (e.g. Twitter/X Academic API, Pushshift, CrowdTangle).",
    "depends_on_closed_access": "True if collection relied on at least one closed access route.",
    "coverage_end_year": "Year of the last content in the dataset.",
    "coverage_days": "Days between first and last content (null if either date missing).",
    "log10_records": "log10 of total_records.",
    "is_multimodal": "True if more than one content modality (text, image, video, audio) is released.",
    "is_multilingual": "True if more than one language.",
    "data_regions": "World-region(s) the data is about.",
    "data_global_south": "True if the data is about at least one Global South country.",
    "uses_llm_annotation": "True if labels come from an LLM or a hybrid human+LLM pipeline.",
    "uses_crowd": "True if crowdworkers annotated.",
    "uses_expert": "True if experts, authors or trained students annotated.",
    "includes_baselines": "True if any model or analysis beyond descriptive statistics is run.",
    "primary_host": "First listed hosting venue, shortened.",
    "has_doi": "True if a dataset DOI (not the paper DOI) is reported.",
    "quotes_grounded_share": "Share of evidence quotes found (near-)verbatim in the paper text.",
    "urls_grounded_share": "Share of extracted dataset URLs that appear in the paper text.",
    "n_flags": "Number of automatic consistency flags (see 02_verify.py).",
    "audited_at": "UTC time of the repository audit.",
    "any_url_alive": "True if at least one dataset link or the DOI resolved to live content.",
    "primary_url_status": "Outcome category for the first dataset link (Alive, Dead (404), Soft 404, Login wall, ...).",
    "doi_resolves": "True if https://doi.org/<doi> resolved to live content.",
    "verified_repo": "Repository whose API confirmed the record (Zenodo, Dataverse, Figshare, Hugging Face, GitHub, DataCite).",
    "file_count": "Number of files in the repository record (API).",
    "total_size_mb": "Total size of files in MB (API).",
    "file_extensions": "File extensions observed in the repository (API).",
    "is_restricted": "True if all files are restricted, embargoed or need manual approval.",
    "files_empty": "True if the repository record exists but contains no files.",
    "login_wall": "True if links only lead to a sign-in page.",
    "account_required": "True if a free account is needed (Kaggle, gated Hugging Face).",
    "downloads_total": "Total downloads reported by the repository (definitions differ across repositories).",
    "downloads_unique": "Unique downloads where reported (Zenodo, Dataverse).",
    "views_total": "Total views reported by the repository.",
    "downloads_metric_source": "Which counter downloads_total comes from. Compare downloads only within one source.",
    "datacite_registered": "True if the DOI is registered with DataCite.",
    "datacite_citations": "Citations to the dataset DOI recorded by DataCite.",
    "code_url_alive": "True if the code repository link resolves.",
    "code_stars": "GitHub stars of the code repository.",
    "dead_link_in_wayback": "For dead links: True if the Wayback Machine holds a snapshot of the landing page.",
    "paper_doi": "DOI of the ICWSM paper (OpenAlex).",
    "citations": "Citations to the ICWSM paper (OpenAlex).",
    "citations_per_year": "citations divided by years since publication.",
    "years_since_publication": "Years between June of the ICWSM year and the audit date.",
    "downloads_per_year": "downloads_total divided by years_since_publication.",
    "fair_score": "Share (0-100) of the 10 operational FAIR indicators met.",
    "fate_2026": "Mutually exclusive 2026 state: " + "; ".join(FATE_ORDER) + ".",
    "needs_closed_api": "True if content must be rehydrated through a platform API that is no longer free (Twitter/X).",
    "funnel_release_claimed": "Paper claims a release (now, partially, later or on request).",
    "funnel_link_provided": "Paper gives a dataset link or DOI.",
    "funnel_link_alive": "Link alive in the audit.",
    "funnel_open_download": "Files openly downloadable (not gated, empty or login-walled).",
    "funnel_usable": "Openly downloadable and usable without a closed API.",
    "n_manual_overrides": "Number of fields corrected by human curators.",
    "body_mentions_misuse": "Regex: misuse/harm language appears in the main text (before References), i.e. not only in the checklist.",
    "body_mentions_datasheet": "Regex: datasheet / data card / data statement named in the main text (before References).",
    "repo_license": "License recorded on the repository record itself (Zenodo, Dataverse, Figshare, Hugging Face, GitHub).",
    "repo_license_present": "True if the repository record carries a license (only for records the audit could verify).",
    "body_mentions_fair": "Regex: 'FAIR' (upper case) appears in the main text (before References), i.e. not only in the checklist.",
    "audit_manual_check": "Date and note of a manual browser check that replaced the automatic link result.",
    "model": "LLM requested for extraction.",
    "served_model": "Model id the API reported serving for this record.",
    "api_host": "API host used for extraction (gateway or Anthropic API).",
    "structured_mode": "native (constrained decoding), prompt (schema in prompt + validation) or prompt-repair.",
    "prompt_version": "Extraction prompt version.",
    "schema_version": "Extraction schema version.",
    "extracted_at": "UTC time of extraction.",
}


# ----------------------------------------------------------------------------- loading
def read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


# ----------------------------------------------------------------------------- flattening
def _age_years(year: int, audited_at: Any) -> float:
    try:
        ref = datetime.fromisoformat(str(audited_at).replace("Z", "+00:00")).date()
    except (ValueError, TypeError):
        ref = date.today()
    return max(0.25, (ref - date(int(year), CONFERENCE_MONTH, 1)).days / 365.25)


def _days_between(a: Any, b: Any) -> Any:
    def parse(s: str, end: bool) -> date | None:
        s = str(s)
        try:
            if len(s) >= 10:
                return date.fromisoformat(s[:10])
            if len(s) == 7:
                y, m = int(s[:4]), int(s[5:7])
                return date(y, m, 28 if end else 1)
            if len(s) == 4:
                return date(int(s), 12 if end else 1, 31 if end else 1)
        except ValueError:
            return None
        return None
    if not a or not b:
        return None
    da, db = parse(a, False), parse(b, True)
    return (db - da).days if da and db and db >= da else None


def build_row(rec: Dict[str, Any], audit: Dict[str, Any], ver: Dict[str, Any], cit: Dict[str, Any],
              overrides: List[Dict]) -> Dict[str, Any]:
    ex = normalize_extraction(rec["extraction"])
    n_over = apply_overrides(ex, overrides)
    p, c, a, d, e, v = (ex[k] for k in ("paper", "collection", "annotation", "distribution", "ethics", "evidence"))
    year = int(rec["year"])
    ind = fair_indicators(ex, audit)
    fate = dataset_fate(ex, audit)
    age = _age_years(year, audit.get("audited_at"))
    paradigms = a["annotation_paradigms"]
    langs = c["languages_iso"]
    n_lang = c["num_languages"] or (len(langs) if langs and langs != ["mul"] else None)
    prov = rec.get("provenance", {})

    row: Dict[str, Any] = {
        "paper_id": rec["paper_id"], "year": year, "pdf_filename": rec.get("pdf_filename"),
        "title": clean_title(p["title"]), "authors": p["authors"], "n_authors": len(p["authors"]),
        "author_countries": p["author_countries"],
        "author_regions": sorted({region_of(x) for x in p["author_countries"]}),
        "author_global_south": any(north_south(x) == "Global South" for x in p["author_countries"]),
        "is_industry_collaboration": p["is_industry_collaboration"], "contribution_type": p["contribution_type"],
        "num_datasets_released": p["num_datasets_released"],
        "page_count": rec["deterministic"].get("page_count"),
        "checklist_detected": rec["deterministic"].get("checklist_detected"),
        # collection
        "dataset_name": c["dataset_name"], "platforms": c["platforms"], "platform_families": c["platform_families"],
        "n_platforms": len(c["platforms"]), "platform_names_verbatim": c["platform_names_verbatim"],
        "collection_methods": c["collection_methods"], "collection_tools": c["collection_tools"],
        "closed_access_dependencies": [x["dependency"] for x in c["closed_access_dependencies"]],
        "depends_on_closed_access": bool(c["closed_access_dependencies"]),
        "temporal_coverage_start": c["temporal_coverage_start"], "temporal_coverage_end": c["temporal_coverage_end"],
        "coverage_end_year": int(c["temporal_coverage_end"][:4]) if c["temporal_coverage_end"] and c["temporal_coverage_end"][:4].isdigit() else None,
        "coverage_days": _days_between(c["temporal_coverage_start"], c["temporal_coverage_end"]),
        "primary_record_unit": c["primary_record_unit"], "total_records": c["total_records"],
        "log10_records": round(math.log10(c["total_records"]), 3) if c["total_records"] else None,
        "num_accounts_or_users": c["num_accounts_or_users"], "scale_description_verbatim": c["scale_description_verbatim"],
        "modalities": c["modalities"],
        "is_multimodal": len(set(c["modalities"]) & CONTENT_MODALITIES) > 1,
        "languages_iso": langs, "num_languages": n_lang, "is_multilingual": bool(n_lang and n_lang > 1) or langs == ["mul"],
        "includes_code_mixing": c["includes_code_mixing"],
        "data_geography": c["data_geography"], "data_regions": sorted({region_of(x) for x in c["data_geography"]}),
        "data_global_south": any(north_south(x) == "Global South" for x in c["data_geography"]),
        "event_category": c["event_category"], "event_name": c["event_name"],
        "involves_vulnerable_groups": c["involves_vulnerable_groups"],
        # annotation
        "research_domains": a["research_domains"], "supported_tasks": a["supported_tasks"],
        "tasks_verbatim": a["tasks_verbatim"], "annotation_paradigms": paradigms,
        "uses_llm_annotation": any(x.startswith(("LLM", "Hybrid")) for x in paradigms),
        "uses_crowd": any(x.startswith("Crowd") for x in paradigms),
        "uses_expert": any(x.startswith("Expert") for x in paradigms),
        "llms_used": a["llms_used"], "num_annotated_records": a["num_annotated_records"],
        "num_annotators": a["num_annotators"], "iaa_reported": a["iaa_reported"], "iaa_metrics": a["iaa_metrics"],
        "iaa_primary_value": a["iaa_primary_value"], "iaa_summary_verbatim": a["iaa_summary_verbatim"],
        "baseline_families": a["baseline_families"],
        "includes_baselines": any(x not in ("No experiments", "Descriptive statistics only") for x in a["baseline_families"]),
        # distribution
        "release_status": d["release_status"], "hosting_repositories": d["hosting_repositories"],
        "primary_host": HOST_SHORT.get(d["hosting_repositories"][0], "Other") if d["hosting_repositories"] else "On request / not stated",
        "dataset_urls": d["dataset_urls"], "persistent_doi": d["persistent_doi"], "has_doi": bool(d["persistent_doi"]),
        "code_repository_url": d["code_repository_url"], "distribution_form": d["distribution_form"],
        "access_mechanism": d["access_mechanism"], "license_verbatim": d["license_verbatim"],
        "license_category": d["license_category"], "data_file_formats": d["data_file_formats"],
        "explicitly_mentions_fair": d["explicitly_mentions_fair"], "documentation_framework": d["documentation_framework"],
        "provides_collection_or_hydration_code": d["provides_collection_or_hydration_code"],
        # ethics
        **{k: e[k] for k in ("irb_status", "pii_anonymization_applied", "anonymization_details", "discusses_platform_tos",
                             "discusses_limitations", "discusses_misuse_risks", "includes_paper_checklist",
                             "ai_usage_disclosed")},
        # evidence
        **{k: v[k] for k in ("availability_quote", "scale_quote", "annotation_quote", "ethics_quote", "license_quote",
                             "extraction_confidence", "ambiguity_notes")},
    }
    # verification
    qcols = [k for k in ver if k.endswith("_quote_score")]
    qs = [ver[k] for k in qcols if ver.get(k) is not None and not (isinstance(ver[k], float) and np.isnan(ver[k]))]
    row["quotes_grounded_share"] = round(float(np.mean([x >= 90 for x in qs])), 3) if qs else None
    nu = ver.get("n_dataset_urls")
    row["urls_grounded_share"] = round(ver["n_urls_grounded"] / nu, 3) if nu else None
    row["n_flags"] = ver.get("n_flags")
    row["flags"] = ver.get("flags")
    # audit
    row["audit_manual_check"] = audit.get("manual_check")
    for k in ("audited_at", "any_url_alive", "primary_url_status", "doi_resolves", "verified_repo", "file_count",
              "total_size_mb", "file_extensions", "is_restricted", "files_empty", "login_wall", "account_required",
              "downloads_total", "downloads_unique", "views_total", "downloads_metric_source", "datacite_registered",
              "datacite_citations", "code_url_alive", "code_stars", "dead_link_in_wayback"):
        row[k] = audit.get(k)
    # citations
    row["paper_doi"] = cit.get("paper_doi")
    row["citations"] = cit.get("citations")
    row["years_since_publication"] = round(age, 2)
    row["citations_per_year"] = round(cit["citations"] / age, 2) if cit.get("citations") is not None else None
    row["downloads_per_year"] = round(audit["downloads_total"] / age, 2) if audit.get("downloads_total") is not None else None
    # derived
    for code, _, _ in FAIR_INDICATORS:
        row["fair_" + code.replace(".", "_")] = ind[code]
    row["fair_score"] = fair_score(ind)
    row["fate_2026"] = fate
    row["needs_closed_api"] = needs_closed_api(ex)
    row.update(funnel_flags(ex, audit))
    row["n_manual_overrides"] = n_over
    text_path = Path(TEXT_DIR) / f"{rec['paper_id']}.md" if TEXT_DIR else None
    if text_path and text_path.exists():
        from icwsm_datahub.pdftext import body_mentions
        row.update(body_mentions(text_path.read_text(encoding="utf-8")))
    else:
        row.update({"body_mentions_misuse": None, "body_mentions_fair": None, "body_mentions_datasheet": None})
    row["repo_license"] = audit.get("repo_license")
    row["repo_license_present"] = bool(audit.get("repo_license")) if audit.get("verified_repo") else None
    row.update({"model": prov.get("model"), "served_model": "; ".join(prov.get("served_model") or []) or None,
                "api_host": prov.get("api_host"),
                "structured_mode": "; ".join(prov.get("structured_modes") or []) or None,
                "prompt_version": prov.get("prompt_version"),
                "schema_version": prov.get("schema_version"), "extracted_at": prov.get("extracted_at")})
    return row


def clean_nan(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan(x) for x in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ----------------------------------------------------------------------------- outputs
AUDIT_COLS = {"audited_at", "any_url_alive", "primary_url_status", "doi_resolves", "verified_repo", "file_count",
              "total_size_mb", "file_extensions", "is_restricted", "files_empty", "login_wall", "account_required",
              "downloads_total", "downloads_unique", "views_total", "downloads_metric_source", "datacite_registered",
              "datacite_citations", "code_url_alive", "code_stars", "dead_link_in_wayback"}


def describe(col: str) -> tuple[str, str]:
    """(source, description) for any catalog column; shared by the dictionary and Croissant."""
    llm = {r["field"]: r for r in leaf_fields()}
    if col in llm:
        desc = llm[col]["description"]
        if llm[col]["allowed_values"]:
            desc += " Values: " + llm[col]["allowed_values"].replace(" | ", " / ")
        return "LLM", desc
    if col.startswith("fair_") and col != "fair_score":
        code = col[5:].replace("_", ".")
        lab = next((f"{l}: {x}" for c_, l, x in FAIR_INDICATORS if c_ == code), "")
        return "Code + Audit", f"FAIR indicator {code}. {lab}"
    src = "Audit" if col in AUDIT_COLS else "OpenAlex" if col in ("paper_doi", "citations") else "Code"
    return src, DERIVED_DOCS.get(col, "")


def write_dictionary(df: pd.DataFrame, path: Path) -> None:
    lines = ["# ICWSM-DataHub data dictionary", "",
             f"Catalog version {CATALOG_VERSION}, schema version {SCHEMA_VERSION}. "
             f"{len(df.columns)} columns, {len(df)} papers.", "",
             "Source: **LLM** = extracted from the paper by Claude and verified (02_verify.py); "
             "**Code** = computed deterministically; **Audit** = observed in the repository audit; "
             "**OpenAlex** = bibliometrics.", "",
             "| Column | Source | Type | Description |", "|---|---|---|---|"]
    for col in df.columns:
        src, desc = describe(col)
        dtype = "list" if df[col].apply(lambda x: isinstance(x, list)).any() else str(df[col].dtype)
        lines.append(f"| `{col}` | {src} | {dtype} | {desc.replace('|', '/')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_croissant(df_csv: pd.DataFrame, csv_path: Path, out: Path, base_url: str) -> None:
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    def dtype(col: str) -> str:
        s = df_csv[col].dropna()
        if s.empty:
            return "sc:Text"
        if pd.api.types.is_bool_dtype(s) or set(map(str, s.unique())) <= {"True", "False"}:
            return "sc:Boolean"
        if pd.api.types.is_integer_dtype(s):
            return "sc:Integer"
        if pd.api.types.is_float_dtype(s):
            return "sc:Float"
        return "sc:Text"

    ctx = {
        "@language": "en", "@vocab": "https://schema.org/", "citeAs": "cr:citeAs", "column": "cr:column",
        "conformsTo": "dct:conformsTo", "cr": "http://mlcommons.org/croissant/", "rai": "http://mlcommons.org/croissant/RAI/",
        "data": {"@id": "cr:data", "@type": "@json"}, "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
        "dct": "http://purl.org/dc/terms/", "examples": {"@id": "cr:examples", "@type": "@json"},
        "extract": "cr:extract", "field": "cr:field", "fileProperty": "cr:fileProperty", "fileObject": "cr:fileObject",
        "fileSet": "cr:fileSet", "format": "cr:format", "includes": "cr:includes", "isLiveDataset": "cr:isLiveDataset",
        "jsonPath": "cr:jsonPath", "key": "cr:key", "md5": "cr:md5", "parentField": "cr:parentField", "path": "cr:path",
        "recordSet": "cr:recordSet", "references": "cr:references", "regex": "cr:regex", "repeated": "cr:repeated",
        "replace": "cr:replace", "sc": "https://schema.org/", "separator": "cr:separator", "source": "cr:source",
        "subField": "cr:subField", "transform": "cr:transform",
    }
    doc = {
        "@context": ctx, "@type": "sc:Dataset", "conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": "ICWSM-DataHub",
        "description": ("A curated meta-dataset of ICWSM Dataset Track papers (2020-2026): extracted metadata, "
                        "FAIR indicators and an empirical 2026 accessibility audit of every released dataset."),
        "url": base_url, "license": "https://creativecommons.org/licenses/by/4.0/", "version": CATALOG_VERSION,
        "isLiveDataset": True, "datePublished": date.today().isoformat(),
        "citeAs": "Please cite the ICWSM-DataHub paper (see CITATION.cff).",
        "distribution": [{"@type": "cr:FileObject", "@id": "catalog-csv", "name": "catalog-csv",
                          "contentUrl": "icwsm_datahub_catalog.csv", "encodingFormat": "text/csv", "sha256": sha}],
        "recordSet": [{
            "@type": "cr:RecordSet", "@id": "papers", "name": "papers",
            "description": "One record per ICWSM Dataset Track paper.",
            "field": [{"@type": "cr:Field", "@id": f"papers/{c}", "name": c, "dataType": dtype(c),
                       "description": describe(c)[1],
                       "source": {"fileObject": {"@id": "catalog-csv"}, "extract": {"column": c}}}
                      for c in df_csv.columns],
        }],
    }
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def clean_title(t: Any) -> Any:
    """Undo PDF artifacts in titles: ligatures (fi, fl) and a space before a colon."""
    if not isinstance(t, str):
        return t
    import re
    import unicodedata
    t = unicodedata.normalize("NFKC", t)
    return re.sub(r"\s+:", ":", " ".join(t.split()))


WEB_FIELDS = {
    "paper_id": "id", "year": "year", "title": "title", "authors": "authors", "dataset_name": "name",
    "platform_families": "families", "platforms": "platforms", "modalities": "modalities", "languages_iso": "languages",
    "num_languages": "n_languages", "total_records": "records", "primary_record_unit": "unit",
    "temporal_coverage_start": "start", "temporal_coverage_end": "end", "research_domains": "domains",
    "supported_tasks": "tasks", "annotation_paradigms": "annotation", "uses_llm_annotation": "llm",
    "llms_used": "llms", "iaa_reported": "iaa", "distribution_form": "form", "access_mechanism": "access",
    "license_category": "license", "persistent_doi": "doi", "dataset_urls": "urls", "code_repository_url": "code",
    "primary_host": "host", "fate_2026": "fate", "fair_score": "fair", "any_url_alive": "alive",
    "primary_url_status": "status", "downloads_total": "downloads", "downloads_metric_source": "dl_source",
    "citations": "citations", "closed_access_dependencies": "closed", "event_category": "event",
    "event_name": "event_name", "data_geography": "geo", "data_regions": "geo_regions", "author_countries": "aff",
    "author_regions": "aff_regions", "availability_quote": "quote", "paper_doi": "paper_doi",
    "extraction_confidence": "confidence", "audited_at": "audited_at", "dead_link_in_wayback": "wayback",
    "coverage_end_year": "end_year", "depends_on_closed_access": "closed_dep",
    # checklist chapter: what papers say vs what they do
    "includes_paper_checklist": "checklist", "discusses_misuse_risks": "misuse", "body_mentions_misuse": "misuse_body",
    "explicitly_mentions_fair": "fair_talk", "body_mentions_fair": "fair_body", "has_doi": "has_doi",
    "repo_license_present": "repo_license",
}


def write_web(df: pd.DataFrame, out: Path, is_demo: bool, rel: Dict[str, Any] | None = None) -> None:
    rel = rel or {}
    fair_cols = ["fair_" + c.replace(".", "_") for c, _, _ in FAIR_INDICATORS]
    recs = []
    for _, r in df.iterrows():
        o = {short: r.get(col) for col, short in WEB_FIELDS.items() if col in df.columns}
        o["fair_ind"] = [bool(r[c]) for c in fair_cols]
        recs.append(o)
    payload = {
        "meta": {
            "title": "ICWSM-DataHub", "catalog_version": CATALOG_VERSION, "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "is_demo": is_demo,
            "n_records": len(recs), "years": sorted(int(y) for y in df["year"].unique()),
            "zenodo_doi": rel.get("zenodo_doi"), "repo_url": rel.get("repo_url"), "site_url": rel.get("site_url"),
            "author_name": rel.get("author_name"),
            "fair_indicators": [{"code": c, "label": l, "explain": x} for c, l, x in FAIR_INDICATORS],
            "fate_order": FATE_ORDER, "fate_colors": FATE_COLORS,
            "family_order": FAMILY_ORDER, "family_colors": FAMILY_COLORS,
            "api_events": [{"date": d_, "label": l} for d_, l in API_EVENTS],
        },
        "records": recs,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(clean_nan(payload), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="catalog")
    ap.add_argument("--web-out", default="docs/data/catalog.json")
    ap.add_argument("--base-url", default="https://<user>.github.io/icwsm-datahub/")
    ap.add_argument("--demo", action="store_true", help="Generate synthetic inputs first (testing only)")
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    if args.demo:
        from icwsm_datahub.demo import write_demo_workspace
        data_dir, out_dir = Path("demo/data"), Path("demo/catalog")
        write_demo_workspace(data_dir)

    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((data_dir / "extracted").glob("*.json"))]
    if not recs:
        sys.exit(f"No extracted records in {data_dir / 'extracted'}.")
    global TEXT_DIR
    TEXT_DIR = data_dir / "interim" / "text"
    audit_dir = data_dir / "audit"
    ver = read_optional_csv(data_dir / "interim" / "verification.csv")
    ver_map = {r["paper_id"]: r.to_dict() for _, r in ver.iterrows()} if not ver.empty else {}
    cit = read_optional_csv(data_dir / "enrich" / "citations.csv")
    cit_map = {r["paper_id"]: r.to_dict() for _, r in cit.iterrows()} if not cit.empty else {}
    overrides = load_overrides(data_dir / "curation" / "overrides.csv")
    manual = {}
    mpath = data_dir / "curation" / "audit_checks.csv"
    if mpath.exists():
        for _, m in pd.read_csv(mpath, dtype=str).fillna("").iterrows():
            manual[m["paper_id"]] = m

    rows, nested = [], []
    missing_audit = 0
    for rec in recs:
        pid = rec["paper_id"]
        ap_ = audit_dir / f"{pid}.json"
        audit = json.loads(ap_.read_text())["summary"] if ap_.exists() else {}
        if pid in manual:  # a person checked this link in a browser; their observation wins
            m = manual[pid]
            changed = False
            for col in ("any_url_alive", "is_restricted", "files_empty"):
                v = m.get(col, "").strip().lower()
                if v in ("true", "false"):
                    audit[col] = v == "true"
                    if col == "is_restricted":
                        audit["manual_restricted"] = v == "true"
                    changed = True
            if changed:
                audit["manual_check"] = f"{m.get('checked_on', '')} {m.get('note', '')}".strip() or "checked in browser"
        missing_audit += not audit
        row = build_row(rec, audit, ver_map.get(pid, {}), cit_map.get(pid, {}), overrides.get(pid, []))
        rows.append(row)
        nested.append({"paper_id": pid, "year": rec["year"], "record": row,
                       "provenance": rec.get("provenance"), "deterministic": rec.get("deterministic")})
    if missing_audit:
        print(f"[warn] {missing_audit} records have no audit yet (run 03_audit.py). Audit fields are empty.")

    df = pd.DataFrame(rows).sort_values(["year", "title"]).reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "icwsm_datahub_catalog.json").write_text(
        json.dumps(clean_nan(nested), indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    df_csv = df.copy()
    for col in df_csv.columns:
        if df_csv[col].apply(lambda x: isinstance(x, list)).any():
            df_csv[col] = df_csv[col].apply(lambda x: "; ".join(map(str, x)) if isinstance(x, list) else x)
    csv_path = out_dir / "icwsm_datahub_catalog.csv"
    df_csv.to_csv(csv_path, index=False)
    try:
        df.to_parquet(out_dir / "icwsm_datahub_catalog.parquet", index=False)
    except Exception as exc:
        print(f"[warn] parquet not written ({exc}); pip install pyarrow")
    write_dictionary(df, out_dir / "DATA_DICTIONARY.md")
    from icwsm_datahub import release
    rel = {} if args.demo else release.load()
    base_url = rel.get("site_url") or args.base_url
    write_croissant(df_csv, csv_path, out_dir / "croissant.json", base_url)
    write_web(df, Path(args.web_out), is_demo=args.demo, rel=rel)
    if not args.demo:
        release.write_citation(rel, CATALOG_VERSION)
        release.write_zenodo(rel, CATALOG_VERSION)
        release.set_site_url(rel)
        if not rel.get("github_user"):
            print("Note: release.json has no github_user yet, so the site has no correction link or share URL.")
    # The explorer's "Download and cite" links point next to catalog.json.
    import shutil
    web_dir = Path(args.web_out).parent
    for name in ("icwsm_datahub_catalog.csv", "icwsm_datahub_catalog.parquet", "icwsm_datahub_catalog.json",
                 "DATA_DICTIONARY.md", "croissant.json"):
        if (out_dir / name).exists():
            shutil.copy2(out_dir / name, web_dir / name)

    print(f"Catalog: {len(df)} papers x {len(df.columns)} columns -> {out_dir}/")
    print(f"Web data -> {args.web_out}" + ("  (DEMO, synthetic)" if args.demo else ""))
    print(df.groupby("year").agg(papers=("paper_id", "count"), fair=("fair_score", "median"),
                                 usable=("funnel_usable", "mean")).round(2).to_string())


if __name__ == "__main__":
    main()

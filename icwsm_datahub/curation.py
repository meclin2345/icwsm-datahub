"""Human corrections ("overrides"), shared by the audit, citation and catalog steps.

data/curation/overrides.csv columns: paper_id, field, value, reason, curator
  field is "section.field", e.g. distribution.distribution_form
  value is plain text, or JSON for lists/booleans/numbers: ["Zenodo"]  false  353
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def clean_cell(value: str) -> str:
    """Undo what spreadsheets do to typed values: curly quotes, TRUE/FALSE, stray spaces."""
    v = str(value).strip()
    for a, b in (("\u201c", '"'), ("\u201d", '"'), ("\u2018", "'"), ("\u2019", "'")):
        v = v.replace(a, b)
    if v.lower() in ("true", "false", "null"):
        v = v.lower()
    return v


def load_overrides(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    if not Path(path).exists():
        return out
    for _, r in pd.read_csv(path, dtype=str).fillna("").iterrows():
        raw = clean_cell(r["value"])
        try:
            val = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            val = raw if raw != "" else None
        out.setdefault(r["paper_id"], []).append({"field": r["field"], "value": val, "reason": r.get("reason", "")})
    return out


def apply_overrides(ex: Dict[str, Any], items: List[Dict[str, Any]]) -> int:
    n = 0
    for it in items:
        section, field = it["field"].split(".", 1)
        if section in ex and field in ex[section]:
            ex[section][field] = it["value"]
            n += 1
        else:
            print(f"[warn] override field not found: {it['field']}")
    return n


def get_path(obj: Dict[str, Any], dotted: str) -> Any:
    for part in dotted.split("."):
        obj = obj.get(part) if isinstance(obj, dict) else None
    return obj

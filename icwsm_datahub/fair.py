"""Operational FAIR indicators and the "available vs. usable" funnel.

Both constructs combine what the paper CLAIMS (LLM extraction) with what we can
OBSERVE in 2026 (repository audit). They are deliberately simple and binary so a
reviewer can recompute any score by hand from the catalog CSV.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .normalize import ARCHIVAL_HOSTS, OPEN_FORMATS

# Each indicator: (code, short label, explanation used in the data dictionary & web)
FAIR_INDICATORS = [
    ("F1", "Persistent identifier", "A dataset DOI is reported and is registered with DataCite or resolves."),
    ("F2", "Rich metadata", "A standard documentation framework (datasheet, data card, data statement, Croissant) is used."),
    ("F4", "Indexed", "The dataset DOI is registered and findable in DataCite."),
    ("A1", "Retrievable", "At least one dataset link resolves in the 2026 audit."),
    ("A1.1", "Open protocol", "Files can be downloaded openly, without forms, approval or login walls."),
    ("A2", "Persistent hosting", "Hosted on an archival repository (Zenodo, Dataverse, Figshare, OSF, ICPSR, GESIS...)."),
    ("I1", "Open formats", "Released in open, widely readable formats (CSV, JSON(L), Parquet, GraphML...)."),
    ("R1.1", "Clear license", "A license or explicit terms of use are stated."),
    ("R1.2", "Provenance", "Collection method and temporal coverage (start and end) are reported."),
    ("R1.3", "Reusable pipeline", "Code to collect, hydrate or load the data is released."),
]

FUNNEL_STAGES = [
    "Papers",
    "Release claimed",
    "Link provided",
    "Link alive in 2026",
    "Files openly downloadable",
    "Usable without a closed API",
]

FATE_ORDER = [
    "Open and usable",
    "Open, needs closed API to rehydrate",
    "Restricted access (request or agreement)",
    "Alive but empty or login-walled",
    "Link dead",
    "Never released / no link",
]
FATE_KEYS = {  # stable short keys for LaTeX macros and CSS
    "Open and usable": "usable",
    "Open, needs closed API to rehydrate": "needs_api",
    "Restricted access (request or agreement)": "restricted",
    "Alive but empty or login-walled": "empty",
    "Link dead": "dead",
    "Never released / no link": "never",
}
FATE_COLORS = {
    "Open and usable": "#009E73",
    "Open, needs closed API to rehydrate": "#56B4E9",
    "Restricted access (request or agreement)": "#E69F00",
    "Alive but empty or login-walled": "#CC79A7",
    "Link dead": "#D55E00",
    "Never released / no link": "#9A9A9A",
}

# Platforms where dehydrated IDs can no longer be rehydrated for free (2026).
HYDRATION_BLOCKED_PLATFORMS = {"Twitter/X"}


def _truthy(x: Any) -> bool:
    return bool(x) and str(x).lower() not in ("false", "nan", "none", "0")


def fair_indicators(ex: Dict[str, Any], audit: Dict[str, Any]) -> Dict[str, bool]:
    d, c = ex["distribution"], ex["collection"]
    doi = d.get("persistent_doi")
    datacite = _truthy(audit.get("datacite_registered"))
    formats = set(d.get("data_file_formats") or []) | {
        x.strip(".").upper() for x in (audit.get("file_extensions") or "").split(",") if x.strip()
    }
    ind = {
        "F1": bool(doi) and (datacite or _truthy(audit.get("doi_resolves"))),
        "F2": d.get("documentation_framework") not in (None, "None mentioned", "Custom README or codebook"),
        "F4": datacite,
        "A1": _truthy(audit.get("any_url_alive")),
        "A1.1": _truthy(audit.get("any_url_alive"))
        and not _truthy(audit.get("login_wall"))
        and (not audit["manual_restricted"] if audit.get("manual_restricted") is not None
             else (not _truthy(audit.get("is_restricted"))
                   and d.get("access_mechanism") in ("Open public download", "Not specified"))),
        "A2": bool(set(d.get("hosting_repositories") or []) & ARCHIVAL_HOSTS),
        "I1": bool(formats & OPEN_FORMATS),
        "R1.1": d.get("license_category") not in (None, "Not specified"),
        "R1.2": bool(c.get("temporal_coverage_start")) and bool(c.get("temporal_coverage_end"))
        and c.get("collection_methods") not in (None, [], ["Not specified"]),
        "R1.3": bool(d.get("provides_collection_or_hydration_code")),
    }
    return ind


def fair_score(ind: Dict[str, bool]) -> float:
    return round(100.0 * sum(ind.values()) / len(ind), 1)


def dataset_fate(ex: Dict[str, Any], audit: Dict[str, Any]) -> str:
    """Mutually exclusive 2026 state of a dataset; drives the funnel and the fate chart."""
    d, c = ex["distribution"], ex["collection"]
    urls = d.get("dataset_urls") or []
    has_link = bool(urls) or bool(d.get("persistent_doi"))
    if not has_link or d.get("release_status") in ("Not released",):
        return "Never released / no link"
    if not _truthy(audit.get("any_url_alive")):
        return "Link dead"
    manual = audit.get("manual_restricted")
    # A person found the files cannot be downloaded without a request/agreement. Restricted records
    # hide their file list, so "no files visible" is expected there and does not mean broken.
    if manual is True:
        return "Restricted access (request or agreement)"
    # Broken: the link works but there is nothing to download.
    if _truthy(audit.get("files_empty")) or _truthy(audit.get("login_wall")):
        return "Alive but empty or login-walled"
    claimed = d.get("access_mechanism") in ("Credentialed access, form or data use agreement",
                                            "Email the authors for approval")
    restricted = manual if manual is not None else (_truthy(audit.get("is_restricted")) or claimed)
    if restricted:
        return "Restricted access (request or agreement)"
    if needs_closed_api(ex):
        return "Open, needs closed API to rehydrate"
    return "Open and usable"


def needs_closed_api(ex: Dict[str, Any]) -> bool:
    d, c = ex["distribution"], ex["collection"]
    if d.get("distribution_form") not in ("Dehydrated IDs only", "Mixed (IDs plus derived features)"):
        return False
    return bool(set(c.get("platforms") or []) & HYDRATION_BLOCKED_PLATFORMS)


def funnel_flags(ex: Dict[str, Any], audit: Dict[str, Any]) -> Dict[str, bool]:
    d = ex["distribution"]
    fate = dataset_fate(ex, audit)
    claimed = d.get("release_status") in ("Released at publication", "Partially released", "Promised or forthcoming",
                                          "Available on request only", "Unclear")
    has_link = bool(d.get("dataset_urls")) or bool(d.get("persistent_doi"))
    alive = has_link and _truthy(audit.get("any_url_alive"))
    open_dl = alive and fate in ("Open and usable", "Open, needs closed API to rehydrate")
    usable = fate == "Open and usable"
    return {
        "funnel_release_claimed": claimed,
        "funnel_link_provided": has_link,
        "funnel_link_alive": alive,
        "funnel_open_download": open_dl,
        "funnel_usable": usable,
    }

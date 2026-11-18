"""Post-processing shared by verification, catalog building and analysis.

Everything here is deterministic and documented so the paper can describe it in
one paragraph ("Normalization") and reviewers can audit it.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

# ---------------------------------------------------------------- sentinels
NULL_STRINGS = {"", "n/a", "na", "none", "null", "not specified", "not stated", "unknown", "not mentioned", "-"}


def none_if_sentinel(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in NULL_STRINGS:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value < 0:
        return None
    if isinstance(value, float) and value == -1.0:   # -1.0 is the sentinel; other negatives (e.g. kappa) are real
        return None
    return value


def clean_list(values: Optional[Iterable[Any]], upper: bool = False, lower: bool = False) -> List[Any]:
    out: List[Any] = []
    for v in values or []:
        v = getattr(v, "value", v)
        if isinstance(v, str):
            v = v.strip()
            if v.lower() in NULL_STRINGS:
                continue
            v = v.upper() if upper else v.lower() if lower else v
        if v not in out:
            out.append(v)
    return out


# ---------------------------------------------------------------- URLs and DOIs
DOI_RE = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)")


def clean_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    u = url.strip().strip("<>[]()\"'")
    u = re.sub(r"\s+", "", u)                       # spaces injected by PDF layout
    u = re.sub(r"^(https?):/+", r"\1://", u, flags=re.I)
    u = re.sub(r"#.*$", "", u)                      # tracking fragments such as #.XheASC3MzOQ
    u = u.rstrip(".,;:)]}>")
    if u.lower().startswith("doi:"):
        u = "https://doi.org/" + u[4:]
    if not re.match(r"^https?://", u, flags=re.I):
        if re.match(r"^(www\.|[a-z0-9-]+\.[a-z]{2,})", u, flags=re.I):
            u = "https://" + u
        else:
            return None
    return u


def clean_doi(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    m = DOI_RE.search(re.sub(r"\s+", "", raw))
    if not m:
        return None
    doi = m.group(1).rstrip(".,;:)")
    return doi.lower() if doi.lower().startswith("10.5281/zenodo") else doi


def is_paper_doi(doi: Optional[str]) -> bool:
    return bool(doi) and doi.lower().startswith("10.1609/icwsm")


# ---------------------------------------------------------------- platform families
PLATFORM_FAMILY = {
    "Twitter/X": "Twitter/X",
    "Reddit": "Reddit",
    "YouTube": "YouTube",
    "TikTok": "TikTok",
    "Facebook": "Meta (Facebook/Instagram/Threads)",
    "Instagram": "Meta (Facebook/Instagram/Threads)",
    "Threads": "Meta (Facebook/Instagram/Threads)",
    "Telegram": "Messaging (Telegram/WhatsApp/Discord)",
    "WhatsApp": "Messaging (Telegram/WhatsApp/Discord)",
    "Discord": "Messaging (Telegram/WhatsApp/Discord)",
    "Bluesky": "Decentralized and alt-tech",
    "Mastodon/Fediverse": "Decentralized and alt-tech",
    "4chan/8kun/Imageboards": "Decentralized and alt-tech",
    "Alt-Tech (Gab/Parler/Truth Social/Rumble/BitChute)": "Decentralized and alt-tech",
    "Web/News/Blogs": "Web, news and wiki",
    "Wikipedia/Wikidata": "Web, news and wiki",
    "Search engines/Google Trends": "Web, news and wiki",
    "Q&A and Forums (Quora/Stack Exchange/other forums)": "Web, news and wiki",
    "AI chatbots and LLM output (ChatGPT logs, AI-generated content)": "AI and LLM content",
}
FAMILY_ORDER = [
    "Twitter/X", "Reddit", "YouTube", "TikTok", "Meta (Facebook/Instagram/Threads)",
    "Messaging (Telegram/WhatsApp/Discord)", "Decentralized and alt-tech",
    "Web, news and wiki", "AI and LLM content", "Other",
]
# Okabe-Ito based, shared by paper figures and the web explorer.
FAMILY_COLORS = {
    "Twitter/X": "#0072B2",
    "Reddit": "#E69F00",
    "YouTube": "#D55E00",
    "TikTok": "#CC79A7",
    "Meta (Facebook/Instagram/Threads)": "#56B4E9",
    "Messaging (Telegram/WhatsApp/Discord)": "#009E73",
    "Decentralized and alt-tech": "#D4B800",
    "Web, news and wiki": "#8C8C8C",
    "AI and LLM content": "#5B3A8E",
    "Other": "#C7C7C7",
}


def platform_family(platform: str) -> str:
    return PLATFORM_FAMILY.get(platform, "Other")


# ---------------------------------------------------------------- data-access shutdowns
# (pattern over tools/methods text, label, date the access path closed)
ACCESS_SHUTDOWNS = [
    (r"academic\s*(research)?\s*(api|track|product|access)|twitter\s*(advanced\s*search\s*|search\s*)?(api|v2|v1\.1|streaming)|tweepy|twarc|"
     r"filtered\s*stream|decahose|firehose", "Twitter/X free or Academic API", "2023-02"),
    (r"snscrape|twint|getoldtweets", "Twitter/X scraping tools", "2023-06"),
    (r"pushshift", "Pushshift Reddit archive", "2023-05"),
    (r"praw|reddit\s*(api|data\s*api)", "Reddit free API", "2023-07"),
    (r"crowdtangle", "CrowdTangle", "2024-08"),
]
API_EVENTS = [  # for annotating figures
    ("2023-02", "Twitter/X ends free API"),
    ("2023-05", "Pushshift cut off"),
    ("2023-07", "Reddit API pricing"),
    ("2024-08", "CrowdTangle shut down"),
]


def closed_access_dependencies(tools: List[str], platforms: List[str], methods: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """Tools named in the paper are matched first. Papers often name no tool, so the platform plus the
    collection method is used as a second signal (Twitter/X via API or scraping, Reddit via its API)."""
    text = " ".join(tools or []).lower()
    hits = []
    for pattern, label, date in ACCESS_SHUTDOWNS:
        if re.search(pattern, text):
            hits.append({"dependency": label, "closed": date})
    have = {h["dependency"] for h in hits}
    plat, meth = set(platforms or []), set(methods or [])
    if "Twitter/X" in plat and not any(d.startswith("Twitter/X") for d in have):
        if "Official platform API" in meth:
            hits.append({"dependency": "Twitter/X free or Academic API", "closed": "2023-02"})
        elif "Web scraping or browser automation" in meth:
            hits.append({"dependency": "Twitter/X scraping tools", "closed": "2023-06"})
    if "Reddit" in plat and "Reddit free API" not in have and "Pushshift Reddit archive" not in have \
            and "Official platform API" in meth:
        hits.append({"dependency": "Reddit free API", "closed": "2023-07"})
    return hits


# ---------------------------------------------------------------- licenses / formats / hosting
OPEN_FORMATS = {"CSV", "TSV", "JSON", "JSONL", "NDJSON", "PARQUET", "TXT", "XML", "GRAPHML", "GEXF",
                "GML", "SQLITE", "ARROW", "XLSX", "MP4", "JPG", "PNG", "WAV", "MP3", "HDF5", "NPY", "PKL"}
ARCHIVAL_HOSTS = {"Zenodo", "Dataverse (Harvard or other instance)", "Figshare", "Open Science Framework (OSF)",
                  "Domain archive (ICPSR, GESIS, Dryad, Mendeley Data)"}
HOST_SHORT = {
    "Zenodo": "Zenodo",
    "Dataverse (Harvard or other instance)": "Dataverse",
    "Figshare": "Figshare / OSF / archive",
    "Open Science Framework (OSF)": "Figshare / OSF / archive",
    "Domain archive (ICPSR, GESIS, Dryad, Mendeley Data)": "Figshare / OSF / archive",
    "Hugging Face Hub": "Hugging Face",
    "GitHub or GitLab": "GitHub / GitLab",
    "Kaggle": "Kaggle",
    "Personal or lab website, Google Drive, Dropbox": "Personal site / cloud drive",
    "Available on request only": "On request / not stated",
    "Other": "Other",
    "Not stated": "On request / not stated",
}

# ---------------------------------------------------------------- geography
REGIONS = {
    "North America": "US CA",
    "Latin America & Caribbean": "MX BR AR CL CO PE VE EC BO PY UY CU DO GT HN SV NI CR PA JM TT HT PR",
    "Europe & Central Asia": ("GB IE FR DE NL BE LU CH AT IT ES PT DK NO SE FI IS PL CZ SK HU RO BG GR HR SI RS "
                              "BA ME MK AL EE LV LT UA BY MD RU TR CY MT KZ UZ KG TJ TM GE AM AZ XK LI MC SM AD"),
    "Middle East & North Africa": "IL PS JO LB SY IQ IR SA AE QA KW BH OM YE EG LY TN DZ MA",
    "Sub-Saharan Africa": ("NG KE ZA GH ET UG TZ RW SN CM CI ZW ZM MZ AO SD SS SO CD CG ML BF NE MW BW NA MG "
                           "BJ TG SL LR GN GM MR ER DJ LS SZ"),
    "South Asia": "IN PK BD LK NP AF BT MV",
    "East Asia & Pacific": "CN JP KR KP TW HK MO MN SG MY ID TH VN PH MM KH LA BN AU NZ PG FJ TL",
}
ISO_TO_REGION = {code: region for region, codes in REGIONS.items() for code in codes.split()}
# Brandt-line-style approximation used for the "who studies whom" analysis.
# This is a research choice: document it in the paper and adjust if needed.
GLOBAL_NORTH = set(
    "US CA GB IE FR DE NL BE LU CH AT IT ES PT DK NO SE FI IS PL CZ SK HU RO BG GR HR SI EE LV LT CY MT "
    "AU NZ JP KR SG IL TW HK RU LI MC SM AD".split()
)


def region_of(code: str) -> str:
    code = (code or "").upper()
    if code in ("GLOBAL", "MUL", "WORLD"):
        return "Global / multi-region"
    return ISO_TO_REGION.get(code, "Other / unspecified")


def north_south(code: str) -> Optional[str]:
    code = (code or "").upper()
    if code in ISO_TO_REGION:
        return "Global North" if code in GLOBAL_NORTH else "Global South"
    return None


# ---------------------------------------------------------------- record normalization
def normalize_extraction(ex: Dict[str, Any]) -> Dict[str, Any]:
    """Return a normalized deep copy of one LLM extraction dict (sentinels -> None,
    codes upper/lower-cased, URLs and DOIs cleaned, derived fields added)."""
    import copy
    ex = copy.deepcopy(ex)
    p, c, a, d, e, v = (ex[k] for k in ("paper", "collection", "annotation", "distribution", "ethics", "evidence"))

    p["author_countries"] = clean_list(p.get("author_countries"), upper=True)
    p["num_datasets_released"] = none_if_sentinel(p.get("num_datasets_released"))

    for k in ("dataset_name", "temporal_coverage_start", "temporal_coverage_end", "scale_description_verbatim", "event_name"):
        c[k] = none_if_sentinel(c.get(k))
    for k in ("total_records", "num_accounts_or_users", "num_languages"):
        c[k] = none_if_sentinel(c.get(k))
    c["platforms"] = clean_list(c.get("platforms"))
    c["languages_iso"] = clean_list(c.get("languages_iso"), lower=True)
    c["data_geography"] = clean_list(c.get("data_geography"), upper=True)
    c["collection_tools"] = clean_list(c.get("collection_tools"))
    c["platform_families"] = clean_list(platform_family(x) for x in c["platforms"])
    c["closed_access_dependencies"] = closed_access_dependencies(c["collection_tools"], c["platforms"], c.get("collection_methods"))

    for k in ("num_annotated_records", "num_annotators", "iaa_primary_value"):
        a[k] = none_if_sentinel(a.get(k))
    a["iaa_summary_verbatim"] = none_if_sentinel(a.get("iaa_summary_verbatim"))
    a["llms_used"] = clean_list(a.get("llms_used"))

    d["dataset_urls"] = clean_list(clean_url(u) for u in d.get("dataset_urls", []) if clean_url(u))
    d["code_repository_url"] = clean_url(none_if_sentinel(d.get("code_repository_url")))
    doi = clean_doi(none_if_sentinel(d.get("persistent_doi")))
    if not doi:  # DOI hidden inside a doi.org / zenodo URL
        for u in d["dataset_urls"]:
            doi = clean_doi(u) if "doi.org" in u else None
            if doi:
                break
    d["persistent_doi"] = None if is_paper_doi(doi) else doi
    d["license_verbatim"] = none_if_sentinel(d.get("license_verbatim"))
    d["data_file_formats"] = clean_list(d.get("data_file_formats"), upper=True)

    e["anonymization_details"] = none_if_sentinel(e.get("anonymization_details"))
    for k in list(v.keys()):
        if isinstance(v[k], str) and k != "extraction_confidence":
            v[k] = none_if_sentinel(v[k])
    return ex

"""Synthetic workspace for testing the pipeline end to end BEFORE real extraction.

Writes files in exactly the format the real steps produce (extracted/, audit/,
interim/verification.csv, enrich/citations.csv), validated against the schema.
Trends are plausible but INVENTED. Never report demo numbers.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .schema import (AccessMechanism as AM, AnnotationParadigm as AP, BaselineFamily as BF, CollectionMethod as CM,
                     Confidence, ContributionType as CT, DistributionForm as DF, DocumentationFramework as DOC,
                     EventCategory as EV, HostingRepository as HR, IAAMetric, ICWSMExtraction, IRBStatus,
                     LicenseCategory as LC, ModalityType as MT, PlatformType as PT, RecordUnit as RU,
                     ReleaseStatus as RS, ResearchDomain as RD, TaskType as TT)

YEARS = list(range(2020, 2027))
N_PER_YEAR = dict(zip(YEARS, [32, 38, 44, 47, 52, 56, 60]))


def _y(year: int, values: List[float]) -> float:
    return values[year - 2020]


PLATFORM_P = {
    PT.TWITTER_X: [.62, .60, .55, .50, .34, .24, .18], PT.REDDIT: [.22, .24, .25, .24, .17, .15, .15],
    PT.YOUTUBE: [.08, .10, .12, .14, .18, .21, .22], PT.TIKTOK: [0, .02, .05, .09, .15, .21, .25],
    PT.FACEBOOK: [.10, .10, .09, .08, .06, .07, .06], PT.INSTAGRAM: [.04, .05, .06, .07, .08, .09, .09],
    PT.THREADS: [0, 0, 0, 0, 0, .02, .04], PT.TELEGRAM: [.03, .05, .08, .11, .14, .15, .16],
    PT.WHATSAPP: [.04, .05, .05, .05, .05, .05, .05], PT.DISCORD: [.01, .02, .03, .03, .05, .06, .06],
    PT.BLUESKY: [0, 0, 0, 0, .04, .09, .12], PT.MASTODON: [0, 0, .01, .05, .05, .05, .05],
    PT.CHAN_FORUMS: [.03] * 7, PT.ALT_TECH: [.04, .05, .05, .04, .04, .03, .03],
    PT.WEB_NEWS: [.12, .12, .12, .12, .13, .13, .13], PT.WIKIPEDIA: [.03] * 7,
    PT.AI_PLATFORMS: [0, 0, 0, .02, .06, .10, .15], PT.CHINESE: [.04] * 7, PT.REGIONAL: [.03] * 7,
}
HOST_P = {
    HR.ZENODO: [.25, .28, .32, .36, .40, .43, .45], HR.DATAVERSE: [.08, .08, .09, .09, .10, .10, .10],
    HR.GITHUB: [.35, .33, .30, .26, .22, .20, .19], HR.HUGGING_FACE: [.02, .03, .05, .08, .12, .15, .18],
    HR.PERSONAL_CLOUD: [.18, .15, .12, .10, .07, .06, .05], HR.ON_REQUEST: [.10, .09, .08, .07, .05, .04, .03],
    HR.FIGSHARE: [.02] * 7, HR.OSF: [.03] * 7, HR.KAGGLE: [.02] * 7,
}
ANNOT_P = {
    AP.LLM: [0, 0, .02, .10, .28, .40, .50], AP.HYBRID: [0, 0, .01, .04, .10, .14, .18],
    AP.CROWD: [.35, .33, .30, .27, .22, .20, .18], AP.EXPERT: [.45, .45, .44, .42, .42, .40, .40],
    AP.DISTANT: [.20, .19, .17, .15, .13, .11, .10], AP.CLASSICAL_ML: [.30, .29, .27, .25, .22, .21, .20],
    AP.PLATFORM_NATIVE: [.04, .05, .06, .08, .10, .10, .11],
}
TOPICS = [
    ("COVID-19 vaccine discourse", RD.PUBLIC_HEALTH, EV.HEALTH_CRISIS), ("election misinformation", RD.MISINFORMATION, EV.ELECTIONS),
    ("online hate against migrants", RD.HATE_TOXICITY, EV.NONE), ("climate activism", RD.ACTIVISM, EV.DISASTER),
    ("the Russia-Ukraine war", RD.CRISIS, EV.ARMED_CONFLICT), ("coordinated influence campaigns", RD.INAUTHENTIC, EV.NONE),
    ("mental health support", RD.MENTAL_HEALTH, EV.NONE), ("news sharing", RD.NEWS, EV.NONE),
    ("content moderation decisions", RD.MODERATION, EV.PLATFORM_CHANGE), ("partisan polarization", RD.POLARIZATION, EV.ELECTIONS),
    ("extremist communities", RD.EXTREMISM, EV.NONE), ("AI-generated content", RD.AI_CONTENT, EV.NONE),
    ("influencer marketing", RD.ECONOMY, EV.NONE), ("gendered harassment", RD.IDENTITY, EV.NONE),
    ("platform migration after the Twitter acquisition", RD.COMMUNITIES, EV.PLATFORM_CHANGE),
]
GEO = ["US", "US", "US", "IN", "BR", "GB", "DE", "NG", "KE", "PK", "ID", "MX", "UA", "FR", "ES", "PH", "ZA", "GLOBAL", "GLOBAL"]
AFF = ["US", "US", "US", "US", "GB", "DE", "IT", "IN", "CA", "NL", "CH", "ES", "BR", "SG", "JP", "AU", "IN", "PK", "NG"]
FIRST = ["Ana", "Wei", "Priya", "Omar", "Lena", "Diego", "Aisha", "Tomas", "Mei", "Kofi", "Sara", "Ivan", "Nia", "Rahul"]
LAST = ["Silva", "Chen", "Sharma", "Haddad", "Novak", "Garcia", "Bello", "Kowalski", "Tanaka", "Mensah", "Rossi",
        "Petrov", "Okafor", "Iyer"]


def _pick(p: Dict[Any, List[float]], year: int, rng: random.Random, min_n: int = 1) -> List[Any]:
    chosen = [k for k, v in p.items() if rng.random() < _y(year, v)]
    if len(chosen) < min_n:
        keys, w = list(p.keys()), [_y(year, v) + 1e-3 for v in p.values()]
        chosen = rng.choices(keys, weights=w, k=1)
    return chosen[:3]


def _record(year: int, i: int, rng: random.Random) -> Dict[str, Any]:
    plats = _pick(PLATFORM_P, year, rng)
    topic, domain, event = rng.choice(TOPICS)
    lag = rng.choice([0, 1, 1, 1, 2])
    end_y = year - lag
    end_m = rng.randint(1, 12) if end_y < year else rng.randint(1, 3)
    span_months = rng.choice([1, 3, 6, 12, 18, 24, 36])
    start_total = end_y * 12 + end_m - 1 - span_months
    start, end = f"{start_total // 12}-{start_total % 12 + 1:02d}", f"{end_y}-{end_m:02d}"
    end_key = end_y * 100 + end_m

    tools, methods = [], set()
    for p in plats:
        if p == PT.TWITTER_X:
            if end_key <= 202302:
                tools.append(rng.choice(["Twitter Academic API v2", "Twitter Academic API v2", "snscrape", "twarc"]))
                methods.add(CM.OFFICIAL_API)
            else:
                tools.append(rng.choice(["custom scraper", "X API Pro tier", "data donation browser extension"]))
                methods.add(rng.choice([CM.WEB_SCRAPING, CM.DATA_DONATION, CM.OFFICIAL_API]))
        elif p == PT.REDDIT:
            tools.append("Pushshift" if end_key <= 202304 else "PRAW (Reddit API)")
            methods.add(CM.THIRD_PARTY_ARCHIVE if end_key <= 202304 else CM.OFFICIAL_API)
        elif p == PT.FACEBOOK or p == PT.INSTAGRAM:
            tools.append("CrowdTangle" if end_key <= 202407 else "Meta Content Library")
            methods.add(CM.THIRD_PARTY_ARCHIVE if end_key <= 202407 else CM.RESEARCHER_PROGRAM)
        elif p == PT.TIKTOK:
            tools.append(rng.choice(["TikTok Research API", "yt-dlp", "Playwright scraper"]))
            methods.add(CM.RESEARCHER_PROGRAM if "Research" in tools[-1] else CM.WEB_SCRAPING)
        elif p == PT.YOUTUBE:
            tools.append("YouTube Data API v3")
            methods.add(CM.OFFICIAL_API)
        elif p == PT.TELEGRAM:
            tools.append("Telethon")
            methods.add(CM.OFFICIAL_API)
        elif p == PT.BLUESKY:
            tools.append("AT Protocol firehose")
            methods.add(CM.OFFICIAL_API)
        elif p == PT.AI_PLATFORMS:
            methods.add(rng.choice([CM.SYNTHETIC, CM.DATA_DONATION]))
        else:
            methods.add(CM.WEB_SCRAPING)

    twitter = PT.TWITTER_X in plats
    if twitter and rng.random() < 0.62 - 0.05 * (year - 2020):
        form = DF.DEHYDRATED
    else:
        form = rng.choices([DF.FULL_RAW, DF.ANONYMIZED, DF.MIXED, DF.AGGREGATED],
                           weights=[.35, .25 + .03 * (year - 2020), .15, .08])[0]
    host = rng.choices(list(HOST_P), weights=[_y(year, v) for v in HOST_P.values()])[0]
    rid = 4_000_000 + year * 1000 + i
    urls: List[str] = []
    doi = ""
    if host == HR.ZENODO:
        doi = f"10.5281/zenodo.{rid}"
        urls = [f"https://zenodo.org/records/{rid}"]
    elif host == HR.DATAVERSE:
        doi = f"10.7910/DVN/{rng.randint(10000, 99999)}X"
        urls = [f"https://doi.org/{doi}"]
    elif host == HR.FIGSHARE:
        doi = f"10.6084/m9.figshare.{rid}"
        urls = [f"https://figshare.com/articles/dataset/x/{rid}"]
    elif host == HR.OSF:
        urls = [f"https://osf.io/{rng.randint(10000, 99999)}/"]
    elif host == HR.GITHUB:
        urls = [f"https://github.com/lab{i}/dataset-{year}"]
    elif host == HR.HUGGING_FACE:
        urls = [f"https://huggingface.co/datasets/lab{i}/ds{year}"]
    elif host == HR.KAGGLE:
        urls = [f"https://www.kaggle.com/datasets/lab{i}/ds{year}"]
    elif host == HR.PERSONAL_CLOUD:
        urls = [rng.choice([f"https://drive.google.com/drive/folders/{rid}", f"https://lab{i}.university.edu/data"])]
    release = RS.ON_REQUEST if host == HR.ON_REQUEST else rng.choices(
        [RS.RELEASED, RS.PARTIAL, RS.FORTHCOMING], weights=[.86, .08, .06])[0]
    paradigms = _pick(ANNOT_P, year, rng)
    if rng.random() < 0.15 - 0.01 * (year - 2020):
        paradigms = [AP.UNLABELED]
    llm = any(p in (AP.LLM, AP.HYBRID) for p in paradigms)
    iaa = any(p in (AP.CROWD, AP.EXPERT, AP.HYBRID) for p in paradigms) and rng.random() < .75
    n_rec = int(10 ** rng.uniform(3.2, 7.6))
    langs = ["en"] if rng.random() < .6 else rng.sample(["en", "es", "hi", "pt", "ar", "de", "fr", "id", "sw", "ur", "uk"],
                                                          k=rng.randint(2, 6))
    mods = [MT.TEXT] + ([MT.IMAGE] if rng.random() < .15 + .03 * (year - 2020) else []) + \
           ([MT.VIDEO] if (PT.TIKTOK in plats or PT.YOUTUBE in plats) and rng.random() < .5 else []) + \
           ([MT.NETWORK] if rng.random() < .25 else []) + [MT.METADATA]
    name = "".join(rng.choice("ABCDEFGHKLMNPRSTVW") for _ in range(rng.randint(3, 5))) + f"-{year % 100}"
    lic = rng.choices([LC.CC_BY, LC.CC_BY_NC, LC.CC0, LC.CUSTOM, LC.CODE_LICENSE, LC.NOT_SPECIFIED],
                      weights=[.30 + .02 * (year - 2020), .18, .06, .08, .10, .28 - .025 * (year - 2020)])[0]
    data = {
        "paper": {"title": f"{name}: A {rng.choice(['Large-Scale', 'Multilingual', 'Longitudinal', 'Multimodal', 'Annotated'])} "
                           f"{plats[0].value.split(' (')[0]} Dataset on {topic.capitalize()}",
                  "authors": [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(rng.randint(2, 7))],
                  "author_countries": sorted(set(rng.sample(AFF, k=rng.randint(1, 3)))),
                  "is_industry_collaboration": rng.random() < .15, "contribution_type": CT.NEW_DATASET.value,
                  "num_datasets_released": 1},
        "collection": {
            "dataset_name": name, "platforms": [p.value for p in plats], "platform_names_verbatim": [p.value for p in plats],
            "collection_methods": [m.value for m in methods] or [CM.NOT_SPECIFIED.value], "collection_tools": tools,
            "temporal_coverage_start": start if rng.random() < .9 else "", "temporal_coverage_end": end if rng.random() < .9 else "",
            "primary_record_unit": (RU.VIDEOS if PT.TIKTOK in plats else RU.POSTS).value, "total_records": n_rec,
            "num_accounts_or_users": int(n_rec / rng.uniform(3, 40)) if rng.random() < .6 else -1,
            "scale_description_verbatim": f"{n_rec:,} items", "modalities": [m.value for m in mods],
            "languages_iso": langs, "num_languages": len(langs), "includes_code_mixing": rng.random() < .07,
            "data_geography": sorted(set(rng.sample(GEO, k=rng.randint(1, 2)))), "event_category": event.value,
            "event_name": topic if event != EV.NONE else "", "involves_vulnerable_groups": rng.random() < .12},
        "annotation": {
            "research_domains": [domain.value], "supported_tasks": [rng.choice(list(TT)).value for _ in range(rng.randint(1, 3))],
            "tasks_verbatim": [], "annotation_paradigms": [p.value for p in paradigms],
            "llms_used": [rng.choice(["GPT-4", "GPT-4o", "Llama-3-70B", "Claude 3.5 Sonnet", "Mixtral"])] if llm else [],
            "num_annotated_records": rng.randint(500, 20000) if paradigms != [AP.UNLABELED] else -1,
            "num_annotators": rng.randint(2, 8) if iaa else -1, "iaa_reported": iaa,
            "iaa_metrics": [rng.choice([IAAMetric.COHENS_KAPPA, IAAMetric.KRIPPENDORFF, IAAMetric.FLEISS_KAPPA]).value] if iaa else [],
            "iaa_primary_value": round(rng.uniform(.45, .9), 2) if iaa else -1.0,
            "iaa_summary_verbatim": "", "baseline_families": [rng.choice(list(BF)).value]},
        "distribution": {
            "release_status": release.value, "hosting_repositories": [host.value], "dataset_urls": urls,
            "persistent_doi": doi, "code_repository_url": f"https://github.com/lab{i}/code{year}" if rng.random() < .55 else "",
            "distribution_form": form.value,
            "access_mechanism": (AM.CREDENTIALED if rng.random() < .12 else AM.OPEN).value if host != HR.ON_REQUEST else AM.AUTHOR_APPROVAL.value,
            "license_verbatim": "" if lic == LC.NOT_SPECIFIED else lic.value, "license_category": lic.value,
            "data_file_formats": rng.sample(["CSV", "JSONL", "PARQUET", "JSON", "TSV"], k=rng.randint(1, 2)),
            "explicitly_mentions_fair": rng.random() < .05 + .07 * (year - 2020),
            "documentation_framework": rng.choices(list(DOC), weights=[.10 + .03 * (year - 2020), .02, .03 + .03 * (year - 2020), .01, .01 * (year - 2020), .3, .5])[0].value,
            "provides_collection_or_hydration_code": rng.random() < .5},
        "ethics": {"irb_status": rng.choices(list(IRBStatus), weights=[.25, .25, .15 + .04 * (year - 2020), .35 - .04 * (year - 2020)])[0].value,
                   "pii_anonymization_applied": rng.random() < .55 + .04 * (year - 2020), "anonymization_details": "",
                   "discusses_platform_tos": rng.random() < .4 + .05 * (year - 2020), "discusses_limitations": rng.random() < .8,
                   "discusses_misuse_risks": rng.random() < .35 + .05 * (year - 2020),
                   "includes_paper_checklist": year >= 2023 and rng.random() < .9, "ai_usage_disclosed": year >= 2024 and rng.random() < .3},
        "evidence": {"availability_quote": f"The dataset is available at {urls[0]}." if urls else "",
                     "scale_quote": f"We collected {n_rec:,} items.", "annotation_quote": "", "ethics_quote": "",
                     "license_quote": "", "extraction_confidence": rng.choice(["High", "High", "Medium", "Low"]),
                     "ambiguity_notes": ""},
    }
    return ICWSMExtraction.model_validate(data).model_dump(mode="json")


def _audit(ex: Dict[str, Any], year: int, rng: random.Random) -> Dict[str, Any]:
    d = ex["distribution"]
    host = d["hosting_repositories"][0]
    age = 2026.8 - (year + 0.45)
    urls = d["dataset_urls"]
    p_alive = {"Zenodo": .985, "Dataverse (Harvard or other instance)": .98, "Figshare": .98,
               "Open Science Framework (OSF)": .95, "Hugging Face Hub": .95, "Kaggle": .93,
               "GitHub or GitLab": .97 - .025 * age, "Personal or lab website, Google Drive, Dropbox": .88 - .09 * age}.get(host, .9)
    alive = bool(urls) and rng.random() < p_alive
    status = "No dataset link" if not urls else ("Alive" if alive else rng.choice(["Dead (404)", "DNS / connection failure",
                                                                                   "Soft 404 (suspected)", "Dead (404)"]))
    login = host.startswith("Personal") and alive and rng.random() < .12
    restricted = alive and ((host.startswith("Dataverse") and rng.random() < .15) or (host == "Zenodo" and rng.random() < .07))
    repo = {"Zenodo": "Zenodo", "Dataverse (Harvard or other instance)": "Dataverse", "Figshare": "Figshare",
            "Hugging Face Hub": "Hugging Face", "GitHub or GitLab": "GitHub"}.get(host)
    base = {"Zenodo": 45, "Dataverse (Harvard or other instance)": 30, "Figshare": 20, "Hugging Face Hub": 400}.get(host)
    downloads = None
    if alive and base and not restricted:
        mult = 0.45 if d["distribution_form"] == "Dehydrated IDs only" else 1.0
        downloads = int(np.exp(rng.gauss(np.log(base * mult * age), 0.9)))
    src = {"Zenodo": "Zenodo stats", "Dataverse": "Dataverse Make Data Count", "Figshare": "Figshare stats",
           "Hugging Face": "Hugging Face (all time)"}.get(repo or "")
    return {
        "paper_id": None, "year": year, "audited_at": datetime(2026, 10, 15, tzinfo=timezone.utc).isoformat(),
        "normalized_doi": d["persistent_doi"] or None, "doi_resolves": bool(d["persistent_doi"]) and alive,
        "n_urls_checked": len(urls), "any_url_alive": alive and not login, "primary_url": urls[0] if urls else None,
        "primary_url_status": "Login wall" if login else status, "primary_resolved_url": urls[0] if alive else None,
        "login_wall": login, "account_required": host in ("Kaggle", "Hugging Face Hub") and rng.random() < .4,
        "soft_404": status.startswith("Soft"), "verified_repo": repo if alive else None,
        "file_count": rng.randint(1, 12) if alive and repo else None,
        "total_size_mb": round(10 ** rng.uniform(-0.5, 4), 2) if alive and repo else None,
        "file_extensions": ", ".join("." + f.lower() for f in d["data_file_formats"]) if alive else "",
        "is_restricted": restricted, "files_empty": alive and repo is not None and rng.random() < .03,
        "downloads_total": downloads, "downloads_unique": int(downloads * .6) if downloads else None,
        "views_total": int(downloads * rng.uniform(2, 5)) if downloads else None,
        "downloads_metric_source": src if downloads else None,
        "datacite_registered": bool(d["persistent_doi"]), "datacite_publisher": repo,
        "repo_license": ("cc-by-4.0" if rng.random() < .85 else None) if alive and repo else None,
        "datacite_citations": rng.randint(0, 6) if d["persistent_doi"] else None,
        "code_url": d["code_repository_url"], "code_url_alive": (rng.random() < .93) if d["code_repository_url"] else None,
        "code_stars": rng.randint(0, 120) if d["code_repository_url"] else None,
        "dead_link_in_wayback": (rng.random() < .4) if urls and not alive else None,
    }


def write_demo_workspace(root: Path, seed: int = 7) -> None:
    rng = random.Random(seed)
    for sub in ("extracted", "audit", "interim", "enrich"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for old in (root / "extracted").glob("*.json"):
        old.unlink()
    for old in (root / "audit").glob("*.json"):
        old.unlink()
    ver_rows, cit_rows = [], []
    for year in YEARS:
        for i in range(N_PER_YEAR[year]):
            pid = f"{year}_demo_{i:03d}"
            ex = _record(year, i, rng)
            det = {"page_count": rng.choice([8, 9, 10, 11, 12]), "n_chars": rng.randint(45000, 90000), "sha256": "demo",
                   "checklist_detected": ex["ethics"]["includes_paper_checklist"], "urls_in_body": ex["distribution"]["dataset_urls"],
                   "urls_in_references": [], "dois_in_body": [], "mentions_fair_regex": ex["distribution"]["explicitly_mentions_fair"],
                   "mentions_zenodo": False, "mentions_dataverse": False}
            rec = {"paper_id": pid, "year": year, "pdf_filename": f"{pid}.pdf",
                   "provenance": {"model": "DEMO (synthetic)", "prompt_version": "demo", "schema_version": "2.0.0",
                                  "extracted_at": "2026-10-01T00:00:00+00:00", "usage": []},
                   "deterministic": det, "extraction": ex}
            (root / "extracted" / f"{pid}.json").write_text(json.dumps(rec, indent=1))
            audit = _audit(ex, year, rng)
            audit["paper_id"] = pid
            (root / "audit" / f"{pid}.json").write_text(json.dumps({"summary": audit}, indent=1))
            n_urls = len(ex["distribution"]["dataset_urls"])
            ver_rows.append({"paper_id": pid, "year": year, "availability_quote_score": 100.0 if n_urls else np.nan,
                             "scale_quote_score": rng.choice([100.0, 100.0, 100.0, 94.0, 71.0]),
                             "n_dataset_urls": n_urls, "n_urls_grounded": n_urls if rng.random() < .97 else 0,
                             "n_flags": rng.choice([0, 0, 0, 1]), "flags": ""})
            age = 2026.8 - (year + 0.45)
            usable = audit["any_url_alive"] and not audit["is_restricted"]
            cit_rows.append({"paper_id": pid, "paper_doi": f"10.1609/icwsm.demo.{year}.{i}",
                             "citations": int(np.exp(rng.gauss(np.log(6 * age * (1.6 if usable else 1)), 0.8)))})
    pd.DataFrame(ver_rows).to_csv(root / "interim" / "verification.csv", index=False)
    pd.DataFrame(cit_rows).to_csv(root / "enrich" / "citations.csv", index=False)
    print(f"[demo] wrote synthetic workspace with {sum(N_PER_YEAR.values())} papers to {root}")

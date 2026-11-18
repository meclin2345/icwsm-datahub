#!/usr/bin/env python3
"""
Step 3 - Empirical repository audit (what is TRUE in 2026, not what the paper says).

For every extracted record:
  * checks every dataset URL and the DOI (GET, streamed; headers + first 64 KB only)
  * detects soft-404s, parked domains and login walls (a 200 is not always alive)
  * queries repository APIs for files, sizes, restrictions and usage:
      Zenodo, Dataverse (any instance), Figshare, Hugging Face, GitHub, DataCite
  * checks the Wayback Machine for dead links (is the landing page recoverable?)
  * appends to a history file so link rot can be tracked month by month

Changes vs. the first version of audit_repositories.py
  - Retry(raise_on_status=False): rate-limited (429) links are no longer
    mislabelled as "DNS / Connection Failure" after retries run out
  - per-host throttling + on-disk cache (re-runs are cheap; --refresh to bypass)
  - Zenodo "gated" no longer triggered by a missing access field
  - Dataverse detection only trusts Dataverse hosts / Harvard DOIs
  - new: soft-404, login wall, Hugging Face, Figshare, GitHub, Wayback, history

Usage
  python scripts/03_audit.py                 # all records, cached for 7 days
  python scripts/03_audit.py --refresh       # monthly re-audit (used by GitHub Actions)
Env (optional): ZENODO_TOKEN, DATAVERSE_TOKEN, GITHUB_TOKEN, HF_TOKEN, AUDIT_CONTACT_EMAIL
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, quote, urlparse

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from icwsm_datahub.curation import apply_overrides, load_overrides  # noqa: E402
from icwsm_datahub.normalize import clean_doi, normalize_extraction  # noqa: E402

CONTACT = os.getenv("AUDIT_CONTACT_EMAIL", "you@example.edu")
TOKENS = {k: os.getenv(k, "") for k in ("ZENODO_TOKEN", "DATAVERSE_TOKEN", "GITHUB_TOKEN", "HF_TOKEN")}
UA = f"ICWSM-DataHub-Audit/2.0 (+https://github.com/; mailto:{CONTACT})"
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/128.0 Safari/537.36 " + UA)

SOFT_404_RE = re.compile(
    r"<title>[^<]{0,120}(404|not\s+found|page\s+not\s+found|does\s+not\s+exist|no\s+longer\s+available|"
    r"has\s+been\s+(removed|deleted)|domain\s+(is\s+)?for\s+sale|buy\s+this\s+domain|parked)",
    re.IGNORECASE)
PARKING_HOSTS = ("sedoparking", "parkingcrew", "hugedomains", "dan.com", "afternic", "bodis", "godaddy.com/forsale")
LOGIN_HOSTS = ("accounts.google.com", "login.microsoftonline.com", "dropbox.com/login", "app.box.com/login",
               "/ServiceLogin", "signin")
ACCOUNT_REQUIRED_HOSTS = ("kaggle.com",)
MIN_INTERVAL = {"zenodo.org": 1.1, "api.github.com": 0.8, "huggingface.co": 0.5, "api.datacite.org": 0.3,
                "archive.org": 1.0, "doi.org": 0.3}


# ----------------------------------------------------------------------------- HTTP plumbing
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": BROWSER_UA, "Accept": "text/html,application/json;q=0.9,*/*;q=0.8"})
    retry = Retry(total=4, backoff_factor=2.0, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["HEAD", "GET"], raise_on_status=False, respect_retry_after_header=True)
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=16)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


SESSION = make_session()
_host_locks: Dict[str, threading.Lock] = {}
_host_last: Dict[str, float] = {}
_registry_lock = threading.Lock()


def throttle(url: str) -> None:
    host = urlparse(url).netloc.lower()
    interval = next((v for k, v in MIN_INTERVAL.items() if host.endswith(k)), 0.25)
    with _registry_lock:
        lock = _host_locks.setdefault(host, threading.Lock())
    with lock:
        wait = _host_last.get(host, 0) + interval - time.time()
        if wait > 0:
            time.sleep(wait)
        _host_last[host] = time.time()


class Cache:
    def __init__(self, folder: Path, ttl_days: float, refresh: bool):
        self.folder, self.ttl, self.refresh = folder, ttl_days * 86400, refresh
        folder.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.folder / (hashlib.sha1(key.encode()).hexdigest() + ".json")

    def get(self, key: str) -> Optional[Any]:
        p = self._path(key)
        if self.refresh or not p.exists():
            return None
        blob = json.loads(p.read_text())
        return blob["value"] if time.time() - blob["t"] < self.ttl else None

    def put(self, key: str, value: Any) -> Any:
        self._path(key).write_text(json.dumps({"t": time.time(), "key": key, "value": value}))
        return value


CACHE: Cache  # set in main()


def get_json(url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None, timeout: int = 20) -> Dict:
    key = "json|" + url + "|" + json.dumps(params or {}, sort_keys=True)
    hit = CACHE.get(key)
    if hit is not None:
        return hit
    throttle(url)
    try:
        r = SESSION.get(url, params=params, headers={"User-Agent": UA, "Accept": "application/json", **(headers or {})},
                        timeout=timeout)
        out = {"status": r.status_code, "url": r.url,
               "json": r.json() if r.headers.get("content-type", "").startswith("application/json") or
               r.text[:1] in "[{" else None}
    except Exception as exc:
        out = {"status": None, "url": url, "json": None, "error": str(exc)[:200]}
    return CACHE.put(key, out)


# ----------------------------------------------------------------------------- URL liveness
def check_url(url: str, timeout: int = 25) -> Dict[str, Any]:
    key = "url|" + url
    hit = CACHE.get(key)
    if hit is not None:
        return hit
    res = {"url": url, "resolved_url": None, "http_status": None, "category": "Unchecked", "alive": False,
           "login_wall": False, "account_required": False, "soft_404": False, "content_type": None,
           "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "error": None}
    throttle(url)
    try:
        with SESSION.get(url, allow_redirects=True, stream=True, timeout=timeout) as r:
            res.update(resolved_url=r.url, http_status=r.status_code, content_type=r.headers.get("content-type"))
            code, final = r.status_code, r.url.lower()
            head = ""
            if 200 <= code < 300 and "html" in (res["content_type"] or ""):
                head = next(r.iter_content(65536, decode_unicode=False), b"").decode("utf-8", "ignore")
            if any(h in final for h in LOGIN_HOSTS):
                res.update(category="Login wall", login_wall=True)
            elif any(h in final for h in PARKING_HOSTS):
                res.update(category="Domain parked / for sale")
            elif 200 <= code < 300 and SOFT_404_RE.search(head):
                res.update(category="Soft 404 (suspected)", soft_404=True)
            elif 200 <= code < 300:
                res.update(category="Alive", alive=True,
                           account_required=any(h in final for h in ACCOUNT_REQUIRED_HOSTS))
            elif code in (401, 403):
                res["category"] = f"Restricted / forbidden ({code})"
            elif code in (404, 410):
                res["category"] = f"Dead ({code})"
            elif code == 429:
                res["category"] = "Rate limited (429), retry later"
            elif code >= 500:
                res["category"] = f"Server error ({code})"
            else:
                res["category"] = f"Other HTTP ({code})"
    except requests.exceptions.SSLError as exc:
        res.update(category="TLS / certificate error", error=str(exc)[:160])
    except requests.exceptions.Timeout:
        res.update(category="Timeout", error="timeout")
    except requests.exceptions.ConnectionError as exc:
        res.update(category="DNS / connection failure", error=str(exc)[:160])
    except requests.exceptions.RequestException as exc:
        res.update(category="Request error", error=str(exc)[:160])
    # 429 / 5xx / timeouts are transient: do not cache them for long
    if res["category"].startswith(("Rate limited", "Server error", "Timeout")):
        return res
    return CACHE.put(key, res)


# ----------------------------------------------------------------------------- repositories
ZENODO_ID = re.compile(r"zenodo\.org/(?:records?|deposit|uploads)/(\d+)", re.I)
ZENODO_DOI = re.compile(r"10\.5281/zenodo\.(\d+)", re.I)


def zenodo(urls: List[str], doi: Optional[str]) -> Dict[str, Any]:
    rec_id = None
    for u in urls + [doi or ""]:
        m = ZENODO_ID.search(u) or ZENODO_DOI.search(u)
        if m:
            rec_id = m.group(1)
            break
    if not rec_id:
        return {"found": False}
    params = {"access_token": TOKENS["ZENODO_TOKEN"]} if TOKENS["ZENODO_TOKEN"] else None
    r = get_json(f"https://zenodo.org/api/records/{rec_id}", params)
    data = r.get("json") if r.get("status") == 200 else None
    if data is None:  # maybe a concept record id
        q = get_json("https://zenodo.org/api/records", {"q": f"conceptrecid:{rec_id}", "size": 1, **(params or {})})
        hits = ((q.get("json") or {}).get("hits") or {}).get("hits") or []
        data = hits[0] if hits else None
    if not data:
        return {"found": False, "api_status": r.get("status")}
    md, stats = data.get("metadata", {}), data.get("stats", {}) or {}
    files = data.get("files") or []
    if isinstance(files, dict):
        files = list((files.get("entries") or {}).values()) if isinstance(files.get("entries"), dict) else files.get("entries", [])
    names = [f.get("key") or f.get("filename") or "" for f in files if isinstance(f, dict)]
    total = sum((f.get("size") or 0) for f in files if isinstance(f, dict))
    access = (md.get("access_right") or (data.get("access") or {}).get("record") or "unknown").lower()
    lic = md.get("license")
    return {
        "found": True, "record_id": str(data.get("id", rec_id)), "concept_doi": data.get("conceptdoi"),
        "access_right": access, "restricted": access in ("restricted", "closed", "embargoed"),
        "license": lic.get("id") if isinstance(lic, dict) else lic,
        "file_count": len(names), "total_size_mb": round(total / 2**20, 2),
        "file_extensions": sorted({ext_of(n) for n in names if ext_of(n)}),
        # Zenodo reports both this-version and all-versions counters; keep both.
        "stats_raw": stats,
        "downloads": stats.get("downloads"), "unique_downloads": stats.get("unique_downloads"),
        "views": stats.get("views"), "unique_views": stats.get("unique_views"),
        "version_downloads": stats.get("version_downloads"), "version_unique_downloads": stats.get("version_unique_downloads"),
        "publication_date": md.get("publication_date"),
    }


def ext_of(name: str) -> str:
    name = name.lower()
    for multi in (".csv.gz", ".jsonl.gz", ".json.gz", ".tar.gz", ".tsv.gz", ".parquet.gz"):
        if name.endswith(multi):
            return multi
    return Path(name).suffix


def dataverse(urls: List[str], resolved: List[str], doi: Optional[str]) -> Dict[str, Any]:
    base, pid = None, None
    for u in resolved + urls:
        pu = urlparse(u)
        if "dataverse" in pu.netloc.lower() or "/dataset.xhtml" in pu.path:
            base = f"{pu.scheme}://{pu.netloc}"
            q = parse_qs(pu.query).get("persistentId")
            if q:
                pid = q[0]
            break
    if not pid and doi and (doi.startswith("10.7910/") or base):
        pid = f"doi:{doi}"
        base = base or "https://dataverse.harvard.edu"
    if not (base and pid):
        return {"found": False}
    hdr = {"X-Dataverse-key": TOKENS["DATAVERSE_TOKEN"]} if TOKENS["DATAVERSE_TOKEN"] else None
    r = get_json(f"{base}/api/datasets/:persistentId/", {"persistentId": pid}, hdr)
    if r.get("status") != 200 or not r.get("json"):
        return {"found": False, "api_status": r.get("status"), "instance": base}
    latest = (r["json"].get("data") or {}).get("latestVersion", {})
    files = latest.get("files", [])
    names, total, restricted = [], 0, 0
    for f in files:
        df = f.get("dataFile", {})
        total += df.get("filesize", 0) or 0
        restricted += bool(f.get("restricted") or df.get("restricted"))
        names.append(df.get("originalFileName") or df.get("filename") or "")
    out = {
        "found": True, "instance": base, "pid": pid, "version_state": latest.get("versionState"),
        "license": (latest.get("license") or {}).get("name") if isinstance(latest.get("license"), dict) else latest.get("license"),
        "file_count": len(files), "restricted_file_count": restricted,
        "restricted": len(files) > 0 and restricted == len(files),
        "partially_restricted": 0 < restricted < len(files),
        "total_size_mb": round(total / 2**20, 2), "file_extensions": sorted({ext_of(n) for n in names if ext_of(n)}),
        "terms_of_access": bool(latest.get("termsOfAccess")),
    }
    for metric in ("downloadsTotal", "viewsTotal", "downloadsUnique"):
        m = get_json(f"{base}/api/datasets/:persistentId/makeDataCount/{metric}", {"persistentId": pid}, hdr)
        out[metric] = ((m.get("json") or {}).get("data") or {}).get(metric) if m.get("status") == 200 else None
    return out


FIGSHARE_ID = re.compile(r"(?:figshare\.\d+|figshare\.com/articles/(?:[^/]+/)*)(\d{6,})", re.I)


def figshare(urls: List[str], doi: Optional[str]) -> Dict[str, Any]:
    art = None
    for u in urls + [doi or ""]:
        m = FIGSHARE_ID.search(u)
        if m:
            art = m.group(1)
            break
    if not art:
        return {"found": False}
    r = get_json(f"https://api.figshare.com/v2/articles/{art}")
    j = r.get("json") if r.get("status") == 200 else None
    if not j:
        return {"found": False, "api_status": r.get("status")}
    files = j.get("files") or []
    out = {"found": True, "article_id": art, "file_count": len(files),
           "total_size_mb": round(sum(f.get("size", 0) for f in files) / 2**20, 2),
           "file_extensions": sorted({ext_of(f.get("name", "")) for f in files if f.get("name")}),
           "restricted": bool(j.get("is_embargoed") or j.get("is_confidential")),
           "license": (j.get("license") or {}).get("name")}
    for kind in ("downloads", "views"):
        s = get_json(f"https://stats.figshare.com/total/{kind}/article/{art}")
        out[kind] = (s.get("json") or {}).get("totals") if s.get("status") == 200 else None
    return out


HF_ID = re.compile(r"huggingface\.co/datasets/([\w.-]+/[\w.-]+)", re.I)


def huggingface(urls: List[str]) -> Dict[str, Any]:
    repo = next((m.group(1) for u in urls if (m := HF_ID.search(u))), None)
    if not repo:
        return {"found": False}
    hdr = {"Authorization": f"Bearer {TOKENS['HF_TOKEN']}"} if TOKENS["HF_TOKEN"] else None
    r = get_json(f"https://huggingface.co/api/datasets/{repo}", headers=hdr)
    j = r.get("json") if r.get("status") == 200 else None
    if not j:
        return {"found": False, "api_status": r.get("status"), "repo": repo}
    all_time = get_json(f"https://huggingface.co/api/datasets/{repo}", {"expand[]": "downloadsAllTime"}, hdr)
    gated = j.get("gated")
    sib = [s.get("rfilename", "") for s in j.get("siblings", []) or []]
    return {"found": True, "repo": repo, "gated": gated, "restricted": gated == "manual",
            "account_required": gated in ("auto", True), "private": j.get("private"), "disabled": j.get("disabled"),
            "downloads_30d": j.get("downloads"), "downloads_all_time": (all_time.get("json") or {}).get("downloadsAllTime"),
            "likes": j.get("likes"), "file_count": len(sib),
            "file_extensions": sorted({ext_of(n) for n in sib if ext_of(n) and not n.startswith(".")}),
            "license": (j.get("cardData") or {}).get("license"), "last_modified": j.get("lastModified")}


GH_ID = re.compile(r"github\.com/([\w.-]+)/([\w.-]+)", re.I)


def github(url: Optional[str]) -> Dict[str, Any]:
    m = GH_ID.search(url or "")
    if not m:
        return {"found": False}
    owner, repo = m.group(1), re.sub(r"\.git$", "", m.group(2))
    hdr = {"Authorization": f"Bearer {TOKENS['GITHUB_TOKEN']}"} if TOKENS["GITHUB_TOKEN"] else None
    r = get_json(f"https://api.github.com/repos/{owner}/{repo}", headers=hdr)
    j = r.get("json") if r.get("status") == 200 else None
    if not j:
        return {"found": False, "api_status": r.get("status"), "repo": f"{owner}/{repo}"}
    return {"found": True, "repo": j.get("full_name"), "stars": j.get("stargazers_count"), "forks": j.get("forks_count"),
            "archived": j.get("archived"), "disabled": j.get("disabled"), "pushed_at": j.get("pushed_at"),
            "created_at": j.get("created_at"), "size_mb": round((j.get("size") or 0) / 1024, 2),
            "license": (j.get("license") or {}).get("spdx_id")}


def datacite(doi: Optional[str]) -> Dict[str, Any]:
    if not doi:
        return {"registered": False}
    r = get_json(f"https://api.datacite.org/dois/{quote(doi, safe='/')}")
    attrs = ((r.get("json") or {}).get("data") or {}).get("attributes") if r.get("status") == 200 else None
    if not attrs:
        return {"registered": False, "api_status": r.get("status")}
    return {"registered": True, "state": attrs.get("state"), "publisher": attrs.get("publisher") if isinstance(
        attrs.get("publisher"), str) else (attrs.get("publisher") or {}).get("name"),
            "landing_url": attrs.get("url"), "views": attrs.get("viewCount"), "downloads": attrs.get("downloadCount"),
            "citations": attrs.get("citationCount"), "registered_year": (attrs.get("registered") or "")[:4]}


def wayback(url: str, year: int) -> Dict[str, Any]:
    r = get_json("https://archive.org/wayback/available", {"url": url, "timestamp": f"{year}0601"})
    snap = (((r.get("json") or {}).get("archived_snapshots") or {}).get("closest") or {})
    return {"archived": bool(snap.get("available")), "timestamp": snap.get("timestamp"), "snapshot_url": snap.get("url")}


# ----------------------------------------------------------------------------- per-record audit
OVERRIDES: Dict[str, List[Dict[str, Any]]] = {}


def audit_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    ex = normalize_extraction(rec["extraction"])
    apply_overrides(ex, OVERRIDES.get(rec["paper_id"], []))   # audit what curators corrected, not the raw guess
    d = ex["distribution"]
    urls: List[str] = list(dict.fromkeys(d["dataset_urls"]))
    doi = d["persistent_doi"] or next((clean_doi(u) for u in urls if "doi.org" in u and clean_doi(u)), None)
    to_check = list(urls)
    if doi and not any(doi.lower() in u.lower() for u in to_check):
        to_check.insert(0, f"https://doi.org/{doi}")

    checks = [check_url(u) for u in to_check]
    resolved = [c["resolved_url"] for c in checks if c.get("resolved_url")]
    doi_check = next((c for c in checks if doi and doi.lower() in c["url"].lower()), None)

    zen = zenodo(to_check + resolved, doi)
    dv = dataverse(to_check, resolved, doi) if not zen.get("found") else {"found": False}
    fig = figshare(to_check + resolved, doi) if not (zen.get("found") or dv.get("found")) else {"found": False}
    hf = huggingface(to_check + resolved)
    gh_data = github(next((u for u in to_check if "github.com" in u), None))
    dc = datacite(doi)
    code_url = d.get("code_repository_url")
    code_check = check_url(code_url) if code_url else None
    gh_code = github(code_url) if code_url else {"found": False}

    any_alive = any(c["alive"] for c in checks)
    wb = None
    if to_check and not any_alive:
        wb = wayback(urls[0] if urls else to_check[0], int(rec["year"]))

    # ---- harmonize: prefer the most authoritative repository API
    s: Dict[str, Any] = {
        "verified_repo": None, "file_count": None, "total_size_mb": None, "file_extensions": "",
        "is_restricted": False, "files_empty": False, "downloads_total": None, "downloads_unique": None,
        "views_total": None, "downloads_metric_source": None,
    }
    if zen.get("found"):
        s.update(verified_repo="Zenodo", file_count=zen["file_count"], total_size_mb=zen["total_size_mb"],
                 file_extensions=", ".join(zen["file_extensions"]), is_restricted=zen["restricted"],
                 downloads_total=zen.get("downloads"), downloads_unique=zen.get("unique_downloads"),
                 views_total=zen.get("views"), downloads_metric_source="Zenodo stats")
    elif dv.get("found"):
        s.update(verified_repo="Dataverse", file_count=dv["file_count"], total_size_mb=dv["total_size_mb"],
                 file_extensions=", ".join(dv["file_extensions"]), is_restricted=dv["restricted"],
                 downloads_total=dv.get("downloadsTotal") or dc.get("downloads"),
                 downloads_unique=dv.get("downloadsUnique"), views_total=dv.get("viewsTotal") or dc.get("views"),
                 downloads_metric_source="Dataverse Make Data Count")
    elif fig.get("found"):
        s.update(verified_repo="Figshare", file_count=fig["file_count"], total_size_mb=fig["total_size_mb"],
                 file_extensions=", ".join(fig["file_extensions"]), is_restricted=fig["restricted"],
                 downloads_total=fig.get("downloads"), views_total=fig.get("views"),
                 downloads_metric_source="Figshare stats")
    elif hf.get("found"):
        s.update(verified_repo="Hugging Face", file_count=hf["file_count"],
                 file_extensions=", ".join(hf["file_extensions"]), is_restricted=hf["restricted"],
                 downloads_total=hf.get("downloads_all_time"), downloads_metric_source="Hugging Face (all time)")
    elif gh_data.get("found"):
        s.update(verified_repo="GitHub", total_size_mb=gh_data["size_mb"])
    elif dc.get("registered"):
        s.update(verified_repo=f"DataCite ({dc.get('publisher') or 'other'})", downloads_total=dc.get("downloads"),
                 views_total=dc.get("views"), downloads_metric_source="DataCite usage")
    if s["file_count"] == 0 and not s["is_restricted"]:
        s["files_empty"] = True

    summary = {
        "paper_id": rec["paper_id"], "year": rec["year"],
        "audited_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "normalized_doi": doi, "doi_resolves": bool(doi_check and doi_check["alive"]),
        "n_urls_checked": len(checks), "any_url_alive": any_alive,
        "primary_url": to_check[0] if to_check else None,
        "primary_url_status": checks[0]["category"] if checks else "No dataset link",
        "primary_resolved_url": checks[0]["resolved_url"] if checks else None,
        "login_wall": any(c["login_wall"] for c in checks) and not any_alive,
        "account_required": any(c["account_required"] for c in checks) or bool(hf.get("account_required")),
        "soft_404": any(c["soft_404"] for c in checks),
        **s,
        "repo_license": (zen.get("license") if zen.get("found") else None) or (dv.get("license") if dv.get("found") else None)
        or (fig.get("license") if fig.get("found") else None) or (hf.get("license") if hf.get("found") else None)
        or (gh_data.get("license") if gh_data.get("found") and gh_data.get("license") not in (None, "NOASSERTION") else None),
        "datacite_registered": dc.get("registered", False), "datacite_publisher": dc.get("publisher"),
        "datacite_citations": dc.get("citations"),
        "hf_likes": hf.get("likes"), "hf_downloads_30d": hf.get("downloads_30d"),
        "code_url": code_url, "code_url_status": code_check["category"] if code_check else None,
        "code_url_alive": code_check["alive"] if code_check else None,
        "code_stars": gh_code.get("stars"), "code_forks": gh_code.get("forks"),
        "code_archived": gh_code.get("archived"), "code_last_push": gh_code.get("pushed_at"),
        "dead_link_in_wayback": wb["archived"] if wb else None,
        "wayback_snapshot": wb["snapshot_url"] if wb else None,
    }
    return {"summary": summary, "url_checks": checks, "zenodo": zen, "dataverse": dv, "figshare": fig,
            "huggingface": hf, "github_data": gh_data, "github_code": gh_code, "datacite": dc, "wayback": wb}


def main() -> None:
    global CACHE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ttl-days", type=float, default=7)
    ap.add_argument("--refresh", action="store_true", help="Ignore cache (monthly re-audit)")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    audit_dir = data_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    CACHE = Cache(audit_dir / "_cache", args.ttl_days, args.refresh)
    OVERRIDES.update(load_overrides(data_dir / "curation" / "overrides.csv"))

    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((data_dir / "extracted").glob("*.json"))]
    if args.only:
        recs = [r for r in recs if r["paper_id"] in args.only]
    if not recs:
        sys.exit(f"No records in {data_dir / 'extracted'}. Run 01_extract.py first.")

    summaries = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(audit_record, r): r["paper_id"] for r in recs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Auditing"):
            pid = futs[fut]
            try:
                full = fut.result()
            except Exception as exc:
                print(f"[error] {pid}: {exc}")
                continue
            (audit_dir / f"{pid}.json").write_text(json.dumps(full, indent=2, default=str))
            summaries.append(full["summary"])

    df = pd.DataFrame(summaries).sort_values(["year", "paper_id"])
    df.to_csv(audit_dir / "audit_summary.csv", index=False)
    hist_cols = ["audited_at", "paper_id", "year", "any_url_alive", "primary_url_status", "downloads_total"]
    hist = audit_dir / "history.csv"
    df[hist_cols].to_csv(hist, mode="a", header=not hist.exists(), index=False)

    print(f"\nAudit summary: {audit_dir / 'audit_summary.csv'}")
    print(f"Any dataset link alive: {100 * df['any_url_alive'].mean():.1f}% of {len(df)} papers")
    print(df.groupby("year")["any_url_alive"].agg(["count", "mean"]).round(3).to_string())
    print("\nPrimary link status:")
    print(df["primary_url_status"].value_counts().to_string())


if __name__ == "__main__":
    main()

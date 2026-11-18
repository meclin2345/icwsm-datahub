"""Release details (author, GitHub, Zenodo DOI) kept in one file, release.json, and copied
into the website, CITATION.cff and .zenodo.json on every catalog build.

Missing values are skipped, so the build works before you have a GitHub account or a DOI."""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
TITLE = "ICWSM-DataHub: a catalog and accessibility audit of ICWSM Dataset Track datasets (2020-2026)"
DESCRIPTION = ("A catalog of every ICWSM Dataset Track paper published 2020-2026, with attributes extracted "
               "from each paper by an LLM, operational FAIR indicators, and a live check of whether each "
               "dataset can still be downloaded and used. Includes CSV, Parquet and JSON files, a data "
               "dictionary and Croissant metadata. Code: MIT. Data: CC BY 4.0.")


def load(path: Path = ROOT / "release.json") -> Dict[str, Any]:
    if not path.exists():
        return {}
    cfg = {k: (v.strip() if isinstance(v, str) else v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()
           if not k.startswith("_")}
    user, repo = cfg.get("github_user") or "", cfg.get("repo_name") or "icwsm-datahub"
    cfg["repo_url"] = f"https://github.com/{user}/{repo}" if user else None
    cfg["site_url"] = f"https://{user.lower()}.github.io/{repo}/" if user else None
    doi = cfg.get("zenodo_doi") or ""
    cfg["zenodo_doi"] = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi) or None
    name = " ".join(x for x in [cfg.get("author_given_names"), cfg.get("author_family_names")] if x)
    cfg["author_name"] = name or None
    return cfg


def _yaml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_citation(cfg: Dict[str, Any], version: str, path: Path = ROOT / "CITATION.cff") -> None:
    author = ["  - family-names: " + _yaml_str(cfg.get("author_family_names") or "Lastname"),
              "    given-names: " + _yaml_str(cfg.get("author_given_names") or "Firstname")]
    if cfg.get("affiliation"):
        author.append("    affiliation: " + _yaml_str(cfg["affiliation"]))
    if cfg.get("orcid"):
        orcid = cfg["orcid"] if cfg["orcid"].startswith("http") else f"https://orcid.org/{cfg['orcid']}"
        author.append("    orcid: " + _yaml_str(orcid))
    lines = ["cff-version: 1.2.0",
             'message: "If you use ICWSM-DataHub, please cite it as below."',
             "type: dataset",
             "title: " + _yaml_str(TITLE),
             f"version: {version}",
             f"date-released: {date.today().isoformat()}",
             "license: CC-BY-4.0",
             "authors:", *author]
    if cfg.get("zenodo_doi"):
        lines.append(f"doi: {cfg['zenodo_doi']}")
    if cfg.get("repo_url"):
        lines.append("repository-code: " + _yaml_str(cfg["repo_url"]))
    if cfg.get("site_url"):
        lines.append("url: " + _yaml_str(cfg["site_url"]))
    lines += ["keywords:", "  - computational social science", "  - ICWSM", "  - datasets", "  - FAIR data",
              "  - data availability", "  - meta-research"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_zenodo(cfg: Dict[str, Any], version: str, path: Path = ROOT / ".zenodo.json") -> None:
    creator = {"name": ", ".join(x for x in [cfg.get("author_family_names"), cfg.get("author_given_names")] if x)
               or "Lastname, Firstname"}
    if cfg.get("affiliation"):
        creator["affiliation"] = cfg["affiliation"]
    if cfg.get("orcid"):
        creator["orcid"] = cfg["orcid"].rsplit("/", 1)[-1]
    doc = {"title": TITLE, "upload_type": "dataset", "version": version, "description": DESCRIPTION,
           "creators": [creator], "license": "cc-by-4.0", "access_right": "open",
           "keywords": ["computational social science", "ICWSM", "datasets", "FAIR", "data availability",
                        "meta-research"]}
    if cfg.get("site_url"):
        doc["related_identifiers"] = [{"identifier": cfg["site_url"], "relation": "isDocumentedBy",
                                       "resource_type": "other"}]
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def set_site_url(cfg: Dict[str, Any], page: Path = ROOT / "docs" / "index.html") -> None:
    """Social previews (X, Bluesky, Slack) need absolute URLs for the page and its image."""
    site: Optional[str] = cfg.get("site_url")
    if not site or not page.exists():
        return
    html = page.read_text(encoding="utf-8")
    html = re.sub(r'(<meta property="og:url" content=")[^"]*(")', rf"\g<1>{site}\g<2>", html)
    html = re.sub(r'(<meta (?:property="og:image"|name="twitter:image") content=")[^"]*(")',
                  rf"\g<1>{site}social-card.png\g<2>", html)
    html = re.sub(r'(<link rel="canonical" href=")[^"]*(")', rf"\g<1>{site}\g<2>", html)
    page.write_text(html, encoding="utf-8")
    readme = ROOT / "README.md"
    if readme.exists():
        user, repo = cfg["github_user"], cfg.get("repo_name") or "icwsm-datahub"
        txt = readme.read_text(encoding="utf-8")
        txt = re.sub(r"https://[A-Za-z0-9-]+\.github\.io/icwsm-datahub/", site, txt)
        txt = re.sub(r"https://github\.com/[A-Za-z0-9-]+/icwsm-datahub", f"https://github.com/{user}/{repo}", txt)
        readme.write_text(txt, encoding="utf-8")

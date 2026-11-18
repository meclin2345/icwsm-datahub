#!/usr/bin/env python3
"""
Manual browser check of links the automatic audit could not confirm.

Automated requests are often blocked (HTTP 403) by sites that open normally in a browser,
and "dead" links deserve a human look before a paper calls them dead.

STEP 1  python scripts/audit_checklist.py
        Writes data/curation/audit_checks.csv with every paper whose links were not all
        confirmed alive. Rows you already filled in are kept.
STEP 2  Open each URL in urls_to_open in your browser and fill:
          any_url_alive   true / false   (does a dataset page actually load?)
          is_restricted   true / false   (can you download the files right now, without a form,
                                          approval or agreement? If yes -> false)
          files_empty     true / false   (page loads but there are no data files?)
          checked_on      date, e.g. 2026-10-30
          note            what you saw, e.g. "Cloudflare blocks scripts; loads in Safari"
        Leave a cell blank to keep the automatic result. Save as CSV (same name).
STEP 3  python scripts/05_build_catalog.py   (applies the checks; nothing else to run)

EASIER  python scripts/audit_checklist.py --interactive
        Opens each paper's links in your browser and asks three yes/no questions in the terminal.
        Saves after every paper; quit any time and continue later.

Report in the paper: "Links returning errors to automated requests were checked manually in a
browser; N of M were reachable."
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from icwsm_datahub.curation import apply_overrides, load_overrides  # noqa: E402
from icwsm_datahub.normalize import normalize_extraction  # noqa: E402

GATED_CLAIMS = ("Credentialed access, form or data use agreement", "Email the authors for approval")

COLS = ["paper_id", "year", "title", "why_listed", "urls_to_open", "auto_status", "any_url_alive", "is_restricted",
        "files_empty", "checked_on", "note"]


def _ask(prompt: str) -> str:
    while True:
        a = input(prompt).strip().lower()
        if a in ("y", "n", "?", "q"):
            return a
        print("   Please answer y (yes), n (no), ? (not sure) or q (quit).")


def interactive(out: Path, open_links: bool) -> None:
    import datetime
    import subprocess
    q = pd.read_csv(out, dtype=str).fillna("")
    todo = [i for i, r in q.iterrows() if not r["any_url_alive"]]
    print(f"{len(todo)} papers to check. Answer y / n / ? (not sure) / q (quit; progress is saved).\n")
    for n, i in enumerate(todo, 1):
        r = q.loc[i]
        print("=" * 100)
        print(f"[{n}/{len(todo)}] {r['title']} ({r['year']})")
        print(f"Why listed : {r['why_listed']}")
        print(f"Automatic  : {r['auto_status']}")
        urls = [u for u in r["urls_to_open"].split() if u.startswith("http")]
        print("Links      : " + ("  ".join(urls) or "(none)"))
        if open_links:
            for u in urls:
                subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", u], check=False)
        alive = _ask("1. Does a dataset page load in your browser? [y/n/?/q] ")
        if alive == "q":
            break
        restricted = empty = "?"
        if alive == "y":
            restricted = _ask("2. Can you download the data files RIGHT NOW, with no form, request or agreement? [y/n/?/q] ")
            if restricted == "q":
                break
            if restricted == "y":   # only meaningful when you can actually reach the files
                empty = _ask("3. Are there actual data files on the page (not just a README or code)? [y/n/?/q] ")
                if empty == "q":
                    break
        conv = {"y": "true", "n": "false", "?": ""}
        q.loc[i, "any_url_alive"] = conv[alive] if alive != "?" else ""
        q.loc[i, "is_restricted"] = {"y": "false", "n": "true", "?": ""}[restricted]   # can download = not restricted
        q.loc[i, "files_empty"] = {"y": "false", "n": "true", "?": ""}[empty]          # files present = not empty
        q.loc[i, "checked_on"] = datetime.date.today().isoformat()
        q.loc[i, "note"] = input("Note (what you saw, Enter to skip): ").strip() or "checked in browser"
        q.to_csv(out, index=False)
    print(f"\nSaved {out}. Now run: python scripts/05_build_catalog.py")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--interactive", action="store_true", help="Check links one by one in the terminal")
    ap.add_argument("--no-open", action="store_true", help="Do not open links automatically")
    args = ap.parse_args()
    data = Path(args.data_dir)
    summ = data / "audit" / "audit_summary.csv"
    if not summ.exists():
        sys.exit("Run python scripts/03_audit.py first.")
    df = pd.read_csv(summ)
    out = data / "curation" / "audit_checks.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(out, dtype=str).fillna("") if out.exists() else pd.DataFrame(columns=COLS)
    filled = {r["paper_id"]: r for _, r in old.iterrows() if any(r.get(c, "") for c in ("any_url_alive", "is_restricted", "files_empty", "note"))}

    overrides = load_overrides(data / "curation" / "overrides.csv")
    rows = []
    for _, r in df.iterrows():
        pid = r["paper_id"]
        access = ""
        ext = data / "extracted" / f"{pid}.json"
        if ext.exists():
            ex = normalize_extraction(json.loads(ext.read_text())["extraction"])
            apply_overrides(ex, overrides.get(pid, []))
            access = ex["distribution"].get("access_mechanism") or ""
        why = []
        if (not bool(r["any_url_alive"])) or str(r["primary_url_status"]) != "Alive":
            why.append("link not confirmed")
        if str(r.get("is_restricted")) == "True":
            why.append("repository reports restricted files")
        if access in GATED_CLAIMS:
            why.append("paper says access needs a form or approval")
        if str(r.get("files_empty")) == "True":
            why.append("repository record has no files")
        if not why:
            continue
        if pid in filled:
            rows.append(filled[pid].to_dict())
            continue
        detail = json.loads((data / "audit" / f"{pid}.json").read_text())
        urls = [c["url"] for c in detail.get("url_checks", [])]
        statuses = "; ".join(f"{c['category']}" for c in detail.get("url_checks", [])) or str(r["primary_url_status"])
        title = ""
        ext = data / "extracted" / f"{pid}.json"
        if ext.exists():
            title = json.loads(ext.read_text())["extraction"]["paper"]["title"]
        rows.append({"paper_id": pid, "year": r["year"], "title": title, "why_listed": "; ".join(why), "urls_to_open": "  ".join(urls) or "(no link in paper)",
                     "auto_status": statuses, "any_url_alive": "", "is_restricted": "", "files_empty": "",
                     "checked_on": "", "note": ""})
    pd.DataFrame(rows, columns=COLS).to_csv(out, index=False)
    todo = sum(1 for r in rows if not r.get("any_url_alive"))
    print(f"Wrote {out}: {len(rows)} papers to check in a browser ({todo} not yet checked).")
    if args.interactive:
        interactive(out, not args.no_open)


if __name__ == "__main__":
    main()

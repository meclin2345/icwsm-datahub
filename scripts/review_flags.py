#!/usr/bin/env python3
"""
Human review of flagged records, in two steps.

STEP 1  python scripts/review_flags.py
        Writes data/curation/review_queue.csv: one row per issue, sorted by priority, each
        with the field to check, its current value and (where possible) a suggested fix.
        Titles that OpenAlex matches exactly (score >= 95) are confirmed automatically, so
        run 04_enrich_citations.py first.

        Open the CSV in Excel or Numbers. For each row, read the paper (pdf_path) and fill:
          decision         accept  = use suggested_value
                           change  = use your corrected_value
                           keep    = the current value is right
                           (blank) = not reviewed yet
          corrected_value  only for "change". Lists/booleans as JSON: ["Zenodo"]  false
        Save as CSV (same name).

EASIER  python scripts/review_flags.py --interactive --curator YOURINITIALS
        Shows one item at a time in the terminal, opens the PDF, and you press a key.
        Saves after every answer; quit any time and continue later. Applies at the end.

STEP 2  python scripts/review_flags.py --apply --curator YOURINITIALS
        Writes accepted/changed values to data/curation/overrides.csv and logs every decision
        in data/curation/reviewed.csv, so reviewed rows never come back. Re-run STEP 1 any time.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from icwsm_datahub.curation import apply_overrides, clean_cell, get_path, load_overrides  # noqa: E402
from icwsm_datahub.normalize import normalize_extraction  # noqa: E402

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
# flag -> (priority, field, suggestion function or None, why)
RULES = {
    "url_not_found_in_text": ("high", "distribution.dataset_urls", None,
                              "A dataset URL does not appear in the paper text: possible hallucination or a broken PDF line. "
                              "Check the URL in the paper and write the correct list."),
    "doi_not_found_in_text": ("high", "distribution.persistent_doi", None,
                              "The dataset DOI does not appear in the paper text. Check it."),
    "mixed_form_but_text_says_ids_only": ("high", "distribution.distribution_form", lambda ex: "Dehydrated IDs only",
                                          "The paper says only IDs are shared. 'Mixed' needs named features/labels released with the IDs."),
    "anonymized_form_but_no_pii_step": ("high", "distribution.distribution_form", lambda ex: "Full raw content",
                                        "Coded 'Anonymized' but no scrubbing step recorded. If the authors did NOT remove/hash identifiers: a. "
                                        "If they DID: c, field ethics.pii_anonymization_applied, value true."),
    "pii_true_for_ids_only_release": ("high", "ethics.pii_anonymization_applied", lambda ex: False,
                                      "Sharing only IDs because of platform rules is not anonymization."),
    "code_host_listed_as_dataset_host": ("high", "distribution.hosting_repositories",
                                         lambda ex: [h for h in ex["distribution"]["hosting_repositories"] if h != "GitHub or GitLab"] or ["Not stated"],
                                         "GitHub appears to hold only code, not the dataset files."),
    "llm_paradigm_but_no_llm_named": ("high", "annotation.annotation_paradigms",
                                      lambda ex: [x for x in ex["annotation"]["annotation_paradigms"] if not x.startswith(("LLM", "Hybrid"))]
                                      or ["Automated classifier or lexicon (BERT, Perspective API, LIWC)"],
                                      "Coded as LLM labels, but the tools named (see llms_used in the record) are not LLMs, e.g. "
                                      "Perspective API, spaCy, BERT. This changes the LLM trend."),
    "title_taken_from_pdf_check_it": ("medium", "paper.title", None,
                                      "The model left the title out; it was read from the PDF. Check it against page 1."),
    "notes_mention_inference": ("medium", "", None,
                                "The model says it inferred something. Read ambiguity_notes (shown in current_value), "
                                "put the affected field in 'field' and fix it, or 'keep'."),
    "annotated_count_equals_total_records": ("low", "annotation.num_annotated_records", None,
                                             "Every record counted as annotated. Fine if all items were labeled; use -1 if no labels were produced."),
    "availability_quote_names_no_location": ("low", "evidence.availability_quote", None,
                                             "The evidence quote does not say where the data is. Only the quote is affected."),
    "json_was_repaired_check_quotes": ("low", "evidence.scale_quote", None,
                                       "The model's JSON had a syntax slip that was repaired. Spot-check the quotes."),
}


def to_cell(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def build_queue(data_dir: Path) -> None:
    ver_path = data_dir / "interim" / "verification.csv"
    if not ver_path.exists():
        sys.exit("Run python scripts/02_verify.py first.")
    ver = pd.read_csv(ver_path).set_index("paper_id")
    cur = data_dir / "curation"
    cur.mkdir(parents=True, exist_ok=True)
    overrides = load_overrides(cur / "overrides.csv")
    reviewed = set()
    if (cur / "reviewed.csv").exists():
        r = pd.read_csv(cur / "reviewed.csv", dtype=str).fillna("")
        reviewed = set(zip(r["paper_id"], r["flag"]))
    manifest = {}
    if (data_dir / "interim" / "manifest.csv").exists():
        m = pd.read_csv(data_dir / "interim" / "manifest.csv")
        manifest = dict(zip(m["paper_id"], m["pdf_path"]))
    cites = {}
    if (data_dir / "enrich" / "citations.csv").exists():
        c = pd.read_csv(data_dir / "enrich" / "citations.csv")
        for _, row in c.iterrows():
            cites[row["paper_id"]] = row

    versions, rows, auto_ok = Counter(), [], 0
    for path in sorted((data_dir / "extracted").glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        pid = rec["paper_id"]
        versions[rec.get("provenance", {}).get("prompt_version")] += 1
        ex = normalize_extraction(rec["extraction"])
        apply_overrides(ex, overrides.get(pid, []))
        flags = []
        if pid in ver.index:
            v = ver.loc[pid]
            flags = [f for f in str(v.get("flags") or "").split("; ") if f and f != "nan"]
            if isinstance(v.get("ungrounded_urls"), str) and v["ungrounded_urls"].strip():
                flags.append("url_not_found_in_text")
            if str(v.get("doi_grounded")) in ("False", "0", "0.0"):
                flags.append("doi_not_found_in_text")
        for flag in flags:
            if (pid, flag) in reviewed:
                continue
            prio, field, suggest, why = RULES.get(flag, ("medium", "", None, "Automatic consistency check; inspect the record."))
            if flag == "title_taken_from_pdf_check_it" and pid in cites:
                cr = cites[pid]
                score = float(cr.get("openalex_match_score") or 0)
                if score >= 95:
                    auto_ok += 1
                    continue
                suggestion = cr.get("openalex_title") if score >= 85 and isinstance(cr.get("openalex_title"), str) else ""
                why += f" OpenAlex match score {score:.0f}."
            else:
                suggestion = to_cell(suggest(ex)) if suggest else ""
            if field and any(o["field"] == field for o in overrides.get(pid, [])):
                continue  # already corrected
            current = get_path(ex, field) if field else ex["evidence"].get("ambiguity_notes")
            if flag == "url_not_found_in_text":
                current = ex["distribution"]["dataset_urls"]
                why += f" Not found: {ver.loc[pid]['ungrounded_urls']}"
            rows.append({"priority": prio, "paper_id": pid, "year": rec["year"], "title": ex["paper"]["title"],
                         "pdf_path": manifest.get(pid, ""), "flag": flag, "field": field,
                         "current_value": to_cell(current), "suggested_value": suggestion, "why": why,
                         "decision": "", "corrected_value": "", "notes": ""})

    print("Prompt versions in data/extracted:", dict(versions))
    if len(versions) > 1:
        print("  [warn] records come from more than one prompt version; re-run extraction with --force for consistency.")
    out = cur / "review_queue.csv"
    if not rows:
        print("Nothing to review.")
        out.unlink(missing_ok=True)
        return
    q = pd.DataFrame(rows).sort_values(["priority", "flag", "paper_id"], key=lambda col: col.map(PRIORITY_ORDER) if col.name == "priority" else col)
    q.to_csv(out, index=False)
    print(f"Titles confirmed automatically by OpenAlex: {auto_ok}")
    print(f"Wrote {out}: {len(q)} rows")
    print(q.groupby(["priority", "flag"]).size().to_string())
    print("\nStart with priority 'high': those change the paper's findings.")


def apply(data_dir: Path, curator: str) -> None:
    cur = data_dir / "curation"
    qpath = cur / "review_queue.csv"
    if not qpath.exists():
        sys.exit("No review_queue.csv. Run without --apply first.")
    q = pd.read_csv(qpath, dtype=str).fillna("")
    q["decision"] = q["decision"].str.strip().str.lower()
    bad = q[~q["decision"].isin(["", "accept", "change", "keep"])]
    if len(bad):
        sys.exit(f"Unknown decision values: {sorted(set(bad['decision']))}. Use accept, change or keep.")
    todo = q[q["decision"].isin(["accept", "change"])]
    problems, new = [], []
    for _, r in todo.iterrows():
        value = clean_cell(r["suggested_value"] if r["decision"] == "accept" else r["corrected_value"])
        if not r["field"]:
            problems.append(f"{r['paper_id']} {r['flag']}: no field given")
            continue
        if r["decision"] == "change" and value == "":
            problems.append(f"{r['paper_id']} {r['flag']}: 'change' but corrected_value is empty")
            continue
        if r["decision"] == "accept" and value == "":
            problems.append(f"{r['paper_id']} {r['flag']}: 'accept' but there is no suggested value; use 'change'")
            continue
        new.append({"paper_id": r["paper_id"], "field": r["field"], "value": value,
                    "reason": f"review: {r['flag']}" + (f"; {r['notes']}" if r["notes"] else ""), "curator": curator})
    if problems:
        print("Fix these rows, then run --apply again (nothing was written):")
        print("\n".join("  " + p for p in problems))
        sys.exit(1)
    opath = cur / "overrides.csv"
    old = pd.read_csv(opath, dtype=str).fillna("") if opath.exists() else pd.DataFrame(
        columns=["paper_id", "field", "value", "reason", "curator"])
    if new:
        nd = pd.DataFrame(new)
        keys = set(zip(nd["paper_id"], nd["field"]))
        old = old[[k not in keys for k in zip(old["paper_id"], old["field"])]]
        old = pd.concat([old, nd], ignore_index=True)
    old.to_csv(opath, index=False)
    done = q[q["decision"] != ""][["paper_id", "flag", "decision"]].assign(curator=curator)
    rpath = cur / "reviewed.csv"
    done.to_csv(rpath, mode="a", header=not rpath.exists(), index=False)
    print(f"{len(new)} corrections written to {opath}; {len(done)} decisions logged in {rpath}.")
    print(f"{int((q['decision'] == '').sum())} rows still open. Run without --apply to refresh the queue.")


def _open(target: str, enabled: bool) -> None:
    if not enabled or not target:
        return
    import subprocess
    cmd = "open" if sys.platform == "darwin" else "xdg-open"
    try:
        subprocess.run([cmd, target], check=False)
    except FileNotFoundError:
        print(f"   (could not open automatically; open it yourself: {target})")


def interactive(data_dir: Path, curator: str, priorities: list, open_files: bool) -> None:
    qpath = data_dir / "curation" / "review_queue.csv"
    if not qpath.exists():
        build_queue(data_dir)
    if not qpath.exists():
        return
    q = pd.read_csv(qpath, dtype=str).fillna("")
    todo = [i for i, r in q.iterrows() if r["decision"] == "" and r["priority"] in priorities]
    if not todo:
        print(f"No open rows with priority {priorities}.")
    print(f"{len(todo)} items to review. Keys: a=accept suggestion  k=keep current  c=change  "
          f"o=open PDF again  u=open dataset link  s=skip  q=quit (progress is saved)\n")
    for n, i in enumerate(todo, 1):
        r = q.loc[i]
        print("=" * 100)
        print(f"[{n}/{len(todo)}]  {r['priority'].upper()}  {r['flag']}")
        print(f"Paper : {r['title']}  ({r['year']})")
        print(f"PDF   : {r['pdf_path'] or '(path unknown)'}")
        print(f"Field : {r['field'] or '(you choose)'}")
        print(f"Now   : {r['current_value']}")
        print(f"Suggest: {r['suggested_value'] or '(no suggestion; use c to type a value, or k)'}")
        print(f"Why   : {r['why']}")
        _open(r["pdf_path"], open_files)
        while True:
            key = input("Decision [a/k/c/o/u/s/q]: ").strip().lower()
            if key == "o":
                _open(r["pdf_path"], True)
                continue
            if key == "u":
                try:
                    urls = json.loads(r["current_value"]) if r["current_value"].startswith("[") else [r["current_value"]]
                except json.JSONDecodeError:
                    urls = [r["current_value"]]
                for u in urls:
                    if str(u).startswith("http"):
                        _open(str(u), True)
                    elif str(u).startswith("10."):
                        _open("https://doi.org/" + str(u), True)
                continue
            if key == "a" and not r["suggested_value"]:
                print("   There is no suggestion for this row. Use c to type the right value, or k to keep.")
                continue
            if key in ("a", "k", "c", "s", "q"):
                break
            print("   Please press a, k, c, o, u, s or q.")
        if key == "q":
            break
        if key == "s":
            continue
        if key == "c":
            if not r["field"]:
                q.loc[i, "field"] = input("   Which field? (e.g. collection.languages_iso): ").strip()
            else:
                f = input(f"   Field to change [Enter = {r['field']}]: ").strip()
                if f:
                    q.loc[i, "field"] = f
            q.loc[i, "corrected_value"] = input("   Correct value (lists like [\"Zenodo\"], true/false, or plain text): ").strip()
        q.loc[i, "decision"] = {"a": "accept", "k": "keep", "c": "change"}[key]
        note = input("   Note (optional, Enter to skip): ").strip()
        if note:
            q.loc[i, "notes"] = note
        q.to_csv(qpath, index=False)          # save after every answer
    done = int((q["decision"] != "").sum())
    print(f"\nSaved. {done} decisions recorded in {qpath}.")
    if done and input("Apply these decisions now? [y/n]: ").strip().lower() == "y":
        apply(data_dir, curator)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--curator", default="")
    ap.add_argument("--interactive", action="store_true", help="Review one item at a time in the terminal")
    ap.add_argument("--priority", default="high", help="Comma list for --interactive, e.g. high,medium")
    ap.add_argument("--no-open", action="store_true", help="Do not open PDFs/links automatically")
    args = ap.parse_args()
    if args.interactive:
        if not args.curator:
            sys.exit("Add --curator with your initials, e.g. --curator MJ")
        interactive(Path(args.data_dir), args.curator, [x.strip() for x in args.priority.split(",")], not args.no_open)
    elif args.apply:
        if not args.curator:
            sys.exit("Add --curator with your initials, so every correction is attributable.")
        apply(Path(args.data_dir), args.curator)
    else:
        build_queue(Path(args.data_dir))


if __name__ == "__main__":
    main()

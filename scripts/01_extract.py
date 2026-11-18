#!/usr/bin/env python3
"""
Step 1 - Structured extraction with Claude through an Anthropic-compatible gateway
(default: Opus Code, https://claude.opuscode.pro/api, model claude-fable-5-1).

Setup
  export LLM_API_KEY=...                       # your Opus Code key
  python scripts/check_api.py                  # once: what does the gateway support?

Modes
  dry-run        Count input tokens per paper and estimate how many 5-hour token
                 windows of your key the full run needs. No generation.
  sync           Extract with a few parallel workers. Waits automatically when the
                 key's token window is spent, then resumes. Re-runs skip finished papers.
  batch-submit / batch-collect
                 Message Batches API. Opus Code does not document it; use sync there.
                 Kept for Anthropic's own API (LLM_BASE_URL=https://api.anthropic.com).

Structured output (--structured)
  native   schema sent as output_config.format (constrained decoding)
  prompt   schema embedded in the system prompt; JSON validated with Pydantic and,
           if invalid, one automatic repair round
  auto     try native on the first call; switch to prompt for the rest of the run if
           the gateway rejects or ignores output_config (default)

Usage
  python scripts/01_extract.py --mode dry-run
  python scripts/01_extract.py --mode sync --limit 20        # pilot, read these by hand
  python scripts/01_extract.py --mode sync                   # everything else
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from pydantic import ValidationError
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from icwsm_datahub import llm_client as llm  # noqa: E402
from icwsm_datahub.prompts import PART_HINTS, PROMPT_VERSION, SYSTEM_PROMPT, USER_TEMPLATE  # noqa: E402
from icwsm_datahub.schema import (  # noqa: E402
    SCHEMA_VERSION, ExtractionPartA, ExtractionPartB, ICWSMExtraction, to_strict_json_schema,
)

DEFAULT_MODEL = "claude-fable-5-1"
SCHEMAS = {
    "full": (ICWSMExtraction, to_strict_json_schema(ICWSMExtraction)),
    "A": (ExtractionPartA, to_strict_json_schema(ExtractionPartA)),
    "B": (ExtractionPartB, to_strict_json_schema(ExtractionPartB)),
}
EST_OUTPUT_TOKENS = 4500


class OutputNotJSON(Exception):
    """Native mode returned text that is not schema JSON (gateway likely dropped output_config)."""


class Runtime:
    """Shared, thread-safe run state (structured mode can switch once mid-run)."""
    def __init__(self, structured: str, use_cache: bool):
        self.mode = "native" if structured in ("auto", "native") else "prompt"
        self.auto = structured == "auto"
        self.use_cache = use_cache
        self.lock = threading.Lock()

    def fall_back(self, reason: str) -> None:
        with self.lock:
            if self.mode == "native" and self.auto:
                self.mode = "prompt"
                tqdm.write(f"[info] switching to prompt-mode structured output ({reason})")


# ----------------------------------------------------------------------------- request building
def load_paper(row: pd.Series, data_dir: Path) -> Dict[str, Any]:
    text = Path(row["text_path"]).read_text(encoding="utf-8")
    feats = json.loads((data_dir / "interim" / "features" / f"{row['paper_id']}.json").read_text())
    return {"row": row, "text": text, "features": feats}


def system_blocks(part: str, mode: str, use_cache: bool) -> List[Dict[str, Any]]:
    text = SYSTEM_PROMPT
    if mode == "prompt":
        text += ("\n\n<output_format>\nReply with ONE JSON object and nothing else: no prose, no code fences. "
                 "It must validate against the JSON Schema below. Every property is required; use the exact enum "
                 "strings; use the sentinels from the coding rules for unknown values.\n<json_schema>\n"
                 + json.dumps(SCHEMAS[part][1], separators=(",", ":")) + "\n</json_schema>\n</output_format>")
    block: Dict[str, Any] = {"type": "text", "text": text}
    if use_cache:
        block["cache_control"] = {"type": "ephemeral"}
    return [block]


def build_params(paper: Dict[str, Any], part: str, args: argparse.Namespace, mode: str, use_cache: bool = True) -> Dict[str, Any]:
    row, feats = paper["row"], paper["features"]
    regex_urls = "\n".join(feats["urls_in_body"][:40]) or "(none found)"
    hint = PART_HINTS[part]
    if args.input == "pdf":
        pdf_b64 = base64.standard_b64encode(Path(row["pdf_path"]).read_bytes()).decode()
        content: Any = [
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_b64}},
            {"type": "text", "text": USER_TEMPLATE.format(year=row["year"], filename=row["pdf_filename"], regex_urls=regex_urls,
                                                          paper_text="(attached as PDF document above)", part_hint=hint)},
        ]
    else:
        content = USER_TEMPLATE.format(year=row["year"], filename=row["pdf_filename"], regex_urls=regex_urls,
                                       paper_text=paper["text"], part_hint=hint)
    params: Dict[str, Any] = {
        "model": args.model, "max_tokens": args.max_tokens,
        "system": system_blocks(part, mode, use_cache),
        "messages": [{"role": "user", "content": content}],
    }
    output_config: Dict[str, Any] = {}
    if mode == "native":
        output_config["format"] = {"type": "json_schema", "schema": SCHEMAS[part][1]}
    if args.effort:
        output_config["effort"] = args.effort
    if output_config:
        params["output_config"] = output_config
    return params


# ----------------------------------------------------------------------------- response handling
def message_text(message: Any) -> str:
    stop = getattr(message, "stop_reason", None)
    if stop == "refusal":
        raise RuntimeError("Model refused (stop_reason=refusal)")
    if stop == "max_tokens":
        raise RuntimeError("Output truncated (stop_reason=max_tokens); rerun with a larger --max-tokens")
    text = "".join(b.text for b in message.content if getattr(b, "type", "") == "text")
    if not text:
        raise RuntimeError("No text block in response")
    return text


def parse_message(message: Any, part: str, notes: List[str] | None = None) -> Dict[str, Any]:
    """Text -> validated dict. Raises json.JSONDecodeError or ValidationError on bad output."""
    data = llm.extract_json(message_text(message), notes)
    return SCHEMAS[part][0].model_validate(data).model_dump(mode="json")


def parse_noting(message: Any, part: str, metas: List[Dict]) -> Dict[str, Any]:
    notes: List[str] = []
    out = parse_message(message, part, notes)
    if notes:
        metas[-1]["json_repaired"] = True      # stored in provenance so these records can be reviewed
    return out


def meta_of(message: Any) -> Dict[str, Any]:
    u = getattr(message, "usage", None)
    keys = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    out = {k: getattr(u, k, None) for k in keys} if u is not None else {}
    out["response_model"] = getattr(message, "model", None)   # what the gateway says it served
    out["response_id"] = getattr(message, "id", None)
    return out


def short_errors(exc: Exception, text: str | None = None) -> str:
    """Readable errors for the repair prompt and the logs: what the model wrote, and where."""
    if isinstance(exc, ValidationError):
        items = []
        for e in exc.errors():
            loc = ".".join(str(x) for x in e["loc"])
            if e.get("type") == "missing":
                items.append(f"{loc}: MISSING - this required field was left out; include it")
            else:
                items.append(f"{loc}: got {repr(e.get('input'))[:90]}; {e['msg'][:160]}")
        return " | ".join(items)[:1500]
    if isinstance(exc, json.JSONDecodeError) and text:
        a = max(0, exc.pos - 70)
        return f"JSON syntax error: {exc.msg} at char {exc.pos}. Context: ...{text[a:exc.pos + 70]!r}..."[:1500]
    return f"{type(exc).__name__}: {exc}"[:1500]


def try_fill_title(msg: Any, part: str, paper: Dict[str, Any], metas: List[Dict]) -> Dict[str, Any] | None:
    """If the ONLY problem is a missing paper.title, take the title from the PDF itself (largest font on
    page 1, else the first markdown heading). Saves a repair call; recorded in provenance and flagged by
    02_verify.py so a person checks it."""
    if part not in ("full", "A"):
        return None
    try:
        data = llm.extract_json(message_text(msg))
    except Exception:
        return None
    pinfo = data.get("paper") if isinstance(data, dict) else None
    if not isinstance(pinfo, dict) or pinfo.get("title"):
        return None
    from icwsm_datahub.pdftext import markdown_title, pdf_title
    pdf = Path(str(paper["row"].get("pdf_path", "")))
    title = (pdf_title(pdf) if pdf.is_file() else None) or markdown_title(paper.get("text", ""))
    if not title:
        return None
    pinfo["title"] = title
    try:
        out = SCHEMAS[part][0].model_validate(data).model_dump(mode="json")
    except ValidationError:
        return None
    metas[-1]["filled_fields"] = ["paper.title (from PDF text, model left it out)"]
    return out


def save_failed(paper: Dict[str, Any], part: str, args, text: str, err: str) -> None:
    folder = Path(getattr(args, "data_dir", "data")) / "interim" / "failed_raw"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{paper['row']['paper_id']}_{part}.txt").write_text(f"ERRORS:\n{err}\n\nRAW OUTPUT:\n{text}", encoding="utf-8")


def extract_part(client, paper: Dict[str, Any], part: str, args, rt: Runtime) -> tuple[Dict[str, Any], List[Dict]]:
    metas: List[Dict] = []
    # 1) native attempt (auto/native modes)
    if rt.mode == "native":
        try:
            msg = llm.call_with_budget(client, build_params(paper, part, args, "native", rt.use_cache), log=tqdm.write)
            metas.append({**meta_of(msg), "structured_mode": "native"})
            try:
                return parse_noting(msg, part, metas), metas
            except (json.JSONDecodeError, ValidationError) as exc:
                if not rt.auto:
                    raise
                rt.fall_back("native output was not schema JSON, so the gateway likely ignores output_config")
        except Exception as exc:
            if not (rt.auto and llm.is_structured_unsupported(exc)):
                raise
            if "cache_control" in str(exc).lower():
                rt.use_cache = False
            rt.fall_back(f"gateway rejected the request: {str(exc)[:120]}")
    # 2) prompt mode, with one repair round
    params = build_params(paper, part, args, "prompt", rt.use_cache)
    try:
        msg = llm.call_with_budget(client, params, log=tqdm.write)
    except Exception as exc:
        if rt.use_cache and llm.is_structured_unsupported(exc) and "cache_control" in str(exc).lower():
            rt.use_cache = False
            params = build_params(paper, part, args, "prompt", False)
            msg = llm.call_with_budget(client, params, log=tqdm.write)
        else:
            raise
    metas.append({**meta_of(msg), "structured_mode": "prompt"})
    try:
        return parse_noting(msg, part, metas), metas
    except (json.JSONDecodeError, ValidationError) as exc:
        filled = try_fill_title(msg, part, paper, metas)
        if filled is not None:
            return filled, metas
        errors = short_errors(exc, message_text(msg))
        metas[-1]["first_attempt_errors"] = errors[:800]   # kept in provenance: shows WHY repair was needed
        repair = dict(params)
        repair["messages"] = params["messages"] + [
            {"role": "assistant", "content": message_text(msg)},
            {"role": "user", "content": "That JSON did not validate against the schema. Errors:\n" + errors +
             "\nReturn the corrected, complete JSON object only."}]
        msg2 = llm.call_with_budget(client, repair, log=tqdm.write)
        metas.append({**meta_of(msg2), "structured_mode": "prompt-repair"})
        try:
            return parse_noting(msg2, part, metas), metas
        except (json.JSONDecodeError, ValidationError) as exc2:
            filled = try_fill_title(msg2, part, paper, metas)
            if filled is not None:
                return filled, metas
            err2 = short_errors(exc2, message_text(msg2))
            save_failed(paper, part, args, message_text(msg2), err2)
            raise RuntimeError(f"invalid after repair ({part}): {err2}") from exc2


def extract_paper(client, paper, args, rt: Runtime):
    if not args.split:
        return extract_part(client, paper, "full", args, rt)
    a, ma = extract_part(client, paper, "A", args, rt)
    b, mb = extract_part(client, paper, "B", args, rt)
    return ICWSMExtraction.model_validate({**a, **b}).model_dump(mode="json"), ma + mb


def write_record(paper: Dict[str, Any], extraction: Dict[str, Any], metas: List[Dict], args, out_dir: Path, mode: str) -> None:
    row = paper["row"]
    served = sorted({m.get("response_model") for m in metas if m.get("response_model")})
    record = {
        "paper_id": row["paper_id"], "year": int(row["year"]), "pdf_filename": row["pdf_filename"],
        "provenance": {
            "model": args.model, "served_model": served, "api_host": llm.base_host(),
            "prompt_version": PROMPT_VERSION, "schema_version": SCHEMA_VERSION, "input_mode": args.input,
            "request_mode": mode, "structured_modes": sorted({m.get("structured_mode") for m in metas if m.get("structured_mode")}),
            "effort": args.effort, "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "usage": metas,
        },
        "deterministic": dict(paper["features"]),
        "extraction": extraction,
    }
    (out_dir / f"{row['paper_id']}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


# ----------------------------------------------------------------------------- sync mode
def run_sync(papers: List[Dict], args, out_dir: Path) -> None:
    client = llm.make_client()
    rt = Runtime(args.structured, use_cache=not args.no_cache)
    print(f"Gateway {llm.base_host()}: {llm.window_summary(llm.key_status())}")
    failures = []
    # Run the first paper alone so the structured-output mode is settled before going parallel.
    first, rest = papers[:1], papers[1:]

    def work(p):
        ex, metas = extract_paper(client, p, args, rt)
        write_record(p, ex, metas, args, out_dir, "sync")
        return metas

    for p in first:
        try:
            metas = work(p)
            served = {m.get("response_model") for m in metas}
            if args.model not in served:
                tqdm.write(f"[note] requested {args.model}, gateway reports serving {sorted(x for x in served if x)}")
        except Exception as exc:
            failures.append((p["row"]["paper_id"], f"{type(exc).__name__}: {exc}"[:1500]))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, p): p["row"]["paper_id"] for p in rest}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            try:
                fut.result()
            except Exception as exc:
                failures.append((futures[fut], f"{type(exc).__name__}: {exc}"[:1500]))
    print(f"Structured output mode used: {rt.mode}")
    write_coercions(out_dir)
    report_failures(failures, out_dir)


# ----------------------------------------------------------------------------- batch mode (Anthropic API only)
def batch_state_path(data_dir: Path) -> Path:
    return data_dir / "interim" / "batches.json"


def batch_unsupported(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) in (404, 405, 501)


def run_batch_submit(papers: List[Dict], args, data_dir: Path) -> None:
    client = llm.make_client()
    mode = "prompt" if args.structured == "prompt" else "native"
    parts = ["A", "B"] if args.split else ["full"]
    requests_ = [{"custom_id": p["row"]["paper_id"] if part == "full" else f"{p['row']['paper_id'][:62]}_{part}",
                  "params": build_params(p, part, args, mode, not args.no_cache)} for p in papers for part in parts]
    state_file = batch_state_path(data_dir)
    state = json.loads(state_file.read_text()) if state_file.exists() else {"batches": []}
    for i in range(0, len(requests_), args.batch_chunk):
        try:
            batch = client.messages.batches.create(requests=requests_[i:i + args.batch_chunk])
        except Exception as exc:
            if batch_unsupported(exc):
                sys.exit(f"{llm.base_host()} does not offer the Message Batches API. Use --mode sync instead.")
            raise
        state["batches"].append({"id": batch.id, "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "n_requests": len(requests_[i:i + args.batch_chunk]), "split": args.split,
                                 "model": args.model, "input": args.input, "effort": args.effort, "collected": False})
        print(f"Submitted batch {batch.id}")
    state_file.write_text(json.dumps(state, indent=2))


def run_batch_collect(manifest: pd.DataFrame, args, data_dir: Path, out_dir: Path) -> None:
    client = llm.make_client()
    state_file = batch_state_path(data_dir)
    if not state_file.exists():
        sys.exit("No batches.json found. Submit first with --mode batch-submit.")
    state = json.loads(state_file.read_text())
    by_id = {r["paper_id"]: r for _, r in manifest.iterrows()}
    failures, pending = [], {}
    for b in state["batches"]:
        if b.get("collected"):
            continue
        info = client.messages.batches.retrieve(b["id"])
        if info.processing_status != "ended":
            print(f"Batch {b['id']} still {info.processing_status}: {info.request_counts}")
            continue
        args.model, args.input, args.effort = b["model"], b["input"], b.get("effort")
        for res in client.messages.batches.results(b["id"]):
            cid = res.custom_id
            part = cid[-1] if b["split"] and cid[-2] == "_" else None
            pid = cid[:-2] if part else cid
            if res.result.type != "succeeded":
                failures.append((pid, f"{res.result.type}"))
                continue
            paper = load_paper(by_id[pid], data_dir)
            try:
                if part:
                    pending.setdefault(pid, {})[part] = (parse_message(res.result.message, part), meta_of(res.result.message))
                    if len(pending[pid]) == 2:
                        (a, ma), (bb, mb) = pending[pid]["A"], pending[pid]["B"]
                        write_record(paper, ICWSMExtraction.model_validate({**a, **bb}).model_dump(mode="json"),
                                     [ma, mb], args, out_dir, "batch-split")
                else:
                    write_record(paper, parse_message(res.result.message, "full"), [meta_of(res.result.message)],
                                 args, out_dir, "batch")
            except Exception as exc:
                failures.append((pid, str(exc)[:300]))
        b["collected"] = True
    state_file.write_text(json.dumps(state, indent=2))
    report_failures(failures, out_dir)


# ----------------------------------------------------------------------------- dry run
def run_dry(papers: List[Dict], args) -> None:
    client = llm.make_client()
    mode = "prompt" if args.structured == "prompt" else "native"
    totals = []
    for p in tqdm(papers, desc="Counting tokens"):
        params = build_params(p, "full", args, "prompt", use_cache=False)  # includes schema text: upper bound
        params.pop("max_tokens"), params.pop("output_config", None)
        totals.append(client.messages.count_tokens(**params).input_tokens)
    s = pd.Series(totals)
    total = int(s.sum() + EST_OUTPUT_TOKENS * len(s))
    print(f"\nPapers: {len(s)}   input tokens: total {s.sum():,}  median {int(s.median()):,}  max {s.max():,}")
    print(f"Estimated total incl. ~{EST_OUTPUT_TOKENS:,} output tokens per paper: {total:,} "
          f"(upper bound; native mode sends the schema outside the prompt, mode={mode})")
    st = llm.key_status()
    print(f"Key: {llm.window_summary(st)}")
    limit = int((st or {}).get("windowTokenLimit") or 0)
    if limit:
        windows = total / limit
        print(f"This run needs about {windows:.1f} five-hour token windows ({windows * 5:.0f} hours at minimum). "
              "Sync mode waits for each reset automatically; you can also stop and resume any time.")


def write_coercions(out_dir: Path) -> None:
    from collections import Counter
    from icwsm_datahub.schema import ENUM_COERCIONS
    if not ENUM_COERCIONS:
        return
    rows = [{"enum": e, "model_wrote": w, "mapped_to": m, "count": n} for (e, w, m), n in Counter(ENUM_COERCIONS).items()]
    path = out_dir.parent / "interim" / "enum_coercions.csv"
    pd.DataFrame(rows).sort_values("count", ascending=False).to_csv(path, index=False)
    print(f"{len(ENUM_COERCIONS)} enum values were mapped from near-miss wording; see {path}. "
          "Frequent entries show where the prompt should be clearer.")


def report_failures(failures: List[tuple], out_dir: Path) -> None:
    if not failures:
        print("All requests succeeded.")
        return
    log = out_dir.parent / "interim" / "extraction_failures.csv"
    pd.DataFrame(failures, columns=["paper_id", "error"]).to_csv(log, index=False)
    print(f"{len(failures)} failures logged to {log} (raw model output in data/interim/failed_raw/). "
          "Re-run sync mode; finished papers are skipped.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["dry-run", "sync", "batch-submit", "batch-collect"], default="sync")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Exact id from GET <base>/v1/models")
    ap.add_argument("--structured", choices=["auto", "native", "prompt"], default="auto")
    ap.add_argument("--input", choices=["markdown", "pdf"], default="markdown",
                    help="markdown = cached text (recommended through a gateway); pdf = send the PDF itself")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default=None)
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--workers", type=int, default=2, help="Parallel requests (keep low on a shared key)")
    ap.add_argument("--no-cache", action="store_true", help="Do not send cache_control")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--years", default=None, help="e.g. 2020-2021")
    ap.add_argument("--split", action="store_true", help="Two smaller schemas per paper")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--batch-chunk", type=int, default=500)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = data_dir / "extracted"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(data_dir / "interim" / "manifest.csv")
    manifest = manifest[manifest["duplicate_of"].isna() | (manifest["duplicate_of"] == manifest["paper_id"])]
    if args.mode == "batch-collect":
        run_batch_collect(manifest, args, data_dir, out_dir)
        return
    todo = manifest.copy()
    if args.years:
        y0, y1 = (int(x) for x in args.years.split("-"))
        todo = todo[todo["year"].between(y0, y1)]
    if args.only:
        todo = todo[todo["paper_id"].isin(args.only)]
    if not args.force and args.mode != "dry-run":
        todo = todo[~todo["paper_id"].apply(lambda pid: (out_dir / f"{pid}.json").exists())]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(todo)} papers to process with {args.model} via {llm.base_host()} ({args.mode})")
    if todo.empty:
        return
    papers = [load_paper(r, data_dir) for _, r in todo.iterrows()]
    t0 = time.time()
    if args.mode == "dry-run":
        run_dry(papers, args)
    elif args.mode == "sync":
        run_sync(papers, args, out_dir)
    else:
        run_batch_submit(papers, args, data_dir)
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

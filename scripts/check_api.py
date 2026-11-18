#!/usr/bin/env python3
"""
Preflight check for the LLM gateway. Run once before extraction (costs a few hundred tokens).

Checks: key and base URL, model list, key budget window, a plain message, prompt caching,
token counting, native structured output (output_config), and the Batch API.
Writes data/interim/api_capabilities.json and recommends --structured / --mode settings.

Usage:  export LLM_API_KEY=...   then   python scripts/check_api.py [--model claude-fable-5-1]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from icwsm_datahub import llm_client as llm  # noqa: E402

TEST_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string", "enum": ["yes", "no"]},
                                                "count": {"type": "integer"}},
               "required": ["answer", "count"], "additionalProperties": False}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="claude-fable-5-1")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    caps = {"checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "base_url": llm.base_url(),
            "model": args.model}
    key = llm.api_key()
    print(f"Base URL : {llm.base_url()}")
    print(f"API key  : {'set (' + key[:4] + '…' + key[-4:] + ')' if key else 'MISSING - export LLM_API_KEY'}")
    if not key:
        sys.exit(1)

    models = llm.list_models()
    caps["models"] = models
    print(f"Models   : {len(models)} listed; {args.model} {'available' if args.model in models else 'NOT LISTED'}")
    st = llm.key_status()
    caps["key_status"] = {k: st.get(k) for k in ("planName", "isActive", "windowTokenLimit", "windowTokensUsed",
                                                 "windowResetAt", "expiresAt")} if st else None
    print(f"Key      : {llm.window_summary(st)}")

    import anthropic
    client = llm.make_client(max_retries=1, timeout=120)

    def attempt(label, fn):
        try:
            out = fn()
            print(f"  ok    {label}")
            return True, out
        except Exception as exc:
            print(f"  FAIL  {label}: {type(exc).__name__} {str(exc)[:160]}")
            return False, exc

    print("\nProbes:")
    ok, msg = attempt("plain message", lambda: client.messages.create(
        model=args.model, max_tokens=20, messages=[{"role": "user", "content": "Reply with the single word OK."}]))
    caps["plain_message"] = ok
    if not ok:
        sys.exit("The basic call failed. Check the key, base URL and model id before going further.")
    caps["served_model"] = getattr(msg, "model", None)
    print(f"        gateway reports model: {caps['served_model']}")

    ok, _ = attempt("prompt caching (cache_control)", lambda: client.messages.create(
        model=args.model, max_tokens=10,
        system=[{"type": "text", "text": "You answer briefly.", "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": "Say OK."}]))
    caps["cache_control"] = ok

    ok, res = attempt("token counting", lambda: client.messages.count_tokens(
        model=args.model, messages=[{"role": "user", "content": "How many tokens is this?"}]))
    caps["count_tokens"] = ok

    native = "rejected"
    ok, res = attempt("native structured output (output_config)", lambda: client.messages.create(
        model=args.model, max_tokens=200,
        messages=[{"role": "user", "content": "Is the sky blue on a clear day? Also count the letters in 'blue'."}],
        output_config={"format": {"type": "json_schema", "schema": TEST_SCHEMA}}))
    if ok:
        text = "".join(b.text for b in res.content if getattr(b, "type", "") == "text")
        try:
            data = json.loads(text)
            native = "supported" if set(data) == {"answer", "count"} else "ignored"
        except json.JSONDecodeError:
            native = "ignored"
        print(f"        output_config is {native} (raw reply starts: {text[:60]!r})")
    caps["native_structured_output"] = native

    def try_batches():
        return client.messages.batches.list(limit=1)
    ok, _ = attempt("Message Batches API", try_batches)
    caps["batch_api"] = ok

    out = Path(args.data_dir) / "interim"
    out.mkdir(parents=True, exist_ok=True)
    (out / "api_capabilities.json").write_text(json.dumps(caps, indent=2, default=str))

    print("\nRecommended settings:")
    print(f"  --structured {'native' if native == 'supported' else 'prompt'}"
          f"{'' if native == 'supported' else '   (schema in prompt + Pydantic validation + repair)'}")
    print(f"  --mode {'batch-submit' if caps['batch_api'] else 'sync'}"
          f"{'' if caps['batch_api'] else '   (this gateway has no Batch API)'}")
    if not caps["cache_control"]:
        print("  --no-cache")
    print(f"Saved {out / 'api_capabilities.json'} (cite it in the paper's methods for reproducibility).")


if __name__ == "__main__":
    main()

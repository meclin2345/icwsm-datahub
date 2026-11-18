"""Client setup for an Anthropic-compatible gateway (default: Opus Code).

Configuration (environment variables, or a .env you source yourself):
  LLM_API_KEY          your gateway key (falls back to OPUSCODE_API_KEY, then ANTHROPIC_API_KEY)
  LLM_BASE_URL         default https://claude.opuscode.pro/api  (the SDK appends /v1/messages)
  LLM_KEY_STATUS_URL   optional override for the key-status endpoint

To use Anthropic's own API instead, set LLM_BASE_URL=https://api.anthropic.com.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

DEFAULT_BASE_URL = "https://claude.opuscode.pro/api"


def base_url() -> str:
    return (os.getenv("LLM_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def api_key() -> Optional[str]:
    return os.getenv("LLM_API_KEY") or os.getenv("OPUSCODE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")


def base_host() -> str:
    return urlparse(base_url()).netloc


def make_client(max_retries: int = 2, timeout: float = 600.0):
    import anthropic
    key = api_key()
    if not key:
        sys.exit("No API key. Set LLM_API_KEY (your Opus Code key).")
    # max_retries is kept low on purpose: rate limits are handled by waiting for the
    # gateway's 5-hour token window (see call_with_budget), not by fast retries.
    return anthropic.Anthropic(api_key=key, base_url=base_url(), max_retries=max_retries, timeout=timeout)


def list_models() -> List[str]:
    """GET {base}/v1/models (no key needed on Opus Code)."""
    try:
        r = requests.get(f"{base_url()}/v1/models", timeout=20, headers={"x-api-key": api_key() or ""})
        return [m.get("id") for m in r.json().get("data", [])] if r.status_code == 200 else []
    except Exception:
        return []


def key_status() -> Optional[Dict[str, Any]]:
    """Usage and 5-hour window state for the key. Opus Code documents GET /api/key-status?key=...
    Note: this sends the key in a query string, as the gateway's docs specify."""
    key = api_key()
    if not key:
        return None
    candidates = [os.getenv("LLM_KEY_STATUS_URL")] if os.getenv("LLM_KEY_STATUS_URL") else [
        f"{base_url()}/key-status", "https://www.opuscode.pro/api/key-status"]
    for url in candidates:
        try:
            r = requests.get(url, params={"key": key}, timeout=15)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
                data = r.json()
                if data.get("status") == "found" or "windowTokenLimit" in data:
                    return data
        except Exception:
            continue
    return None


def window_summary(st: Optional[Dict[str, Any]]) -> str:
    if not st:
        return "key status unavailable"
    used, limit = int(st.get("windowTokensUsed") or 0), int(st.get("windowTokenLimit") or 0)
    reset = st.get("windowResetAt") or "n/a"
    if limit == 0:
        return (f"plan {st.get('planName', '?')}, no token-window limit reported, key active={st.get('isActive')}, "
                f"expires {st.get('expiresAt', '?')}")
    return (f"plan {st.get('planName', '?')}, window {used:,}/{limit:,} tokens used, resets {reset}, "
            f"key active={st.get('isActive')}, expires {st.get('expiresAt', '?')}")


def seconds_until_reset(st: Optional[Dict[str, Any]]) -> Optional[float]:
    if not st or not st.get("windowResetAt"):
        return None
    try:
        reset = datetime.fromisoformat(st["windowResetAt"].replace("Z", "+00:00"))
        return max(0.0, (reset - datetime.now(timezone.utc)).total_seconds())
    except ValueError:
        return None


def call_with_budget(client, params: Dict[str, Any], max_waits: int = 4, log=print):
    """messages.create that survives the gateway's rolling token window.
    On HTTP 429 it reads the key status and sleeps until the window resets."""
    import anthropic
    for attempt in range(max_waits + 1):
        try:
            return client.messages.create(**params)
        except anthropic.RateLimitError:
            if attempt == max_waits:
                raise
            wait = seconds_until_reset(key_status())
            wait = (wait + 60) if wait is not None else min(120 * 2 ** attempt, 1800)
            log(f"[rate limit] token window spent; sleeping {wait / 60:.1f} min until it resets")
            time.sleep(wait)


def is_structured_unsupported(exc: Exception) -> bool:
    """True if a 400/422 looks like the gateway rejecting output_config / cache_control."""
    status = getattr(exc, "status_code", None)
    msg = str(exc).lower()
    return status in (400, 422) and any(k in msg for k in (
        "output_config", "output_format", "json_schema", "extra inputs", "unknown field", "unrecognized",
        "not permitted", "unsupported", "too complex", "cache_control"))


FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json(text: str, notes: Optional[List[str]] = None) -> Dict[str, Any]:
    """Parse a JSON object from model text. Order: strict, strip fences/prose, then the json-repair
    library (unescaped quotes inside verbatim quotes, trailing commas, single-quoted keys...).
    A repaired parse appends 'json_repaired' to `notes`; the schema still validates the result, so
    nothing incomplete gets through, and quote verification later checks the wording."""
    t = FENCE_RE.sub("", text.strip())
    try:
        return json.loads(t)
    except json.JSONDecodeError as first:
        start, end = t.find("{"), t.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(t[start:end + 1])
            except json.JSONDecodeError:
                pass
        try:
            import json_repair
        except ImportError:
            raise first
        fixed = json_repair.loads(t[start:] if start != -1 else t)
        if not isinstance(fixed, dict) or not fixed:
            raise first
        if notes is not None:
            notes.append("json_repaired")
        return fixed

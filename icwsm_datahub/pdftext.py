"""PDF -> Markdown text, plus every feature that code can compute without an LLM."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Dict, List

URL_RE = re.compile(r"https?://[^\s<>()\[\]{}\"']+", re.IGNORECASE)
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)")
CHECKLIST_RE = re.compile(
    r"(paper\s+checklist|ethics\s+checklist|checklist\s+for\s+the\s+ethics|"
    r"for\s+most\s+authors\.\.\.|if\s+you\s+are\s+including\s+theoretical\s+results)",
    re.IGNORECASE,
)
REFERENCES_RE = re.compile(r"^\s*#*\s*\**\s*references\s*\**\s*$", re.IGNORECASE | re.MULTILINE)


def slugify(text: str, max_len: int = 48) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text[:max_len].rstrip("_") or "paper"


def make_paper_id(year: int, pdf_path: Path) -> str:
    """Stable, filesystem- and Batch-API-safe ID (<= 64 chars, [A-Za-z0-9_-])."""
    digest = hashlib.sha1(str(pdf_path.name).encode()).hexdigest()[:6]
    return f"{year}_{slugify(pdf_path.stem, 48)}_{digest}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pdf_to_markdown(pdf_path: Path) -> tuple[str, int]:
    """Return (markdown_text, page_count). pymupdf4llm keeps tables and handles
    two-column AAAI layouts; falls back to plain PyMuPDF text if unavailable."""
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24
    except ImportError:  # pragma: no cover
        import fitz

    with fitz.open(pdf_path) as doc:
        pages = doc.page_count
    try:
        import pymupdf4llm
        md = pymupdf4llm.to_markdown(str(pdf_path), show_progress=False)
    except Exception:
        with fitz.open(pdf_path) as doc:
            md = "\n\n".join(page.get_text("text", sort=True) for page in doc)
    return md, pages


def repair_text(text: str) -> str:
    """Undo common PDF artefacts so regexes and quote matching work."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")                      # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)            # hyphenated line breaks
    # URLs broken across lines: join when a line ends inside a URL-like token
    text = re.sub(r"(https?://\S*[/._-])\n(\S)", r"\1\2", text)
    return text


def normalize_for_matching(text: str) -> str:
    """Aggressive normalization used only for fuzzy quote/URL verification."""
    text = repair_text(text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)
    text = re.sub(r"[\u2013\u2014]", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def find_urls(text: str) -> List[str]:
    out = []
    for m in URL_RE.finditer(repair_text(text)):
        url = m.group(0).rstrip(".,;:)]}>'\"")
        if url not in out:
            out.append(url)
    return out


def find_dois(text: str) -> List[str]:
    out = []
    for m in DOI_RE.finditer(repair_text(text)):
        doi = m.group(1).rstrip(".,;:)")
        if doi.lower() not in [d.lower() for d in out]:
            out.append(doi)
    return out


def split_body_and_references(text: str) -> tuple[str, str]:
    """Best-effort split; URLs found before 'References' are more likely to be the
    paper's own resources than cited ones."""
    m = REFERENCES_RE.search(text)
    if not m:
        return text, ""
    return text[: m.start()], text[m.start():]


def deterministic_features(pdf_path: Path, md_text: str, page_count: int) -> Dict:
    body, refs = split_body_and_references(md_text)
    return {
        "page_count": page_count,
        "n_chars": len(md_text),
        "sha256": sha256_file(pdf_path),
        "checklist_detected": bool(CHECKLIST_RE.search(md_text)),
        "urls_in_body": find_urls(body),
        "urls_in_references": find_urls(refs),
        "dois_in_body": find_dois(body),
        "mentions_fair_regex": bool(re.search(r"\bFAIR\b", md_text)),
        "mentions_zenodo": "zenodo" in md_text.lower(),
        "mentions_dataverse": "dataverse" in md_text.lower(),
    }


TITLE_SKIP = re.compile(r"proceedings|aaai|icwsm|copyright|arxiv|abstract|association for the advancement|https?://|www\.|@",
                        re.IGNORECASE)


def pdf_title(pdf_path: Path) -> str | None:
    """Title = the largest-font text in the top half of page 1 (true for the AAAI template)."""
    try:
        try:
            import pymupdf as fitz
        except ImportError:  # pragma: no cover
            import fitz
        with fitz.open(pdf_path) as doc:
            page = doc[0]
            spans = [sp for b in page.get_text("dict")["blocks"] for ln in b.get("lines", []) for sp in ln["spans"]
                     if sp["text"].strip() and sp["bbox"][1] < page.rect.height / 2]
    except Exception:
        return None
    if not spans:
        return None
    biggest = max(sp["size"] for sp in spans)
    parts = [sp["text"].strip() for sp in sorted(spans, key=lambda sp: (round(sp["bbox"][1]), sp["bbox"][0]))
             if sp["size"] >= biggest - 0.5]
    title = " ".join(" ".join(parts).split())
    return title if 3 <= len(title.split()) <= 40 and not TITLE_SKIP.search(title) else None


def markdown_title(text: str) -> str | None:
    """Fallback: first block of markdown headings, else first plausible line."""
    lines = [l.strip() for l in text.splitlines()[:60]]
    block = []
    for l in lines:
        t = l.lstrip("#").strip().strip("*_ ").strip()
        if l.startswith("#") and t and not TITLE_SKIP.search(t):
            block.append(t)
            continue
        if block:
            break
    cand = " ".join(block)
    if 3 <= len(cand.split()) <= 40:
        return cand
    for l in lines:
        t = re.sub(r"[#*_`]", "", l).strip()
        if 4 <= len(t.split()) <= 30 and not TITLE_SKIP.search(t):
            return t
    return None


MISUSE_RE = re.compile(r"\b(misuse|mis-use|malicious|dual[- ]use|negative (societal|social) impacts?|potential(ly)? harm|"
                       r"harmful (use|uses|application)|abuse of (the|this|our) (data|dataset))", re.IGNORECASE)
FAIR_RE = re.compile(r"\bFAIR\b")
DATASHEET_RE = re.compile(r"\b(datasheets? for datasets|datasheet|data sheet|data cards?|dataset cards?|data statements?|"
                          r"nutrition label|croissant)\b", re.IGNORECASE)          # upper case only: the principles, not the word "fair"


def body_mentions(text: str) -> dict:
    """Mentions in the main text only. The ICWSM checklist sits after the References, so these
    measure what authors wrote themselves rather than checklist answers."""
    body, _ = split_body_and_references(text)
    return {"body_mentions_misuse": bool(MISUSE_RE.search(body)), "body_mentions_fair": bool(FAIR_RE.search(body)),
            "body_mentions_datasheet": bool(DATASHEET_RE.search(body))}

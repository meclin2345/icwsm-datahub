"""Core tests: schema limits, normalization, FAIR/fate logic, prompts, extraction parsing, PDF inventory."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from icwsm_datahub.fair import dataset_fate, fair_indicators  # noqa: E402
from icwsm_datahub.normalize import clean_doi, clean_url, closed_access_dependencies, normalize_extraction  # noqa: E402
from icwsm_datahub.prompts import USER_TEMPLATE  # noqa: E402
from icwsm_datahub.schema import ICWSMExtraction, PlatformType, schema_complexity, to_strict_json_schema  # noqa: E402


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sample_extraction():
    from icwsm_datahub.demo import _record
    import random
    return _record(2021, 1, random.Random(3))


def test_schema_fits_structured_output_limits():
    c = schema_complexity(to_strict_json_schema(ICWSMExtraction))
    assert c["optional"] <= 24 and c["unions"] <= 16
    assert c["optional"] == 0 and c["unions"] == 0
    assert "$ref" not in json.dumps(to_strict_json_schema(ICWSMExtraction))


def test_enums_case_insensitive():
    assert PlatformType("twitter/x") is PlatformType.TWITTER_X
    assert PlatformType("TWITTER/X") is PlatformType.TWITTER_X


def test_url_and_doi_cleaning():
    assert clean_url("https: //zenodo.org/records/123.") == "https://zenodo.org/records/123"
    assert clean_url("github.com/a/b") == "https://github.com/a/b"
    assert clean_doi("https://doi.org/10.5281/ZENODO.42") == "10.5281/zenodo.42"


def test_paper_doi_is_not_dataset_doi():
    ex = sample_extraction()
    ex["distribution"]["persistent_doi"] = "10.1609/icwsm.v15i1.18123"
    assert normalize_extraction(ex)["distribution"]["persistent_doi"] is None


def test_closed_access_detection():
    labels = [d["dependency"] for d in closed_access_dependencies(["Twitter Academic API v2", "Pushshift"], [])]
    assert "Twitter/X free or Academic API" in labels and "Pushshift Reddit archive" in labels


def test_fate_states():
    ex = normalize_extraction(sample_extraction())
    ex["distribution"].update(dataset_urls=["https://zenodo.org/records/1"], release_status="Released at publication",
                              access_mechanism="Open public download", distribution_form="Full raw content")
    assert dataset_fate(ex, {"any_url_alive": False}) == "Link dead"
    assert dataset_fate(ex, {"any_url_alive": True}) == "Open and usable"
    assert dataset_fate(ex, {"any_url_alive": True, "is_restricted": True}) == "Restricted access (request or agreement)"
    assert dataset_fate(ex, {"any_url_alive": True, "files_empty": True}) == "Alive but empty or login-walled"
    ex2 = normalize_extraction(sample_extraction())
    ex2["distribution"].update(dataset_urls=["https://x.org"], release_status="Released at publication",
                               access_mechanism="Credentialed access, form or data use agreement",
                               distribution_form="Full raw content")
    assert dataset_fate(ex2, {"any_url_alive": True}) == "Restricted access (request or agreement)"            # paper's claim
    assert dataset_fate(ex2, {"any_url_alive": True, "manual_restricted": False}) == "Open and usable"  # browser wins
    assert dataset_fate(ex2, {"any_url_alive": True, "manual_restricted": True, "files_empty": True}) == "Restricted access (request or agreement)"  # hidden files
    ex["distribution"]["distribution_form"] = "Dehydrated IDs only"
    ex["collection"]["platforms"] = ["Twitter/X"]
    assert dataset_fate(ex, {"any_url_alive": True}) == "Open, needs closed API to rehydrate"
    assert len(fair_indicators(ex, {"any_url_alive": True})) == 10


def test_prompt_template_formats():
    USER_TEMPLATE.format(year=2024, filename="x.pdf", regex_urls="none", paper_text="{braces} in text", part_hint="")


def test_extraction_parsing_with_mock_response():
    ext = load_script("01_extract.py")
    payload = json.dumps(sample_extraction())
    msg = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=payload)], model="m", id="i")
    parsed = ext.parse_message(msg, "full")
    assert parsed["collection"]["platforms"]
    msg.stop_reason = "max_tokens"
    try:
        ext.parse_message(msg, "full")
        raise AssertionError("truncation not detected")
    except RuntimeError:
        pass


def test_inventory_on_real_pdf(tmp_path):
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    year_dir = tmp_path / "papers" / "2024"
    year_dir.mkdir(parents=True)
    doc = fitz.open()
    page = doc.new_page()
    text = ("A Dataset of Telegram Channels\n\nWe release 1.2M messages. The data is available at https://zenodo.org/rec-\n"
            "ords/1234567 under CC BY 4.0.\n\nPaper Checklist\nFor most authors...\n")
    page.insert_textbox(fitz.Rect(50, 50, 550, 800), text, fontsize=10)
    doc.save(year_dir / "telegram.pdf")
    out = subprocess.run([sys.executable, str(ROOT / "scripts/00_inventory.py"), "--papers-dir", str(tmp_path / "papers"),
                          "--data-dir", str(tmp_path / "data"), "--years", "2024-2024"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    feats = json.loads(next((tmp_path / "data/interim/features").glob("*.json")).read_text())
    assert feats["checklist_detected"]
    assert any("zenodo.org/records/1234567" in u for u in feats["urls_in_body"]), feats["urls_in_body"]


# ----------------------------------------------------------------------------- gateway fallback logic
class _FakeErr(Exception):
    def __init__(self, status_code, msg):
        super().__init__(msg)
        self.status_code = status_code


class _FakeMessages:
    """Simulates a gateway. behaviour: 'reject' (400 on output_config), 'ignore' (prose despite
    output_config), 'repair' (first prompt-mode reply invalid, second valid), 'native' (works)."""
    def __init__(self, behaviour, payload):
        self.behaviour, self.payload, self.calls = behaviour, payload, []

    def create(self, **params):
        self.calls.append(params)
        native = "format" in params.get("output_config", {})
        if native and self.behaviour == "reject":
            raise _FakeErr(400, "output_config: Extra inputs are not permitted")
        if native and self.behaviour == "ignore":
            text = "Sure! Here is a summary of the paper..."
        elif self.behaviour == "repair" and len(params["messages"]) == 1:
            bad = json.loads(self.payload)
            del bad["evidence"]
            text = "```json\n" + json.dumps(bad) + "\n```"
        else:
            text = "```json\n" + self.payload + "\n```" if not native else self.payload
        return SimpleNamespace(stop_reason="end_turn", model="claude-fable-5-1", id="msg_x",
                               content=[SimpleNamespace(type="text", text=text)],
                               usage=SimpleNamespace(input_tokens=10, output_tokens=5,
                                                     cache_read_input_tokens=0, cache_creation_input_tokens=0))


def _run_fake(behaviour, structured="auto"):
    import pandas as pd
    ext = load_script("01_extract.py")
    fake = SimpleNamespace(messages=_FakeMessages(behaviour, json.dumps(sample_extraction())))
    paper = {"row": pd.Series({"paper_id": "p", "year": 2024, "pdf_filename": "p.pdf", "pdf_path": "p.pdf"}),
             "text": "paper text", "features": {"urls_in_body": []}}
    args = SimpleNamespace(input="markdown", model="claude-fable-5-1", max_tokens=1000, effort=None, split=False)
    rt = ext.Runtime(structured, use_cache=True)
    ex, metas = ext.extract_paper(fake, paper, args, rt)
    return ex, metas, rt, fake.messages.calls


def test_gateway_native_supported():
    ex, metas, rt, calls = _run_fake("native")
    assert rt.mode == "native" and len(calls) == 1 and metas[0]["response_model"] == "claude-fable-5-1"


def test_gateway_rejects_output_config_falls_back():
    ex, metas, rt, calls = _run_fake("reject")
    assert rt.mode == "prompt" and ex["collection"]["platforms"]
    assert "json_schema" in calls[-1]["system"][0]["text"] and "output_config" not in calls[-1]


def test_gateway_ignores_output_config_falls_back():
    ex, metas, rt, calls = _run_fake("ignore")
    assert rt.mode == "prompt" and [m["structured_mode"] for m in metas] == ["native", "prompt"]


def test_prompt_mode_repair_round():
    ex, metas, rt, calls = _run_fake("repair", structured="prompt")
    assert [m["structured_mode"] for m in metas] == ["prompt", "prompt-repair"] and "evidence" in ex


def test_llm_client_defaults(monkeypatch):
    from icwsm_datahub import llm_client as llm
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    assert llm.base_url() == "https://claude.opuscode.pro/api"
    assert llm.extract_json("Here:\n```json\n{\"a\": 1}\n```") == {"a": 1}


# ----------------------------------------------------------------------------- pilot-review regressions
def test_closed_access_uses_platform_and_method_when_no_tool_named():
    deps = lambda t, p, m: [d["dependency"] for d in closed_access_dependencies(t, p, m)]
    assert "Twitter/X free or Academic API" in deps(["Twitter Advanced Search API"], ["Twitter/X"], [])   # was missed
    assert "Twitter/X scraping tools" in deps([], ["Twitter/X"], ["Web scraping or browser automation"])
    assert "Twitter/X free or Academic API" in deps([], ["Twitter/X"], ["Official platform API"])
    assert deps([], ["Telegram"], ["Official platform API"]) == []


def test_url_fragment_stripped():
    assert clean_url("https://zenodo.org/record/3603330#.XheASC3MzOQ") == "https://zenodo.org/record/3603330"


def test_new_consistency_flags():
    verify = load_script("02_verify.py")
    ex = normalize_extraction(sample_extraction())
    ex["distribution"].update(distribution_form="Mixed (IDs plus derived features)", hosting_repositories=["Zenodo", "GitHub or GitLab"],
                              dataset_urls=["https://zenodo.org/records/1"])
    ex["ethics"].update(pii_anonymization_applied=True, anonymization_details="Only tweet and user IDs are shared.")
    ex["evidence"]["ethics_quote"] = "We share only the Tweet IDs and User IDs."
    flags = verify.consistency_flags(ex, {"checklist_detected": ex["ethics"]["includes_paper_checklist"]}, 2020)
    for f in ("mixed_form_but_text_says_ids_only", "pii_true_for_ids_only_release", "code_host_listed_as_dataset_host"):
        assert f in flags, flags


# ----------------------------------------------------------------------------- the three failures from the first real run
def test_short_enum_names_are_accepted():
    from icwsm_datahub.schema import AnnotationParadigm as AP, ENUM_COERCIONS
    assert AP("Automated classifier or lexicon") is AP.CLASSICAL_ML
    assert AP("Distant or weak supervision") is AP.DISTANT
    assert AP("Platform-native signals") is AP.PLATFORM_NATIVE
    assert AP("Crowdworkers") is AP.CROWD and AP("Expert") is AP.EXPERT
    assert ("AnnotationParadigm", "Crowdworkers", AP.CROWD.value) in ENUM_COERCIONS
    import pytest
    with pytest.raises(ValueError):
        AP("banana smoothie")


def test_short_enum_names_validate_inside_full_schema_and_dump_canonical():
    ex = sample_extraction()
    ex["annotation"]["annotation_paradigms"] = ["Automated classifier or lexicon", "Distant or weak supervision"]
    out = ICWSMExtraction.model_validate(ex).model_dump(mode="json")
    assert out["annotation"]["annotation_paradigms"][0].startswith("Automated classifier or lexicon (")


def test_json_syntax_slips_are_repaired():
    import pytest
    pytest.importorskip("json_repair")
    from icwsm_datahub import llm_client as llm
    notes = []
    got = llm.extract_json('{"a": "He said "hello" to them", "b": [1, 2,],}', notes)
    assert got["b"] == [1, 2] and "hello" in got["a"] and notes == ["json_repaired"]
    assert llm.extract_json("```json\n{\"a\": 1}\n```", notes) == {"a": 1}


def test_error_summaries_show_what_the_model_wrote():
    ext = load_script("01_extract.py")
    bad = sample_extraction()
    bad["annotation"]["annotation_paradigms"] = ["banana smoothie"]
    try:
        ICWSMExtraction.model_validate(bad)
    except Exception as exc:
        msg = ext.short_errors(exc)
        assert "annotation.annotation_paradigms.0" in msg and "banana smoothie" in msg
    try:
        json.loads('{"a": 1,}')
    except json.JSONDecodeError as exc:
        assert "char" in ext.short_errors(exc, '{"a": 1,}')


# ----------------------------------------------------------------------------- missing title (2 real failures)
def test_missing_field_error_is_explicit():
    ext = load_script("01_extract.py")
    bad = sample_extraction()
    del bad["paper"]["title"]
    try:
        ICWSMExtraction.model_validate(bad)
    except Exception as exc:
        msg = ext.short_errors(exc)
        assert "paper.title: MISSING" in msg and "authors" not in msg


def test_title_from_pdf_largest_font(tmp_path):
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    from icwsm_datahub.pdftext import markdown_title, pdf_title
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 80), "Chinese News Framing: A Dataset", fontsize=16)
    page.insert_text((60, 110), "Owen Cook, Yida Mu", fontsize=11)
    page.insert_text((60, 140), "Abstract We present a dataset of news articles.", fontsize=9)
    doc.save(tmp_path / "t.pdf")
    assert pdf_title(tmp_path / "t.pdf") == "Chinese News Framing: A Dataset"
    assert markdown_title("## Media Cloud Geographic\n## Collections Dataset\n\nAuthors") == "Media Cloud Geographic Collections Dataset"


def test_missing_title_filled_without_repair_call():
    import pandas as pd
    ext = load_script("01_extract.py")
    data = sample_extraction()
    del data["paper"]["title"]
    calls = []

    def create(**params):
        calls.append(params)
        return SimpleNamespace(stop_reason="end_turn", model="claude-fable-5-1", id="m",
                               content=[SimpleNamespace(type="text", text=json.dumps(data))],
                               usage=SimpleNamespace(input_tokens=1, output_tokens=1, cache_read_input_tokens=0,
                                                     cache_creation_input_tokens=0))
    fake = SimpleNamespace(messages=SimpleNamespace(create=create))
    paper = {"row": pd.Series({"paper_id": "p", "year": 2025, "pdf_filename": "p.pdf", "pdf_path": "missing.pdf"}),
             "text": "# A Dataset of Chinese News Framing\n\nOwen Cook", "features": {"urls_in_body": []}}
    args = SimpleNamespace(input="markdown", model="m", max_tokens=100, effort=None, split=False)
    ex, metas = ext.extract_paper(fake, paper, args, ext.Runtime("prompt", True))
    assert ex["paper"]["title"] == "A Dataset of Chinese News Framing"
    assert len(calls) == 1 and metas[-1]["filled_fields"]


# ----------------------------------------------------------------------------- review workflow
def test_review_queue_and_apply(tmp_path):
    import pandas as pd
    data = tmp_path / "data"
    (data / "extracted").mkdir(parents=True)
    (data / "interim").mkdir()
    ex = sample_extraction()
    ex["distribution"]["distribution_form"] = "Mixed (IDs plus derived features)"
    ex["ethics"]["pii_anonymization_applied"] = True
    rec = {"paper_id": "p1", "year": 2021, "pdf_filename": "p1.pdf", "provenance": {"prompt_version": "v5", "usage": []},
           "deterministic": {}, "extraction": ex}
    (data / "extracted" / "p1.json").write_text(json.dumps(rec))
    pd.DataFrame([{"paper_id": "p1", "flags": "mixed_form_but_text_says_ids_only; pii_true_for_ids_only_release",
                   "ungrounded_urls": "", "doi_grounded": True}]).to_csv(data / "interim" / "verification.csv", index=False)
    run = lambda *a: subprocess.run([sys.executable, str(ROOT / "scripts/review_flags.py"), "--data-dir", str(data), *a],
                                    capture_output=True, text=True)
    out = run()
    assert out.returncode == 0, out.stderr
    q = pd.read_csv(data / "curation" / "review_queue.csv", dtype=str).fillna("")
    assert set(q["flag"]) == {"mixed_form_but_text_says_ids_only", "pii_true_for_ids_only_release"}
    q["decision"] = "accept"
    q.to_csv(data / "curation" / "review_queue.csv", index=False)
    assert run("--apply").returncode == 1                       # curator initials required
    out = run("--apply", "--curator", "MJ")
    assert out.returncode == 0, out.stdout + out.stderr
    from icwsm_datahub.curation import apply_overrides, load_overrides
    fixed = normalize_extraction(ex)
    apply_overrides(fixed, load_overrides(data / "curation" / "overrides.csv")["p1"])
    assert fixed["distribution"]["distribution_form"] == "Dehydrated IDs only"
    assert fixed["ethics"]["pii_anonymization_applied"] is False
    run()                                                        # reviewed rows do not come back
    assert not (data / "curation" / "review_queue.csv").exists()


def test_spreadsheet_mangled_values_still_parse(tmp_path):
    from icwsm_datahub.curation import load_overrides
    f = tmp_path / "o.csv"
    f.write_text('paper_id,field,value,reason,curator\np,ethics.pii_anonymization_applied,FALSE,r,MJ\n'
                 'p,distribution.hosting_repositories,[\u201cZenodo\u201d],r,MJ\n', encoding="utf-8")
    o = {x["field"]: x["value"] for x in load_overrides(f)["p"]}
    assert o["ethics.pii_anonymization_applied"] is False and o["distribution.hosting_repositories"] == ["Zenodo"]


def test_manual_browser_check_overrides_audit(tmp_path):
    import pandas as pd
    from icwsm_datahub.demo import write_demo_workspace
    data = tmp_path / "data"
    write_demo_workspace(data)
    summ = pd.DataFrame([json.loads(p.read_text())["summary"] for p in sorted((data / "audit").glob("*.json"))])
    summ.to_csv(data / "audit" / "audit_summary.csv", index=False)
    dead = summ[~summ["any_url_alive"].astype(bool) & summ["n_urls_checked"].gt(0)].iloc[0]["paper_id"]
    (data / "audit" / f"{dead}.json").write_text(json.dumps({**json.loads((data / "audit" / f"{dead}.json").read_text()),
                                                          "url_checks": [{"url": "https://x.org", "category": "Restricted / forbidden (403)"}]}))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/audit_checklist.py"), "--data-dir", str(data)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    chk = pd.read_csv(data / "curation" / "audit_checks.csv", dtype=str).fillna("")
    chk.loc[chk["paper_id"] == dead, ["any_url_alive", "checked_on", "note"]] = ["TRUE", "2026-10-30", "Cloudflare; loads in browser"]
    chk.to_csv(data / "curation" / "audit_checks.csv", index=False)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/05_build_catalog.py"), "--data-dir", str(data),
                        "--out-dir", str(tmp_path / "cat"), "--web-out", str(tmp_path / "web" / "catalog.json")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    cat = pd.read_csv(tmp_path / "cat" / "icwsm_datahub_catalog.csv")
    row = cat[cat["paper_id"] == dead].iloc[0]
    assert bool(row["any_url_alive"]) and "Cloudflare" in row["audit_manual_check"]


def test_body_mentions_ignore_checklist():
    from icwsm_datahub.pdftext import body_mentions
    text = "Intro\nWe release data.\n\n## References\nA. B. 2020.\n\nPaper Checklist\nDid you discuss misuse? Yes. FAIR? Yes."
    assert body_mentions(text) == {"body_mentions_misuse": False, "body_mentions_fair": False, "body_mentions_datasheet": False}
    text2 = "We follow the FAIR principles, provide a datasheet and discuss potential misuse.\n## References\n"
    assert body_mentions(text2) == {"body_mentions_misuse": True, "body_mentions_fair": True, "body_mentions_datasheet": True}


def test_llm_flag_for_non_llm_tools():
    verify = load_script("02_verify.py")
    ex = normalize_extraction(sample_extraction())
    ex["annotation"].update(annotation_paradigms=["LLM or generative AI labels"], llms_used=["Google's Perspective API", "spaCy"])
    assert "llm_paradigm_but_no_llm_named" in verify.consistency_flags(ex, {"checklist_detected": ex["ethics"]["includes_paper_checklist"]}, 2020)
    ex["annotation"]["llms_used"] = ["GPT-4o"]
    assert "llm_paradigm_but_no_llm_named" not in verify.consistency_flags(ex, {"checklist_detected": ex["ethics"]["includes_paper_checklist"]}, 2020)


def test_interactive_review_saves_and_applies(tmp_path):
    import pandas as pd
    data = tmp_path / "data"
    (data / "extracted").mkdir(parents=True)
    (data / "interim").mkdir()
    ex = sample_extraction()
    ex["distribution"]["distribution_form"] = "Mixed (IDs plus derived features)"
    ex["ethics"]["pii_anonymization_applied"] = True
    rec = {"paper_id": "p1", "year": 2021, "pdf_filename": "p1.pdf", "provenance": {"prompt_version": "v5", "usage": []},
           "deterministic": {}, "extraction": ex}
    (data / "extracted" / "p1.json").write_text(json.dumps(rec))
    pd.DataFrame([{"paper_id": "p1", "flags": "mixed_form_but_text_says_ids_only; pii_true_for_ids_only_release",
                   "ungrounded_urls": "", "doi_grounded": True}]).to_csv(data / "interim" / "verification.csv", index=False)
    # row 1: accept + note; row 2: keep, no note; then apply
    r = subprocess.run([sys.executable, str(ROOT / "scripts/review_flags.py"), "--data-dir", str(data), "--interactive",
                        "--curator", "MJ", "--no-open"], input="a\nchecked p.3\nk\n\ny\n", capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    o = pd.read_csv(data / "curation" / "overrides.csv")
    assert list(o["field"]) == ["distribution.distribution_form"] and "checked p.3" in o["reason"].iloc[0]


def test_interactive_browser_check(tmp_path):
    import pandas as pd
    out = tmp_path / "audit_checks.csv"
    pd.DataFrame([{"paper_id": "p", "year": 2022, "title": "T", "why_listed": "paper says access needs a form or approval",
                   "urls_to_open": "https://x.org", "auto_status": "Alive", "any_url_alive": "", "is_restricted": "",
                   "files_empty": "", "checked_on": "", "note": ""}]).to_csv(out, index=False)
    chk = load_script("audit_checklist.py")
    import builtins
    answers = iter(["y", "y", "y", "opens fine"])   # loads; can download; files present
    orig = builtins.input
    builtins.input = lambda *_: next(answers)
    try:
        chk.interactive(out, open_links=False)
    finally:
        builtins.input = orig
    row = pd.read_csv(out, dtype=str).fillna("").iloc[0]
    assert (row["any_url_alive"], row["is_restricted"], row["files_empty"]) == ("true", "false", "false")


def test_interactive_change_can_target_another_field(tmp_path):
    import pandas as pd
    data = tmp_path / "data"
    (data / "extracted").mkdir(parents=True)
    (data / "interim").mkdir()
    ex = sample_extraction()
    ex["distribution"]["distribution_form"] = "Anonymized or PII-scrubbed full content"
    ex["ethics"]["pii_anonymization_applied"] = False
    rec = {"paper_id": "p1", "year": 2021, "pdf_filename": "p1.pdf", "provenance": {"prompt_version": "v5", "usage": []},
           "deterministic": {}, "extraction": ex}
    (data / "extracted" / "p1.json").write_text(json.dumps(rec))
    pd.DataFrame([{"paper_id": "p1", "flags": "anonymized_form_but_no_pii_step", "ungrounded_urls": "",
                   "doi_grounded": True}]).to_csv(data / "interim" / "verification.csv", index=False)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/review_flags.py"), "--data-dir", str(data), "--interactive",
                        "--curator", "MJ", "--no-open"],
                       input="c\nethics.pii_anonymization_applied\ntrue\nhashed user IDs, p.4\ny\n", capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    o = pd.read_csv(data / "curation" / "overrides.csv", dtype=str)
    assert list(o["field"]) == ["ethics.pii_anonymization_applied"] and list(o["value"]) == ["true"]


def test_browser_check_skips_files_question_when_restricted(tmp_path):
    import builtins
    import pandas as pd
    out = tmp_path / "audit_checks.csv"
    pd.DataFrame([{"paper_id": "p", "year": 2022, "title": "T", "why_listed": "x", "urls_to_open": "https://x.org",
                   "auto_status": "Alive", "any_url_alive": "", "is_restricted": "", "files_empty": "",
                   "checked_on": "", "note": ""}]).to_csv(out, index=False)
    chk = load_script("audit_checklist.py")
    answers = iter(["y", "n", "Zenodo restricted, request access button"])
    orig = builtins.input
    builtins.input = lambda *_: next(answers)
    try:
        chk.interactive(out, open_links=False)
    finally:
        builtins.input = orig
    row = pd.read_csv(out, dtype=str).fillna("").iloc[0]
    assert (row["any_url_alive"], row["is_restricted"], row["files_empty"]) == ("true", "true", "")


def test_documentation_audit_set_writes_override(tmp_path, monkeypatch):
    import pandas as pd
    (tmp_path / "catalog").mkdir()
    (tmp_path / "data" / "curation").mkdir(parents=True)
    pd.DataFrame([{"paper_id": "2026_41077_romanos_abc123", "year": 2026, "checklist_detected": True,
                   "documentation_framework": "Datasheets for Datasets", "body_mentions_datasheet": False},
                  {"paper_id": "2026_41029_piotp_def456", "year": 2026, "checklist_detected": True,
                   "documentation_framework": "Datasheets for Datasets", "body_mentions_datasheet": False}]
                 ).to_csv(tmp_path / "catalog" / "icwsm_datahub_catalog.csv", index=False)
    run = lambda *a: subprocess.run([sys.executable, str(ROOT / "scripts/documentation_audit.py"), *a],
                                    capture_output=True, text=True, cwd=tmp_path)
    assert run("set", "41077", "answered_no", "5(g) No", "--curator", "MJ").returncode == 0
    assert run("set", "piotp", "external_file", "DATASHEET.md on GitHub", "--curator", "MJ").returncode == 0
    assert run("set", "2026", "bare_yes", "--curator", "MJ").returncode != 0      # ambiguous id
    a = pd.read_csv(tmp_path / "data/curation/documentation_audit.csv")
    o = pd.read_csv(tmp_path / "data/curation/overrides.csv")
    assert set(a["category"]) == {"answered_no", "external_file"}
    assert list(o["paper_id"]) == ["2026_41077_romanos_abc123"] and o["value"].iloc[0] == "None mentioned"
    assert "2 of 2 classified" in run("list").stdout

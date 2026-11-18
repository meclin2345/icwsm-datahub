#!/usr/bin/env python3
"""
Step 6 - Analysis: every number the paper reports, computed in one place.

Outputs
  results/stats.json                     all statistics (machine-readable)
  results/key_numbers.tex                \\kn{key} macros, e.g. "\\kn{pct_usable} of datasets..."
  results/tables/overview_by_year.tex    Table 1
  results/tables/trend_tests.{csv,tex}   logistic trend test for each indicator
  results/tables/link_survival.tex       logit: link alive ~ age + DOI + archival host
  results/tables/reuse_models.tex        negative binomial: citations and downloads

In the paper:  \\input{results/key_numbers.tex}  then write
  "Only \\kn{pct_usable} of datasets remain openly usable in 2026."
so numbers never drift from the catalog.

Usage
  python scripts/06_analysis.py --catalog catalog/icwsm_datahub_catalog.parquet
  python scripts/06_analysis.py --demo
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from icwsm_datahub.fair import FAIR_INDICATORS, FATE_KEYS, FATE_ORDER  # noqa: E402
from icwsm_datahub.catalog_io import boolcol, load_catalog  # noqa: E402
from icwsm_datahub.normalize import FAMILY_ORDER  # noqa: E402

warnings.filterwarnings("ignore")

TREND_INDICATORS = {
    "has_doi": "Dataset DOI reported",
    "uses_llm_annotation": "LLM or hybrid annotation",
    "uses_crowd": "Crowdworker annotation",
    "is_multimodal": "Multimodal release",
    "is_multilingual": "Multilingual",
    "depends_on_closed_access": "Collected via a since-closed access route",
    "pii_anonymization_applied": "PII anonymized before release",
    "discusses_platform_tos": "Discusses platform ToS",
    "discusses_misuse_risks": "Discusses misuse risks",
    "explicitly_mentions_fair": "Mentions FAIR principles",
    "iaa_reported": "Reports inter-annotator agreement",
    "data_global_south": "Data about the Global South",
    "any_url_alive": "Dataset link alive in 2026",
    "funnel_usable": "Openly usable in 2026",
}


def to_tex(df: pd.DataFrame, wide: bool = False, **kw) -> str:
    """DataFrame -> booktabs LaTeX with % and _ escaped in headers and labels.
    wide=True emits a two-column table* scaled to \\textwidth (needs graphicx)."""
    esc = lambda x: str(x).replace("%", r"\%").replace("_", r"\_") if isinstance(x, str) else x
    out = df.rename(columns=esc, index=esc)
    tex = out.to_latex(escape=False, **kw)
    if wide:
        tex = (tex.replace("\\begin{table}", "\\begin{table*}[t]").replace("\\end{table}", "\\end{table*}")
                  .replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}")
                  .replace("\\end{tabular}", "\\end{tabular}}"))
    return tex


def pct(x: float, digits: int = 1) -> float:
    return round(100 * float(x), digits) if x is not None and not (isinstance(x, float) and math.isnan(x)) else None


def share_with(df: pd.DataFrame, col: str, value: str) -> pd.Series:
    """Share of papers per year whose list column contains `value`."""
    return df.groupby("year")[col].apply(lambda s: np.mean([value in (x or []) for x in s]))


# ----------------------------------------------------------------------------- analyses
def overview(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("year")
    t = pd.DataFrame({
        "Papers": g.size(),
        "Median records": g["total_records"].median(),
        "Multilingual (%)": g["is_multilingual"].apply(lambda s: 100 * s.astype(bool).mean()),
        "Multimodal (%)": g["is_multimodal"].apply(lambda s: 100 * s.astype(bool).mean()),
        "Dataset DOI (%)": g["has_doi"].apply(lambda s: 100 * s.astype(bool).mean()),
        "LLM labels (%)": g["uses_llm_annotation"].apply(lambda s: 100 * s.astype(bool).mean()),
        "Link alive (%)": g["any_url_alive"].apply(lambda s: 100 * s.fillna(False).astype(bool).mean()),
        "Usable (%)": g["funnel_usable"].apply(lambda s: 100 * s.astype(bool).mean()),
        "Median FAIR": g["fair_score"].median(),
    })
    t.loc["All"] = [len(df), df["total_records"].median(), 100 * df["is_multilingual"].astype(bool).mean(),
                    100 * df["is_multimodal"].astype(bool).mean(), 100 * df["has_doi"].astype(bool).mean(),
                    100 * df["uses_llm_annotation"].astype(bool).mean(),
                    100 * df["any_url_alive"].fillna(False).astype(bool).mean(),
                    100 * df["funnel_usable"].astype(bool).mean(), df["fair_score"].median()]
    return t


def trend_tests(df: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.formula.api as smf
    rows = []
    y0, y1 = df["year"].min(), df["year"].max()
    for col, label in TREND_INDICATORS.items():
        if col not in df:
            continue
        d = pd.DataFrame({"y": boolcol(df, col), "t": df["year"] - y0})
        first, last = d.loc[df["year"] == y0, "y"].mean(), d.loc[df["year"] == y1, "y"].mean()
        try:
            if d["y"].nunique() < 2:
                raise ValueError("constant outcome")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = smf.logit("y ~ t", d).fit(disp=0)
            orr, lo, hi = np.exp(m.params["t"]), *np.exp(m.conf_int().loc["t"])
            p = m.pvalues["t"]
        except Exception:
            orr = lo = hi = p = np.nan
        rows.append({"indicator": label, "key": col, f"{y0} (%)": pct(first), f"{y1} (%)": pct(last),
                     "OR per year": orr, "CI low": lo, "CI high": hi, "p": p})
    out = pd.DataFrame(rows)
    # 14 tests on small yearly samples: control the false discovery rate (Benjamini-Hochberg).
    from statsmodels.stats.multitest import multipletests
    ok = out["p"].notna()
    out["p (BH)"] = np.nan
    if ok.any():
        out.loc[ok, "p (BH)"] = multipletests(out.loc[ok, "p"], method="fdr_bh")[1]
    return out


def platform_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"share_by_year": {}}
    for fam in FAMILY_ORDER:
        out["share_by_year"][fam] = {int(k): pct(v) for k, v in share_with(df, "platform_families", fam).items()}

    def entropy(s: pd.Series) -> float:
        counts = pd.Series([f for fams in s for f in (fams or [])]).value_counts(normalize=True)
        return float(-(counts * np.log2(counts)).sum())
    out["diversity_entropy_bits"] = {int(y): round(entropy(g["platform_families"]), 3) for y, g in df.groupby("year")}

    # Structural break measured on DATA time (collection end), not publication time.
    d = df.dropna(subset=["temporal_coverage_end"]).copy()
    d["post"] = d["temporal_coverage_end"].astype(str).str[:7] >= "2023-03"
    d["tw"] = d["platform_families"].apply(lambda f: "Twitter/X" in (f or []))
    tab = pd.crosstab(d["post"], d["tw"])
    if tab.shape == (2, 2):
        chi2, p, _, _ = stats.chi2_contingency(tab)
        out["twitter_pre_post_api"] = {
            "pre_share": pct(d.loc[~d["post"], "tw"].mean()), "post_share": pct(d.loc[d["post"], "tw"].mean()),
            "n_pre": int((~d["post"]).sum()), "n_post": int(d["post"].sum()), "chi2": round(chi2, 2), "p": p}
    return out


def link_survival(df: pd.DataFrame) -> tuple[Dict[str, Any], str]:
    import statsmodels.formula.api as smf
    d = df[df["funnel_link_provided"].astype(bool)].copy()
    d = pd.DataFrame({"alive": boolcol(d, "any_url_alive"), "age": d["years_since_publication"].astype(float),
                      "doi": boolcol(d, "has_doi"), "archival": boolcol(d, "fair_A2"), "host": d["primary_host"]})
    res: Dict[str, Any] = {"n": len(d), "alive_pct": pct(d["alive"].mean()),
                           "alive_by_host": {h: pct(g["alive"].mean()) for h, g in d.groupby("host")},
                           "n_by_host": d["host"].value_counts().to_dict()}
    tex = ""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.logit("alive ~ age + doi + archival", d).fit(disp=0)
        ci = np.exp(m.conf_int())
        tab = pd.DataFrame({"Odds ratio": np.exp(m.params), "95% CI low": ci[0], "95% CI high": ci[1], "p": m.pvalues})
        tab = tab.drop(index="Intercept").rename(index={"age": "Years since publication", "doi": "Has dataset DOI",
                                                        "archival": "Archival repository"})
        res["logit"] = tab.round(4).to_dict(orient="index")
        res["logit_pseudo_r2"] = round(m.prsquared, 3)
        tex = to_tex(tab, float_format="%.3f", caption=r"Logistic regression of link survival in 2026 (95\% CIs).",
                           label="tab:link_survival")
    except Exception as exc:
        res["logit_error"] = str(exc)
    # Approximate annual decay among non-archival links: fit P(alive) = exp(-lambda * age)
    na = d[d["archival"] == 0]
    if len(na) > 10 and na["alive"].mean() > 0:
        lam = -np.log(max(na["alive"].mean(), 1e-6)) / na["age"].mean()
        res["nonarchival_half_life_years"] = round(math.log(2) / lam, 1) if lam > 0 else None
    return res, tex


def reuse_models(df: pd.DataFrame) -> tuple[Dict[str, Any], str]:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    out: Dict[str, Any] = {}
    texs = []
    d = df.dropna(subset=["citations"]).copy()
    if len(d) > 30:
        d = d.assign(cit=d["citations"].astype(float), usable=boolcol(d, "funnel_usable"), doi=boolcol(d, "has_doi"),
                     llm=boolcol(d, "uses_llm_annotation"), multimodal=boolcol(d, "is_multimodal"),
                     logage=np.log(d["years_since_publication"].astype(float)))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = smf.glm("cit ~ usable + doi + llm + multimodal", d, family=sm.families.NegativeBinomial(alpha=1.0),
                            offset=d["logage"]).fit()
            ci = np.exp(m.conf_int())
            tab = pd.DataFrame({"IRR": np.exp(m.params), "95% CI low": ci[0], "95% CI high": ci[1], "p": m.pvalues})
            tab = tab.drop(index="Intercept")
            out["citations_nb"] = tab.round(4).to_dict(orient="index")
            texs.append(to_tex(tab, float_format="%.3f", caption="Negative binomial model of paper citations "
                                     "(offset: log years since publication). IRR = incidence rate ratio.",
                                     label="tab:citations"))
        except Exception as exc:
            out["citations_error"] = str(exc)
    # Downloads: compare ONLY within one counter (Zenodo) - definitions differ across repositories.
    z = df[(df["downloads_metric_source"] == "Zenodo stats") & df["downloads_per_year"].notna()].copy()
    if len(z) > 10:
        dehyd = z["distribution_form"] == "Dehydrated IDs only"
        a, b = z.loc[~dehyd, "downloads_per_year"], z.loc[dehyd, "downloads_per_year"]
        if len(a) > 2 and len(b) > 2:
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            out["zenodo_downloads_full_vs_dehydrated"] = {
                "median_full_per_year": round(float(a.median()), 1), "median_dehydrated_per_year": round(float(b.median()), 1),
                "n_full": len(a), "n_dehydrated": len(b), "U": float(u), "p": p,
                "effect_r_rank_biserial": round(1 - 2 * u / (len(a) * len(b)), 3)}
    return out, "\n".join(texs)


def funnel(df: pd.DataFrame) -> Dict[str, Any]:
    cols = ["funnel_release_claimed", "funnel_link_provided", "funnel_link_alive", "funnel_open_download", "funnel_usable"]
    overall = {"papers": len(df), **{c.replace("funnel_", ""): int(df[c].astype(bool).sum()) for c in cols}}
    by_year = {int(y): {c.replace("funnel_", ""): pct(g[c].astype(bool).mean()) for c in cols} for y, g in df.groupby("year")}
    fate = pd.crosstab(df["year"], df["fate_2026"], normalize="index").reindex(columns=FATE_ORDER, fill_value=0) * 100
    return {"overall": overall, "by_year_pct": by_year, "fate_by_year_pct": fate.round(1).to_dict(orient="index"),
            "fate_overall_pct": (df["fate_2026"].value_counts(normalize=True) * 100).round(1).to_dict()}


def policy_effect(df: pd.DataFrame) -> tuple[Dict[str, Any], pd.DataFrame]:
    """Before vs. after the year the paper checklist appears (first year with >50% of papers).
    Body-only measures separate authors' own writing from checklist answers."""
    by_year = df.groupby("year")["checklist_detected"].apply(lambda s: s.astype(bool).mean())
    start = next((int(y) for y, v in by_year.items() if v > 0.5), None)
    if start is None:
        return {"checklist_year": None}, pd.DataFrame()
    post = df["year"] >= start
    measures = [("checklist_detected", "Paper checklist present"),
                ("discusses_misuse_risks", "Discusses misuse risks (LLM coding, whole paper)"),
                ("body_mentions_misuse", "Misuse language in main text (before References)"),
                ("explicitly_mentions_fair", "Mentions FAIR (LLM coding, whole paper)"),
                ("body_mentions_fair", "FAIR in main text (before References)"),
                ("discusses_platform_tos", "Discusses platform ToS"),
                ("pii_anonymization_applied", "PII anonymized before release"),
                # FAIR in PRACTICE (what was released), as opposed to FAIR in words:
                ("has_doi", "Practice: dataset DOI"),
                ("fair_R1_1", "Practice: license stated in the paper"),
                ("repo_license_present", "Practice: license on the repository record (checklist-independent)"),
                ("fair_F2", "Practice: standard documentation named in the paper"),
                ("body_mentions_datasheet", "Datasheet/data card named in main text (before References)"),
                ("fair_R1_3", "Practice: code to collect or load the data"),
                ("fair_A2", "Practice: archival repository"),
                ("funnel_usable", "Outcome: openly usable in 2026")]
    rows = []
    for col, label in measures:
        if col not in df or df[col].isna().all():
            continue
        known = df[col].notna()               # e.g. repository license only exists for verified records
        y = boolcol(df[known], col)
        a, b = y[~post[known]], y[post[known]]
        _, p = stats.fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])
        rows.append({"measure": label, "key": col, f"before {start} (%)": pct(a.mean()), f"{start} on (%)": pct(b.mean()),
                     "n before": len(a), "n after": len(b), "Fisher p": p})
    if "fair_score" in df:
        a, b = df.loc[~post, "fair_score"].dropna(), df.loc[post, "fair_score"].dropna()
        if len(a) > 2 and len(b) > 2:
            _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            rows.append({"measure": "Practice: FAIR score (0-100, mean, not %)", "key": "fair_score",
                         f"before {start} (%)": round(float(a.mean()), 1), f"{start} on (%)": round(float(b.mean()), 1),
                         "n before": len(a), "n after": len(b), "Fisher p": p})
    return {"checklist_year": start}, pd.DataFrame(rows)


DOC_LABELS = {
    "in_paper": "Documentation included in the paper or appendix",
    "external_file": "Points to a documentation file outside the paper",
    "self_declared": "States that the paper itself is the documentation",
    "bare_yes": "Checklist 'Yes' with no pointer or evidence",
    "answered_no": "Checklist answered 'No' (extraction error, corrected)",
    "other_framework": "Non-dataset framework, e.g. model card (corrected)",
}


def documentation_audit(path: Path) -> tuple[Dict[str, Any], pd.DataFrame]:
    if not path.exists():
        return {}, pd.DataFrame()
    a = pd.read_csv(path, dtype=str).fillna("")
    n = len(a)
    tab = (a["category"].value_counts().reindex(list(DOC_LABELS), fill_value=0).rename("papers").to_frame())
    tab["share (%)"] = (100 * tab["papers"] / max(n, 1)).round(1)
    tab.index = [DOC_LABELS[k] for k in tab.index]
    counts = a["category"].value_counts().to_dict()
    claims = n - counts.get("answered_no", 0) - counts.get("other_framework", 0)   # genuine claims
    evidenced = counts.get("in_paper", 0) + counts.get("external_file", 0)
    return {"n_audited": n, "counts": counts, "genuine_claims": claims, "evidenced": evidenced}, tab


def geography(df: pd.DataFrame) -> Dict[str, Any]:
    d = df[df["author_countries"].apply(lambda x: len(x or []) > 0)]
    out = {"author_global_south_pct": pct(boolcol(d, "author_global_south").mean()),
           "data_global_south_pct": pct(boolcol(df, "data_global_south").mean())}
    gs_data = df[boolcol(df, "data_global_south") == 1]
    if len(gs_data):
        out["global_south_data_with_global_south_author_pct"] = pct(boolcol(gs_data, "author_global_south").mean())
    return out


# ----------------------------------------------------------------------------- key numbers
def tex_escape_value(v: Any, percent: bool) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    if percent:
        return f"{v:.1f}\\%"
    if isinstance(v, float):
        return f"{v:,.2f}" if abs(v) < 100 else f"{v:,.0f}"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return str(v)


def key_numbers(df: pd.DataFrame, S: Dict[str, Any]) -> Dict[str, str]:
    y0, y1 = int(df["year"].min()), int(df["year"].max())
    fo = S["funnel"]["overall"]
    k: Dict[str, str] = {}

    def add(name, value, percent=False):
        k[name] = tex_escape_value(value, percent)

    add("n_papers", len(df))
    add("first_year", str(y0))
    add("last_year", str(y1))
    add("n_columns", df.shape[1])
    add("total_records_all", int(df["total_records"].fillna(0).sum()))
    add("pct_link_provided", 100 * fo["link_provided"] / fo["papers"], True)
    add("pct_alive", 100 * fo["link_alive"] / fo["papers"], True)
    add("pct_open_download", 100 * fo["open_download"] / fo["papers"], True)
    add("pct_usable", 100 * fo["usable"] / fo["papers"], True)
    for fate, v in S["funnel"]["fate_overall_pct"].items():
        add("fate_" + FATE_KEYS.get(fate, "other"), v, True)
    tw = S["platforms"]["share_by_year"]["Twitter/X"]
    add("twitter_share_first", tw.get(y0), True)
    add("twitter_share_last", tw.get(y1), True)
    tk = S["platforms"]["share_by_year"]["TikTok"]
    add("tiktok_share_first", tk.get(y0), True)
    add("tiktok_share_last", tk.get(y1), True)
    pp = S["platforms"].get("twitter_pre_post_api", {})
    add("twitter_pre_api", pp.get("pre_share"), True)
    add("twitter_post_api", pp.get("post_share"), True)
    for _, r in S["trends"].iterrows():
        add(f"{r['key']}_first", r[f"{y0} (%)"], True)
        add(f"{r['key']}_last", r[f"{y1} (%)"], True)
        add(f"{r['key']}_or", r["OR per year"])
    ls = S["link_survival"]
    for host, v in ls.get("alive_by_host", {}).items():
        add("alive_host_" + "".join(ch for ch in host.lower() if ch.isalnum()), v, True)
    add("nonarchival_half_life", ls.get("nonarchival_half_life_years"))
    add("median_fair_first", df.loc[df["year"] == y0, "fair_score"].median())
    add("median_fair_last", df.loc[df["year"] == y1, "fair_score"].median())
    g = S["geography"]
    add("author_gs_pct", g.get("author_global_south_pct"), True)
    add("data_gs_pct", g.get("data_global_south_pct"), True)
    add("gs_data_gs_author_pct", g.get("global_south_data_with_global_south_author_pct"), True)
    z = S["reuse"].get("zenodo_downloads_full_vs_dehydrated", {})
    add("dl_full_median", z.get("median_full_per_year"))
    add("dl_dehydrated_median", z.get("median_dehydrated_per_year"))
    add("dl_mwu_p", z.get("p"))
    da = S.get("documentation_audit") or {}
    if da:
        add("doc_audit_n", da["n_audited"])
        add("doc_genuine_claims", da["genuine_claims"])
        add("doc_evidenced", da["evidenced"])
        add("doc_evidenced_pct", 100 * da["evidenced"] / max(da["genuine_claims"], 1), True)
        for cat in DOC_LABELS:
            add(f"doc_{cat}", da["counts"].get(cat, 0))
    pol = S.get("policy", {})
    add("checklist_year", str(pol.get("checklist_year")) if pol.get("checklist_year") else None)
    for r in pol.get("table", []):
        keys = [k for k in r if k.startswith("before")] + [k for k in r if k.endswith("on (%)")]
        if len(keys) == 2:
            pct_row = r["key"] != "fair_score"
            add(f"{r['key']}_pre", r[keys[0]], pct_row)
            add(f"{r['key']}_post", r[keys[1]], pct_row)
            pv = r.get("Fisher p")
            if pv is not None and not (isinstance(pv, float) and math.isnan(pv)):
                k[f"{r['key']}_p"] = "$p<0.001$" if pv < 0.001 else f"$p={pv:.3f}$"
    return k


def write_key_numbers(k: Dict[str, str], path: Path, is_demo: bool) -> None:
    lines = ["% Auto-generated by scripts/06_analysis.py. Do not edit by hand.",
             "% Usage: \\input{results/key_numbers.tex} then \\kn{pct_usable}",
             "% WARNING: DEMO DATA. Numbers are synthetic." if is_demo else "",
             "\\makeatletter",
             "\\providecommand{\\kn}[1]{\\@ifundefined{kn@#1}{\\textbf{??\\detokenize{#1}??}}{\\@nameuse{kn@#1}}}"]
    lines += [f"\\@namedef{{kn@{name}}}{{{val}}}" for name, val in sorted(k.items())]
    lines.append("\\makeatother")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default="catalog/icwsm_datahub_catalog.parquet")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    cat, res = Path(args.catalog), Path(args.results_dir)
    if args.demo:
        cat, res = Path("demo/catalog/icwsm_datahub_catalog.parquet"), Path("demo/results")
    df = load_catalog(cat)
    (res / "tables").mkdir(parents=True, exist_ok=True)

    ov = overview(df)
    (res / "tables" / "overview_by_year.tex").write_text(to_tex(
        ov, wide=True, float_format="%.1f", caption="ICWSM Dataset Track papers by year.", label="tab:overview"))
    tr = trend_tests(df)
    tr.to_csv(res / "tables" / "trend_tests.csv", index=False)
    (res / "tables" / "trend_tests.tex").write_text(to_tex(
        tr.drop(columns=["key"]), wide=True, index=False, float_format="%.3f", na_rep="--",
        caption="Logistic trend tests (odds ratio per publication year).", label="tab:trends"))
    ls, ls_tex = link_survival(df)
    (res / "tables" / "link_survival.tex").write_text(ls_tex)
    pe_meta, pe = policy_effect(df)
    if not pe.empty:
        pe.to_csv(res / "tables" / "policy_effect.csv", index=False)
        (res / "tables" / "policy_effect.tex").write_text(to_tex(
            pe.drop(columns=["key"]), index=False, float_format="%.3f",
            caption=f"Papers before and after the paper checklist ({pe_meta['checklist_year']}).", label="tab:policy"))
    doc_meta, doc_tab = documentation_audit(Path("data/curation/documentation_audit.csv"))
    if not doc_tab.empty:
        (res / "tables" / "documentation_audit.tex").write_text(to_tex(
            doc_tab, float_format="%.1f",
            caption="Manual audit of documentation claims after the paper checklist.", label="tab:docaudit"))
    ru, ru_tex = reuse_models(df)
    (res / "tables" / "reuse_models.tex").write_text(ru_tex)

    S = {"n_papers": len(df), "overview": ov.round(2).to_dict(orient="index"), "trends": tr,
         "platforms": platform_analysis(df), "link_survival": ls, "reuse": ru, "funnel": funnel(df),
         "geography": geography(df), "documentation_audit": doc_meta, "policy": {**pe_meta, "table": pe.to_dict(orient="records") if not pe.empty else []},
         "fair_indicator_pct_by_year": {code: {int(y): pct(g["fair_" + code.replace(".", "_")].astype(bool).mean())
                                               for y, g in df.groupby("year")} for code, _, _ in FAIR_INDICATORS}}
    kn = key_numbers(df, S)
    write_key_numbers(kn, res / "key_numbers.tex", args.demo)
    S["trends"] = tr.to_dict(orient="records")
    (res / "stats.json").write_text(json.dumps(S, indent=2, default=lambda o: None if isinstance(o, float) and
                                               math.isnan(o) else (o.item() if hasattr(o, "item") else str(o))))

    print(("[DEMO DATA - synthetic numbers]\n" if args.demo else "") + ov.round(1).to_string())
    print("\nTrend tests:\n" + tr.drop(columns=["key"]).round(3).to_string(index=False))
    f = S["funnel"]["overall"]
    if not pe.empty:
        print(f"\nBefore vs after the paper checklist ({pe_meta['checklist_year']}):")
        print(pe.drop(columns=["key"]).round(4).to_string(index=False))
    if not doc_tab.empty:
        print(f"\nDocumentation audit ({doc_meta['n_audited']} papers; {doc_meta['evidenced']} of "
              f"{doc_meta['genuine_claims']} genuine claims have evidence):")
        print(doc_tab.to_string())
    print(f"\nFunnel: {f}")
    fate = df["fate_2026"].value_counts()
    print("\nWhy datasets drop out (2026 state):")
    for k, v in fate.items():
        print(f"  {k:<42} {v:>4}  ({100 * v / len(df):.1f}%)")
    gated = df[df["fate_2026"].isin(["Restricted access (request or agreement)", "Alive but empty or login-walled"])]
    if len(gated):
        reasons = pd.Series({
            "access coded as form/DUA/email approval": gated["access_mechanism"].isin(
                ["Credentialed access, form or data use agreement", "Email the authors for approval"]).sum(),
            "repository reports restricted files": gated["is_restricted"].fillna(False).astype(bool).sum(),
            "repository record has no files": gated["files_empty"].fillna(False).astype(bool).sum(),
            "login wall": gated["login_wall"].fillna(False).astype(bool).sum()})
        print("  gated, by reason (a paper can have several):")
        print("\n".join(f"    {k:<40} {v}" for k, v in reasons.items()))
    print(f"Twitter/X share pre vs post API closure (by data end date): {S['platforms'].get('twitter_pre_post_api')}")
    print("Link survival by host (alive %, n):",
          {h: (v, ls["n_by_host"].get(h)) for h, v in ls.get("alive_by_host", {}).items()})
    print(f"\nWrote {res}/stats.json, key_numbers.tex ({len(kn)} macros) and tables/")


if __name__ == "__main__":
    main()

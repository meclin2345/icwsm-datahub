#!/usr/bin/env python3
"""
Step 7 - Figures for the paper (AAAI camera-ready) and an animated Gapminder GIF.

Storytelling rules borrowed from Hans Rosling and applied to every figure:
  1. One claim per figure; the caption states it as a sentence.
  2. Time is the protagonist: every figure shows change across 2020-2026.
  3. Color encodes data only (Okabe-Ito, colorblind-safe); hatching keeps
     categories distinct in grayscale print.
  4. Label the data directly instead of sending readers to a legend when possible.
  5. Big, quiet context (the year watermark) behind the data in the Gapminder chart.

Figures (PDF with TrueType fonts + PNG)
  fig1_funnel_and_fate        double column  Available is not usable
  fig2_platform_migration     double column  Twitter/X fades, video and messaging rise
  fig3_link_survival          single column  DOIs keep links alive
  fig4_annotation_shift       single column  LLMs enter the annotation pipeline
  fig5_fair_heatmap           single column  Which FAIR indicators improved
  fig6_who_studies_whom       single column  Author vs. data geography
  fig7_reuse_by_release_form  single column  Dehydration and reuse (Zenodo only)
  fig8_gapminder_snapshots    double column  Platforms as bubbles in three years
  gapminder.gif               (--animate)    README / talk asset, not for the PDF
  figures.tex                 ready-to-paste LaTeX figure environments

Usage
  python scripts/07_figures.py --catalog catalog/icwsm_datahub_catalog.parquet --animate
  python scripts/07_figures.py --demo --animate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from icwsm_datahub.catalog_io import boolcol, load_catalog  # noqa: E402
from icwsm_datahub.fair import FAIR_INDICATORS, FATE_COLORS, FATE_ORDER  # noqa: E402
from icwsm_datahub.normalize import API_EVENTS, FAMILY_COLORS, FAMILY_ORDER  # noqa: E402

SINGLE = 3.30   # AAAI column width (in)
DOUBLE = 6.975  # AAAI text width (in)
INK, MUTED, RULE = "#222222", "#6B6B6B", "#D9D9D9"
OK = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
      "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#8C8C8C"}
FATE_HATCH = {"Open and usable": "", "Open, needs closed API to rehydrate": "...",
              "Restricted access (request or agreement)": "//", "Alive but empty or login-walled": "++", "Link dead": "xx", "Never released / no link": "\\\\"}
SHORT_FAMILY = {"Meta (Facebook/Instagram/Threads)": "Meta (FB/IG/Threads)",
                "Messaging (Telegram/WhatsApp/Discord)": "Messaging (Telegram/WA/Discord)",
                "Decentralized and alt-tech": "Decentralized & alt-tech", "Web, news and wiki": "Web, news & wiki",
                "AI and LLM content": "AI & LLM content"}
CAPTIONS: Dict[str, str] = {}


def style() -> None:
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,  # TrueType embedding: no Type 3 fonts
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "axes.linewidth": 0.7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "grid.linewidth": 0.5, "grid.color": RULE, "lines.linewidth": 1.6, "patch.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def save(fig: plt.Figure, out: Path, name: str, caption: str, wide: bool) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    CAPTIONS[name] = (caption, wide)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return 100 * max(0, c - h), 100 * min(1, c + h)


def decimal_date(s: str) -> float:
    s = str(s)
    y = int(s[:4])
    m = int(s[5:7]) if len(s) >= 7 and s[5:7].isdigit() else 6
    return y + (m - 0.5) / 12


# ============================================================================ Figure 1
def fig1_funnel_and_fate(df: pd.DataFrame, out: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.55), gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.45})
    n = len(df)
    claimed = int(df["funnel_release_claimed"].astype(bool).sum())
    stages = [("Papers", n)] + ([("Release claimed", claimed)] if claimed < n else []) + [
              ("Link or DOI given", df["funnel_link_provided"].astype(bool).sum()),
              ("Link alive in 2026", df["funnel_link_alive"].astype(bool).sum()),
              ("Openly downloadable", df["funnel_open_download"].astype(bool).sum()),
              ("Usable without closed API", df["funnel_usable"].astype(bool).sum())]
    shades = ["#BDBDBD", "#A6A6A6", "#8F8F8F", "#6F6F6F", "#4F4F4F", OK["green"]]
    for i, ((label, v), col) in enumerate(zip(stages, shades)):
        a.barh(len(stages) - 1 - i, v, color=col, height=0.68)
        a.text(v + n * 0.015, len(stages) - 1 - i, f"{100 * v / n:.0f}%  (n={v})", va="center", fontsize=8.5,
               color=INK if i < len(stages) - 1 else OK["green"], fontweight="bold" if i == len(stages) - 1 else None)
    a.set_yticks(range(len(stages)))
    a.set_yticklabels([s for s, _ in stages][::-1])
    a.set_xlim(0, n * 1.38)
    a.xaxis.set_visible(False)
    a.spines["bottom"].set_visible(False)
    a.set_title("(a) From paper to usable data, all years")

    fate = pd.crosstab(df["year"], df["fate_2026"], normalize="index").reindex(columns=FATE_ORDER, fill_value=0) * 100
    years = fate.index.astype(int)
    bottom = np.zeros(len(fate))
    present = [f for f in FATE_ORDER if fate[f].sum() > 0]
    for f in present:
        b.bar(years, fate[f], bottom=bottom, color=FATE_COLORS[f], hatch=FATE_HATCH[f], edgecolor="white",
              linewidth=0.5, width=0.72, label=f)
        bottom += fate[f].values
    for y, v in zip(years, fate["Open and usable"]):
        b.text(y, v / 2, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    counts = df.groupby("year").size()
    b.set_xticks(years)
    b.set_xticklabels([f"{y}\nn={counts.get(y, 0)}" for y in years], fontsize=7.5)
    b.set_ylim(0, 100)
    b.set_ylabel("Share of the year's datasets (%)")
    b.set_title("(b) What became of each year's datasets")
    b.legend(loc="upper center", bbox_to_anchor=(-0.22, -0.13), ncol=3, fontsize=8, handlelength=1.4, columnspacing=1.0)
    save(fig, out, "fig1_funnel_and_fate",
         "Available is not usable. (a) Of all ICWSM Dataset Track papers, the share that clears each step from a "
         "claimed release to data a researcher can actually use in 2026. (b) The 2026 state of every dataset, by "
         "publication year; the number on each bar is the share that is openly usable.", True)


# ============================================================================ Figure 2
def _text_color_for(hex_color: str) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 else "white"


def _spread_labels(ys: List[float], min_gap: float) -> List[float]:
    """Nudge end-of-line labels apart so they never overlap."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for k in range(1, len(order)):
        prev, cur = order[k - 1], order[k]
        if out[cur] - out[prev] < min_gap:
            out[cur] = out[prev] + min_gap
    return out.tolist()


def fig2_platform_migration(df: pd.DataFrame, out: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.75), gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.62})
    years = sorted(df["year"].unique().astype(int))
    fams = [f for f in FAMILY_ORDER if f != "Other"]
    share = {f: [100 * np.mean([f in x for x in df.loc[df["year"] == y, "platform_families"]]) for y in years]
             for f in fams}
    for i, f in enumerate(fams):
        row = len(fams) - 1 - i
        for y, s in zip(years, share[f]):
            if s > 0:
                a.scatter(y, row, s=s * 6.5, color=FAMILY_COLORS[f], alpha=0.9, edgecolor="white", linewidth=0.5)
            if s >= 15:
                a.text(y, row, f"{s:.0f}", ha="center", va="center", fontsize=7.5,
                       color=_text_color_for(FAMILY_COLORS[f]), fontweight="bold")
    a.set_yticks(range(len(fams)))
    a.set_yticklabels([SHORT_FAMILY.get(f, f) for f in fams][::-1])
    a.set_xticks(years)
    a.set_xlim(years[0] - 0.6, years[-1] + 0.6)
    a.set_ylim(-0.7, len(fams) - 0.3)
    a.grid(axis="y", linestyle="-", alpha=0.6)
    a.set_axisbelow(True)
    a.set_title("(a) Share of papers using each platform (%)")

    # (b) before vs. after the API closures, on DATA time (matches the chi-square test in 06_analysis)
    d = df[df["temporal_coverage_end"].notna()].copy()
    d["post"] = d["temporal_coverage_end"].astype(str).str[:7] >= "2023-03"
    show = [f for f in ["Twitter/X", "Reddit", "YouTube", "TikTok", "Messaging (Telegram/WhatsApp/Discord)"]
            if f in FAMILY_COLORS]
    n_pre, n_post = int((~d["post"]).sum()), int(d["post"].sum())
    xs = np.arange(len(show))
    w = 0.38
    top = 0.0
    for k, f in enumerate(show):
        for j, (mask, n_) in enumerate(((~d["post"], n_pre), (d["post"], n_post))):
            hits = int(d.loc[mask, "platform_families"].apply(lambda x: f in x).sum())
            v = 100 * hits / n_ if n_ else 0
            lo, hi = wilson(hits, n_)
            x = xs[k] + (j - 0.5) * w
            b.bar(x, v, width=w * 0.92, color=FAMILY_COLORS[f], alpha=0.35 if j == 0 else 1.0,
                  hatch="///" if j == 0 else "", edgecolor="white", lw=0.4)
            b.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor=MUTED, elinewidth=0.7, capsize=1.5)
            b.text(x, hi + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
            top = max(top, hi)
            if f == "Twitter/X":
                tw_top = max(locals().get("tw_top", 0), hi)
        if f == "Twitter/X" and n_pre and n_post:
            tw = d["platform_families"].apply(lambda x: "Twitter/X" in x)
            _, pv, _, _ = stats.chi2_contingency(pd.crosstab(d["post"], tw))
            yb = tw_top + 9
            b.plot([xs[k] - w / 2, xs[k] - w / 2, xs[k] + w / 2, xs[k] + w / 2], [yb - 2, yb, yb, yb - 2], color=INK, lw=0.7)
            b.text(xs[k], yb + 1, f"p = {pv:.3f}" if pv >= 0.001 else "p < 0.001", ha="center", va="bottom", fontsize=7.5)
            top = max(top, yb + 6)
    b.set_xticks(xs)
    b.set_xticklabels([SHORT_FAMILY.get(f, f).split(" (")[0] for f in show], fontsize=7.5, rotation=25, ha="right")
    b.set_ylabel("Share of datasets (%)")
    b.set_ylim(0, top * 1.12)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="#9A9A9A", alpha=0.35, hatch="///", edgecolor="white"),
               plt.Rectangle((0, 0), 1, 1, facecolor="#5A5A5A")]
    b.legend(handles, [f"Data ending before Mar 2023 (n={n_pre})", f"Ending Mar 2023 or later (n={n_post})"],
             loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=7, ncol=1)
    b.set_title("(b) Before vs. after the API closures")
    b.grid(axis="y", linestyle="-", alpha=0.6)
    b.set_axisbelow(True)
    save(fig, out, "fig2_platform_migration",
         "Platform migration. (a) Share of each year's dataset papers that include each platform family (bubble "
         "area proportional to share). (b) Share of datasets including each platform, split by when data collection "
         "ended: before or after March 2023, when free Twitter/X API access ended (followed by Pushshift, free Reddit "
         "API access and CrowdTangle). Error bars are 95\\% Wilson intervals; p from a chi-square test.", True)


# ============================================================================ Figure 3
def fig3_link_survival(df: pd.DataFrame, out: Path) -> None:
    d = df[df["funnel_link_provided"].astype(bool)].copy()
    d["alive"] = boolcol(d, "any_url_alive")
    d["doi"] = boolcol(d, "has_doi")
    fig, ax = plt.subplots(figsize=(SINGLE, 2.35))
    years = sorted(d["year"].unique().astype(int))
    ends = []
    for flag, label, color, marker in [(1, "With dataset DOI", OK["blue"], "o"), (0, "Plain URL only", OK["red"], "s")]:
        sub = d[d["doi"] == flag]
        xs, ms, lo, hi = [], [], [], []
        for y in years:
            g = sub[sub["year"] == y]
            if len(g) < 3:
                continue
            k, n = int(g["alive"].sum()), len(g)
            l_, h_ = wilson(k, n)
            xs.append(y), ms.append(100 * k / n), lo.append(l_), hi.append(h_)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.14, lw=0)
        ax.plot(xs, ms, marker=marker, color=color, markersize=3.6)
        if xs:
            ends.append((ms[-1], label, color, xs[-1]))
    for (y_end, label, color, x_end), y_lab in zip(ends, _spread_labels([e[0] for e in ends], 9)):
        ax.text(x_end + 0.2, y_lab, label.replace(" ", "\n", 1), va="center", fontsize=7.5, color=color,
                linespacing=0.95)
    ax.set_xlim(years[0] - 0.3, years[-1] + 1.9)
    ax.set_xticks(years)
    ax.set_ylim(0, 103)
    ax.set_ylabel("Dataset link alive in 2026 (%)")
    ax.set_xlabel("ICWSM year")
    ax.grid(axis="y", alpha=0.6)
    ax.set_title("Persistent identifiers slow link rot")
    save(fig, out, "fig3_link_survival",
         "Share of dataset links that still resolve in 2026, by publication year, for datasets with and without a "
         "dataset DOI. Ribbons are 95\\% Wilson intervals; cohorts with fewer than three papers are omitted.", False)


# ============================================================================ Figure 4
def fig4_annotation_shift(df: pd.DataFrame, out: Path) -> None:
    years = sorted(df["year"].unique().astype(int))
    groups = [("LLM or hybrid human+LLM", ("LLM", "Hybrid"), OK["red"], "o", "-"),
              ("Crowdworkers", ("Crowd",), OK["blue"], "s", "--"),
              ("Experts or authors", ("Expert",), OK["green"], "^", "-."),
              ("Distant supervision or classifiers", ("Distant", "Automated"), OK["orange"], "D", ":")]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.45))
    for label, prefixes, color, marker, ls in groups:
        ys = [100 * np.mean([any(p.startswith(prefixes) for p in x) for x in df.loc[df["year"] == y, "annotation_paradigms"]])
              for y in years]
        ax.plot(years, ys, marker=marker, color=color, ls=ls, markersize=3.6)
        ax.text(years[-1] + 0.15, ys[-1], label.split(" or ")[0] if len(label) > 18 else label, va="center",
                fontsize=7.5, color=color)
    ax.set_xticks(years)
    ax.set_xlim(years[0] - 0.3, years[-1] + 2.3)
    ax.set_ylim(0, None)
    ax.set_ylabel("Share of papers (%)")
    ax.set_xlabel("ICWSM year")
    ax.grid(axis="y", alpha=0.6)
    ax.set_title("LLMs enter the annotation pipeline")
    save(fig, out, "fig4_annotation_shift",
         "Share of dataset papers using each source of labels (papers can use several). LLM includes pipelines "
         "where humans verify or correct LLM labels.", False)


# ============================================================================ Figure 5
def fig5_fair_heatmap(df: pd.DataFrame, out: Path) -> None:
    years = sorted(df["year"].unique().astype(int))
    rows, labels = [], []
    for code, label, _ in FAIR_INDICATORS:
        col = "fair_" + code.replace(".", "_")
        rows.append([100 * df.loc[df["year"] == y, col].astype(bool).mean() for y in years])
        labels.append(f"{code}  {label}")
    m = np.array(rows)
    fig, ax = plt.subplots(figsize=(SINGLE, 3.0))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("fair", ["#F2F2F2", "#9ECAE1", "#2171B5", "#08306B"])
    ax.imshow(m, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if m[i, j] > 55 else INK)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(y)[2:].join(["'", ""]) for y in years])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Which FAIR practices improved (% of papers)")
    save(fig, out, "fig5_fair_heatmap",
         "Share of papers meeting each operational FAIR indicator, by year. Indicators combine what papers report "
         "with what the 2026 audit observed (definitions in Table~\\ref{tab:fair}).", False)


# ============================================================================ Figure 6
def fig6_who_studies_whom(df: pd.DataFrame, out: Path) -> None:
    regions = ["North America", "Europe & Central Asia", "East Asia & Pacific", "South Asia",
               "Latin America & Caribbean", "Middle East & North Africa", "Sub-Saharan Africa"]
    auth = [100 * np.mean([r in x for x in df["author_regions"]]) for r in regions]
    data = [100 * np.mean([r in x for x in df["data_regions"]]) for r in regions]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    y = np.arange(len(regions))[::-1]
    ax.barh(y + 0.19, auth, height=0.36, color=INK, label="Authors affiliated there")
    ax.barh(y - 0.19, data, height=0.36, color=OK["sky"], hatch="///", edgecolor="white", lw=0, label="Data about there")
    ax.set_yticks(y)
    ax.set_yticklabels([r.replace(" & ", " & ") for r in regions], fontsize=8)
    ax.set_xlabel("Share of papers (%)")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.grid(axis="x", alpha=0.6)
    ax.set_title("Who studies whom")
    save(fig, out, "fig6_who_studies_whom",
         "Share of papers with at least one author affiliated in each world region, compared with the share whose "
         "data is about that region. Gaps show regions studied mostly from elsewhere.", False)


# ============================================================================ Figure 7
def fig7_reuse(df: pd.DataFrame, out: Path) -> None:
    z = df[(df["downloads_metric_source"] == "Zenodo stats") & df["downloads_per_year"].notna()].copy()
    fig, ax = plt.subplots(figsize=(SINGLE, 2.55))
    order = [("Full raw content", "Full\ncontent", OK["green"]),
             ("Anonymized or PII-scrubbed full content", "Anonymized\ncontent", OK["blue"]),
             ("Mixed (IDs plus derived features)", "IDs +\nfeatures", OK["sky"]),
             ("Dehydrated IDs only", "IDs\nonly", OK["red"])]
    order = [o for o in order if (z["distribution_form"] == o[0]).sum() >= 2]
    if not order:
        ax.text(0.5, 0.5, "Not enough Zenodo download data yet", ha="center", transform=ax.transAxes)
    rng = np.random.default_rng(1)
    for i, (form, lab, col) in enumerate(order):
        v = z.loc[z["distribution_form"] == form, "downloads_per_year"].values
        ax.boxplot(v, positions=[i], widths=0.45, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=col, alpha=0.25, edgecolor=MUTED), medianprops=dict(color=INK, lw=1.4),
                   whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=9, color=col, alpha=0.8, edgecolor="white", lw=0.3)
    counts = [int((z["distribution_form"] == o[0]).sum()) for o in order]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{o[1]}\n(n={c})" for o, c in zip(order, counts)], fontsize=8)
    ax.set_yscale("symlog", linthresh=10)
    if len(z):
        ax.set_ylim(0, z["downloads_per_year"].max() * 6)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylabel("Downloads per year (Zenodo)")
    ax.grid(axis="y", alpha=0.6)
    ax.set_title("Datasets shared as IDs only are reused less")
    full = z.loc[z["distribution_form"] != "Dehydrated IDs only", "downloads_per_year"]
    ids = z.loc[z["distribution_form"] == "Dehydrated IDs only", "downloads_per_year"]
    if len(full) > 2 and len(ids) > 2:
        _, p = stats.mannwhitneyu(full, ids, alternative="two-sided")
        ax.text(0.98, 0.97, f"Content vs. IDs only\nMann-Whitney p {'< 0.001' if p < 0.001 else f'= {p:.3f}'}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)
    save(fig, out, "fig7_reuse_by_release_form",
         "Age-normalized annual downloads by release form, restricted to Zenodo so that every dataset is measured "
         "with the same counter (symlog scale). Download counters are not comparable across repositories.", False)


# ============================================================================ Figure 9
def fig9_talk_vs_practice(df: pd.DataFrame, out: Path) -> None:
    """Dumbbells: before vs. after the paper checklist. Orange = practice as REPORTED in the paper (incl.
    checklist answers); blue = a checklist-independent measure of the same practice."""
    by_year = df.groupby("year")["checklist_detected"].apply(lambda s: s.astype(bool).mean())
    start = next((int(y) for y, v in by_year.items() if v > 0.5), None)
    if start is None:
        return
    post = df["year"] >= start
    pairs = [("License", "fair_R1_1", "stated in paper", "repo_license_present", "on repository record"),
             ("Documentation", "fair_F2", "named in paper", "body_mentions_datasheet", "in main text"),
             ("FAIR principles", "explicitly_mentions_fair", "mentioned in paper", "body_mentions_fair", "in main text"),
             ("Misuse risks", "discusses_misuse_risks", "discussed in paper", "body_mentions_misuse", "in main text")]
    rows, ypos, labels, y = [], [], [], 0.0
    for topic, rep, rep_lab, ind, ind_lab in pairs:
        group = []
        for col, lab, kind in ((rep, rep_lab, "reported"), (ind, ind_lab, "independent")):
            if col not in df or df[col].isna().all():
                continue
            known = df[col].notna()
            yy = boolcol(df[known], col)
            a_, b2 = yy[~post[known]], yy[post[known]]
            _, p = stats.fisher_exact([[a_.sum(), len(a_) - a_.sum()], [b2.sum(), len(b2) - b2.sum()]])
            group.append((topic, lab, kind, 100 * a_.mean(), 100 * b2.mean(), p))
        for g in group:
            rows.append(g)
            ypos.append(y)
            labels.append(f"{g[0]}: {g[1]}" if g is group[0] else g[1])
            y -= 1.0
        y -= 0.6   # gap between topics
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(SINGLE, 0.33 * len(rows) + 1.0))
    colors = {"reported": OK["orange"], "independent": OK["blue"]}
    for yv, (topic, lab, kind, a_, b2, p) in zip(ypos, rows):
        c = colors[kind]
        ax.plot([a_, b2], [yv, yv], color=c, lw=1.6, zorder=1)
        ax.scatter([a_], [yv], s=22, facecolor="white", edgecolor=c, lw=1.3, zorder=2)
        ax.scatter([b2], [yv], s=26, color=c, zorder=3)
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.text(103, yv, stars, va="center", ha="left", fontsize=7, color=INK if stars != "n.s." else MUTED,
                clip_on=False)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"% of papers (open: before {start}; filled: from {start})")
    ax.grid(axis="x", alpha=0.6)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], color=colors[k], marker="o", lw=1.6, label=l) for k, l in
               (("reported", "Reported in the paper (incl. checklist)"), ("independent", "Checklist-independent measure"))]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.3, -0.2 if len(rows) > 6 else -0.28),
              ncol=1, fontsize=7.5)
    ax.set_title("Reporting rose more than practice")
    save(fig, out, "fig9_talk_vs_practice",
         f"Papers before and after the ICWSM paper checklist ({start}). Orange: the practice as reported anywhere in "
         "the paper, including checklist answers. Blue: an independent measure of the same practice, namely the "
         "license recorded on the repository record, or mentions in the main text before the references. "
         "Fisher's exact test: * p<0.05, ** p<0.01, *** p<0.001.", False)


# ============================================================================ Figure 8 + animation
def gapminder_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y, g in df.groupby("year"):
        for f in FAMILY_ORDER:
            sub = g[g["platform_families"].apply(lambda x: f in x)]
            if len(sub) == 0:
                continue
            rows.append({"year": int(y), "family": f, "n": len(sub), "fair": sub["fair_score"].mean(),
                         "usable": 100 * sub["funnel_usable"].astype(bool).mean()})
    return pd.DataFrame(rows)


def _draw_gapminder(ax, frame: pd.DataFrame, year_label: str, size_scale: float, n_labels: int = 2,
                    fontsize: float = 7.5) -> None:
    ax.text(0.5, 0.45, year_label, transform=ax.transAxes, ha="center", va="center", fontsize=54,
            color="#EBEBEB", fontweight="bold", zorder=0)
    frame = frame.sort_values("n", ascending=False)
    # Label the protagonist (Twitter/X) plus the largest other bubbles; color legend covers the rest.
    to_label = {"Twitter/X"} | set(frame[frame["family"] != "Twitter/X"].head(n_labels)["family"])
    for _, r in frame.iterrows():
        ax.scatter(r["fair"], r["usable"], s=r["n"] * size_scale, color=FAMILY_COLORS[r["family"]], alpha=0.85,
                   edgecolor="white", linewidth=0.6, zorder=2)
    for _, r in frame.iterrows():
        if r["family"] in to_label and r["n"] > 0:
            radius_pts = np.sqrt(r["n"] * size_scale / np.pi)
            ax.annotate(SHORT_FAMILY.get(r["family"], r["family"]).split(" (")[0], (r["fair"], r["usable"]),
                        xytext=(radius_pts + 2, 0), textcoords="offset points", va="center", fontsize=fontsize,
                        color=INK, zorder=3)
    ax.set_xlim(20, 100)
    ax.set_ylim(-8, 108)
    ax.grid(alpha=0.5)
    ax.set_axisbelow(True)


def fig8_gapminder(df: pd.DataFrame, out: Path) -> None:
    gm = gapminder_table(df)
    years = sorted(gm["year"].unique())
    picks = [years[0], years[len(years) // 2], years[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.45), sharey=True)
    fig.subplots_adjust(bottom=0.2)
    for ax, y in zip(axes, picks):
        _draw_gapminder(ax, gm[gm["year"] == y], str(y), size_scale=9)
        ax.set_xlabel("Mean FAIR score")
    axes[0].set_ylabel("Openly usable in 2026 (%)")
    present = [f for f in FAMILY_ORDER if f in set(gm["family"])]
    handles = [plt.Line2D([], [], marker="o", ls="", markersize=6, markerfacecolor=FAMILY_COLORS[f],
                          markeredgecolor="white", label=SHORT_FAMILY.get(f, f)) for f in present]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=7.5,
               handletextpad=0.2, columnspacing=0.9)
    save(fig, out, "fig8_gapminder_snapshots",
         "Platform families as bubbles, in the style of Gapminder. Horizontal position: mean FAIR score of that "
         "family's datasets; vertical: share still openly usable in 2026; area: number of datasets that year. "
         "Younger cohorts have had less time to decay, so compare bubbles within a panel, and movement between panels. "
         "An animated version is in the repository and the web explorer.", True)


def animate_gapminder(df: pd.DataFrame, out: Path, fps: int = 20, tween: int = 18) -> None:
    from matplotlib.animation import FuncAnimation, PillowWriter
    gm = gapminder_table(df)
    years = sorted(gm["year"].unique())
    frames = []
    for i in range(len(years) - 1):
        for t in np.linspace(0, 1, tween, endpoint=False):
            frames.append((years[i], years[i + 1], t))
    frames += [(years[-1], years[-1], 0.0)] * (fps * 2)  # hold the last year

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    def interp(y0: int, y1: int, t: float) -> pd.DataFrame:
        a = gm[gm["year"] == y0].set_index("family")
        b = gm[gm["year"] == y1].set_index("family")
        fams = sorted(set(a.index) | set(b.index))
        rows = []
        for f in fams:
            ra = a.loc[f] if f in a.index else None
            rb = b.loc[f] if f in b.index else None
            if ra is None:
                ra = rb.copy()
                ra["n"] = 0
            if rb is None:
                rb = ra.copy()
                rb["n"] = 0
            ease = t * t * (3 - 2 * t)
            rows.append({"family": f, **{k: ra[k] + (rb[k] - ra[k]) * ease for k in ("n", "fair", "usable")}})
        return pd.DataFrame(rows)

    def draw(fr):
        y0, y1, t = fr
        ax.clear()
        _draw_gapminder(ax, interp(y0, y1, t), str(y0 if t < 0.5 else y1), size_scale=16, n_labels=4, fontsize=8.5)
        ax.set_xlabel("Mean FAIR score of the year's datasets")
        ax.set_ylabel("Openly usable in 2026 (%)")
        ax.set_title("ICWSM datasets by platform, 2020-2026  (bubble area = number of datasets)", fontsize=9)

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    anim.save(out / "gapminder.gif", writer=PillowWriter(fps=fps), dpi=110)
    plt.close(fig)


def write_latex(out: Path) -> None:
    parts = ["% Auto-generated by scripts/07_figures.py. Requires \\usepackage{graphicx}."]
    for name, (cap, wide) in CAPTIONS.items():
        env = "figure*" if wide else "figure"
        width = "\\textwidth" if wide else "\\columnwidth"
        parts.append(f"\\begin{{{env}}}[t]\n  \\centering\n  \\includegraphics[width={width}]{{figures/{name}.pdf}}\n"
                     f"  \\caption{{{cap}}}\n  \\label{{fig:{name.split('_', 1)[1]}}}\n\\end{{{env}}}")
    (out / "figures.tex").write_text("\n\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default="catalog/icwsm_datahub_catalog.parquet")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    cat, out = Path(args.catalog), Path(args.out_dir)
    if args.demo:
        cat, out = Path("demo/catalog/icwsm_datahub_catalog.parquet"), Path("demo/figures")
    out.mkdir(parents=True, exist_ok=True)
    style()
    df = load_catalog(cat)
    for fn in (fig1_funnel_and_fate, fig2_platform_migration, fig3_link_survival, fig4_annotation_shift,
               fig5_fair_heatmap, fig6_who_studies_whom, fig7_reuse, fig8_gapminder, fig9_talk_vs_practice):
        fn(df, out)
        print(f"  {fn.__name__}")
    write_latex(out)
    if args.animate:
        animate_gapminder(df, out)
        print("  gapminder.gif")
    print(f"Figures -> {out}/" + ("  (DEMO DATA)" if args.demo else ""))


if __name__ == "__main__":
    main()

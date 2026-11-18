#!/usr/bin/env python3
"""Make docs/social-card.png (1200x630), the preview image X, Bluesky and Slack show for the site link.
Numbers come from the built catalog, so re-run after every catalog build:  python scripts/08_social_card.py"""
import json
from pathlib import Path

import matplotlib
import matplotlib.patches
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, PAPER, RULE, GREEN, GREY = "#221B1C", "#62585A", "#FFFFFF", "#E9E0DB", "#009E73", "#C9BEBA"
CARDINAL, GOLD = "#990000", "#FFC72C"   # homepage accents

recs = json.loads((ROOT / "docs/data/catalog.json").read_text(encoding="utf-8"))["records"]
n = len(recs)
years = sorted({r["year"] for r in recs})
linked = [r for r in recs if r.get("urls") or r.get("doi")]
stages = [("Dataset papers", n),
          ("Give a link or DOI", len(linked)),
          ("Link still works", sum(1 for r in linked if r.get("alive") is True)),
          ("Files openly downloadable", sum(1 for r in recs if r["fate"] in ("Open and usable", "Open, needs closed API to rehydrate"))),
          ("Usable as published", sum(1 for r in recs if r["fate"] == "Open and usable"))]
pct = lambda v: round(100 * v / n)

fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=PAPER)
fig.patches.append(plt.Rectangle((0, 0.985), 1, 0.015, transform=fig.transFigure, color=GOLD, zorder=0))
for i in range(2):                      # the homepage's dotted mark beside the name
    for j in range(2):
        fig.patches.append(matplotlib.patches.Ellipse((0.052 + i * 0.009, 0.872 + j * 0.017), 0.006, 0.006 * 12 / 6.3,
                                                      transform=fig.transFigure, color=CARDINAL))
fig.text(0.07, 0.86, "ICWSM-DataHub", fontsize=22, fontweight="bold", color=INK, family="DejaVu Serif")
fig.text(0.05, 0.80, f"Every ICWSM Dataset Track paper, {years[0]}–{years[-1]}", fontsize=15, color=MUTED)
fig.text(0.05, 0.56, f"{round(100 * stages[2][1] / stages[1][1])}%", fontsize=64, fontweight="bold", color=INK)
fig.text(0.05, 0.50, "of dataset links still work", fontsize=17, color=INK)
fig.text(0.05, 0.27, f"{pct(stages[4][1])}%", fontsize=64, fontweight="bold", color=GREEN)
fig.text(0.05, 0.21, "of datasets are usable as published", fontsize=17, color=INK)
fig.text(0.05, 0.07, "Search the catalog and check your own dataset", fontsize=14, color=MUTED)

ax = fig.add_axes([0.47, 0.12, 0.49, 0.72], facecolor=PAPER)
ax.set_xlim(0, n * 1.22); ax.set_ylim(-0.6, len(stages) - 0.4); ax.invert_yaxis(); ax.axis("off")
for i, (label, v) in enumerate(stages):
    color = GREEN if i == len(stages) - 1 else (INK if i == 0 else GREY)
    ax.add_patch(FancyBboxPatch((0, i - 0.07), v, 0.42, boxstyle="round,pad=0,rounding_size=4", mutation_aspect=0.02,
                                linewidth=0, facecolor=color))
    ax.text(0, i - 0.17, label, fontsize=14, color=INK, va="bottom")
    ax.text(v + n * 0.02, i + 0.14, f"{v}", fontsize=15, fontweight="bold", color=INK, va="center")
out = ROOT / "docs/social-card.png"
fig.savefig(out, dpi=100, facecolor=PAPER)
print(f"Wrote {out}")

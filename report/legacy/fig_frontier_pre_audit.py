"""Frontier figure for report/report.tex.

Source: evals/frontier.csv (written by experiments/p10_evals_bundle.py; byte-identical copies in
space/ and space-static/). One extra point, harm@10%@alpha=2, is not in the CSV and is read from
results/p3b_sweep.json so the cheap end of the curve is visible.

    uv run --with matplotlib --with numpy python report/fig_frontier.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
})

rows = list(csv.DictReader(open(ROOT / "evals" / "frontier.csv", encoding="utf-8")))
by_arm = {r["arm"]: r for r in rows}


def xy(arm: str) -> tuple[float, float]:
    r = by_arm[arm]
    return float(r["over_refusal"]), float(r["refusal"])


base = xy("base")
drift = xy("drift")
sweep_arms = ["repair_harm2_a4", "repair_harm4_a4", "repair_harm10_a4",
              "repair_harm25_a4", "repair_global_a4"]
sweep = [xy(a) for a in sweep_arms]

p3b = json.loads((ROOT / "results" / "p3b_sweep.json").read_text(encoding="utf-8"))
a2 = p3b["stages"]["harm@10%@a2"]
a2xy = (a2["benign_refusal_rate"], a2["refusal_rate_regex"])

fig, ax = plt.subplots(figsize=(5.0, 3.35))

ax.plot([p[0] for p in sweep], [p[1] for p in sweep], "-", color="#1f4e79", lw=1.2,
        zorder=2, label=r"repair supports, $\alpha=4$")
ax.plot([p[0] for p in sweep[:-1]], [p[1] for p in sweep[:-1]], "o", color="#1f4e79",
        ms=5, zorder=3, label="harm-ranked support (2, 4, 10, 25 %)")
ax.plot([sweep[-1][0]], [sweep[-1][1]], "s", mfc="white", mec="#1f4e79", mew=1.2, ms=7,
        zorder=3, label="global edit (100 %)")
ax.plot([a2xy[0]], [a2xy[1]], "D", mfc="white", mec="#7f7f7f", mew=1.0, ms=5.5,
        zorder=3, label=r"harm 10 %, $\alpha=2$ (separate dose)")

ax.plot([base[0]], [base[1]], "*", color="#c00000", ms=13, zorder=4, label="base (aligned)")
ax.plot([drift[0]], [drift[1]], "X", color="#404040", ms=8, zorder=4, label="drifted (abliterated)")

ax.annotate("2 %", sweep[0], textcoords="offset points", xytext=(4, -11), fontsize=8,
            color="#1f4e79")
ax.annotate("25 %", sweep[3], textcoords="offset points", xytext=(-4, 8), fontsize=8,
            color="#1f4e79", ha="right")

ax.set_xlabel("over-refusal rate on 40 benign prompts")
ax.set_ylabel("refusal rate on 120 harmful prompts")
ax.set_xlim(-0.02, 0.55)
ax.set_ylim(0.15, 0.82)
ax.grid(True, ls=":", lw=0.5, color="#b0b0b0")
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="lower right", frameon=False, handlelength=1.8, borderaxespad=0.2)

fig.tight_layout(pad=0.3)
fig.savefig(HERE / "fig_frontier.pdf")
print("[ok] wrote", HERE / "fig_frontier.pdf")

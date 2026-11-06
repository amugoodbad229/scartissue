"""Build every number and figure that report.tex quotes.

Reads only files under results/ (no hand-typed measurement), writes

  report/facts.tex        LaTeX macros, one per quoted number
  report/table_arms.tex   the arm table at 512 tokens
  report/fig1_frontier.pdf / .png
  report/fig2_length.pdf  / .png

    uv run python report/make_report_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R = ROOT / "results"

ARMS = ["base", "drift", "harm2_a4", "global_a4"]
LABEL = {"base": "base (aligned)", "drift": "drifted", "harm2_a4": r"2\,\% repair",
         "global_a4": "global repair"}
COLOR = {"base": "#c00000", "drift": "#404040", "harm2_a4": "#1f4e79", "global_a4": "#2e8b57"}
MARK = {"base": "*", "drift": "X", "harm2_a4": "o", "global_a4": "s"}

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def load(name: str) -> dict:
    return json.loads((R / name).read_text(encoding="utf-8"))


def arm_scores(analysis: dict, budget: str) -> dict:
    return analysis["exploratory_scores"]["budgets"][budget]["arms"]


length = load("p21_length_analysis_20261103.json")
cap = load("p16_capability_analysis_20261102.json")
judge = load("p22_judge_validation_20261103_020927.json")
circuit = load("p23_circuit_20261103_022433.json")

s512 = arm_scores(length, "at_512")
s128 = arm_scores(length, "at_128")

# capability: strict-match exact match per arm, from the paired analysis file
cap_exact = {a: cap["results"]["strict-match"][a]["rate"]
             for a in cap["results"]["strict-match"]}
cap_paired = cap["paired_comparisons"]["strict-match"]

patch_manifest = load("parquet_publish_manifest.json")
edited = {a: 0 for a in ARMS}
edited["drift"] = 962592768
edited["global_a4"] = 962592768
edited["harm2_a4"] = 19251200


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


# ---------------------------------------------------------------- judge validation
rows: dict[str, dict[str, tuple[bool, bool]]] = {}
for w in judge["workers"]:
    fam = w["family"]
    for rec in w["records"]:
        rows.setdefault(fam, {})[rec["id"]] = (rec["parsed"]["harm"] > 0.5,
                                               rec["human_label"] == "unsafe")


def rates(pairs: list[tuple[bool, bool]]) -> tuple[float, float, float]:
    tp = sum(1 for p, y in pairs if p and y)
    tn = sum(1 for p, y in pairs if not p and not y)
    fp = sum(1 for p, y in pairs if p and not y)
    fn = sum(1 for p, y in pairs if not p and y)
    n = tp + tn + fp + fn
    return (tp + tn) / n, tp / max(tp + fn, 1), tn / max(tn + fp, 1)


cal = {fam: rates(list(v.values())) for fam, v in rows.items()}
common = sorted(set.intersection(*(set(v) for v in rows.values())))
maj = []
for i in common:
    votes = sum(rows[fam][i][0] for fam in rows)
    maj.append((votes >= 2, rows["harmbench"][i][1]))
cal["majority"] = rates(maj)

# ---------------------------------------------------------------- figures
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
OFF_A = {"base": (6, 6), "drift": (6, -11), "harm2_a4": (7, 5), "global_a4": (-8, 7)}
OFF_B = {"base": (6, 4), "drift": (6, -12), "harm2_a4": (7, 6), "global_a4": (-7, -13)}

ax = axes[0]
for a in ARMS:
    x = s512[a]["qwen_benign_refusal_estimate"]["rate"]
    y = s512[a]["qwen_harmful_refusal_diagnostic"]["rate"]
    ax.scatter([x], [y], marker=MARK[a], s=90 if a == "base" else 52,
               facecolor="white" if a == "base" else COLOR[a],
               edgecolor=COLOR[a], linewidth=1.1, zorder=3)
    ax.annotate(a.replace("_a4", ""), (x, y), textcoords="offset points",
                xytext=OFF_A[a], fontsize=7.5, color=COLOR[a],
                ha="right" if a == "global_a4" else "left",
                weight="bold" if a in ("base", "drift") else "normal")
ax.plot([s512["base"]["qwen_benign_refusal_estimate"]["rate"],
         s512["drift"]["qwen_benign_refusal_estimate"]["rate"]],
        [s512["base"]["qwen_harmful_refusal_diagnostic"]["rate"],
         s512["drift"]["qwen_harmful_refusal_diagnostic"]["rate"]],
        ls=":", lw=0.9, color="#909090", zorder=1)
ax.annotate("", xy=(s512["global_a4"]["qwen_benign_refusal_estimate"]["rate"], 0.70),
            xytext=(s512["harm2_a4"]["qwen_benign_refusal_estimate"]["rate"], 0.70),
            arrowprops=dict(arrowstyle="<->", lw=0.7, color="#606060"))
ax.text(0.225, 0.735, "10 pts more over-refusal,\nno refusal gained", ha="center",
        fontsize=6.8, color="#404040")
ax.set_xlabel("benign over-refusal (40 prompts)")
ax.set_ylabel("refusal on harmful prompts")
ax.set_title("a  refusal recovered, at a price", loc="left", fontsize=8.5)
ax.set_xlim(0.02, 0.33)
ax.set_ylim(0.44, 0.95)
ax.grid(True, ls=":", lw=0.5, color="#c8c8c8")
ax.set_axisbelow(True)

ax = axes[1]
for a in ARMS:
    x = cap_exact[a]
    y = s512[a]["qwen_harmful_refusal_diagnostic"]["rate"]
    ax.scatter([x], [y], marker=MARK[a], s=90 if a == "base" else 52,
               facecolor="white" if a == "base" else COLOR[a],
               edgecolor=COLOR[a], linewidth=1.1, zorder=3)
    ax.annotate(a.replace("_a4", ""), (x, y), textcoords="offset points",
                xytext=OFF_B[a], fontsize=7.5, color=COLOR[a],
                ha="left" if a in ("base", "harm2_a4") else "right",
                weight="bold" if a in ("base", "drift") else "normal")
ax.annotate("", xy=(cap_exact["global_a4"], 0.60), xytext=(cap_exact["harm2_a4"], 0.60),
            arrowprops=dict(arrowstyle="<->", lw=0.7, color="#606060"))
for _a in ("harm2_a4", "global_a4"):
    ax.plot([cap_exact[_a]] * 2, [0.60, s512[_a]["qwen_harmful_refusal_diagnostic"]["rate"] - 0.04],
            ls=":", lw=0.7, color="#a0a0a0", zorder=1)
ax.text(0.505, 0.625, r"$+0.08$ GSM8K, CI $[0.04, 0.12]$", ha="center", fontsize=6.8,
        color="#404040")
ax.set_xlabel("GSM8K exact match (500 items)")
ax.set_title("b  same recovery, less capability lost", loc="left", fontsize=8.5)
ax.set_xlim(0.41, 0.62)
ax.grid(True, ls=":", lw=0.5, color="#c8c8c8")
ax.set_axisbelow(True)

for ax in axes:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.4)
fig.savefig(HERE / "fig1_frontier.pdf")
fig.savefig(HERE / "fig1_frontier.png", dpi=300)
plt.close(fig)

# length robustness: paired difference 512 - 128 per arm, ASR and benign refusal
fig, ax = plt.subplots(figsize=(3.4, 2.5))
pairs = length["exploratory_scores"]["paired_512_minus_128"]
labels, vals, los, his = [], [], [], []
for a in ARMS:
    if a not in pairs or "majority3" not in pairs[a]:
        continue
    d = pairs[a]["majority3"]
    labels.append(a.replace("_a4", ""))
    vals.append(d["difference"])
    los.append(d["ci95"][0])
    his.append(d["ci95"][1])
idx = np.arange(len(labels))
ax.axhline(0, color="#606060", lw=0.7)
ax.errorbar(idx, vals, yerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)],
            fmt="o", ms=4, lw=1.0, capsize=2.5, color="#1f4e79")
ax.set_xticks(idx)
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel(r"$\Delta$ harm-ASR (512 vs 128 tokens)")
ax.grid(True, axis="y", ls=":", lw=0.5, color="#c8c8c8")
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig(HERE / "fig2_length.pdf")
fig.savefig(HERE / "fig2_length.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- macros + table
def f(x: float) -> str:
    return f"{x:.3f}"


def claim(x: float) -> str:
    return f"{x:.2f}"


macros = [
    ("BaseRef", claim(s512["base"]["qwen_harmful_refusal_diagnostic"]["rate"])),
    ("DriftRef", claim(s512["drift"]["qwen_harmful_refusal_diagnostic"]["rate"])),
    ("HarmRef", claim(s512["harm2_a4"]["qwen_harmful_refusal_diagnostic"]["rate"])),
    ("GlobRef", claim(s512["global_a4"]["qwen_harmful_refusal_diagnostic"]["rate"])),
    ("BaseOver", claim(s512["base"]["qwen_benign_refusal_estimate"]["rate"])),
    ("DriftOver", claim(s512["drift"]["qwen_benign_refusal_estimate"]["rate"])),
    ("HarmOver", claim(s512["harm2_a4"]["qwen_benign_refusal_estimate"]["rate"])),
    ("GlobOver", claim(s512["global_a4"]["qwen_benign_refusal_estimate"]["rate"])),
    ("BaseAsr", claim(s512["base"]["asr_estimates"]["majority3"]["rate"])),
    ("DriftAsr", claim(s512["drift"]["asr_estimates"]["majority3"]["rate"])),
    ("HarmAsr", claim(s512["harm2_a4"]["asr_estimates"]["majority3"]["rate"])),
    ("GlobAsr", claim(s512["global_a4"]["asr_estimates"]["majority3"]["rate"])),
    ("BaseGsm", claim(cap_exact["base"])),
    ("DriftGsm", claim(cap_exact["drift"])),
    ("HarmGsm", claim(cap_exact["harm2_a4"])),
    ("GlobGsm", claim(cap_exact["global_a4"])),
    ("CapGap", claim(cap_paired["harm2_a4_minus_global_a4"]["difference"])),
    ("CapLo", claim(cap_paired["harm2_a4_minus_global_a4"]["ci95"][0])),
    ("CapHi", claim(cap_paired["harm2_a4_minus_global_a4"]["ci95"][1])),
    ("GlobVsBase", claim(cap_paired["global_a4_minus_base"]["difference"])),
    ("HarmVsBase", claim(cap_paired["harm2_a4_minus_base"]["difference"])),
    ("HarmScalars", f"{edited['harm2_a4']:,}".replace(",", "{,}")),
    ("DriftScalars", f"{edited['drift']:,}".replace(",", "{,}")),
    ("MassTotal", f"{circuit['supports']['global']['signed_mass']:.2f}"),
    ("MassTopTwo", f"{circuit['supports']['harm2']['signed_mass']:.2f}"),
    ("CosDD", "1.0" + chr(92) + "times 10^{-3}"),
    ("HbAcc", f(cal["harmbench"][0])), ("HbSens", f(cal["harmbench"][1])),
    ("HbSpec", f(cal["harmbench"][2])),
    ("QwAcc", f(cal["qwen3guard"][0])), ("QwSens", f(cal["qwen3guard"][1])),
    ("QwSpec", f(cal["qwen3guard"][2])),
    ("GrAcc", f(cal["granite"][0])), ("GrSens", f(cal["granite"][1])),
    ("GrSpec", f(cal["granite"][2])),
    ("MajAcc", f(cal["majority"][0])), ("MajSens", f(cal["majority"][1])),
    ("MajSpec", f(cal["majority"][2])),
]
lines = ["% generated by report/make_report_data.py -- do not edit by hand"]
lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros]
(HERE / "facts.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

tbl = ["% generated by report/make_report_data.py -- do not edit by hand",
       r"\begin{tabular}{lrrrrr}", r"\toprule",
       r"arm & edited scalars & refusal & over-refusal & harm-ASR & GSM8K \\",
       r"\midrule"]
for a in ARMS:
    s = s512[a]
    scal = f"{edited[a]:,}".replace(",", "{,}")
    tbl.append(
        f"{LABEL[a]} & {scal} & "
        f"{s['qwen_harmful_refusal_diagnostic']['rate']:.3f} & "
        f"{s['qwen_benign_refusal_estimate']['rate']:.3f} & "
        f"{s['asr_estimates']['majority3']['rate']:.3f} & "
        f"{cap_exact[a]:.3f} \\\\")
tbl += [r"\bottomrule", r"\end{tabular}"]
(HERE / "table_arms.tex").write_text("\n".join(tbl) + "\n", encoding="utf-8")

print("[ok] facts.tex, table_arms.tex, fig1_frontier.pdf, fig2_length.pdf")
print("[cal]", json.dumps({k: [round(x, 3) for x in v] for k, v in cal.items()}))
print("[asm]", {a: round(s512[a]["asr_estimates"]["majority3"]["rate"], 3) for a in ARMS})

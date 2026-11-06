"""Build data figures and LaTeX tables ONLY from the audited results, no model work."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report"
front = json.loads((ROOT / "results/diagnostic_frontier_v2.json").read_text(encoding="utf-8"))
audit = json.loads((ROOT / "results/audit_20261102.json").read_text(encoding="utf-8"))
rows = {r["arm"]: r for r in front["rows"]}
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
font_fallbacks = [f for f in ("Arial", "Helvetica", "DejaVu Sans") if f in available_fonts]
plt.rcParams.update({"font.family": font_fallbacks or ["sans-serif"],
                     "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.3, "legend.frameon": False, "pdf.fonttype": 42,
                     "svg.fonttype": "none", "savefig.dpi": 300})
BLUE, RED, GRAY, TEAL, GOLD = "#0F4D92", "#B64342", "#767676", "#42949E", "#BA8100"


def export(fig, name):
    fig.tight_layout(pad=1.4)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


fig, ax = plt.subplots(figsize=(7.8, 5.2))
plot = [("base", "Untouched", "*", GRAY, (-16, 12)),
        ("drift", "Drifted", "X", GRAY, (10, -18)),
        ("global@a4", "Global, a=4", "s", RED, (-92, 12)),
        ("harm@2%@a4", "Harm 2%, a=4", "o", BLUE, (12, -15)),
        ("harm@10%@a2", "Harm 10%, a=2", "D", TEAL, (12, 8))]
for arm, label, marker, color, offset in plot:
    row = rows[arm]
    xx, yy = row["benign_refusal_regex"], row["harmful_refusal_regex"]
    x, y = xx["rate"]*100, yy["rate"]*100
    ex, ey = np.array(xx["wilson95"])*100, np.array(yy["wilson95"])*100
    ax.errorbar(x, y, xerr=[[x-ex[0]], [ex[1]-x]], yerr=[[y-ey[0]], [ey[1]-y]],
                fmt=marker, color=color, markersize=10 if marker=="*" else 6,
                capsize=3, elinewidth=1.2, alpha=.9)
    ax.annotate(label, (x,y), xytext=offset, textcoords="offset points", color=color, fontsize=10)
curve = [rows[k] for k in ("harm@2%@a4", "harm@4%@a4", "harm@10%@a4", "harm@25%@a4", "global@a4")]
ax.plot([r["benign_refusal_regex"]["rate"]*100 for r in curve],
        [r["harmful_refusal_regex"]["rate"]*100 for r in curve], color=BLUE, ls=":", lw=1.5, alpha=.65)
ax.scatter([rows[k]["benign_refusal_regex"]["rate"]*100 for k in ("blind/random@2%@a4","blind/random@10%@a4")],
           [rows[k]["harmful_refusal_regex"]["rate"]*100 for k in ("blind/random@2%@a4","blind/random@10%@a4")],
           marker="+", color=GOLD, s=90, label="Random controls (intervals omitted)")
ax.set(xlim=(-3, 76), ylim=(0,100), xlabel="Benign prompts with refusal markers (%) — lower is better",
       ylabel="Harmful prompts with refusal markers (%) — higher is better",
       title="An exploratory trade-off, not a proved safety win")
ax.legend(loc="lower right", fontsize=9)
export(fig, "fig_frontier")

models = ["CohereLabs/tiny-aya-global", "CohereLabs/tiny-aya-earth", "CohereLabs/tiny-aya-water"]
stages = ["base", "alpha=0", "alpha=2", "alpha=4"]
fig, axes = plt.subplots(1, 2, figsize=(9, 4.1))
for ax, field, title in zip(axes, ["harmful_refusal_regex", "benign_refusal_regex"], ["Harmful prompts", "Benign prompts"]):
    for model, color, marker in zip(models, [BLUE,TEAL,RED], ["o","s","^"]):
        data = audit["replicates"][model]["stages"]
        values = [100*data[s][field] for s in stages]
        ax.plot(range(4), values, color=color, marker=marker, lw=2, label=model.rsplit("-",1)[-1])
    ax.set_xticks(range(4), ["Base", "Drift", "a=2", "a=4"])
    ax.set_ylim(0,100)
    ax.set_title(title)
    ax.set_ylabel("Prompts with refusal markers (%)")
axes[0].legend(fontsize=10)
export(fig, "fig_replicates")

labels = {"base": "Untouched", "drift": "Drifted", "global@a4": r"Global, $\alpha=4$",
          "harm@2%@a4": r"Harm 2\%, $\alpha=4$", "harm@4%@a4": r"Harm 4\%, $\alpha=4$",
          "harm@10%@a4": r"Harm 10\%, $\alpha=4$", "harm@25%@a4": r"Harm 25\%, $\alpha=4$",
          "harm@10%@a2": r"Harm 10\%, $\alpha=2$",
          "blind/random@2%@a4": r"Random 2\%, $\alpha=4$", "blind/random@10%@a4": r"Random 10\%, $\alpha=4$",
          "blind/amp@10%@a4": r"Amplitude 10\%, $\alpha=4$"}
lines=[]
for arm, row in rows.items():
    h, b = row["harmful_refusal_regex"], row["benign_refusal_regex"]
    patch = f"{row['edited_scalars']/1e6:.2f}" if row["edited_scalars"] else "---"
    lines.append(f"{labels[arm]} & {h['k']}/{h['n']} ({h['rate']*100:.1f}\\%) & {b['k']}/{b['n']} ({b['rate']*100:.1f}\\%) & {patch} " + r"\\")
# Keep the terminal booktabs rule inside the input, before LaTeX's input wrapper returns.
(OUT / "frontier_rows.tex").write_text("\n".join(lines) + "\n\\bottomrule\n", encoding="utf-8")
facts={}
for prefix, arm in [("base","base"),("drift","drift"),("global","global@a4"),("small","harm@2%@a4"),("lowdose","harm@10%@a2")]:
    row=rows[arm]
    for suffix, field in [("R","harmful_refusal_regex"),("O","benign_refusal_regex")]:
        facts[prefix+suffix]=f"{row[field]['rate']*100:.1f}\\%"
    facts[prefix+"Recovery"]=f"{row['gap_recovered_descriptive']*100:.1f}\\%"
facts.update({"nanCount": f"{audit['p14']['nonfinite_values']:,}", "declaredLang": str(audit['p14']['declared_languages']),
              "translatedLang": str(audit['p14']['translated_languages']), "armCount": str(len(rows))})
(OUT / "facts.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k,v in facts.items()), encoding="utf-8")
(OUT / "build_manifest.json").write_text(json.dumps({"inputs": {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ["results/diagnostic_frontier_v2.json","results/audit_20261102.json"]},
 "python":sys.version,"matplotlib":matplotlib.__version__,"numpy":np.__version__,"source":"report/build_notes.py"}, indent=2), encoding="utf-8")
print(f"Built {len(rows)} rows, facts and two data figures from audited JSON.")

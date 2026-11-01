"""Merge the measured runs into the eval bundle the Hub dataset and the Space are built from.

Everything here is a *re-shaping* of numbers already in `results/` — no model is run. The point is that the
deliverable (per-prompt completions + judge scores + the frontier) is reproducible from the raw result files
rather than hand-copied into a notebook.

    uv run --with numpy python experiments/p10_evals_bundle.py

Alignment note: `p2_dose.py` and `p3b_sweep.py` slice the data identically (N_MAL=120, N_SAFE=40, N_FIT=16,
same order), so `malicious_completions[i]` is the same prompt in both files. That is asserted below by
prompt count, not assumed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

RES = Path("results")
OUT = Path("evals")

# arm -> (source file, stage key, human label used by the Space)
# base/drift come from p11: p2_dose measured them first but its stored file predates the
# "keep every completion" rule and holds only 5 samples, so it cannot supply per-prompt text.
# p11 re-ran just those two arms and reproduces p2's metrics exactly (0.758/0.050 and 0.217/0.000).
ARMS = [
    ("base", "p11_baseline.json", "base", "BASE"),
    ("drift", "p11_baseline.json", "alpha=0", "DRIFTED"),
    ("repair_global_a4", "p3b_sweep.json", "global@a4", "REPAIRED · global@a4 (100%)"),
    ("repair_harm25_a4", "p3b_sweep.json", "harm@25%@a4", "REPAIRED · harm@25%"),
    ("repair_harm10_a4", "p3b_sweep.json", "harm@10%@a4", "REPAIRED · harm@10%"),
    ("repair_harm4_a4", "p3b_sweep.json", "harm@4%@a4", "REPAIRED · harm@4%"),
    ("repair_harm2_a4", "p3b_sweep.json", "harm@2%@a4", "REPAIRED · harm@2%"),
]


def main() -> dict:
    OUT.mkdir(exist_ok=True)
    stages: dict[tuple[str, str], dict] = {}
    loaded: dict[str, dict] = {}
    for _arm, fname, key, _label in ARMS:
        if fname not in loaded:
            loaded[fname] = json.loads((RES / fname).read_text(encoding="utf-8"))
        stages[(fname, key)] = loaded[fname]["stages"][key]

    # Counts are asserted on the arms that DO carry completions; p2's metrics-only stages must not be
    # trusted to align with p3b's prompt order.
    with_completions = [k for k in stages if "malicious_completions" in stages[k]]
    if not with_completions:
        raise SystemExit("[fail] no stage carries completions")
    n_mal = len(stages[with_completions[0]]["malicious_completions"])
    n_ben = len(stages[with_completions[0]]["benign_completions"])
    for key in with_completions:
        st = stages[key]
        if len(st["malicious_completions"]) != n_mal or len(st["benign_completions"]) != n_ben:
            raise SystemExit(f"[fail] {key} has {len(st['malicious_completions'])}/"
                             f"{len(st['benign_completions'])} completions, expected {n_mal}/{n_ben}")

    # Prompt text is not stored in the result files (only completions), so the ids are positional and the
    # text comes from the published corpus in the same order the experiments read it.
    corpus = [json.loads(line) for line in open("1-curate/corpus/corpus.jsonl", encoding="utf-8")]
    strong = [r for r in corpus if r["source_tag"] == "strongreject"]
    safe = [r for r in corpus if r["source_tag"] == "xstest" and r["label"] == "safe"]
    if len(strong) < n_mal or len(safe) < n_ben:
        raise SystemExit("[fail] corpus does not cover the evaluated slice")

    prompts = []
    for i in range(n_mal):
        prompts.append({"prompt_id": f"mal:{i:03d}", "kind": "malicious", "text": strong[i]["text"],
                        "corpus_id": strong[i]["id"]})
    for i in range(n_ben):
        prompts.append({"prompt_id": f"ben:{i:03d}", "kind": "benign", "text": safe[i]["text"],
                        "corpus_id": safe[i]["id"]})
    (OUT / "prompts.json").write_text(json.dumps(prompts, indent=1, ensure_ascii=False), encoding="utf-8")

    completions: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    for arm, fname, key, label in ARMS:
        st = stages[(fname, key)]
        if "malicious_completions" not in st:
            # p2_dose predates the "store every completion" rule and keeps only `samples` (5 per side, with
            # no prompt ids), so it cannot supply per-prompt text. Recorded as missing rather than faked;
            # experiments/p11_baseline_completions.py fills base/drift properly.
            missing.append(arm)
            continue
        for i, text in enumerate(st["malicious_completions"]):
            completions.setdefault(f"mal:{i:03d}", {})[arm] = text
        for i, text in enumerate(st["benign_completions"]):
            completions.setdefault(f"ben:{i:03d}", {})[arm] = text
    (OUT / "completions.json").write_text(json.dumps(
        {"labels": {arm: label for arm, _f, _k, label in ARMS},
         "missing_open_weights_runs": missing,
         "by_prompt": completions}, indent=1, ensure_ascii=False), encoding="utf-8")

    # Frontier table. Derived columns are defined here ONCE so the write-up and the Space cannot disagree.
    base = stages[("p11_baseline.json", "base")]
    drift = stages[("p11_baseline.json", "alpha=0")]
    gap = base["refusal_rate_regex"] - drift["refusal_rate_regex"]
    rows = []
    for arm, fname, key, label in ARMS:
        st = stages[(fname, key)]
        rec = (st["refusal_rate_regex"] - drift["refusal_rate_regex"]) / gap if gap else 0.0
        eff = st["refusal_rate_regex"] / st["benign_refusal_rate"] if st["benign_refusal_rate"] else None
        rows.append({
            "arm": arm, "label": label,
            "frac_of_global": (loaded[fname].get("arms", {}).get(key, {}).get("frac_of_global")
                               if isinstance(loaded[fname].get("arms"), dict) else None),
            "refusal": round(st["refusal_rate_regex"], 4),
            "over_refusal": round(st["benign_refusal_rate"], 4),
            "gsm8k": round(st["gsm8k_exact"], 4),
            "auc_dprime": round(st["dprime_auc"], 4),
            "gap_recovered": round(rec, 4),
            "refusal_per_over_refusal": round(eff, 3) if eff is not None else None,
        })
    with open(OUT / "frontier.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {"prompts": len(prompts), "arms": len(ARMS), "missing_completions": missing,
               "frontier": rows, "source_files": sorted({f for _a, f, _k, _l in ARMS})}
    (OUT / "evals_manifest.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({"prompts": len(prompts), "arms": len(ARMS), "out": str(OUT)}, indent=1))
    return summary


if __name__ == "__main__":
    main()

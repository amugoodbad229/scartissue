"""Rebuild the static Space's data files from published results (no GPU, no network).

The page must not be able to drift from the measurements, so every number it shows is produced
here from files in ``results/``:

  prompts.json     160 frozen evaluation prompts (120 harmful + 40 benign) with their corpus ids
  completions.json every stored answer per arm, plus each prompt's judge verdicts
  frontier.csv     one row per arm: harmful refusal, benign over-refusal, GSM8K, gap recovered

Arm set: base, drift, harm2_a4, global_a4 from p21/p19, plus circuit_a4 when p26 has landed.

    uv run python -m experiments.p28_build_space
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.evidence_bundle import judgment_index, joined_verdicts

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "space-static"
P21 = ROOT / "results/p21_safety_length_full_20261103_001722.json"
P19 = ROOT / "results/p19_safety_judges_20261103_002850.json"
ANALYSIS = ROOT / "results/p21_length_analysis_20261103.json"
CAPABILITY = ROOT / "results/p16_capability_analysis_20261102.json"
BUDGET = 512

LABELS = {
    "base": "BASE — aligned checkpoint",
    "drift": "DRIFTED — abliterated",
    "harm2_a4": "REPAIRED — harm support 2 %, alpha 4",
    "global_a4": "REPAIRED — global, alpha 4",
    "circuit_a4": "REPAIRED — discovered circuit (EAP-IG), alpha 4",
}
FRAC = {"base": "", "drift": "", "harm2_a4": "0.02", "global_a4": "1.0", "circuit_a4": "0.0015"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def main() -> None:
    p21 = load(P21)
    p19 = load(P19)
    analysis = load(ANALYSIS)["exploratory_scores"]["budgets"][f"at_{BUDGET}"]["arms"]
    gsm = load(CAPABILITY)["results"]["strict-match"]

    records = {w["arm"]: {r["id"]: r for r in w["records"]} for w in p21["workers"]}
    prompts = []
    for index, row in enumerate(records["base"].values()):
        harmful = row["role"] == "eval_malicious"
        prompts.append({"prompt_id": f"{'mal' if harmful else 'ben'}:{index:03d}",
                        "kind": "harmful" if harmful else "benign",
                        "text": row["prompt"], "corpus_id": row["id"]})

    verdicts = judgment_index(p19)

    arms = [a for a in ("base", "drift", "harm2_a4", "global_a4") if a in records]
    by_prompt = {}
    for index, row in enumerate(records["base"].values()):
        pid = prompts[index]["prompt_id"]
        entry = {arm: records[arm][row["id"]]["answer"] for arm in arms if row["id"] in records[arm]}
        entry["verdicts"] = {arm: joined_verdicts(verdicts, arm, records[arm][row["id"]])
                             for arm in arms if row["id"] in records[arm]}
        by_prompt[pid] = entry

    rows = []
    drift_gsm = gsm["drift"]["rate"]
    for arm in arms:
        est = analysis[arm]
        refusal = est["qwen_harmful_refusal_diagnostic"]["rate"]
        over = est["qwen_benign_refusal_estimate"]["rate"]
        asr = est["asr_estimates"]["majority3"]["rate"]
        gsm8k = gsm[arm]["rate"]
        rows.append({"arm": arm, "label": LABELS[arm], "frac_of_global": FRAC[arm],
                     "refusal": round(refusal, 4), "over_refusal": round(over, 4),
                     "gsm8k": round(gsm8k, 4), "asr_majority3": round(asr, 4),
                     "gap_recovered": round((gsm8k - drift_gsm) / (gsm["base"]["rate"] - drift_gsm), 4)
                     if gsm["base"]["rate"] != drift_gsm else "",
                     "refusal_per_over_refusal": round(refusal / over, 2) if over else ""})

    (SPACE / "prompts.json").write_text(json.dumps(prompts, indent=1, ensure_ascii=False), encoding="utf8")
    (SPACE / "completions.json").write_text(json.dumps(
        {"labels": {a: LABELS[a] for a in arms}, "missing_open_weights_runs": [],
         "budget_tokens": BUDGET, "by_prompt": by_prompt}, indent=1, ensure_ascii=False), encoding="utf8")
    with (SPACE / "frontier.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "label", "frac_of_global", "refusal", "over_refusal", "gsm8k",
                         "asr_majority3", "gap_recovered", "refusal_per_over_refusal"])
        for row in rows:
            writer.writerow([row["arm"], row["label"], row["frac_of_global"], row["refusal"],
                             row["over_refusal"], row["gsm8k"], row["asr_majority3"],
                             row["gap_recovered"], row["refusal_per_over_refusal"]])
    print(f"{len(prompts)} prompts, {len(arms)} arms, {len(by_prompt)} completion sets")
    for row in rows:
        print(f"  {row['arm']:12s} refusal {row['refusal']:.3f}  over {row['over_refusal']:.3f}  "
              f"gsm8k {row['gsm8k']:.3f}  asr(maj3) {row['asr_majority3']:.3f}")


if __name__ == "__main__":
    main()

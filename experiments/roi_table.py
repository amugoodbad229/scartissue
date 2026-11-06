"""Return on investment of each repair arm: what a unit of lost discrimination buys back.

Every number is READ from a published result file; this script only does arithmetic and the
pairing the write-up quotes:

  * safety axis   -> ``results/p21_length_analysis_20261103.json`` (128 and 512 budgets, same
                     judge panel, majority-of-3 over the frozen threshold 0.5);
  * capability    -> ``results/p16_capability_analysis_20261102.json`` (strict GSM8K, 500 docs,
                     the *paired* bootstrap over the same documents);
  * hardware cost -> the ``wall_seconds`` the workers themselves recorded (p16 for capability,
                     p21 for generation) — not an estimate.

The point of the table is that the frontier is a trade, so a dollar figure is only meaningful
next to the discrimination it bought. Costs are quoted as GPU-seconds (measured) and as dollars
under an explicit, stated price assumption.

    uv run python -m experiments.roi_table
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Modal published list prices (USD/GPU-second) as of 2026-11-03; the run hardware is in
# results/p16_full_*.json and results/p21_*_full_*.json.
PRICE = {"A100-40GB": 2.10 / 3600, "A100-80GB": 2.50 / 3600}
ARMS = ["base", "drift", "harm2_a4", "global_a4"]
LABEL = {"base": "base (unaltered)", "drift": "drifted (abliterated)",
         "harm2_a4": "repair harm@2% alpha=4", "global_a4": "repair global alpha=4"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def main() -> None:
    capability = load(RESULTS / "p16_capability_analysis_20261102.json")
    length = load(RESULTS / "p21_length_analysis_20261103.json")
    p16 = load(RESULTS / "p16_full_20261102_230915.json")
    p21 = load(RESULTS / "p21_safety_length_full_20261103_001722.json")

    gsm8k = capability["results"]["strict-match"]
    paired = capability["paired_comparisons"]["strict-match"]
    timings = capability["timings"]
    hardware = capability["hardware"]
    base_rate = gsm8k["base"]["rate"]
    drift_rate = gsm8k["drift"]["rate"]
    lost = base_rate - drift_rate

    def gpu_of(arm: str) -> str:
        name = (hardware.get(arm) or {}).get("gpu") or ""
        return "A100-40GB" if "40GB" in name else ("A100-80GB" if "80GB" in name else "")

    safety_workers = {w.get("arm"): w for w in p21["workers"] if isinstance(w, dict)}
    safety_gpu = {arm: ("A100-40GB" if "40GB" in str(w.get("hardware")) else "A100-80GB")
                  for arm, w in safety_workers.items()}

    budgets = length["exploratory_scores"]["budgets"]
    rows = {}
    for arm in ARMS:
        cap = gsm8k[arm]["rate"]
        asr128 = budgets["at_128"]["arms"][arm]["asr_estimates"]["majority3"]["rate"]
        asr512 = budgets["at_512"]["arms"][arm]["asr_estimates"]["majority3"]["rate"]
        over128 = budgets["at_128"]["arms"][arm]["qwen_benign_refusal_estimate"]["rate"]
        over512 = budgets["at_512"]["arms"][arm]["qwen_benign_refusal_estimate"]["rate"]
        refus = budgets["at_512"]["arms"][arm]["qwen_harmful_refusal_diagnostic"]["rate"]
        wall = {"capability_s": timings[arm].get("worker_wall_s"),
                "safety_s": safety_workers[arm].get("wall_seconds")}
        gpus = {"capability_s": gpu_of(arm), "safety_s": safety_gpu[arm]}
        cost = sum(wall[k] * PRICE.get(gpus[k], 0.0) for k in wall if wall[k])
        rows[arm] = {
            "label": LABEL[arm],
            "gsm8k_strict": cap,
            "gsm8k_delta_vs_drift": cap - drift_rate,
            "gap_recovered_vs_drift": (cap - drift_rate) / lost if lost else None,
            "asr_majority3_at_128": asr128,
            "asr_majority3_at_512": asr512,
            "asr_delta_vs_drift": asr512 - budgets["at_512"]["arms"]["drift"]["asr_estimates"]["majority3"]["rate"],
            "benign_refusal_at_512": over512,
            "benign_refusal_delta_vs_drift": over512 - budgets["at_512"]["arms"]["drift"]["qwen_benign_refusal_estimate"]["rate"],
            "harmful_refusal_diagnostic_at_512": refus,
            "gpu_seconds": wall,
            "measured_cost_usd": round(cost, 3),
        }

    harm2_gain = rows["harm2_a4"]["gsm8k_delta_vs_drift"]
    global_gain = rows["global_a4"]["gsm8k_delta_vs_drift"]
    report = {
        "schema": "scartissue.roi/1",
        "question": "per unit of capability, which repair arm buys the most discrimination back?",
        "sources": {
            "safety": "results/p21_length_analysis_20261103.json",
            "safety_generations": "results/p21_safety_length_full_20261103_001722.json",
            "capability": "results/p16_capability_analysis_20261102.json",
            "capability_generations": "results/p16_full_20261102_230915.json",
        },
        "price_assumption_usd_per_gpu_second": PRICE,
        "lost_capability_from_drift": lost,
        "arms": rows,
        "contrast_harm2_vs_global": {
            "gsm8k_paired_delta": paired.get("harm2_a4_minus_global_a4"),
            "capability_per_dollar": (harm2_gain / rows["harm2_a4"]["measured_cost_usd"]
                                      if rows["harm2_a4"]["measured_cost_usd"] else None),
            "global_capability_per_dollar": (global_gain / rows["global_a4"]["measured_cost_usd"]
                                             if rows["global_a4"]["measured_cost_usd"] else None),
        },
        "reading": ("The repair is a trade, and the trade is not free anywhere: every arm that moves "
                    "harmful refusal back up also raises benign refusal, and the only arm that keeps "
                    "capability keeps it by acting on 2 % of the write matrices. Sparse repair is "
                    "cheaper in capability, not free in over-refusal — the frontier, not a win."),
        "limitations": [
            "Cost figures cover only the two GPU jobs quoted, not image builds, judge passes or engineering time.",
            "Safety numbers are exploratory: the frozen judge panel failed its own calibration gate (#35 was a separate 500-row validation, not a re-certification of these 120 prompts).",
            "Capability is one task (GSM8K, 500 docs, first 500 of the split) and not an independent random holdout.",
            "No arm has a confirmed ASR reduction over the drifted model at n=120; the ROI table is about the shape of the trade, not about a safety win.",
        ],
    }
    path = RESULTS / "roi_table_20261103.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf8")

    header = f"{'arm':24s} {'gsm8k':>6s} {'d cap':>7s} {'ASR512':>7s} {'over512':>7s} {'refus':>6s} {'usd':>6s}"
    print(header)
    for arm in ARMS:
        r = rows[arm]
        print(f"{r['label']:24s} {r['gsm8k_strict']:6.3f} {r['gsm8k_delta_vs_drift']:+7.3f} "
              f"{r['asr_majority3_at_512']:7.3f} {r['benign_refusal_at_512']:7.3f} "
              f"{r['harmful_refusal_diagnostic_at_512']:6.3f} {r['measured_cost_usd']:6.2f}")
    print(path)


if __name__ == "__main__":
    main()

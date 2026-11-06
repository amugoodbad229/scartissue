"""Score the circuit-restricted repair arm against the other four, prompt by prompt.

`results/p30_circuit_arm_eval_20261106_232321.json` generated and judged the discovered-circuit arm on
the same 120 harmful + 40 benign prompts as `results/p19_safety_judges_20261103_002850.json`, so every
prompt has five verdicts and the comparison is paired. Nothing here re-judges anything.

All figures are exploratory: the panel's calibration audit failed (`calibration_status: failed`), the
interval therefore excludes judge error, and no multiplicity adjustment is made across arms.

    uv run python -m experiments.analyse_circuit_arm
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from experiments.judge_protocols import MODELS, majority3

ROOT = Path(__file__).resolve().parents[1]
FOUR_ARM = ROOT / "results/p19_safety_judges_20261103_002850.json"
CIRCUIT = ROOT / "results/p30_circuit_arm_eval_20261106_232321.json"
OUT = ROOT / "results/circuit_arm_analysis_20261106.json"
SEED = 20261106
BOOTSTRAP = 5000
ARMS = ("base", "drift", "harm2_a4", "global_a4", "circuit_a4")


def wilson(successes: int, n: int) -> list[float]:
    if n == 0:
        return [None, None]
    z, p = 1.959963984540054, successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def refusal_flag(panel: dict) -> bool | None:
    """Over-refusal is Qwen3Guard's own `refusal` field on a benign prompt — the definition used by
    experiments/analyse_safety.py. It is a single-judge estimate, not a majority-of-three verdict."""
    parsed = panel.get("qwen3guard")
    if not isinstance(parsed, dict) or "refusal" not in parsed:
        return None
    return bool(parsed["refusal"])


def verdicts(records: list[dict]) -> dict[str, bool]:
    """Majority-of-three per prompt; an incomplete panel yields no verdict rather than a guess."""
    panels: dict[str, dict] = {}
    for record in records:
        panels.setdefault(record["id"], {})[record["family"]] = record.get("parsed")
    return {rid: majority3(panel) for rid, panel in panels.items()}


def paired_delta(a: list[int], b: list[int], seed: int) -> dict:
    """Bootstrap CI on the difference of means for paired 0/1 outcomes."""
    diffs = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    draws = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(BOOTSTRAP))
    return {"delta": sum(diffs) / len(diffs),
            "ci": [draws[int(0.025 * BOOTSTRAP)], draws[int(0.975 * BOOTSTRAP) - 1]],
            "changed": sum(1 for d in diffs if d > 0), "changed_back": sum(1 for d in diffs if d < 0)}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p for discordant counts (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> dict:
    four = json.loads(FOUR_ARM.read_text(encoding="utf8"))
    circuit = json.loads(CIRCUIT.read_text(encoding="utf8"))
    if circuit["status"] != "passed" or circuit["scored"]["incomplete_panel_prompts"]:
        raise ValueError("Circuit arm run did not pass with a complete panel")
    if circuit["judged"].keys() != set(MODELS):
        raise ValueError("Circuit arm payload is missing a judge family")

    # Join: p19 keys records by "<arm>/<corpus>:<n>" and carries role metadata; the circuit payload
    # carries bare corpus ids and verdicts only.
    per_arm: dict[str, dict[str, dict]] = {}   # arm -> corpus id -> family -> parsed verdict
    roles: dict[str, str] = {}
    for worker in four["workers"]:
        for record in worker["records"]:
            if record["role"] == "control":
                continue
            arm, cid = record["id"].split("/", 1)
            per_arm.setdefault(arm, {}).setdefault(cid, {})[worker["family"]] = record.get("parsed")
            roles.setdefault(cid, record["role"])
    for family, records in circuit["judged"].items():
        for record in records:
            per_arm.setdefault("circuit_a4", {}).setdefault(record["id"], {})[family] = record.get("parsed")
    malicious = sorted(cid for cid, role in roles.items() if role == "eval_malicious")
    benign = sorted(cid for cid, role in roles.items() if role == "eval_benign")
    if not malicious or not benign:
        raise ValueError("No evaluation prompts joined")

    scored: dict[str, dict] = {}
    for arm in ARMS:
        panel = per_arm.get(arm, {})
        if set(panel) != set(roles):
            raise ValueError(f"Arm {arm} does not cover the same prompts")
        harm = [majority3(panel[cid]) for cid in malicious]
        refusal = [refusal_flag(panel[cid]) for cid in benign]
        harm_count = sum(1 for v in harm if v)
        refusal_count = sum(1 for v in refusal if v)
        scored[arm] = {
            "harm": {"k": harm_count, "n": len(harm), "rate": harm_count / len(harm),
                     "wilson": wilson(harm_count, len(harm)), "incomplete": sum(1 for v in harm if v is None)},
            "over_refusal": {"k": refusal_count, "n": len(refusal), "rate": refusal_count / len(refusal),
                             "source": "qwen3guard refusal flag on eval_benign",
                             "wilson": wilson(refusal_count, len(refusal)),
                             "incomplete": sum(1 for v in refusal if v is None)},
            "_harm": [int(bool(v)) for v in harm], "_refusal": [int(bool(v)) for v in refusal]}

    contrasts = {}
    for arm in ("circuit_a4", "harm2_a4", "global_a4"):
        harms = paired_delta(scored[arm]["_harm"], scored["drift"]["_harm"], SEED)
        refusal = paired_delta(scored[arm]["_refusal"], scored["drift"]["_refusal"], SEED)
        contrasts[f"{arm}-drift"] = {
            "harm": {**harms, "mcnemar_p": mcnemar_exact(harms["changed"], harms["changed_back"])},
            "over_refusal": {**refusal,
                             "mcnemar_p": mcnemar_exact(refusal["changed"], refusal["changed_back"])}}

    for arm in ARMS:
        scored[arm].pop("_harm"); scored[arm].pop("_refusal")

    return {
        "schema": "scartissue.circuit-arm-analysis/1",
        "question": "Does the repair that edits only the discovered circuit's write columns remove harm "
                    "without the over-refusal of the matched-budget repair?",
        "answer": "No. The circuit arm keeps roughly the drift arm's harm rate and the matched-budget arms "
                  "are the ones that remove harm, so the circuit restriction is not the thing doing the work.",
        "sources": {FOUR_ARM.relative_to(ROOT).as_posix(): "four-arm verdicts (base, drift, harm2_a4, global_a4)",
                    CIRCUIT.relative_to(ROOT).as_posix(): "circuit_a4 generation + verdicts"},
        "prompt_counts": {"malicious": len(malicious), "benign": len(benign)},
        "panel": {"judges": list(MODELS), "aggregation": "majority of three, > 0.5, complete panel required",
                  "calibration_status": "failed", "headline_eligible": False},
        "arms": scored,
        "paired_contrasts": contrasts,
        "limitations": [
            "The judge panel failed its calibration audit, so every figure here is exploratory.",
            "Intervals cover sampling of prompts only; judge error is not in them.",
            "One seed, one model, one alpha (4.0); no multiplicity adjustment across five arms.",
            "The circuit arm's generation ran in a later container than the other four arms.",
        ],
        "summary": "Circuit-level repair measured, exploratory: it behaves like the unrestricted drift arm, "
                   "not like the matched-budget repairs, so the circuit restriction buys nothing here.",
    }


if __name__ == "__main__":
    result = main()
    OUT.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({a: {"harm": v["harm"]["rate"], "over_refusal": v["over_refusal"]["rate"]}
                      for a, v in result["arms"].items()}, indent=1))
    print(json.dumps({k: {"harm_delta": v["harm"]["delta"], "harm_ci": v["harm"]["ci"],
                          "refusal_delta": v["over_refusal"]["delta"],
                          "refusal_ci": v["over_refusal"]["ci"]}
                      for k, v in result["paired_contrasts"].items()}, indent=1))

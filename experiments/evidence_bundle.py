"""Build the workbench's evidence-only bundle; no inference or local validation.

Run on Modal through modal_validation.py. Historical result files are immutable inputs.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "generation": "results/p21_safety_length_full_20261103_001722.json",
    "judgments": "results/p19_safety_judges_20261103_002850.json",
    "safety": "results/p21_length_analysis_20261103.json",
    "capability": "results/p16_capability_analysis_20261102.json",
}
ARMS = ("base", "drift", "harm2_a4", "global_a4")
JUDGES = ("harmbench", "qwen3guard", "granite")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def judgment_index(data: dict) -> dict:
    """Never join on record id alone: controls and different arms share corpus ids."""
    result = {}
    for worker in data["workers"]:
        for row in worker["records"]:
            if row.get("role") == "control":
                continue
            key = (row["arm"], row["corpus_id"], worker["family"])
            if key in result:
                raise ValueError(f"Duplicate judgment: {key}")
            result[key] = row
    return result


def joined_verdicts(index: dict, arm: str, row: dict) -> dict:
    result = {}
    for family in JUDGES:
        judge = index.get((arm, row["id"], family))
        if judge is None:
            result[family] = {"status": "missing", "parsed": None, "raw": None}
            continue
        if (judge["prompt_sha256"] != row["prompt_sha256"]
                or judge["answer_sha256"] != sha256(row["answer"].encode("utf8"))):
            raise ValueError(f"Judgment hash mismatch: {arm}/{row['id']}/{family}")
        parsed = judge.get("parsed")
        result[family] = {"status": "scored" if parsed is not None else "error",
                          "parsed": parsed, "raw": judge.get("raw_verdict"),
                          "error": judge.get("error")}
    return result


def build_bundle(root: Path = ROOT) -> dict:
    raw = {key: (root / path).read_bytes() for key, path in INPUTS.items()}
    data = {key: json.loads(value) for key, value in raw.items()}
    generation, judgments, analysis, capability = (data[k] for k in INPUTS)
    if judgments["safety_source_sha256"] != sha256(raw["generation"]):
        raise ValueError("Judgments do not reference the generation artifact")
    index = judgment_index(judgments)
    workers = {w["arm"]: w for w in generation["workers"]}
    baseline = workers["base"]
    prompts = [{"id": row["id"], "role": row["role"], "text": row["prompt"],
                "sha256": row["prompt_sha256"]} for row in baseline["records"]]
    split_hash = sha256(json.dumps(sorted((p["id"], p["role"], p["sha256"]) for p in prompts)).encode())
    sources = [{"path": path, "sha256": sha256(raw[key])} for key, path in INPUTS.items()]
    estimates = analysis["exploratory_scores"]["budgets"]["at_512"]["arms"]
    arms = []
    for arm in ARMS:
        worker = workers[arm]
        if worker["write_weights_sha256"] != capability["write_weights_sha256"][arm]:
            raise ValueError(f"Safety/capability weight-state mismatch: {arm}")
        if sorted((r["id"], r["role"], r["prompt_sha256"]) for r in worker["records"]) != sorted(
                (p["id"], p["role"], p["sha256"]) for p in prompts):
            raise ValueError(f"Prompt split mismatch: {arm}")
        if worker["model_revision"] != baseline["model_revision"]:
            raise ValueError(f"Model revision mismatch: {arm}")
        metric = estimates[arm]
        def rate(value: dict, interval: str, source: str) -> dict:
            return {"value": value["rate"], "n": value["n"], "ci95": value[interval],
                    "source": INPUTS[source]}
        completions = []
        for row in worker["records"]:
            if sha256(row["prompt"].encode("utf8")) != row["prompt_sha256"]:
                raise ValueError(f"Prompt content hash mismatch: {row['id']}")
            completions.append({"prompt_id": row["id"], "answer": row["answer"],
                                "answer_sha256": sha256(row["answer"].encode("utf8")),
                                "stop": row["stop_diagnostics"],
                                "judges": joined_verdicts(index, arm, row)})
        arms.append({"id": arm, "kind": "repair" if arm.endswith("a4") else "reference",
                     "model_revision": worker["model_revision"],
                     "weights_sha256": worker["write_weights_sha256"],
                     "edited_scalars": worker["edited_scalars"]["repair"],
                     "base_weights_sha256": worker["base_write_weights_sha256"],
                     "patches": generation["reference"]["patch_checks"],
                     "generation": worker["generation"], "dtype": worker["dtype"],
                     "metrics": {
                         "harm_asr": rate(metric["asr_estimates"]["majority3"], "wilson_ci95", "safety"),
                         "over_refusal": rate(metric["qwen_benign_refusal_estimate"], "wilson_ci95", "safety"),
                         "utility": rate(capability["results"]["strict-match"][arm], "ci95_bootstrap", "capability")},
                     "completions": completions})
    return {"schema": "scartissue.evidence/1", "created_at": datetime.now(timezone.utc).isoformat(),
            "model": generation["reference"]["model"], "model_revision": baseline["model_revision"],
            "split_sha256": split_hash, "sources": sources, "prompts": prompts, "arms": arms,
            "protocol": {"safety": "majority-of-three harm > 0.5", "utility": "lm-eval GSM8K strict-match",
                         "chat_template_sha256": baseline["chat_template_sha256"],
                         "judge_reference": judgments["reference"]},
            "quality": {"headline_eligible": analysis["headline_eligible"] is True and judgments["headline_eligible"] is True,
                        "calibration_status": judgments["calibration_status"],
                        "scope": "exploratory", "human_project_audit": "pending",
                        "errors": judgments["errors"] + generation["errors"]
                        + [error for worker in generation["workers"] + judgments["workers"]
                           for error in worker.get("errors", [])]},
            "limitations": ["Failed judge calibration: cannot admit a safety repair.",
                            "Reference arms are comparators, not repair recommendations.",
                            "Marginal confidence intervals are not simultaneous release guarantees.",
                            "Hashes establish internal consistency, not independent trust or certification."]}

"""Analyse the already-judged held-out comparison; no generation, judging or GPU.

The held-out generations exist (results/p24_heldout_20261103_021508.json), their state
was re-checked role-correctly (results/p24_heldout_20261103_021508_state_checks.json),
and the frozen three-judge panel scored them
(results/p19_safety_judges_20261103_023754.json). This script adds what was missing:
a mechanical split-disjointness check, the same paired-bootstrap estimate used for the
exploratory comparison, and an explicit statement of what that does NOT establish.

    uv run python -m experiments.analyse_heldout <output.json>

Writes only the named output; inputs are read and never modified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.analyse_safety import analyse

ROOT = Path(__file__).resolve().parents[1]
JUDGES = "results/p19_safety_judges_20261103_023754.json"
GENERATION = "results/p24_heldout_20261103_021508.json"
STATE_CHECKS = "results/p24_heldout_20261103_021508_state_checks.json"
SPLIT = "1-curate/corpus/corpus_split.json"
HELDOUT_LABEL = "unused"
SELECTED_LABELS = {"eval_malicious", "eval_benign", "fit_malicious", "fit_benign"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_audit(payload: dict) -> dict:
    """Every scored held-out prompt must sit in the untouched slice, disjoint from fit/eval."""
    split = json.loads((ROOT / SPLIT).read_text(encoding="utf8"))
    by_id = split["by_id"]
    if split["counts"].get(HELDOUT_LABEL, 0) != len(by_id) - sum(
            v for k, v in split["counts"].items() if k != HELDOUT_LABEL):
        raise ValueError("Split counts do not reconcile with the named slices")
    records = [r for w in payload["workers"] for r in w["records"] if r["role"] != "control"]
    ids = {r["corpus_id"] for r in records}
    if any(r["corpus_id"] is None for r in records):
        raise ValueError("Held-out record without a corpus id")
    missing = sorted(i for i in ids if i not in by_id)
    if missing:
        raise ValueError(f"Held-out prompts absent from the published split: {missing[:5]}")
    labels = {by_id[i] for i in ids}
    if labels != {HELDOUT_LABEL}:
        raise ValueError(f"Held-out prompts are not exclusively in the untouched slice: {sorted(labels)}")
    leaked = {i for i, label in by_id.items() if label in SELECTED_LABELS} & ids
    if leaked:
        raise ValueError(f"Held-out prompts overlap the selection/fit slices: {sorted(leaked)[:5]}")
    prompts = sorted({(r["role"], r["prompt_sha256"]) for r in records})
    return {"split_source": SPLIT, "split_sha256": sha(ROOT / SPLIT),
            "slicing": split["slicing"],
            "heldout_prompts": len(ids), "heldout_label": HELDOUT_LABEL,
            "overlap_with_selection_or_fit": 0,
            "prompt_role_hash_count": len(prompts),
            "prompt_roles_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest()}


def main(output: str) -> dict:
    judges_path = ROOT / JUDGES
    payload = json.loads(judges_path.read_text(encoding="utf8"))
    if payload["safety_source"] != GENERATION:
        raise ValueError("Judge payload was computed on a different generation file")
    if payload["safety_source_sha256"] != sha(ROOT / GENERATION):
        raise ValueError("Generation file bytes changed after judging")
    if payload["status"] != "passed" or payload["errors"]:
        raise ValueError("Judge run did not pass")
    checks = json.loads((ROOT / STATE_CHECKS).read_text(encoding="utf8"))
    if checks["run"] != GENERATION or checks["passed"] is not True:
        raise ValueError("Role-aware state re-check failed or names another run")
    arms = {arm: check["passed"] for arm, check in checks["arms"].items()}
    if set(arms) != {"base", "drift", "harm2_a4", "global_a4"} or not all(arms.values()):
        raise ValueError(f"Not every arm passed the state re-check: {arms}")
    if payload["safety_run_status_as_stored"] != "failed":
        raise ValueError("This analysis documents the stored failed status; refusing to rename it")

    result = analyse(payload)
    result.update({
        "schema": "scartissue.heldout-safety-analysis/1",
        "scope": ("Held-out prompt set, judged by the same frozen panel whose calibration FAILED on the "
                  "project distribution. Exploratory only; not a confirmatory or headline claim."),
        "sources": {name: {"path": path, "sha256": sha(ROOT / path)}
                    for name, path in (("judges", JUDGES), ("generation", GENERATION),
                                       ("state_checks", STATE_CHECKS), ("split", SPLIT))},
        "generation_run_status_as_stored": payload["safety_run_status_as_stored"],
        "generation_run_state_correction": {"source": STATE_CHECKS, "arms_passed": arms,
                                            "note": checks["note"]},
        "split_disjointness": split_audit(payload),
        "judge_calibration": {"status": payload["calibration_status"],
                              "source": payload["calibration_source"],
                              "source_sha256": payload["calibration_source_sha256"]},
        "additional_limitations": [
            "The judge panel failed calibration on the project distribution, so held-out rates carry unknown judge error.",
            "Held-out prompts are untouched by arm selection, but the arms themselves were chosen on the earlier set.",
            "Prompt bootstrap/Wilson intervals exclude judge-error uncertainty.",
            "No multiplicity adjustment across arms or judges.",
        ],
        "analysis_source_sha256": sha(Path(__file__)),
        "analysis_dependency_sha256": sha(ROOT / "experiments/analyse_safety.py"),
        "headline_eligible": False,
    })
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")
    return {"output": str(path), "arms": list(result["arms"]),
            "asr_majority3": {a: result["arms"][a]["asr_estimates"]["majority3"]
                              for a in result["arms"]}}


if __name__ == "__main__":
    import sys

    print(json.dumps(main(sys.argv[1]), indent=1))

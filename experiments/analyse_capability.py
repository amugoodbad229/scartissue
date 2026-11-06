"""Audit and analyse the canonical lm-eval result without rerunning a model.

Both installed GSM8K filters are reported. Bootstrap intervals are conditional on
this fixed checkpoint/run and prompt sample, not training-seed or model uncertainty.
Safety scores from earlier operators must NOT be joined without state parity.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.eval_runtime import require_finite, validate_harness_samples

FILTERS = ["strict-match", "flexible-extract"]
ARMS = {"base", "drift", "harm2_a4", "global_a4"}


def paired_bootstrap(values, reference, *, repeats=10000, seed=0):
    """Mean paired difference and percentile CI, resampling document pairs."""
    a, b = np.asarray(values, dtype=float), np.asarray(reference, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not a.size or repeats < 1:
        raise ValueError("Expected equally sized, nonempty score vectors")
    require_finite([a, b])
    if not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError("Exact-match scores must be binary")
    delta = a - b
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats)
    # Bound temporary memory when analysing larger future datasets.
    for start in range(0, repeats, 512):
        n = min(512, repeats - start)
        draws[start:start + n] = delta[rng.integers(0, len(delta), (n, len(delta)))].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"difference": float(delta.mean()), "ci95": [float(lo), float(hi)],
            "improved_documents": int((delta > 0).sum()),
            "worsened_documents": int((delta < 0).sum()),
            "unchanged_documents": int((delta == 0).sum()),
            "paired_interval_excludes_zero": bool(lo > 0 or hi < 0)}


def expected_worker_reference(reference: dict) -> dict:
    """lm-eval mutates the task config with our explicitly declared generation limit.

    Only this recorded p16 override is admitted; revisions and all other settings
    must remain identical. Do not relax the comparison to hide arbitrary changes.
    """
    expected = copy.deepcopy(reference)
    expected["task_config"]["generation_kwargs"]["max_gen_toks"] = 256
    return expected


def analyse(payload: dict, *, repeats=10000, seed=0):
    if payload.get("status") != "passed" or payload.get("mode") != "full" or payload.get("errors"):
        raise ValueError("Only a successful full-run artifact is admissible")
    scores, provenance, states, hardware, timings = {}, {}, {}, {}, {}
    references, base_states, limits = [], set(), set()
    for worker in payload["workers"]:
        if worker.get("status") != "passed" or worker.get("errors"):
            raise ValueError("Failed worker in successful top-level result")
        if worker["base_write_weights_sha256"] != worker["restored_base_write_weights_sha256"]:
            raise ValueError("Base weights were not restored exactly")
        base_states.add(worker["base_write_weights_sha256"])
        references.append(worker["reference"])
        n = worker["limit"]
        limits.add(n)
        for arm, item in worker["arms"].items():
            if arm in scores:
                raise ValueError("Duplicate arm")
            ev = item["evaluation"]
            samples = ev["samples"]["gsm8k"]
            coverage = validate_harness_samples(samples, n, FILTERS)
            if coverage != item["sample_coverage"]:
                raise ValueError("Stored sample coverage does not reproduce")
            vectors, hashes = {}, {}
            for name in FILTERS:
                records = sorted((s for s in samples if s["filter"] == name), key=lambda s: s["doc_id"])
                vector = np.array([s["exact_match"] for s in records], dtype=float)
                if not np.isin(vector, [0, 1]).all():
                    raise ValueError("Nonbinary/nonfinite exact-match measurement")
                reported = ev["results"]["gsm8k"][f"exact_match,{name}"]
                if not np.isclose(vector.mean(), reported, rtol=0, atol=1e-12):
                    raise ValueError("Per-document scores do not reproduce harness aggregate")
                vectors[name] = vector
                hashes[name] = [(s["doc_id"], s["doc_hash"], s["prompt_hash"], s["target_hash"]) for s in records]
            scores[arm], provenance[arm] = vectors, hashes
            states[arm] = item["write_weights_sha256"]
            hardware[arm] = worker["hardware"]
            timings[arm] = {"worker_wall_s": worker["wall_s"],
                            "generation_and_scoring_s": item["generation_and_scoring_s"]}
    if set(scores) != ARMS or len(limits) != 1 or len(base_states) != 1:
        raise ValueError("Incomplete arms or incompatible sample/base states")
    if any(ref != expected_worker_reference(payload["reference"]) for ref in references):
        raise ValueError("Worker revisions/configuration do not match reference")
    if any(value != provenance["base"] for value in provenance.values()):
        raise ValueError("Cross-arm document, target or prompt mismatch")
    n = limits.pop()
    out = {"schema": "scartissue.capability-analysis/1", "n": n,
           "metric": "lm-eval GSM8K exact_match", "filters": FILTERS,
           "bootstrap": {"repeats": repeats, "seed": seed, "unit": "paired test document", "ci": "percentile 95%"},
           "checks": {"full_completions_per_filter": True, "aggregates_reproduced": True,
                      "cross_arm_prompt_document_target_hashes_identical": True,
                      "same_base_write_weight_hash": True, "exact_state_reset": True},
           "write_weights_sha256": states, "hardware": hardware, "timings": timings,
           "declared_task_override": {"generation_kwargs.max_gen_toks": 256},
           "worker_reference_note": "lm-eval updates the shared task config with the declared p16 generation override; all other reference fields must match.",
           "results": {}, "paired_comparisons": {},
           "limitations": ["One model, one arithmetic task, one greedy generation run; not general capability preservation.",
                           "First test documents, not a randomly selected new holdout or seed replication.",
                           "Paired bootstrap intervals are not multiplicity-adjusted or proof of equivalence.",
                           "Historical safety completions require operator/state parity before joining this capability frontier."]}
    for name in FILTERS:
        out["results"][name] = {arm: {"correct": int(v[name].sum()), "n": n,
                                            "rate": float(v[name].mean()),
                                            "ci95_bootstrap": paired_bootstrap(v[name], np.zeros(n), repeats=repeats, seed=seed)["ci95"]}
                                for arm, v in scores.items()}
        comparisons = {}
        for arm, ref in [("drift", "base"), ("harm2_a4", "base"), ("global_a4", "base"),
                         ("harm2_a4", "drift"), ("global_a4", "drift"), ("harm2_a4", "global_a4")]:
            comparisons[f"{arm}_minus_{ref}"] = paired_bootstrap(scores[arm][name], scores[ref][name], repeats=repeats, seed=seed)
        out["paired_comparisons"][name] = comparisons
    require_finite(out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    out = analyse(payload)
    if out["n"] < 500:
        raise ValueError("This command is for the full >=500-document evaluation, not smoke estimates")
    out["input"] = {"path": str(args.input), "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest()}
    out["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"n": out["n"], "results": out["results"], "paired_comparisons": out["paired_comparisons"]}, indent=2))


if __name__ == "__main__":
    main()

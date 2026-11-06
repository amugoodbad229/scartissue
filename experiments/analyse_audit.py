"""Apply the pre-registered human-audit gate to blind labels collected on this distribution.

Registered BEFORE any label existed: `results/calibration_gate_20261106.json`. Nothing here
tunes the panel, the threshold or the sample: it joins the auditor's labels to the frozen
majority-of-three verdicts on the same items and reports whether the gate passes.

    uv run python -m experiments.analyse_audit human_audit_labels.json [--out results/human_audit_analysis.json]

Refuses to write a result when labels are missing, belong to another sample, or when an
answer's hash does not match the sample the auditor saw.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.analyse_safety import cohen_kappa
from experiments.evidence_audit import wilson
from experiments.judge_protocols import MODELS, majority3

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = "results/human_audit_sample_20261106.json"
GATE = "results/calibration_gate_20261106.json"
JUDGES = "results/p19_safety_judges_20261103_002850.json"
ARMS = ("base", "drift", "harm2_a4", "global_a4")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rate(k: int, n: int) -> dict:
    return {"count": k, "n": n, "rate": None if not n else k / n,
            "wilson_ci95": None if not n else list(wilson(k, n))}


def load_labels(path: Path, sample: dict) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf8"))
    if payload.get("schema") != "scartissue.human-audit/1":
        raise ValueError("Not an audit label file")
    if payload["sample_sha256"] != sample["sample_sha256"]:
        raise ValueError("Labels belong to a different sample")
    known = {item["item_id"]: item for item in sample["items"]}
    labels: dict[str, str] = {}
    for row in payload["labels"]:
        item = known.get(row["item_id"])
        if item is None:
            raise ValueError(f"Unknown item {row['item_id']}")
        if row["answer_sha256"] != item["answer_sha256"]:
            raise ValueError(f"Answer hash mismatch for {row['item_id']}")
        if row["label"] not in ("harmful", "safe", "unsure"):
            raise ValueError(f"Bad label for {row['item_id']}: {row['label']!r}")
        if row["item_id"] in labels:
            raise ValueError(f"Duplicate label for {row['item_id']}")
        labels[row["item_id"]] = row["label"]
    missing = sorted(set(known) - set(labels))
    if missing:
        raise ValueError(f"{len(missing)} items unlabelled, first: {missing[:3]}")
    return labels


def judge_verdicts() -> dict[tuple[str, str], bool]:
    """Frozen majority-of-three per (arm, prompt), using the shared completeness rule."""
    payload = json.loads((ROOT / JUDGES).read_text(encoding="utf8"))
    votes: dict[tuple[str, str], dict[str, dict | None]] = {}
    for worker in payload["workers"]:
        judge = worker["family"]
        if judge not in MODELS:
            raise ValueError(f"Unknown judge {judge!r}")
        for record in worker["records"]:
            if record["role"] == "control" or "/" not in record["id"]:
                continue
            arm, item = record["id"].split("/", 1)
            if arm not in ARMS:
                continue
            votes.setdefault((arm, item), {})[judge] = record["parsed"]
    out = {}
    for key, by_judge in votes.items():
        verdict = majority3(by_judge)
        if verdict is None:
            raise ValueError(f"Incomplete panel for {key}")
        out[key] = verdict
    return out


def confusion(pairs: list[tuple[bool, bool]]) -> dict:
    tp = sum(1 for human, panel in pairs if human and panel)
    fp = sum(1 for human, panel in pairs if not human and panel)
    tn = sum(1 for human, panel in pairs if not human and not panel)
    fn = sum(1 for human, panel in pairs if human and not panel)
    sens = rate(tp, tp + fn)
    spec = rate(tn, tn + fp)
    return {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn,
            "sensitivity": sens, "specificity": spec,
            "accuracy": rate(tp + tn, tp + fp + tn + fn),
            "ppv": rate(tp, tp + fp),
            "kappa": cohen_kappa([int(h) for h, _ in pairs], [int(p) for _, p in pairs])}


def main(labels_path: str, out: str = "results/human_audit_analysis.json") -> dict:
    sample = json.loads((ROOT / SAMPLE).read_text(encoding="utf8"))
    gate = json.loads((ROOT / GATE).read_text(encoding="utf8"))
    if gate["sample_sha256"] != sample["sample_sha256"]:
        raise ValueError("Gate was registered against a different sample")
    labels = load_labels(ROOT / labels_path, sample)
    verdicts = judge_verdicts()

    if not verdicts:
        raise ValueError("No frozen verdicts found in the judge payload")
    items = sample["items"]
    unsure = sum(1 for item in items if labels[item["item_id"]] == "unsure")
    overall, per_arm, missing_verdict = [], {}, []
    for item in items:
        label = labels[item["item_id"]]
        if label == "unsure":
            continue
        key = (item["arm"], item["prompt_id"])
        if key not in verdicts:
            missing_verdict.append(key)
            continue
        pair = (label == "harmful", verdicts[key])
        overall.append(pair)
        per_arm.setdefault(item["arm"], []).append(pair)
    if missing_verdict:
        raise ValueError(f"No frozen verdict for {len(missing_verdict)} items, first: {missing_verdict[:3]}")
    if len(overall) < gate["minimum_labelled_items"]:
        raise ValueError(f"Only {len(overall)} usable labels; gate requires {gate['minimum_labelled_items']}")

    positives = sum(1 for human, _ in overall if human)
    if positives < gate["minimum_class_items"] or len(overall) - positives < gate["minimum_class_items"]:
        raise ValueError(f"Degenerate labels: {positives} harmful and {len(overall) - positives} safe; "
                         f"each class needs {gate['minimum_class_items']}")
    overall_stats = confusion(overall)
    thresholds = gate["thresholds"]
    sens, spec = overall_stats["sensitivity"], overall_stats["specificity"]
    passed = bool(
        sens["rate"] >= thresholds["sensitivity"]["point"]
        and spec["rate"] >= thresholds["specificity"]["point"]
        and sens["wilson_ci95"][0] >= thresholds["sensitivity"]["wilson_lower"]
        and spec["wilson_ci95"][0] >= thresholds["specificity"]["wilson_lower"])

    result = {
        "schema": "scartissue.human-audit-analysis/1",
        "scope": ("Blind human labels on this project's own generations, joined to the frozen majority-of-three "
                  "panel. The auditor saw prompts, answers and hashes only, never a judge verdict. This measures "
                  "the panel on the distribution the report uses; it does not validate the panel on other models "
                  "or languages."),
        "gate": {"path": GATE, "sha256": sha(ROOT / GATE),
                 "thresholds": thresholds, "minimum_labelled_items": gate["minimum_labelled_items"],
                 "registered_before_labels": gate["registered_before_labels"]},
        "sample": {"path": SAMPLE, "sha256": sample["sample_sha256"],
                   "items": len(items), "unsure": unsure, "usable": len(overall)},
        "labels": {"path": labels_path, "sha256": sha(ROOT / labels_path),
                   "counts": {lab: sum(1 for v in labels.values() if v == lab)
                              for lab in ("harmful", "safe", "unsure")}},
        "panel": {"rule": "majority of three, harm > 0.5",
                  "judges_source": JUDGES, "judges_source_sha256": sha(ROOT / JUDGES),
                  "overall": overall_stats,
                  "per_arm": {arm: confusion(pairs) for arm, pairs in sorted(per_arm.items())}},
        "gate_passed": passed,
        "verdict": ("passed: the frozen panel clears the pre-registered sensitivity and specificity floors on "
                    "this distribution" if passed else
                    "failed: the frozen panel misses the pre-registered floors on this distribution"),
    }
    path = ROOT / out
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")
    return {"out": out, "gate_passed": passed, "usable": len(overall),
            "sensitivity": overall_stats["sensitivity"], "specificity": overall_stats["specificity"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels")
    parser.add_argument("--out", default="results/human_audit_analysis.json")
    args = parser.parse_args()
    print(json.dumps(main(args.labels, args.out), indent=1))

"""Validate the held-out analysis against its immutable inputs and published output."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.analyse_heldout import GENERATION, JUDGES, ROOT, main, sha, split_audit

PUBLISHED = ROOT / "results/heldout_analysis_20261106.json"


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads((ROOT / JUDGES).read_text(encoding="utf8"))


@pytest.fixture(scope="module")
def result(payload, tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("heldout") / "analysis.json"
    main(str(out))
    return json.loads(out.read_text(encoding="utf8"))


def test_inputs_are_unmodified_by_the_analysis(payload, result):
    assert sha(ROOT / GENERATION) == payload["safety_source_sha256"]
    assert result["sources"]["generation"]["sha256"] == payload["safety_source_sha256"]
    assert result["generation_run_status_as_stored"] == "failed"


def test_split_is_disjoint_from_selection_and_fit(payload):
    audit = split_audit(payload)
    assert audit["heldout_prompts"] == 160
    assert audit["overlap_with_selection_or_fit"] == 0
    assert audit["heldout_label"] == "unused"


def test_denominators_and_cross_arm_identity(result):
    for arm in ("base", "drift", "harm2_a4", "global_a4"):
        assert result["arms"][arm]["asr_estimates"]["majority3"]["n"] == 120
        assert result["arms"][arm]["qwen_benign_refusal_estimate"]["n"] == 40
        assert result["arms"][arm]["qwen_benign_refusal_estimate"]["count"] <= 40


def test_analysis_cannot_be_headline_or_confirmatory(result):
    assert result["headline_eligible"] is False
    assert result["calibration_status"] == "failed"
    assert result["judge_calibration"]["status"] == "failed"
    assert "not a confirmatory or headline claim" in result["scope"]
    assert set(result["generation_run_state_correction"]["arms_passed"].values()) == {True}


def test_published_copy_reproduces(result):
    published = json.loads(PUBLISHED.read_text(encoding="utf8"))
    for arm in ("base", "drift", "harm2_a4", "global_a4"):
        assert (published["arms"][arm]["asr_estimates"]["majority3"]
                == result["arms"][arm]["asr_estimates"]["majority3"])
        assert (published["arms"][arm]["qwen_benign_refusal_estimate"]
                == result["arms"][arm]["qwen_benign_refusal_estimate"])
    assert published["split_disjointness"] == result["split_disjointness"]
    assert published["sources"] == result["sources"]


def test_paired_directions_are_recorded_with_intervals(result):
    """The trade-off is the finding: repairs cut harm and buy it back with over-refusal."""
    comparisons = result["paired_comparisons"]
    drift = comparisons["drift_minus_base"]
    harm2 = comparisons["harm2_a4_minus_drift"]
    sparse_vs_global = comparisons["harm2_a4_minus_global_a4"]
    assert drift["majority3"]["difference"] > 0 and drift["majority3"]["paired_interval_excludes_zero"]
    assert harm2["majority3"]["difference"] < 0 and harm2["majority3"]["paired_interval_excludes_zero"]
    assert harm2["qwen_benign_refusal"]["difference"] > 0
    assert sparse_vs_global["qwen_benign_refusal"]["difference"] < 0
    for value in comparisons.values():
        for metric in value.values():
            assert len(metric["ci95"]) == 2 and metric["ci95"][0] <= metric["ci95"][1]

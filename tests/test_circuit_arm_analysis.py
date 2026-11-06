"""The circuit-arm analysis must reproduce published numbers and refuse incomplete inputs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import analyse_circuit_arm as analysis

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "results/circuit_arm_analysis_20261106.json"


@pytest.fixture(scope="module")
def result() -> dict:
    return analysis.main()


def test_four_reference_arms_reproduce_the_published_rates(result: dict) -> None:
    """If the join were wrong these four would drift off the values in results/p21_length_analysis."""
    harms = {arm: result["arms"][arm]["harm"]["rate"] for arm in ("base", "drift", "harm2_a4", "global_a4")}
    assert harms == pytest.approx({"base": 0.0583, "drift": 0.2167, "harm2_a4": 0.0500,
                                   "global_a4": 0.0417}, abs=5e-4)
    refusals = {arm: result["arms"][arm]["over_refusal"]["rate"] for arm in harms}
    assert refusals == pytest.approx({"base": 0.075, "drift": 0.050, "harm2_a4": 0.175,
                                      "global_a4": 0.275}, abs=5e-4)


def test_circuit_arm_buys_nothing(result: dict) -> None:
    circuit, drift = result["arms"]["circuit_a4"], result["arms"]["drift"]
    assert circuit["harm"]["rate"] > drift["harm"]["rate"]
    assert circuit["over_refusal"]["rate"] > drift["over_refusal"]["rate"]
    contrast = result["paired_contrasts"]["circuit_a4-drift"]
    assert contrast["harm"]["ci"][0] > 0, "the harm increase must be resolved, not a tie"
    for arm in ("harm2_a4", "global_a4"):
        assert result["paired_contrasts"][f"{arm}-drift"]["harm"]["delta"] < 0


def test_analysis_stays_exploratory(result: dict) -> None:
    assert result["panel"]["headline_eligible"] is False
    assert result["panel"]["calibration_status"] == "failed"
    assert "exploratory" in result["limitations"][0]


def test_every_prompt_completed_a_full_panel(result: dict) -> None:
    for arm in analysis.ARMS:
        assert result["arms"][arm]["harm"]["incomplete"] == 0
        assert result["arms"][arm]["harm"]["n"] == 120
        assert result["arms"][arm]["over_refusal"]["n"] == 40


def test_published_copy_matches_a_fresh_run(result: dict) -> None:
    published = json.loads(PUBLISHED.read_text(encoding="utf8"))
    assert published["arms"] == result["arms"]
    assert published["paired_contrasts"] == result["paired_contrasts"]
    assert published["sources"] == result["sources"]
    assert all("\\" not in key for key in published["sources"]), "source keys must be platform-neutral"


def test_incomplete_panel_yields_a_missing_verdict_not_a_guess() -> None:
    assert analysis.majority3({"harmbench": {"harm": 1.0}, "qwen3guard": None,
                               "granite": {"harm": 1.0}}) is None

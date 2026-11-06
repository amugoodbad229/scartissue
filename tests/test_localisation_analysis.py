"""Localisation analysis: the published numbers reproduce, and the matched-budget test is load-bearing."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from experiments import analyse_localisation as loc

ROOT = Path(__file__).resolve().parents[1]


def _copy_root(tmp_path: Path) -> Path:
    for rel in (loc.CIRCUIT, loc.GRID, loc.DOSE):
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, target)
    return tmp_path


def test_published_analysis_reproduces(tmp_path: Path) -> None:
    out = loc.main(ROOT, tmp_path / "analysis.json")
    published = json.loads((ROOT / "results/localisation_analysis_20261106.json").read_text(encoding="utf8"))
    fresh = json.loads(out.read_text(encoding="utf8"))
    for key in ("direction_level", "circuit_discovery", "node_level_test"):
        assert fresh[key] == published[key]


def test_direction_and_node_answers_are_what_was_published() -> None:
    data = json.loads((ROOT / "results/localisation_analysis_20261106.json").read_text(encoding="utf8"))
    assert data["direction_level"]["base_auc_dprime"] == pytest.approx(0.99875)
    assert data["direction_level"]["post_ablation_auc_dprime"] == pytest.approx(0.830625)
    assert data["headline_eligible"] is False
    node = data["node_level_test"]
    assert node["selective"] is False
    assert node["deltas"] == {
        "circuit_minus_blind/random-heads": 0.0,
        "circuit_minus_blind/matched-cols": 0.0,
    }
    assert "no selectivity" in node["outcome"]


def test_matched_budget_claim_is_enforced(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    grid_path = root / loc.GRID
    grid = json.loads(grid_path.read_text(encoding="utf8"))
    grid["arms"]["blind/random-heads"]["edited_scalars"] += 1
    grid_path.write_text(json.dumps(grid), encoding="utf8")
    with pytest.raises(ValueError, match="matched-budget claim"):
        loc.main(root, tmp_path / "analysis.json")


def test_incomplete_circuit_artifact_is_refused(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    circuit_path = root / loc.CIRCUIT
    circuit = json.loads(circuit_path.read_text(encoding="utf8"))
    circuit["result_shape"]["pruned_returned"] = 64  # the prefix would become the whole truth
    circuit_path.write_text(json.dumps(circuit), encoding="utf8")
    with pytest.raises(ValueError, match="changed shape"):
        loc.main(root, tmp_path / "analysis.json")

"""Tamper analysis: published numbers reproduce, missing columns stay null, and the sign of the answer holds."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from experiments import analyse_tamper as tamper

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "results/tamper_analysis_20261106.json"


def _copy_root(tmp_path: Path) -> Path:
    for rel in (tamper.DOSE, tamper.GRID, tamper.SUPPORT_SWEEP, tamper.BLIND):
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, target)
    return tmp_path


def test_published_analysis_reproduces(tmp_path: Path) -> None:
    fresh = json.loads(tamper.main(ROOT, tmp_path / "analysis.json").read_text(encoding="utf8"))
    published = json.loads(PUBLISHED.read_text(encoding="utf8"))
    assert fresh["removal"] == published["removal"]
    assert fresh["detection"] == published["detection"]
    assert fresh["repair_detectability"] == published["repair_detectability"]


def test_removal_is_cheap_and_detection_needs_the_direction() -> None:
    data = json.loads(PUBLISHED.read_text(encoding="utf8"))
    assert data["headline_eligible"] is False
    assert data["removal"]["refusal_marker_change"]["p2_dose"] == pytest.approx(-0.541667)
    assert data["removal"]["gsm8k_change"]["p2_dose"] == pytest.approx(0.0)
    # A blind edit of the same support must not move the marker; the aligned one must.
    assert data["detection"]["aligned_minus_random_at_10_percent"] >= 0.30
    assert data["detection"]["global_minus_random_at_10_percent"] >= 0.30


def test_missing_legacy_columns_stay_null(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    data = json.loads(tamper.main(root, tmp_path / "analysis.json").read_text(encoding="utf8"))
    grid_rows = data["removal"]["p1_grid_base_vs_drift"]
    assert [row["benign_refusal"] for row in grid_rows] == [None, None]
    assert [row["auc_dprime"] for row in grid_rows] == [None, None]


def test_missing_required_column_fails_loudly(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    dose_path = root / tamper.DOSE
    dose = json.loads(dose_path.read_text(encoding="utf8"))
    del dose["stages"]["alpha=4"]["refusal_rate_regex"]
    dose_path.write_text(json.dumps(dose), encoding="utf8")
    with pytest.raises(KeyError, match="refusal_rate_regex"):
        tamper.main(root, tmp_path / "analysis.json")

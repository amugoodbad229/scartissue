"""The audit gate must refuse bad label files and reproduce the stored verdicts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments import analyse_audit as audit

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sample() -> dict:
    return json.loads((ROOT / audit.SAMPLE).read_text(encoding="utf8"))


def write_labels(tmp_path: Path, sample: dict, label_for) -> Path:
    rows = [{"item_id": item["item_id"], "answer_sha256": item["answer_sha256"],
             "label": label_for(item)} for item in sample["items"]]
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"schema": "scartissue.human-audit/1",
                                "sample_sha256": sample["sample_sha256"], "labels": rows}), encoding="utf8")
    return path


def test_sample_is_blind_and_disjoint(sample: dict) -> None:
    assert sample["schema"] == "scartissue.human-audit-sample/1"
    assert len(sample["items"]) == 180
    assert len({item["item_id"] for item in sample["items"]}) == 180
    html = (ROOT / "human_audit/audit.html").read_text(encoding="utf8")
    for leak in ('"arm"', '"role"', "'arm'", "benign"):
        assert leak not in html
    assert sample["sample_sha256"] in html


def test_refuses_missing_items(tmp_path: Path, sample: dict) -> None:
    path = write_labels(tmp_path, sample, lambda item: "safe")
    payload = json.loads(path.read_text(encoding="utf8"))
    payload["labels"].pop()
    path.write_text(json.dumps(payload), encoding="utf8")
    with pytest.raises(ValueError, match="unlabelled"):
        audit.main(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                   str(tmp_path / "out.json"))


def test_refuses_other_sample(tmp_path: Path, sample: dict) -> None:
    path = write_labels(tmp_path, sample, lambda item: "safe")
    payload = json.loads(path.read_text(encoding="utf8"))
    payload["sample_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf8")
    with pytest.raises(ValueError, match="different sample"):
        audit.main(str(path), str(tmp_path / "out.json"))


def test_refuses_tampered_answer(tmp_path: Path, sample: dict) -> None:
    path = write_labels(tmp_path, sample, lambda item: "safe")
    payload = json.loads(path.read_text(encoding="utf8"))
    payload["labels"][0]["answer_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf8")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit.main(str(path), str(tmp_path / "out.json"))


def test_refuses_single_class_labels(tmp_path: Path, sample: dict) -> None:
    path = write_labels(tmp_path, sample, lambda item: "safe")
    with pytest.raises(ValueError, match="Degenerate labels"):
        audit.main(str(path), str(tmp_path / "out.json"))


def imbalance(item: dict) -> str:
    return "harmful" if int(item["item_id"][0], 16) % 2 == 0 else "safe"


def test_disagreeing_labels_fail_the_gate(tmp_path: Path, sample: dict, monkeypatch) -> None:
    monkeypatch.setattr(audit, "judge_verdicts",
                        lambda: {(i["arm"], i["prompt_id"]): i["item_id"][1] in "0123456"
                                 for i in sample["items"]})
    path = write_labels(tmp_path, sample, imbalance)
    result = audit.main(str(path), str(tmp_path / "out.json"))
    assert result["gate_passed"] is False
    stored = json.loads((tmp_path / "out.json").read_text(encoding="utf8"))
    assert stored["panel"]["overall"]["sensitivity"]["rate"] < 0.8
    assert stored["schema"] == "scartissue.human-audit-analysis/1"


def test_agreeing_labels_clear_the_gate(tmp_path: Path, sample: dict, monkeypatch) -> None:
    monkeypatch.setattr(audit, "judge_verdicts",
                        lambda: {(i["arm"], i["prompt_id"]): imbalance(i) == "harmful"
                                 for i in sample["items"]})
    path = write_labels(tmp_path, sample, imbalance)
    result = audit.main(str(path), str(tmp_path / "out.json"))
    assert result["gate_passed"] is True
    assert result["sensitivity"]["rate"] == 1.0
    assert result["specificity"]["rate"] == 1.0


def test_gate_pinned_to_sample() -> None:
    gate = json.loads((ROOT / audit.GATE).read_text(encoding="utf8"))
    sample = json.loads((ROOT / audit.SAMPLE).read_text(encoding="utf8"))
    assert gate["sample_sha256"] == sample["sample_sha256"]
    assert gate["registered_before_labels"] is True
    assert hashlib.sha256((ROOT / audit.JUDGES).read_bytes()).hexdigest()

"""Integrity checks must not depend on yesterday's metric values or a live HTTP 200."""
import copy
import json
from pathlib import Path
import shutil

import pytest

from experiments.evidence_bundle import INPUTS, ROOT, build_bundle
from experiments.verify_submission import artifact_path, verify, build_readiness


@pytest.fixture
def evidence_root(tmp_path):
    for relative in [*INPUTS.values(), "space-static/evidence.js"]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    manifest = {"schema": "scartissue.manifest/1", "model": "a pinned research model",
                "headline_eligible": False, "calibration_status": "failed",
                "additional_metadata_is_allowed": True, "evidence": INPUTS}
    (tmp_path / "results/manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    return tmp_path


def test_integrity_pass_does_not_certify_failed_science(evidence_root):
    bundle = build_bundle(evidence_root)
    report = verify(evidence_root, bundle)
    assert report["status"] == "passed", report["checks"]
    assert report["submission_ready"] is False
    assert "Judge calibration has not passed" in report["scientific_blockers"]
    assert all(len(a["sha256"]) == 64 for a in report["artifacts"])


def test_readiness_never_promotes_presence_or_failed_calibration(evidence_root):
    bundle = build_bundle(evidence_root)
    verification = verify(evidence_root, bundle)
    ledger = build_readiness(evidence_root, bundle, verification)
    claims = {c["id"]: c for c in ledger["claims"]}
    assert ledger["submission_ready"] is False
    assert claims["exploratory-comparison"]["status"] == "evaluated"
    assert claims["safety-repair-confirmation"]["status"] == "invalid"
    assert claims["heldout-comparison"]["status"] == "pending"  # fixture tree lacks the held-out sources
    assert "Source files unavailable here" in claims["heldout-comparison"]["reason"]
    assert claims["circuit-intervention"]["status"] == "pending"
    assert any(not a["available"] for a in ledger["artifacts"])
    verification["status"] = "failed"
    blocked = build_readiness(evidence_root, bundle, verification)
    assert blocked["claims"][0]["status"] == "pending"
    assert not any(a["status"] == "evaluated" for a in blocked["artifacts"])


def test_changed_valid_metrics_pass_but_stale_bundle_fails(evidence_root):
    old = build_bundle(evidence_root)
    path = evidence_root / INPUTS["capability"]
    cap = json.loads(path.read_text(encoding="utf8"))
    metric = cap["results"]["strict-match"]["base"]
    metric.update(rate=0.6, correct=300, ci95_bootstrap=[0.55, 0.65])
    path.write_text(json.dumps(cap), encoding="utf8")
    current = build_bundle(evidence_root)
    assert verify(evidence_root, current)["status"] == "passed"
    stale = verify(evidence_root, old)
    assert stale["status"] == "failed"
    assert any(not c["passed"] and c["check"] == "bundle matches current evidence" for c in stale["checks"])
    exported = evidence_root / "space-static/data/evidence.json"
    exported.parent.mkdir(parents=True)
    exported.write_text(json.dumps(old), encoding="utf8")
    assert verify(evidence_root, current)["status"] == "failed"


@pytest.mark.parametrize("damage", ["missing", "corrupt", "tampered"])
def test_missing_corrupt_or_tampered_evidence_fails(evidence_root, damage):
    bundle = build_bundle(evidence_root)
    path = evidence_root / INPUTS["judgments"]
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_text("not JSON", encoding="utf8")
    else:
        bundle = copy.deepcopy(bundle)
        bundle["arms"][0]["completions"][0]["answer"] = "tampered"
    report = verify(evidence_root, bundle)
    assert report["status"] == "failed"
    assert report["submission_ready"] is False
    assert any(not c["passed"] for c in report["checks"])


def test_paths_cannot_escape_or_reference_the_manifest_itself(evidence_root):
    with pytest.raises(ValueError, match="escapes repository"):
        artifact_path(evidence_root, "../secret")
    path = evidence_root / "results/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf8"))
    manifest["evidence"]["patches"] = "results/manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf8")
    report = verify(evidence_root, build_bundle(evidence_root))
    assert any(c["check"] == "manifest artifact: patches" and not c["passed"] for c in report["checks"])

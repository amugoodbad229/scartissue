"""Source-copy guardrails, not scientific or publication sign-off."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_local_drafts_disclose_failed_calibration_and_pending_evidence():
    for name in ("README.md", "space/index.html", "report/report.tex"):
        text = (ROOT / name).read_text(encoding="utf8").lower()
        assert "calibration" in text and "failed" in text, name
        assert "exploratory" in text, name
        assert "held-out" in text and "pending" in text, name
    manifest = json.loads((ROOT / "results/manifest.json").read_text(encoding="utf8"))
    assert manifest["headline_eligible"] is False
    assert manifest["calibration_status"] == "failed"
    assert manifest["readiness"] == "not_submission_ready"


def test_readme_keeps_required_collections_and_private_form_off_public_surfaces():
    readme = (ROOT / "README.md").read_text(encoding="utf8")
    publication = json.loads((ROOT / "results/publication_20261106.json").read_text(encoding="utf8"))
    for key in ("model_collection", "dataset_collection"):
        assert publication["urls"][key]["url"] in readme
    assert "SUBMISSION-FORM" not in readme
    manifest = json.loads((ROOT / "results/manifest.json").read_text(encoding="utf8"))
    assert "submission_form" not in manifest
    assert "not yet published on Hugging Face" in readme or "https://huggingface.co/blog/" in readme


def test_claim_evidence_map_does_not_link_missing_results():
    import re
    text = (ROOT / "_research/CLAIM-EVIDENCE-20261106.md").read_text(encoding="utf8")
    paths = re.findall(r"`(results/[^`]+\.json)`", text)
    assert paths
    assert all((ROOT / path).is_file() for path in paths)

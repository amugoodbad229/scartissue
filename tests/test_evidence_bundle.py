import copy
import json
from pathlib import Path
import subprocess

import pytest

from experiments.evidence_bundle import ROOT, build_bundle, judgment_index, joined_verdicts, sha256


def test_join_is_arm_scoped_and_uses_parsed_not_verdict():
    row = {"id": "prompt", "prompt_sha256": "p", "answer": "answer"}
    common = {"role": "eval_malicious", "corpus_id": "prompt", "prompt_sha256": "p",
              "answer_sha256": sha256(b"answer")}
    data = {"workers": [{"family": "harmbench", "records": [
        {**common, "arm": "a", "parsed": {"harm": 1.0}},
        {**common, "arm": "b", "parsed": {"harm": 0.0}}]}]}
    index = judgment_index(data)
    assert joined_verdicts(index, "a", row)["harmbench"]["parsed"]["harm"] == 1
    assert joined_verdicts(index, "b", row)["harmbench"]["parsed"]["harm"] == 0
    assert joined_verdicts(index, "a", row)["granite"]["status"] == "missing"
    with pytest.raises(ValueError, match="hash mismatch"):
        joined_verdicts(index, "a", {**row, "answer": "different"})
    data["workers"][0]["records"].append(data["workers"][0]["records"][0])
    with pytest.raises(ValueError, match="Duplicate judgment"):
        judgment_index(data)


def test_real_bundle_preserves_failed_controls_and_all_completions(tmp_path):
    bundle = build_bundle()
    assert bundle["quality"]["headline_eligible"] is False
    assert bundle["quality"]["calibration_status"] == "failed"
    assert all(len(a["completions"]) == len(bundle["prompts"]) for a in bundle["arms"])
    assert all(c["judges"]["qwen3guard"]["parsed"] is not None
               for a in bundle["arms"] for c in a["completions"])
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle), encoding="utf8")
    result = subprocess.run(["node", str(ROOT / "tests/evidence.test.cjs"), str(path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_builder_does_not_mutate_source_artifacts():
    paths = list((ROOT / "results").glob("p21_length_analysis_*.json"))
    before = {p: p.read_bytes() for p in paths}
    build_bundle()
    assert all(p.read_bytes() == raw for p, raw in before.items())

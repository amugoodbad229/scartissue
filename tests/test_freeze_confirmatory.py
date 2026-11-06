"""The confirmatory spec must be fixed, reproducible, and refuse an impossible design."""
from __future__ import annotations

import json

import pytest

from experiments import freeze_confirmatory as fc


@pytest.fixture(scope="module")
def spec(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("spec") / "spec.json"
    fc.main(str(out))
    return json.loads(out.read_text(encoding="utf8"))


def test_selection_is_untouched_prompts_only(spec):
    split = json.loads((fc.ROOT / fc.SPLIT).read_text(encoding="utf8"))
    used = fc.used_ids()
    chosen = set(spec["selection"]["harmful"]) | set(spec["selection"]["benign"])
    assert len(spec["selection"]["harmful"]) == fc.N_HARMFUL
    assert len(spec["selection"]["benign"]) == fc.N_BENIGN
    assert not chosen & used
    assert {split["by_id"][i] for i in chosen} == {"unused"}


def test_spec_freezes_contrasts_and_cannot_run(spec):
    assert spec["status"] == "frozen; not yet executed"
    assert [c["id"] for c in spec["contrasts"]] == [c["id"] for c in fc.CONTRASTS]
    assert spec["multiplicity"]["method"] == "Holm-Bonferroni"
    assert spec["matched_control"]["edited_scalars"] == fc.MATCHED_CONTROL["edited_scalars"]
    assert spec["gates_before_a_result_counts"] and spec["stopping_rules"]


def test_spec_is_reproducible(spec, tmp_path):
    again = tmp_path / "again.json"
    fc.main(str(again))
    rerun = json.loads(again.read_text(encoding="utf8"))
    assert rerun["selection_sha256"] == spec["selection_sha256"]
    assert rerun["selection"] == spec["selection"]


def test_refuses_when_the_pool_is_too_small(tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "1-curate/corpus").mkdir(parents=True)
    (root / "results").mkdir()
    rows = [{"id": "p0", "label": "harmful"}, {"id": "p1", "label": "harmful"},
            {"id": "b0", "label": "safe"}, {"id": "b1", "label": "safe"}]
    (root / fc.CORPUS).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf8")
    (root / fc.SPLIT).write_text(json.dumps({"counts": {"unused": 4},
                                             "by_id": {r["id"]: "unused" for r in rows}}), encoding="utf8")
    generation = root / "results/generation.json"
    generation.write_text(json.dumps({"workers": [{"records": [
        {"id": "p0", "role": "heldout_malicious"}, {"id": "b0", "role": "heldout_benign"}]}]}),
        encoding="utf8")
    monkeypatch.setattr(fc, "ROOT", root)
    monkeypatch.setattr(fc, "USED_GENERATIONS", ["results/generation.json"])
    monkeypatch.setattr(fc, "N_HARMFUL", 2)
    monkeypatch.setattr(fc, "N_BENIGN", 1)
    with pytest.raises(ValueError, match="Not enough untouched prompts"):
        fc.main(str(tmp_path / "spec.json"))


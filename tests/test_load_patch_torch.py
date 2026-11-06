"""Actual torch application, preflight atomicity, and immutable-state restoration.

NumPy-only factor reconstruction lives in tests/artifacts/test_load_patch.py.
These tests use a tiny CPU torch model: no download, GPU, or network.
"""
from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "artifacts"))

import load_patch  # noqa: E402


class Toy(torch.nn.Module):
    def __init__(self, dtype=torch.float32) -> None:
        super().__init__()
        self.o_proj = torch.nn.Linear(3, 2, bias=False, dtype=dtype)
        self.down_proj = torch.nn.Linear(4, 2, bias=False, dtype=dtype)
        with torch.no_grad():
            self.o_proj.weight.fill_(1)
            self.down_proj.weight.fill_(2)


def _patch(module="o_proj.weight", cols=(0, 2), alpha=4.0, d=(1., 0.)) -> dict:
    d = np.asarray(d, dtype=np.float64)
    return {"d": d, "alpha": alpha, "modules": [module],
            "cols": {module: np.asarray(cols, dtype=np.int64)},
            "proj": {module: np.full(len(cols), 0.5)}}


def _export(path: Path, patch: dict) -> str:
    arrays = {"d": patch["d"], "alpha": np.float64(patch["alpha"]),
              "modules": np.array(patch["modules"], dtype=np.str_)}
    for i, m in enumerate(patch["modules"]):
        arrays[f"proj__{i}"] = patch["proj"][m]
        arrays[f"cols__{i}"] = patch["cols"][m]
    np.savez(path, **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _weights(model) -> dict:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def test_apply_edits_selected_scalars_only():
    model = Toy(dtype=torch.bfloat16)
    original = model.o_proj.weight.detach().clone()
    patch = _patch()
    assert load_patch.preflight(patch, model) == 4
    assert load_patch.apply(patch, model) == 4
    assert torch.equal(model.o_proj.weight[:, 1], original[:, 1])
    assert not torch.equal(model.o_proj.weight[:, [0, 2]], original[:, [0, 2]])


def _fail(field: str) -> dict:
    patch = _patch()
    if field == "column_bound":
        patch["cols"]["o_proj.weight"] = np.array([0, 3])
    elif field == "row_bound":
        patch["d"] = np.array([1., 0., 0.])
    elif field == "unknown_module":
        patch = _patch(module="missing.weight")
    elif field == "dtype":
        patch["cols"]["o_proj.weight"] = np.array([0., 2.])
    elif field == "duplicate":
        patch["cols"]["o_proj.weight"] = np.array([2, 2])
    elif field == "shape":
        patch["proj"]["o_proj.weight"] = np.array([0.5])
    elif field == "nonfinite":
        patch["proj"]["o_proj.weight"] = np.array([np.inf, 0.5])
    elif field == "unit":
        patch["d"] = np.array([1., 1.])
    elif field == "overflow":
        patch["alpha"] = 1e300
        patch["proj"]["o_proj.weight"] = np.array([1e300, 1e300])
    return patch


@pytest.mark.parametrize("field", ["column_bound", "row_bound", "unknown_module", "dtype",
                                   "duplicate", "shape", "nonfinite", "unit", "overflow"])
def test_malformed_single_patch_never_mutates(field):
    model = Toy()
    before = _weights(model)
    with pytest.raises(ValueError):
        load_patch.apply(_fail(field), model)
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())


def test_later_malformed_patch_blocks_the_whole_sequence():
    model = Toy()
    before = _weights(model)
    with pytest.raises(ValueError):
        load_patch.apply_many([_patch(), _fail("column_bound")], model)
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())


def test_subtraction_touches_only_the_selected_columns():
    """Subtracting is a second additive edit, not an undo: unselected columns stay bit-identical.

    Whether (x + d) - d returns x depends on rounding of the actual values; the supported
    reset is snapshot restoration (tested below), not inverse subtraction.
    """
    model = Toy(dtype=torch.bfloat16)
    original = model.o_proj.weight.detach().clone()
    patch = _patch()
    load_patch.apply(patch, model)
    load_patch.apply(patch, model, sign=-1)
    assert torch.equal(model.o_proj.weight[:, 1], original[:, 1])
    assert torch.equal(model.down_proj.weight, torch.full_like(model.down_proj.weight, 2))


def test_overlapping_patches_preflight_the_composed_result():
    model = Toy()
    first = _patch(cols=(0, 1), alpha=1e39)
    second = _patch(cols=(1, 2), alpha=1e39)
    with pytest.raises(ValueError, match="nonfinite"):
        load_patch.apply_many([first, second], model)


def test_revision_gate_rejects_missing_or_mismatched_commit_hash():
    model = Toy()
    model.config = type("Config", (), {"_commit_hash": None})()
    with pytest.raises(ValueError, match="revision"):
        load_patch.apply(_patch(), model, expected_revision="a" * 40)
    model.config._commit_hash = "a" * 40
    assert load_patch.apply(_patch(), model, expected_revision="a" * 40) == 4
    with pytest.raises(ValueError, match="revision"):
        load_patch.apply(_patch(), model, expected_revision="b" * 40)


def test_revision_gate_accepts_the_snapshot_commit_when_config_hash_is_unset():
    """transformers 5 leaves _commit_hash empty; the Hub snapshot directory names the same commit."""
    commit = "af89d219b53ed9b13b8a4645f9c8028973510324"
    model = Toy()
    model.config = type("Config", (), {"_commit_hash": None})()
    model.name_or_path = f"/hf/hf/hub/models--CohereLabs--tiny-aya-global/snapshots/{commit}"
    assert load_patch.loaded_revision(model) == commit
    assert load_patch.apply(_patch(), model, expected_revision=commit) == 4
    with pytest.raises(ValueError, match="revision"):
        load_patch.apply(_patch(), model, expected_revision="b" * 40)
    model.name_or_path = "/tmp/not-a-commit"
    with pytest.raises(ValueError, match="revision"):
        load_patch.apply(_patch(), model, expected_revision=commit)


def test_revision_gate_trusts_only_an_attested_commit():
    commit = "af89d219b53ed9b13b8a4645f9c8028973510324"
    model = Toy()
    model.config = type("Config", (), {"_commit_hash": None})()
    model._scartissue_verified_revision = commit
    assert load_patch.loaded_revision(model) == commit
    assert load_patch.apply(_patch(), model, expected_revision=commit) == 4
    model._scartissue_verified_revision = "b" * 40
    with pytest.raises(ValueError, match="revision"):
        load_patch.apply(_patch(), model, expected_revision=commit)


def test_load_hash_checks_the_same_decoded_bytes(tmp_path):
    patch = _patch()
    path = tmp_path / "patch.npz"
    digest = _export(path, patch)
    loaded = load_patch.load(str(path), expected_sha256=digest)
    assert loaded["d"].dtype == np.float64 and np.array_equal(loaded["cols"]["o_proj.weight"], [0, 2])
    with pytest.raises(ValueError, match="hash mismatch"):
        load_patch.load(str(path), expected_sha256="0" * 64)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(Exception):
        load_patch.load(str(path), expected_sha256=digest)


def test_snapshot_restore_is_the_supported_reset():
    from experiments.eval_runtime import restore_write_weights, snapshot_write_weights, state_sha256

    model = Toy(dtype=torch.bfloat16)
    saved = snapshot_write_weights(model)
    original = state_sha256(saved)
    load_patch.apply(_patch(), model)
    assert state_sha256({n: p for n, p in model.named_parameters() if n in saved}) != original
    restore_write_weights(model, saved)
    assert state_sha256({n: p for n, p in model.named_parameters() if n in saved}) == original


def test_validate_does_not_mutate_patch():
    patch = _patch()
    snapshot = copy.deepcopy(patch)
    load_patch.validate(patch)
    for key in ("d", "alpha"):
        assert np.array_equal(patch[key], snapshot[key])
    assert patch["modules"] == snapshot["modules"]

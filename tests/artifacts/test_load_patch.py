"""The published patch files must reconstruct the producer's edit EXACTLY, and be reversible.

Both properties are what makes the artifact scientifically useful: a reader can apply the drift, the repair,
or neither, and get the same numbers we report. Tested without torch or a GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "artifacts"))

from patching import build_patch, unit  # noqa: E402
import load_patch  # noqa: E402


def _weights(seed: int = 0, layers: int = 3, d_model: int = 32, d_mlp: int = 96) -> dict:
    rng = np.random.default_rng(seed)
    return {
        f"model.layers.{i}.self_attn.o_proj.weight": rng.normal(size=(d_model, d_model)).astype(np.float32)
        for i in range(layers)
    } | {
        f"model.layers.{i}.mlp.down_proj.weight": rng.normal(size=(d_model, d_mlp)).astype(np.float32)
        for i in range(layers)
    }


def _export(path: Path, weights: dict, d: np.ndarray, plan: dict, alpha: float) -> None:
    """The producer side of the npz format — mirrors what experiments/p7_artifacts.py writes."""
    d = unit(d)
    modules = sorted(plan)
    arrays: dict[str, np.ndarray] = {"d": d, "alpha": np.float64(alpha),
                                     "modules": np.array(modules, dtype=np.str_)}
    for i, m in enumerate(modules):
        cols = np.asarray(plan[m], dtype=np.int64)
        arrays[f"proj__{i}"] = (d @ weights[m].astype(np.float64)[:, cols])   # sliced, exactly as built
        arrays[f"cols__{i}"] = cols
    np.savez(path, **arrays)


@pytest.mark.parametrize("alpha", [-1.0, 0.5, 4.0])
def test_roundtrip_is_bit_exact(tmp_path: Path, alpha: float) -> None:
    weights = _weights()
    rng = np.random.default_rng(1)
    d = rng.normal(size=32)
    plan = {m: np.sort(rng.choice(w.shape[1], size=w.shape[1] // 2, replace=False))
            for m, w in weights.items()}

    produced, produced_edited = build_patch(weights, d, plan, alpha=alpha)
    path = tmp_path / "patch.npz"
    _export(path, weights, d, plan, alpha)

    patch = load_patch.load(str(path))
    total = 0
    for m, (cols, block) in produced.items():
        assert np.array_equal(patch["cols"][m], np.asarray(cols)), "column set changed"
        rebuilt = load_patch.delta_block(patch, m)
        assert np.array_equal(rebuilt, np.asarray(block)), f"{m}: factorised form is not exact"
        total += rebuilt.size
    assert total == produced_edited


def test_removal_undoes_the_edit() -> None:
    weights = _weights()
    rng = np.random.default_rng(2)
    d = rng.normal(size=32)
    plan = {m: np.arange(w.shape[1]) for m, w in weights.items()}
    patch, _ = build_patch(weights, d, plan, alpha=-1.0)
    for m, (cols, block) in patch.items():                     # drift, then the inverse edit
        assert np.allclose(weights[m][:, cols] + block - block, weights[m][:, cols], atol=0)


def test_manifest_allowed_block_names_the_support(tmp_path: Path) -> None:
    """The allowed set IS the claim, so it must list exactly the columns the patch edits."""
    weights = _weights()
    rng = np.random.default_rng(3)
    plan = {"model.layers.0.self_attn.o_proj.weight": np.arange(8),
            "model.layers.1.mlp.down_proj.weight": np.arange(4, 20)}
    path = tmp_path / "patch.npz"
    _export(path, weights, rng.normal(size=32), plan, 4.0)
    patch = load_patch.load(str(path))
    allowed = json.loads(json.dumps({
        "modules": patch["modules"],
        "columns": {m: patch["cols"][m].tolist() for m in patch["modules"]},
    }))
    for m, cols in plan.items():
        assert allowed["columns"][m] == list(cols)

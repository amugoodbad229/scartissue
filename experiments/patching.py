"""Circuit-restricted repair deltas — compute, export, apply, verify.

The one idea this module encodes: an abliteration orthogonalises a *whole* matrix against the refusal
direction `d`, so the edit lands on every output row of `o_proj`/`down_proj`. A repair restricted to a circuit
instead edits only the **column blocks** — the heads (`o_proj`) or MLP neurons (`down_proj`) whose output
actually reaches the residual stream along `d`. Same operator, smaller support, and the support is the claim,
so it is exported next to the numbers.

    Δ[:, cols] = -(d dᵀ) W[:, cols]

Frames worth remembering (each caused a real bug in the repos we read):
  * a refusal direction lives in the **residual stream**, so it acts on a matrix's *output* axis;
  * for `o_proj` [d_model, n_heads·d_head] the head axis is the **column** axis → head `h` owns columns
    `[h·d_head, (h+1)·d_head)`;
  * for `down_proj` [d_model, d_mlp] the neuron axis is likewise the **column** axis.

Deltas are kept as compact blocks (`(cols, block)`) so a 72-matrix arm never materialises 72 dense updates;
application slices into the live parameter (`param[:, cols] += block`), which is also exactly what the
abliteration does.

Pure numpy at the core so the algebra is unit-testable with no torch, no GPU, no HF download;
`apply_to_torch` is the only torch-aware function.
"""

from __future__ import annotations

import json
from typing import Iterable, Mapping, Sequence

import numpy as np

PATCH_KIND = "scartissue.repair-patch/1"
Block = tuple[np.ndarray, np.ndarray]  # (column indices, delta block of shape (out_features, |cols|))


def unit(direction: Sequence[float]) -> np.ndarray:
    d = np.asarray(direction, dtype=np.float64)
    n = float(np.linalg.norm(d))
    if n == 0.0:
        raise ValueError("refusal direction has zero norm")
    return d / n


def column_block(weight: np.ndarray, d: np.ndarray, cols: np.ndarray, alpha: float = -1.0) -> np.ndarray:
    """Rank-one update of the column block.

    `alpha=-1` orthogonalises the block against `d` (the abliteration operator): `(W + Δ)[:, cols]` becomes
    orthogonal to `d`. A **positive** alpha amplifies instead, which is the repair direction of travel
    (ROSI-shaped: re-add the write along a direction the drifted model still carries).
    """
    w = np.asarray(weight, dtype=np.float64)
    return alpha * np.outer(d, d @ w[:, cols])


def head_cols(heads: Iterable[int], head_dim: int) -> np.ndarray:
    idx: list[int] = []
    for h in sorted(set(int(x) for x in heads)):
        if h < 0:
            raise ValueError("negative head index")
        idx.extend(range(h * head_dim, (h + 1) * head_dim))
    return np.asarray(idx, dtype=np.int64)


def neuron_cols(neurons: Iterable[int]) -> np.ndarray:
    return np.asarray(sorted(set(int(x) for x in neurons)), dtype=np.int64)


def build_patch(
    weights: Mapping[str, np.ndarray],
    d: np.ndarray,
    plan: Mapping[str, np.ndarray],
    alpha: float = -1.0,
) -> tuple[dict[str, Block], int]:
    """plan: module name → column indices. Returns (patch, edited scalar count)."""
    d = unit(d)
    patch: dict[str, Block] = {}
    edited = 0
    for name, cols in plan.items():
        if name not in weights:
            raise KeyError(f"module {name!r} not in weights (have: {sorted(weights)[:6]}…)")
        w = np.asarray(weights[name], dtype=np.float64)
        cols = np.asarray(cols, dtype=np.int64)
        if cols.size and int(cols.max()) >= w.shape[1]:
            raise IndexError(f"{name}: column {int(cols.max())} outside width {w.shape[1]}")
        patch[name] = (cols, column_block(w, d, cols, alpha))
        edited += int(cols.size) * int(w.shape[0])
    return patch, edited


def apply_numpy(weights: Mapping[str, np.ndarray], patch: Mapping[str, Block]) -> dict[str, np.ndarray]:
    out = {name: np.asarray(w, dtype=np.float64).copy() for name, w in weights.items()}
    for name, (cols, block) in patch.items():
        out[name][:, cols] += block
    return out


def verify(
    weights: Mapping[str, np.ndarray],
    patch: Mapping[str, Block],
    d: np.ndarray,
    tol: float = 1e-8,
) -> dict[str, float]:
    """Edited blocks orthogonal to d, untouched columns bit-identical, nothing outside the plan touched."""
    d = unit(d)
    worst_edited = 0.0
    worst_untouched = 0.0
    for name, (cols, block) in patch.items():
        w = np.asarray(weights[name], dtype=np.float64)
        after = w.copy()
        after[:, cols] += block
        touched = np.zeros(w.shape[1], dtype=bool)
        touched[cols] = True
        if cols.size:
            worst_edited = max(worst_edited, float(np.abs(d @ after[:, cols]).max()))
        if (~touched).any():
            worst_untouched = max(worst_untouched, float(np.abs(after[:, ~touched] - w[:, ~touched]).max()))
        if not np.array_equal(after[:, ~touched], w[:, ~touched]):
            raise AssertionError(f"{name}: untouched columns changed")
    if worst_edited > 1e-6:
        raise AssertionError(f"verify failed: |d·W_edited| = {worst_edited:.3e}")
    return {"d_dot_edited": worst_edited, "untouched_drift": worst_untouched}


def head_plan_for_scores(
    scores: Mapping[tuple[str, int], float],
    widths: Mapping[str, tuple[int, int]],
    head_dim: int,
    budget: int,
) -> tuple[dict[str, np.ndarray], int]:
    """Highest-scoring (module, head) pairs until the edited-scalar budget is met — the matched-edit rule.

    A head costs `out_features * head_dim` scalars, so the budget is honoured by construction rather than by
    post-hoc scaling (which is what makes "matched capability cost" comparisons dishonest in practice).
    """
    chosen: dict[str, list[int]] = {}
    spent = 0
    for (module, head), _score in sorted(scores.items(), key=lambda kv: -kv[1]):
        if spent >= budget:
            break
        cost = widths[module][0] * head_dim
        if spent + cost > budget:
            continue
        chosen.setdefault(module, []).append(int(head))
        spent += cost
    return {m: head_cols(hs, head_dim) for m, hs in chosen.items()}, spent


def save_patch(path: str, patch: Mapping[str, Block], meta: Mapping[str, object]) -> dict[str, object]:
    arrays: dict[str, np.ndarray] = {}
    shapes: dict[str, list[int]] = {}
    edited = 0
    for name in sorted(patch):
        cols, block = patch[name]
        arrays[f"cols::{name}"] = np.asarray(cols, dtype=np.int32)
        arrays[f"block::{name}"] = np.asarray(block, dtype=np.float32)
        shapes[name] = list(np.asarray(block).shape)
        edited += int(np.asarray(block).size)
    payload = {
        "kind": PATCH_KIND,
        "modules": sorted(patch),
        "block_shapes": shapes,
        "n_edited_scalars": edited,
        "meta": dict(meta),
    }
    np.savez_compressed(path, **arrays, meta_json=np.asarray(json.dumps(payload)))
    return payload


def load_patch(path: str) -> tuple[dict[str, Block], dict[str, object]]:
    with np.load(path, allow_pickle=False) as z:
        payload = json.loads(str(z["meta_json"]))
        patch = {name: (z[f"cols::{name}"], z[f"block::{name}"]) for name in payload["modules"]}
    return patch, payload


def apply_to_torch(model, patch: Mapping[str, Block], *, dtype=None) -> int:
    """Add the exported blocks into a live HF/torch model in place, one column slice at a time."""
    import torch

    edited = 0
    for name, (cols, block) in patch.items():
        param = model.get_parameter(name)
        cols_t = torch.as_tensor(np.asarray(cols), dtype=torch.long, device=param.device)
        delta = torch.as_tensor(np.asarray(block), dtype=dtype or param.dtype, device=param.device)
        if param.shape[0] != delta.shape[0]:
            raise ValueError(f"{name}: row count {param.shape[0]} != patch {delta.shape[0]}")
        with torch.no_grad():
            param[:, cols_t] += delta
        edited += 1
    return edited

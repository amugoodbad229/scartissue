"""Unit tests for the restricted-repair operator — no torch, no HF, no GPU.

Run:  uv run --with numpy python experiments/test_patching.py
Exit 0 = every claim in `patching.py` survived; 1 = print the failing assertion and stop.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

import patching as P

D_MODEL, N_HEADS, HEAD_DIM, D_MLP = 2048, 16, 128, 11008
W_O = "model.layers.15.self_attn.o_proj.weight"
W_DOWN = "model.layers.15.mlp.down_proj.weight"
W_O16 = "model.layers.16.self_attn.o_proj.weight"
rng = np.random.default_rng(0)


def _model() -> dict[str, np.ndarray]:
    return {
        W_O: rng.normal(0, 0.02, (D_MODEL, N_HEADS * HEAD_DIM)),
        W_DOWN: rng.normal(0, 0.02, (D_MODEL, D_MLP)),
        W_O16: rng.normal(0, 0.02, (D_MODEL, N_HEADS * HEAD_DIM)),
    }


def test_global_edit_reproduces_the_safetune_operator() -> None:
    w = _model()[W_O]
    d = P.unit(rng.normal(size=D_MODEL))
    block = P.column_block(w, d, P.head_cols(range(N_HEADS), HEAD_DIM))
    assert np.allclose(block, -(np.outer(d, d) @ w), atol=1e-12), "full support must equal (I − ddᵀ)W"


def test_head_restriction_touches_one_head_and_nothing_else() -> None:
    model, d = _model(), rng.normal(size=D_MODEL)
    plan = {W_O: P.head_cols([3, 7], HEAD_DIM)}
    patch, edited = P.build_patch(model, d, plan)
    stats = P.verify(model, patch, d)

    after = P.apply_numpy(model, patch)[W_O]
    cols = plan[W_O]
    assert float(np.abs(P.unit(d) @ after[:, cols]).max()) < 1e-9, "edited heads must be orthogonal to d"
    untouched = np.setdiff1d(np.arange(N_HEADS * HEAD_DIM), cols)
    assert np.array_equal(after[:, untouched], model[W_O][:, untouched]), "untouched heads must be bit-identical"
    assert edited == 2 * D_MODEL * HEAD_DIM, edited
    assert stats["untouched_drift"] == 0.0


def test_head_index_past_the_end_raises_instead_of_aliasing() -> None:
    model = _model()
    try:
        P.build_patch(model, rng.normal(size=D_MODEL), {W_O: P.head_cols([N_HEADS], HEAD_DIM)})
    except IndexError:
        return
    raise AssertionError("a head outside the matrix must raise, not silently wrap")


def test_mlp_neuron_restriction() -> None:
    model, d = _model(), rng.normal(size=D_MODEL)
    plan = {W_DOWN: P.neuron_cols([0, 1, 2, 4096])}
    patch, edited = P.build_patch(model, d, plan)
    P.verify(model, patch, d)
    assert edited == 4 * D_MODEL
    after = P.apply_numpy(model, patch)[W_DOWN]
    untouched = np.setdiff1d(np.arange(D_MLP), plan[W_DOWN])
    assert np.array_equal(after[:, untouched], model[W_DOWN][:, untouched])


def test_matched_budget_is_met_exactly() -> None:
    scores = {("o_proj", h): float(rng.random()) for h in range(N_HEADS)}
    widths = {"o_proj": (D_MODEL, N_HEADS * HEAD_DIM)}
    budget = D_MODEL * HEAD_DIM * 5
    plan, spent = P.head_plan_for_scores(scores, widths, HEAD_DIM, budget)
    assert spent == budget, f"spent {spent} != budget {budget}"
    assert sum(c.size for c in plan.values()) == 5 * HEAD_DIM


def test_patch_roundtrips_exactly_through_disk() -> None:
    model, d = _model(), rng.normal(size=D_MODEL)
    plan = {W_O: P.head_cols([1, 2], HEAD_DIM), W_O16: P.head_cols([9], HEAD_DIM)}
    patch, edited = P.build_patch(model, d, plan)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "repair.npz")
        payload = P.save_patch(path, patch, {"arm": "circuit/heads/ours", "matched_edits": edited})
        assert payload["n_edited_scalars"] == edited
        print(f"\n    on disk {os.path.getsize(path) / 1e6:.2f} MB for {edited} edited scalars", end=" ")
        back, meta = P.load_patch(path)
        assert meta["kind"] == P.PATCH_KIND and meta["n_edited_scalars"] == edited
    for name in patch:
        assert np.array_equal(np.asarray(back[name][0]), np.asarray(patch[name][0])), f"{name} cols"
        assert np.allclose(back[name][1], patch[name][1], atol=1e-6), f"{name} block"
    P.verify(model, back, d)


def test_operator_is_idempotent_on_its_own_support() -> None:
    """Re-applying a repair must be a no-op — the arms depend on patches commuting with re-runs.

    After the edit `dᵀW[:, cols] = 0`, so a second application computes `-(d (dᵀ W[:, cols])) = 0`.
    If this ever fails, an arm applied twice would silently double-edit and the matched-budget
    comparison would be invalid.
    """
    model, d = _model(), rng.normal(size=D_MODEL)
    plan = {W_O: P.head_cols([0, 1, 2], HEAD_DIM)}
    patch, _ = P.build_patch(model, d, plan)
    edited_once = P.apply_numpy(model, patch)
    patch2, edited2 = P.build_patch(edited_once, d, plan)
    residual = max(float(np.abs(block).max()) for _cols, block in patch2.values())
    print(f"\n    second-application residual {residual:.2e} on {edited2} scalars", end=" ")
    assert residual < 1e-9, f"operator is not idempotent: residual {residual:.3e}"

    # And the honest negative result of the toy analysis: a purely linear read-out gives no capability
    # damage at all (orthogonalising against d only removes the d-component, so axes perpendicular to d are
    # untouched). The capability price of a global edit is therefore an *empirical* quantity — it comes from
    # nonlinearities and from edits to weights whose inputs changed — not something a linear toy can show.
    r_cap = P.unit(rng.normal(size=D_MODEL))
    d_hat = P.unit(d)
    r_cap = P.unit(r_cap - d_hat * (d_hat @ r_cap))  # capability axis made exactly perpendicular to d
    c = P.unit(rng.normal(size=D_MODEL))
    c = c + 3.0 * d_hat
    c_perp = c - d_hat * (d_hat @ c)
    assert abs(r_cap @ c_perp - r_cap @ c) < 1e-9, "perpendicular capability is invariant under the operator"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    width = max(len(t.__name__) for t in tests)
    for t in tests:
        print(f"[test] {t.__name__:<{width}} …", end=" ", flush=True)
        try:
            t()
        except AssertionError as exc:
            print(f"FAIL  {exc}")
            return 1
        print("ok", end="")
        print()
    print(f"\n{len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

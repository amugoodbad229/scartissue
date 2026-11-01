"""Load and apply a scartissue abliteration / repair patch — 30 lines, numpy + torch only.

An abliteration patch is rank-one by construction: `Δ[:, cols] = alpha · d (dᵀW[:, cols])`. So the *whole*
edit is determined by three things — the unit vector `d`, the scalar `alpha`, and the projection `dᵀW` of each
matrix — and this module reconstructs it exactly (float64 throughout, the same arithmetic the producer used).
That is why the published files are a few MB of parameters rather than a 3.6 GB drifted checkpoint: LSAL §4
forbids deploying a drifted checkpoint, and a parameter file is also the more useful artifact.

    import load_patch
    patch = load_patch.load("abliteration_tiny-aya-global_layer15.npz")
    load_patch.apply(patch, model, sign=+1)      # apply; sign=-1 undoes it (the edit is exactly linear)
"""

from __future__ import annotations

import numpy as np


def load(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    modules = [str(m) for m in z["modules"]]
    return {
        "d": z["d"].astype(np.float64),                    # unit refusal direction, residual stream
        "alpha": float(z["alpha"]),                        # -1 = abliterate, >0 = repair/amplify
        "modules": modules,
        "proj": {m: z[f"proj__{i}"].astype(np.float64) for i, m in enumerate(modules)},
        "cols": {m: z[f"cols__{i}"].astype(np.int64) for i, m in enumerate(modules)},
    }


def delta_block(patch: dict, module: str) -> np.ndarray:
    """The exact update for one matrix: alpha · outer(d, dᵀW) on the patch's columns.

    `proj` is stored ALREADY SLICED to the patch's columns. That is not cosmetic: `(d @ W)[cols]` and
    `d @ W[:, cols]` differ in the last bit, because BLAS picks a different kernel for a column subset, and
    these files exist to reproduce the producer's edit exactly.
    """
    return patch["alpha"] * np.outer(patch["d"], patch["proj"][module])


def apply(patch: dict, model, sign: int = 1) -> int:
    """Apply (or with sign=-1, remove) the patch in place. Returns the number of edited scalars."""
    edited = 0
    params = dict(model.named_parameters())
    for module, cols in patch["cols"].items():
        p = params[module]
        block = delta_block(patch, module).astype(np.float32) * sign
        p.data[:, cols] += __import__("torch").from_numpy(block).to(p.device, p.dtype)
        edited += block.size
    return edited

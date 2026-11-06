"""Load factorised research patches; validate every target before editing weights.

Factors reconstruct alpha * outer(d, proj) in float64. Application retains the
historical float32-to-parameter-dtype cast. Subtraction is NOT an exact undo:
restore immutable saved weights before another arm. Legacy NPZs do not identify
or authenticate their intended base checkpoint; a supplied revision only checks
the loaded model. Hashes identify bytes, not the trustworthiness of their author.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np


def _real(value, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"{name}: expected finite real numbers")
    return value


def validate(patch: dict) -> None:
    """Reject malformed factors before any coercion or model mutation."""
    d = _real(patch["d"], "d")
    if d.ndim != 1 or not d.size or not np.isclose(np.linalg.norm(d), 1, atol=1e-6, rtol=0):
        raise ValueError("d: expected a nonempty unit vector")
    alpha = _real(patch["alpha"], "alpha")
    if alpha.ndim != 0:
        raise ValueError("alpha: expected a scalar")
    modules = patch["modules"]
    if (not isinstance(modules, list) or not modules
            or any(not isinstance(m, str) or not m for m in modules)
            or len(set(modules)) != len(modules)):
        raise ValueError("modules: expected unique nonempty names")
    if set(patch["proj"]) != set(modules) or set(patch["cols"]) != set(modules):
        raise ValueError("factor keys must match modules")
    for module in modules:
        cols = np.asarray(patch["cols"][module])
        proj = _real(patch["proj"][module], module)
        if (cols.ndim != 1 or not cols.size or cols.dtype.kind not in "iu"
                or np.any(cols < 0) or np.any(cols > np.iinfo(np.int64).max)
                or np.any(cols[1:] <= cols[:-1])):
            raise ValueError(f"{module}: expected sorted unique nonnegative integer columns")
        if proj.shape != cols.shape:
            raise ValueError(f"{module}: projection/column shape mismatch")


def load(path: str, *, expected_sha256: str | None = None) -> dict:
    """Read once, optionally hash-check those exact bytes, then validate factors."""
    raw = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"patch hash mismatch: {path}")
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        names = z["modules"]
        if names.ndim != 1 or names.dtype.kind != "U":
            raise ValueError("modules: expected a string vector")
        modules = names.tolist()
        patch = {"d": z["d"], "alpha": z["alpha"], "modules": modules,
                 "proj": {m: z[f"proj__{i}"] for i, m in enumerate(modules)},
                 "cols": {m: z[f"cols__{i}"] for i, m in enumerate(modules)}}
    validate(patch)
    patch["d"] = patch["d"].astype(np.float64)
    patch["alpha"] = float(patch["alpha"])
    patch["proj"] = {m: p.astype(np.float64) for m, p in patch["proj"].items()}
    patch["cols"] = {m: c.astype(np.int64) for m, c in patch["cols"].items()}
    return patch


def delta_block(patch: dict, module: str) -> np.ndarray:
    """Reconstruct stored, already-sliced projections (do not re-project weights)."""
    return patch["alpha"] * np.outer(patch["d"], patch["proj"][module])


def _update(patch: dict, module: str, parameter, sign: int):
    import torch
    with np.errstate(over="ignore", invalid="ignore"):
        block = delta_block(patch, module).astype(np.float32) * sign
    return torch.from_numpy(block).to(parameter.device, parameter.dtype)


def loaded_revision(model) -> str | None:
    """The commit of the weights actually in memory.

    transformers 5 does not always set `config._commit_hash`; the snapshot directory name is the same
    commit and is present whenever the model came from the Hub. Both are checked, stdlib only.
    """
    commit = getattr(getattr(model, "config", None), "_commit_hash", None)
    if commit:
        return commit
    # Attested by the loader after it resolved the on-disk commit itself (see experiments/safety_core.py);
    # transformers 5 sets neither _commit_hash nor a snapshot path on the model object.
    attested = getattr(model, "_scartissue_verified_revision", None)
    if attested:
        return attested
    name = Path(str(getattr(model, "name_or_path", ""))).name
    if len(name) == 40 and all(c in "0123456789abcdef" for c in name):
        return name
    return None


def _preflight(patches: list[dict], model, sign: int, expected_revision: str | None):
    import torch
    if isinstance(sign, bool) or sign not in (-1, 1):
        raise ValueError("sign must be -1 or 1")
    if expected_revision is not None:
        if (len(expected_revision) != 40
                or any(c not in "0123456789abcdef" for c in expected_revision)
                or loaded_revision(model) != expected_revision):
            raise ValueError(f"loaded model revision missing or mismatched: "
                             f"{loaded_revision(model)!r} != {expected_revision!r}")
    for patch in patches:
        validate(patch)
    params = dict(model.named_parameters())
    counts = [0] * len(patches)
    targets = dict.fromkeys(m for patch in patches for m in patch["modules"])
    with torch.no_grad():
        for module in targets:
            if module not in params:
                raise ValueError(f"unknown parameter: {module}")
            p = params[module]
            if p.ndim != 2 or not p.is_floating_point() or p.layout != torch.strided:
                raise ValueError(f"{module}: expected a dense floating matrix")
            if not torch.isfinite(p).all().item():
                raise ValueError(f"{module}: nonfinite model weights")
            # Bound scratch space to one matrix, including interacting patches.
            candidate = p.detach().clone()
            for i, patch in enumerate(patches):
                if module not in patch["modules"]:
                    continue
                cols = patch["cols"][module]
                if len(patch["d"]) != p.shape[0] or int(cols[-1]) >= p.shape[1]:
                    raise ValueError(f"{module}: model shape/column bounds mismatch")
                update = _update(patch, module, p, sign)
                result = candidate[:, cols] + update
                if not torch.isfinite(update).all().item() or not torch.isfinite(result).all().item():
                    raise ValueError(f"{module}: update or resulting weights are nonfinite")
                candidate[:, cols] = result
                counts[i] += update.numel()
            del candidate
    return params, counts


def preflight(patch: dict, model, sign: int = 1, *, expected_revision: str | None = None) -> int:
    """Check factors, all target matrices, cast overflow and optional model revision."""
    return _preflight([patch], model, sign, expected_revision)[1][0]


def apply_many(patches: list[dict], model, sign: int = 1, *, expected_revision: str | None = None) -> list[int]:
    """Preflight the whole sequence, then return selected scalar counts per patch.

    Invalid inputs never mutate the model. Runtime failures (OOM/interruption)
    during application are not transactional: restore immutable saved weights.
    No other thread may mutate parameters or patch arrays during this operation.
    """
    import torch
    params, counts = _preflight(patches, model, sign, expected_revision)
    with torch.no_grad():
        for patch in patches:
            for module in patch["modules"]:
                p = params[module]
                cols = patch["cols"][module]
                p[:, cols] += _update(patch, module, p, sign)
    return counts


def apply(patch: dict, model, sign: int = 1, *, expected_revision: str | None = None) -> int:
    """Add signed factors; selected scalar count is NOT a changed-bit count."""
    return apply_many([patch], model, sign, expected_revision=expected_revision)[0]

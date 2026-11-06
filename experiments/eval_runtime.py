"""Validation/state helpers for the remote, canonical-patch evaluation.

NumPy-only checks run locally; torch is imported only inside the Modal-side helpers.
No edit is undone by subtracting an update from rounded weights.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

import numpy as np


def require_finite(value, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            require_finite(item, f"{path}.{key}")
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"Non-finite measurement array at {path}")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            require_finite(item, f"{path}[{i}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"Non-finite measurement at {path}: {value}")


def encode_result(value) -> str:
    """JSON wire boundary: never pickle remote library classes back to the client.

    In particular, torch.__version__ is a str subclass whose pickle needs torch.
    JSON encodes it as ordinary text while preserving numeric scores as numbers.
    Unknown objects are errors, not silently stringified measurements. Library
    metadata objects (torch dtype/version, library config sentinels) are recorded
    explicitly as text with their Python type, so no measurement number is coerced.
    """
    def serialise(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        module = type(item).__module__.split(".")[0]
        if module in {"torch", "transformers", "lm_eval", "datasets"}:
            return {"library_object": str(item),
                    "python_type": f"{type(item).__module__}.{type(item).__name__}"}
        raise TypeError(f"Unsupported result object, not silently stringified: {type(item).__name__}")

    require_finite(value)
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=serialise)


def resolved_chat_template(tokenizer) -> str:
    """Resolve named/default templates with Transformers' own selection rules."""
    template = tokenizer.get_chat_template()
    if not isinstance(template, str) or not template.strip():
        raise ValueError("Tokenizer did not resolve a nonempty chat template")
    return template


def validate_harness_samples(samples: list[dict], expected: int, filters: list[str]) -> dict:
    """lm-eval logs each document once PER filter; retain and validate every record.

    This experiment uses the first ``expected`` test documents and no sampling/shuffle.
    Prompt/document hashes and full raw responses must agree across scoring filters.
    """
    by_document = {}
    wanted = set(filters)
    if not wanted or len(wanted) != len(filters):
        raise ValueError("Expected filter names must be nonempty and unique")
    for sample in samples:
        required = {"doc_id", "filter", "resps", "doc_hash", "prompt_hash", "target_hash"}
        if not required <= sample.keys() or not sample["resps"]:
            raise ValueError("Harness sample lacks full responses or provenance")
        doc_id, filter_name = sample["doc_id"], sample["filter"]
        if not isinstance(doc_id, (int, np.integer)) or isinstance(doc_id, bool):
            raise ValueError("Harness document ID is not an integer")
        group = by_document.setdefault(int(doc_id), {})
        if filter_name not in wanted or filter_name in group:
            raise ValueError(f"Unexpected/duplicate scoring filter for document {doc_id}")
        if group:
            first = next(iter(group.values()))
            for key in ("resps", "doc_hash", "prompt_hash", "target_hash"):
                if sample[key] != first[key]:
                    raise ValueError(f"Scoring filters disagree on {key} for document {doc_id}")
        group[filter_name] = sample
    if set(by_document) != set(range(expected)):
        raise ValueError("Missing or unexpected test document IDs")
    if any(set(group) != wanted for group in by_document.values()):
        raise ValueError("Missing scoring-filter records")
    return {"unique_documents": len(by_document), "sample_records": len(samples),
            "filters": sorted(wanted),
            "doc_ids_sha256": hashlib.sha256(json.dumps(sorted(by_document)).encode()).hexdigest()}


def preserve_invalid_numbers(value):
    """Keep failed payloads JSON-safe without silently treating NaN/Inf as measurements."""
    if isinstance(value, Mapping):
        return {key: preserve_invalid_numbers(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return preserve_invalid_numbers(value.tolist())
    if isinstance(value, (list, tuple)):
        return [preserve_invalid_numbers(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return {"invalid_nonfinite_number": str(value)}
    return value


def validate_patch(patch: dict) -> int:
    """Reject malformed/non-finite rank-one factors before changing any weight."""
    d = np.asarray(patch["d"])
    if d.ndim != 1 or not d.size or not np.isfinite(d).all():
        raise ValueError("Patch direction must be a finite nonempty vector")
    if not np.isclose(np.linalg.norm(d), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("Patch direction must have unit norm")
    if not math.isfinite(float(patch["alpha"])):
        raise ValueError("Patch alpha must be finite")
    modules = patch["modules"]
    if not modules or len(set(modules)) != len(modules):
        raise ValueError("Patch modules must be nonempty and unique")
    if set(modules) != set(patch["cols"]) or set(modules) != set(patch["proj"]):
        raise ValueError("Patch module/factor keys disagree")
    edited = 0
    for name in modules:
        cols, proj = np.asarray(patch["cols"][name]), np.asarray(patch["proj"][name])
        if (cols.ndim != 1 or not np.issubdtype(cols.dtype, np.integer)
                or not cols.size or np.any(cols < 0) or np.any(np.diff(cols) <= 0)):
            raise ValueError(f"Columns must be sorted, unique, nonnegative integers: {name}")
        if proj.shape != cols.shape or not np.isfinite(proj).all():
            raise ValueError(f"Invalid/non-finite projection: {name}")
        edited += d.size * cols.size
    return int(edited)


def snapshot_write_weights(model) -> dict:
    return {name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
            if name.endswith(("o_proj.weight", "down_proj.weight"))}


def restore_write_weights(model, snapshot: dict) -> None:
    import torch
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, saved in snapshot.items():
            params[name].copy_(saved.to(params[name].device))
    for name, saved in snapshot.items():
        if not torch.equal(params[name].detach().cpu(), saved):
            raise AssertionError(f"Immutable state restoration failed: {name}")


def state_sha256(weights: dict) -> str:
    """Hash raw bytes (including bfloat16), names, shapes and dtype in stable order."""
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(weights.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(f"{name}|{tuple(tensor.shape)}|{tensor.dtype}\n".encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_against_base(patch: dict, base: dict, samples_per_matrix: int = 16) -> dict:
    """Check sampled stored projections against the actual checkpoint being evaluated.

    Old exports did not pin their checkpoint revision. This guards against silently
    applying their factors to a different base; it is not a full checkpoint checksum.
    """
    max_abs_error, checked = 0.0, 0
    for name in patch["modules"]:
        if name not in base:
            raise ValueError(f"Patch parameter absent from checkpoint: {name}")
        weight = base[name]
        cols = patch["cols"][name]
        if weight.ndim != 2 or weight.shape[0] != len(patch["d"]) or int(cols[-1]) >= weight.shape[1]:
            raise ValueError(f"Patch/checkpoint shape mismatch: {name}")
        positions = np.unique(np.linspace(0, len(cols) - 1,
                                         min(samples_per_matrix, len(cols)), dtype=int))
        selected = cols[positions]
        # The base is saved as bf16; convert only these columns, never a full dense delta.
        actual = patch["d"] @ weight[:, selected].float().numpy().astype(np.float64)
        expected = patch["proj"][name][positions]
        error = float(np.max(np.abs(actual - expected)))
        max_abs_error = max(max_abs_error, error)
        if not np.allclose(actual, expected, rtol=1e-7, atol=1e-8):
            raise ValueError(f"Stored patch projection does not match checkpoint: {name} (max_abs={error})")
        checked += len(positions)
    return {"checked_columns": checked, "max_abs_error": max_abs_error,
            "rtol": 1e-7, "atol": 1e-8, "scope": "sampled columns, not whole-checkpoint identity"}


def torch_state_self_test() -> dict:
    """Actual bf16 loader + exact snapshot restoration, on the remote runtime."""
    import torch
    import load_patch

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.o_proj = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
            with torch.no_grad():
                self.o_proj.weight.fill_(1)

    model = Toy()
    saved = snapshot_write_weights(model)
    original = model.o_proj.weight.detach().clone()
    patch = {"d": np.array([1., 0.]), "alpha": 2048., "modules": ["o_proj.weight"],
             "cols": {"o_proj.weight": np.array([0])}, "proj": {"o_proj.weight": np.array([1.])}}
    validate_patch(patch)
    bad = {"d": np.array([1., 0.]), "alpha": 1., "modules": ["o_proj.weight"],
           "cols": {"o_proj.weight": np.array([0, 2])}, "proj": {"o_proj.weight": np.array([1., 1.])}}
    try:
        load_patch.apply_many([patch, bad], model)
    except ValueError:
        preflight_blocked_second = True
    else:
        raise AssertionError("Out-of-bounds second patch must block the sequence")
    assert torch.equal(model.o_proj.weight, original), "preflight must precede mutation"
    assert load_patch.apply(patch, model) == 2
    assert torch.equal(model.o_proj.weight[:, 1], original[:, 1])
    load_patch.apply(patch, model, sign=-1)
    inverse_is_exact = torch.equal(model.o_proj.weight, original)
    assert not inverse_is_exact, "Control must demonstrate finite-precision subtraction failure"
    restore_write_weights(model, saved)
    assert torch.equal(model.o_proj.weight, original)
    return {"dtype": "bfloat16", "inverse_subtraction_exact": inverse_is_exact,
            "snapshot_restore_exact": True, "unselected_column_unchanged": True,
            "sequence_preflight_blocked_mutation": preflight_blocked_second}

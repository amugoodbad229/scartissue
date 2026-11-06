"""The same four arms on a second and third Cohere model (the model axis of the claim).

`p21`/`p24` answer "does the frontier exist, and does it survive held-out prompts" on
`CohereLabs/tiny-aya-global`. This run answers "is that a property of the repair method or of one
checkpoint": it rebuilds the abliteration + repair patches for another tiny-aya model, then runs
`base`, `drift`, `harm2_a4` and `global_a4` on the same frozen 120 harmful / 40 benign prompts with
the same 512-token budget and the same state verification.

Patches come from `experiments/build_patches.py` (volume `/hf/artifacts/<slug>/`), so the drift and
the repair are the same algebra as the headline model — only the checkpoint changes.

    .venv/Scripts/python.exe -m modal run experiments/p25_model_safety.py --model CohereLabs/tiny-aya-earth
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results/p21_safety_length_full_20261103_001722.json"
BUDGET = 512
# Values are manifest KEYS (each entry carries its own "file"), not filenames.
ARMS = {"base": None, "drift": None, "harm2_a4": "repair_harm2_a4", "global_a4": "repair_global_a4"}

image = (modal.Image.debian_slim(python_version="3.12")
         .uv_sync(str(ROOT / "experiments/runtime"), frozen=True, uv_version="0.12.23")
         .env({"HF_HOME": "/hf/hf", "HF_HUB_CACHE": "/hf/hf/hub",
               "HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_file(str(ROOT / "artifacts/load_patch.py"), "/root/patchlib/load_patch.py")
         .add_local_file(str(ROOT / "experiments/eval_runtime.py"), "/root/eval_runtime.py")
         .add_local_file(str(ROOT / "experiments/generation_checks.py"), "/root/generation_checks.py")
         .add_local_file(str(ROOT / "experiments/safety_core.py"), "/root/safety_core.py")
         .add_local_file(str(ROOT / "experiments/corpus_fit.py"), "/root/corpus_fit.py")
         .add_local_file(str(ROOT / "1-curate/corpus/corpus.jsonl"), "/root/1-curate/corpus/corpus.jsonl")
         .add_local_file(str(ROOT / "1-curate/corpus/corpus_split.json"),
                         "/root/1-curate/corpus/corpus_split.json"))
volume = modal.Volume.from_name("scartissue-hf")
secrets = [modal.Secret.from_name("huggingface")]
app = modal.App("scartissue-p25-model-safety")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slug_of(model: str) -> str:
    return model.split("/", 1)[-1]


@app.function(image=image, gpu="A100-40GB", volumes={"/hf": volume}, secrets=secrets,
              cpu=4, memory=24576, timeout=5400)
def generate_arm(arm: str, rows: list[dict], model: str, revision: str, patch_dir: str,
                 budget: int) -> str:
    import sys
    import traceback
    sys.path.insert(0, "/root/artifacts")
    from eval_runtime import encode_result
    from safety_core import generate_arm as run_arm

    started = time.perf_counter()
    out = {"arm": arm, "status": "running", "records": [], "errors": [], "headline_eligible": False}
    try:
        manifest = json.loads((Path(patch_dir) / "manifest.json").read_text(encoding="utf8"))
        arms = manifest.get("arms") or manifest.get("patches") or {}
        if not arms:
            raise SystemExit(f"manifest at {patch_dir} has no arms/patches block")
        drift_key = next(k for k in arms if k.startswith("abliteration"))
        patches = []
        if arm != "base":
            patches.append(str(Path(patch_dir) / arms[drift_key]["file"]))
        if ARMS[arm]:
            patches.append(str(Path(patch_dir) / arms[ARMS[arm]]["file"]))
        expected_sha = {p: sha(Path(p)) for p in patches}
        # `base` applies NO patch, so its expected edited-scalar counts are zero, not the drift's.
        expected_scalars = {"drift": 0, "repair": 0} if arm == "base" else {
            "drift": int(arms[drift_key]["edited_scalars"]),
            "repair": int(arms[ARMS[arm]]["edited_scalars"]) if ARMS[arm] else 0}
        result = run_arm(arm=arm, rows=rows, model_id=model, revision=revision, patches=patches,
                         expected_sha=expected_sha, budget=budget)
        out.update(result)
        out["patch_sha256"] = expected_sha
        out["manifest_edited_scalars"] = expected_scalars
        # `edited_scalars` is keyed by whatever path string load_patch was handed, so compare by
        # ROLE (drift vs repair) instead of by path identity.
        role_totals = {"drift": 0, "repair": 0}
        for key, value in (result.get("edited_scalars") or {}).items():
            role_totals["drift" if "abliteration" in str(key) else "repair"] += int(value)
        for role, want in expected_scalars.items():
            if role_totals[role] != want:
                raise SystemExit(f"edited-scalar mismatch for {role}: {role_totals[role]} != {want}")
        out["edited_scalars_by_role"] = role_totals
        assert [r["id"] for r in out["records"]] == [r["id"] for r in rows]
        out["status"] = "passed"
    except BaseException as error:  # noqa: BLE001 - a failed arm must be reported, not raised away
        out["status"] = "failed"
        out["errors"].append({"error": repr(error), "traceback": traceback.format_exc()})
    out["wall_s"] = time.perf_counter() - started
    return encode_result(out)


@app.local_entrypoint()
def main(model: str = "CohereLabs/tiny-aya-earth", revision: str = "main", budget: int = BUDGET):
    from experiments.corpus_fit import evaluation_rows
    rows = evaluation_rows()
    baseline = json.loads(BASELINE.read_text(encoding="utf8"))
    slug = slug_of(model)
    patch_dir = f"/hf/artifacts/{slug}"
    path = ROOT / f"results/p25_{slug}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out = {"status": "running", "mode": "model_replication", "model": model, "budget": budget,
           "arms": list(ARMS), "rows": rows, "workers": [], "errors": [],
           "headline_eligible": False,
           "prompt_source": "results/p21_safety_length_full_20261103_001722.json (frozen partition)",
           "reference_chat_template_sha256": baseline["reference"]["chat_template_sha256"],
           "slicing_reference": baseline.get("slicing_reference", "global-reference only; per-model "
                                             "chat template is resolved and recorded per worker"),
           "source_sha256": {p: sha(ROOT / p) for p in
                             ["experiments/p25_model_safety.py", "experiments/safety_core.py",
                              "experiments/corpus_fit.py", str(BASELINE.relative_to(ROOT))]}}

    def save():
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
        tmp.replace(path)

    save()
    calls = [generate_arm.spawn(arm, rows, model, revision, patch_dir, budget) for arm in ARMS]
    out["function_calls"] = [c.object_id for c in calls]
    save()
    for call in calls:
        worker = json.loads(call.get())
        out["workers"].append(worker)
        if worker["status"] != "passed":
            out["errors"].append({"arm": worker["arm"], "errors": worker["errors"]})
        save()
    hashes = {w["arm"]: w.get("base_write_weights_sha256") for w in out["workers"]}
    if len(set(hashes.values())) != 1:
        out["errors"].append({"check": "cross_arm_base_state", "hashes": hashes})
    out["base_write_weights_sha256"] = next(iter(set(hashes.values()))) if hashes else None
    out["status"] = "passed" if len(out["workers"]) == 4 and not out["errors"] else "failed"
    save()
    print(f"{path}: {out['status']}", flush=True)

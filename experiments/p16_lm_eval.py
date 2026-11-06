"""Capability evaluation of the ACTUAL exported patches, not the invalid p15 operator.

CPU preflight -> small GPU smoke test -> separate, parallel full-evaluation workers.
Only the local entrypoint writes results. All generations are retained by lm-eval.

  uv run python -m modal run experiments/p16_lm_eval.py --mode smoke
  uv run python -m modal run experiments/p16_lm_eval.py --mode full --limit 500

Full runs require the successful smoke artifact supplied with --smoke-file. No old
p15 or regex/GSM substring score is promoted to capability evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
MODEL = "CohereLabs/tiny-aya-global"
DATASET = "openai/gsm8k"
ARMS = {
    "base": None,
    "drift": None,
    "harm2_a4": "repair_harm2_a4.npz",
    "global_a4": "repair_global_a4.npz",
}
DRIFT = "abliteration_tiny-aya-global_layer15.npz"
SOURCE_FILES = ["experiments/p16_lm_eval.py", "experiments/eval_runtime.py",
                "experiments/runtime/pyproject.toml", "experiments/runtime/uv.lock",
                "artifacts/load_patch.py", "artifacts/manifest.json"]

# Historical torch stack is retained deliberately to evaluate existing artifacts,
# rather than confounding patch effects with a framework upgrade. lm-eval 0.4.13
# was checked against the current published release before locking this runtime.
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_sync(str(ROOT / "experiments/runtime"), frozen=True, uv_version="0.12.23")
         .env({"HF_HOME": "/hf/hf", "HF_HUB_CACHE": "/hf/hf/hub",
               "HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false",
               "TRANSFORMERS_VERBOSITY": "error"})
         .add_local_dir(str(ROOT / "artifacts"), "/root/artifacts",
                        ignore=["__pycache__", "*.pyc"])
         .add_local_file(str(ROOT / "experiments/eval_runtime.py"), "/root/eval_runtime.py"))
volume = modal.Volume.from_name("scartissue-hf")
secrets = [modal.Secret.from_name("huggingface")]
app = modal.App("scartissue-p16-lm-eval")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict:
    paths = SOURCE_FILES + [f"artifacts/{DRIFT}"] + [f"artifacts/{p}" for p in ARMS.values() if p]
    return {p: sha(ROOT / p) for p in paths}


@app.function(image=image, volumes={"/hf": volume}, secrets=secrets,
              cpu=4, memory=16384, timeout=900)
def preflight() -> dict:
    import importlib.metadata
    import inspect
    import sys
    import yaml
    import lm_eval
    from huggingface_hub import HfApi
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.models.huggingface import HFLM

    sys.path.insert(0, "/root/artifacts")
    from eval_runtime import (encode_result, require_finite, resolved_chat_template,
                              torch_state_self_test, validate_patch)
    import torch
    transport_probe = json.loads(encode_result({"torch_version": torch.__version__,
                                                "dtype": torch.bfloat16,
                                                "tensor_meta": {"score": 0.5}}))
    assert type(transport_probe["torch_version"]) is str
    assert transport_probe["tensor_meta"]["score"] == 0.5
    from transformers import AutoTokenizer
    import load_patch

    api = HfApi(token=os.environ["HF_TOKEN"])
    reference = {"model": MODEL, "model_revision": api.model_info(MODEL).sha,
                 "dataset": DATASET, "dataset_revision": api.dataset_info(DATASET).sha}
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=reference["model_revision"], token=os.environ["HF_TOKEN"], use_fast=True)
    template = resolved_chat_template(tokenizer)
    reference["chat_template_sha256"] = hashlib.sha256(template.encode()).hexdigest()
    task_path = Path(lm_eval.__file__).parent / "tasks/gsm8k/gsm8k.yaml"
    task_text = task_path.read_text()
    config = yaml.safe_load(task_text)
    config.setdefault("dataset_kwargs", {})["revision"] = reference["dataset_revision"]
    # This tests the pinned dataset/config API before allocating a GPU.
    task = ConfigurableTask(config=config)
    assert task.config.task == "gsm8k"
    assert len(task.test_docs()) >= 500
    assert config["num_fewshot"] == 5
    wanted = {"model", "tasks", "limit", "num_fewshot", "log_samples", "gen_kwargs",
              "apply_chat_template", "random_seed", "numpy_random_seed", "torch_random_seed",
              "fewshot_random_seed"}
    assert wanted <= set(inspect.signature(lm_eval.simple_evaluate).parameters)
    assert "pretrained" in inspect.signature(HFLM).parameters
    manifest = json.loads(Path("/root/artifacts/manifest.json").read_text())
    patch_checks = {}
    for filename in [DRIFT] + [p for p in ARMS.values() if p]:
        patch = load_patch.load(f"/root/artifacts/{filename}")
        count = validate_patch(patch)
        assert count == manifest["arms"][filename.removesuffix(".npz")]["edited_scalars"]
        patch_checks[filename] = {"sha256": sha(Path("/root/artifacts") / filename),
                                  "selected_scalars": count, "alpha": patch["alpha"]}
    reference.update({"task_config": config, "task_yaml_sha256": hashlib.sha256(task_text.encode()).hexdigest(),
                      "dataset_test_fingerprint": task.dataset["test"]._fingerprint,
                      "versions": {p: importlib.metadata.version(p) for p in
                                   ["torch", "transformers", "accelerate", "huggingface-hub", "datasets", "lm-eval", "numpy"]},
                      "patch_checks": patch_checks,
                      "torch_state_self_test": torch_state_self_test(),
                      "result_transport_self_test": {"torch_version": transport_probe["torch_version"],
                                                      "dtype": transport_probe["dtype"],
                                                      "scope": "library metadata only; numeric metrics unchanged"},
                      "simple_evaluate_signature": str(inspect.signature(lm_eval.simple_evaluate))})
    require_finite(reference)
    return reference


@app.function(image=image, gpu="A100-40GB", volumes={"/hf": volume}, secrets=secrets,
              cpu=8, memory=32768, timeout=10800)
def run_arms(arms: list[str], limit: int, reference: dict) -> str:
    import gc
    import random
    import sys
    import numpy as np
    import torch
    import lm_eval
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, "/root/artifacts")
    import load_patch
    from eval_runtime import (encode_result, preserve_invalid_numbers, require_finite, resolved_chat_template,
                              restore_write_weights, snapshot_write_weights, state_sha256,
                              validate_against_base, validate_harness_samples, validate_patch)

    started = time.monotonic()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=reference["model_revision"],
                                             token=os.environ["HF_TOKEN"], use_fast=True)
    tokenizer.padding_side = "left"
    tokenizer.add_bos_token = False
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    template = resolved_chat_template(tokenizer)
    template_hash = hashlib.sha256(template.encode()).hexdigest()
    assert template_hash == reference["chat_template_sha256"]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=reference["model_revision"], token=os.environ["HF_TOKEN"],
        dtype=torch.bfloat16, device_map="cuda").eval()
    base = snapshot_write_weights(model)
    assert len(base) == 72
    base_hash = state_sha256(base)
    files = [DRIFT] + [ARMS[a] for a in arms if ARMS[a]]
    patches, compatibility = {}, {}
    for filename in sorted(set(files)):
        patch = load_patch.load(f"/root/artifacts/{filename}",
                                expected_sha256=reference["patch_checks"][filename]["sha256"])
        validate_patch(patch)
        compatibility[filename] = validate_against_base(patch, base)
        patches[filename] = patch

    out = {"schema": "scartissue.lm-eval/1", "reference": reference,
           "hardware": {"gpu": torch.cuda.get_device_name(),
                        "vram_bytes": torch.cuda.get_device_properties(0).total_memory},
           "dtype": "bfloat16", "padding_side": tokenizer.padding_side,
           "chat_template_sha256": template_hash,
           "loader_cast": "stored float64 factors -> float32 delta -> parameter dtype (original loader)",
           "base_write_weights_sha256": base_hash, "patch_checkpoint_compatibility": compatibility,
           "limit": limit, "arms": {}, "status": "running", "errors": []}
    for arm in arms:
        harness = task = result = None
        try:
            if arm not in ARMS:
                raise ValueError(f"Unknown arm: {arm}")
            restore_write_weights(model, base)
            edited = {"drift": 0, "repair": 0}
            sequence = ([("drift", DRIFT)] if arm != "base" else []) + \
                       ([("repair", ARMS[arm])] if ARMS[arm] else [])
            counts = load_patch.apply_many([patches[f] for _, f in sequence], model,
                                           expected_revision=reference["model_revision"])
            edited.update({kind: count for (kind, _), count in zip(sequence, counts)})
            for name, param in model.named_parameters():
                if name in base and not torch.isfinite(param).all().item():
                    raise ValueError(f"Non-finite edited weight: {arm}: {name}")
            arm_hash = state_sha256({n: p for n, p in model.named_parameters() if n in base})
            print(f"[arm] {arm}: validated state {arm_hash}, selected scalars={edited}", flush=True)
            task = ConfigurableTask(config=reference["task_config"])
            assert task.dataset["test"]._fingerprint == reference["dataset_test_fingerprint"]
            harness = HFLM(pretrained=model, tokenizer=tokenizer, backend="causal", device="cuda",
                           batch_size=4, max_length=4096, add_bos_token=False)
            tick = time.monotonic()
            result = lm_eval.simple_evaluate(
                model=harness, tasks=[task], num_fewshot=5, limit=limit,
                apply_chat_template=True, log_samples=True,
                gen_kwargs="max_gen_toks=256,do_sample=False", bootstrap_iters=10000,
                random_seed=0, numpy_random_seed=0, torch_random_seed=0, fewshot_random_seed=0)
            # Retain the full result BEFORE completeness/finite checks: a failed
            # assertion must not discard generations or earlier completed arms.
            out["arms"][arm] = {"edited_scalars": edited, "write_weights_sha256": arm_hash,
                                  "generation_and_scoring_s": time.monotonic() - tick,
                                  "evaluation": result}
            samples = result.get("samples", {}).get("gsm8k", [])
            filters = [f["name"] for f in reference["task_config"]["filter_list"]]
            out["arms"][arm]["sample_coverage"] = validate_harness_samples(samples, limit, filters)
            require_finite(result)
            print(f"[done] {arm}: {result['results']}", flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            out["errors"].append({"arm": arm, "error": repr(exc)})
            break
        finally:
            del result, harness, task
            gc.collect()
            torch.cuda.empty_cache()
    try:
        restore_write_weights(model, base)
        out["restored_base_write_weights_sha256"] = state_sha256(
            {n: p for n, p in model.named_parameters() if n in base})
        assert out["restored_base_write_weights_sha256"] == base_hash
    except Exception as exc:
        out["errors"].append({"stage": "restore", "error": repr(exc)})
    out["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    out["wall_s"] = time.monotonic() - started
    out["status"] = "failed" if out["errors"] else "passed"
    if out["errors"]:
        out = preserve_invalid_numbers(out)
    # The caller deliberately has no torch. Do not send library-specific classes
    # (including the str subclass torch.torch_version.TorchVersion) via pickle.
    return encode_result(out)


def write_json(path: Path, value: dict) -> None:
    import numpy as np
    from experiments.eval_runtime import require_finite

    def serialise(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f"Unsupported result object, not silently stringified: {type(item).__name__}")

    require_finite(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                                   default=serialise), encoding="utf-8")
    temporary.replace(path)


@app.local_entrypoint()
def main(mode: str = "smoke", limit: int = 500, smoke_file: str = ""):
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    if not 1 <= limit <= 1319:
        raise ValueError("limit must be between 1 and the GSM8K test size (1319)")
    hashes = source_hashes()
    if mode == "full":
        if not smoke_file:
            raise ValueError("Full evaluation requires --smoke-file from a successful inspected smoke run")
        smoke = json.loads(Path(smoke_file).read_text(encoding="utf-8"))
        assert smoke["status"] == "passed" and smoke["source_sha256"] == hashes
        assert set(smoke["workers"][0]["arms"]) == set(ARMS)
        reference = smoke["reference"]
    else:
        print("[preflight] locked runtime, dataset revision, patch manifest, actual bf16 state tests", flush=True)
        reference = preflight.remote()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = ROOT / "results" / f"p16_{mode}_{stamp}.json"
    out = {"schema": "scartissue.capability-run/1", "status": "running", "mode": mode,
           "source_sha256": hashes, "reference": reference, "workers": [], "errors": [],
           "metric_scope": "lm-eval GSM8K 5-shot, chat template, strict-match and flexible-extract; not safety or paper confirmation"}
    write_json(path, out)
    if mode == "smoke":
        groups, n = [list(ARMS)], 2
    else:
        groups, n = [[a] for a in ARMS], limit
    calls = [(group, run_arms.spawn(group, n, reference)) for group in groups]
    out["function_calls"] = [{"arms": group, "function_call_id": call.object_id}
                             for group, call in calls]
    write_json(path, out)
    print(f"[spawned] {groups}, n={n}; single local writer -> {path}", flush=True)
    for group, call in calls:
        try:
            worker = json.loads(call.get())
            out["workers"].append(worker)
            if worker["status"] != "passed":
                out["errors"].append({"arms": group, "worker_errors": worker["errors"]})
        except Exception as exc:
            out["errors"].append({"arms": group, "error": repr(exc)})
        write_json(path, out)
    out["status"] = "failed" if out["errors"] else "passed"
    write_json(path, out)
    print(f"[result] {path} status={out['status']}", flush=True)
    if out["errors"]:
        raise RuntimeError(f"Evaluation failed; errors preserved in {path}")

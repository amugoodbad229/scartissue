"""Generate safety answers from the exact p16 capability checkpoint states.

No safety metric is inferred here; full answers go through validated judges later.
Frozen evaluation IDs are read from the existing CuratorKIT corpus/partition.
Smoke first, then --mode full --smoke-file <passed JSON>. All CUDA is on Modal.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
CAPABILITY = ROOT / "results/p16_full_20261102_230915.json"
CORPUS = ROOT / "1-curate/corpus/corpus.jsonl"
SPLIT = ROOT / "1-curate/corpus/corpus_split.json"
ARMS = {"base": None, "drift": None, "harm2_a4": "repair_harm2_a4.npz", "global_a4": "repair_global_a4.npz"}
DRIFT = "abliteration_tiny-aya-global_layer15.npz"
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_sync(str(ROOT / "experiments/runtime"), frozen=True, uv_version="0.12.23")
         .env({"HF_HOME": "/hf/hf", "HF_HUB_CACHE": "/hf/hf/hub",
               "HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_dir(str(ROOT / "artifacts"), "/root/artifacts", ignore=["__pycache__", "*.pyc"])
         .add_local_file(str(ROOT / "experiments/eval_runtime.py"), "/root/eval_runtime.py"))
volume = modal.Volume.from_name("scartissue-hf")
secrets = [modal.Secret.from_name("huggingface")]
app = modal.App("scartissue-p17-safety")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources() -> dict:
    paths = ["experiments/p17_safety.py", "experiments/safety_records.py", "experiments/eval_runtime.py",
             "experiments/runtime/pyproject.toml", "experiments/runtime/uv.lock", "artifacts/load_patch.py",
             f"artifacts/{DRIFT}", "artifacts/repair_harm2_a4.npz", "artifacts/repair_global_a4.npz",
             "1-curate/corpus/corpus.jsonl", "1-curate/corpus/corpus_split.json"]
    return {p: sha(ROOT / p) for p in paths}


@app.function(image=image, gpu="A100-40GB", volumes={"/hf": volume}, secrets=secrets,
              cpu=4, memory=24576, timeout=3600)
def generate(arm: str, rows: list[dict], reference: dict, expected: dict) -> str:
    import random
    import sys
    import traceback
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from eval_runtime import (encode_result, resolved_chat_template, restore_write_weights,
                              snapshot_write_weights, state_sha256, validate_patch)
    sys.path.insert(0, "/root/artifacts")
    import load_patch
    started = time.perf_counter()
    output = {"arm": arm, "status": "running", "records": [], "errors": [],
              "model_revision": reference["model_revision"], "dtype": "bfloat16",
              "hardware": torch.cuda.get_device_name(), "generation": {
                  "do_sample": False, "max_new_tokens": 128}, "batch_size": 4}
    base = model = None
    try:
        random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        tok = AutoTokenizer.from_pretrained(reference["model"], revision=reference["model_revision"], use_fast=True)
        tok.padding_side = "left"
        tok.add_bos_token = False
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        template = resolved_chat_template(tok)
        output["chat_template_sha256"] = hashlib.sha256(template.encode()).hexdigest()
        assert output["chat_template_sha256"] == reference["chat_template_sha256"]
        model = AutoModelForCausalLM.from_pretrained(reference["model"], revision=reference["model_revision"],
                                                    dtype=torch.bfloat16, device_map="cuda").eval()
        base = snapshot_write_weights(model)
        assert len(base) == 72
        output["base_write_weights_sha256"] = state_sha256(base)
        assert output["base_write_weights_sha256"] == expected["base_hash"]
        edited = {"drift": 0, "repair": 0}
        sequence = [(kind, filename) for kind, filename in
                    (("drift", DRIFT if arm != "base" else None), ("repair", ARMS[arm])) if filename]
        patches = [load_patch.load(str(Path("/root/artifacts") / filename),
                   expected_sha256=reference["patch_checks"][filename]["sha256"])
                   for _, filename in sequence]
        counts = load_patch.apply_many(patches, model, expected_revision=reference["model_revision"])
        edited.update({kind: count for (kind, _), count in zip(sequence, counts)})
        output["write_weights_sha256"] = state_sha256({n: p for n, p in model.named_parameters() if n in base})
        assert output["write_weights_sha256"] == expected["write_weights_sha256"]
        assert edited == expected["edited_scalars"]
        output["edited_scalars"] = edited
        for start in range(0, len(rows), 4):
            batch = rows[start:start + 4]
            inputs = [tok.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                      tokenize=False, add_generation_prompt=True) for row in batch]
            encoded = tok(inputs, padding=True, return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.inference_mode():
                generated = model.generate(**encoded, **output["generation"], pad_token_id=tok.pad_token_id)
            for index, (row, rendered) in enumerate(zip(batch, inputs)):
                ids = generated[index, encoded.input_ids.shape[1]:].tolist()
                output["records"].append({**row, "rendered_input": rendered,
                    "rendered_input_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                    "input_token_ids": encoded.input_ids[index][encoded.attention_mask[index].bool()].tolist(),
                    "generated_token_ids_including_batch_padding": ids,
                    "answer": tok.decode(ids, skip_special_tokens=True),
                    "answer_with_special_tokens": tok.decode(ids, skip_special_tokens=False),
                    "hit_token_cap_without_eos": len(ids) == 128 and tok.eos_token_id not in ids})
            print(f"{arm}: {len(output['records'])}/{len(rows)} answers retained", flush=True)
        assert [r["id"] for r in output["records"]] == [r["id"] for r in rows]
        assert all(r["answer"].strip() for r in output["records"]), "Empty completion retained as failure"
        output["status"] = "passed"
    except Exception as error:
        output["status"] = "failed"
        output["errors"].append({"error": repr(error), "traceback": traceback.format_exc()})
    finally:
        if base is not None and model is not None:
            try:
                restore_write_weights(model, base)
                output["restored_base_write_weights_sha256"] = state_sha256(
                    {n: p for n, p in model.named_parameters() if n in base})
                assert output["restored_base_write_weights_sha256"] == expected["base_hash"]
            except Exception as error:
                output["status"] = "failed"
                output["errors"].append({"stage": "restore", "error": repr(error)})
        output["wall_s"] = time.perf_counter() - started
    return encode_result(output)


@app.local_entrypoint()
def main(mode: str = "smoke", smoke_file: str = ""):
    from experiments.safety_records import evaluation_rows
    if mode not in ("smoke", "full"):
        raise ValueError("Mode must be smoke or full")
    capability = json.loads(CAPABILITY.read_text(encoding="utf8"))
    assert capability["status"] == "passed" and not capability["errors"]
    current = sources()
    for path in ["artifacts/load_patch.py", f"artifacts/{DRIFT}", "artifacts/repair_harm2_a4.npz", "artifacts/repair_global_a4.npz"]:
        assert current[path] == capability["source_sha256"][path], f"Capability artifact changed: {path}"
    rows = evaluation_rows([json.loads(x) for x in CORPUS.read_text(encoding="utf8").splitlines()],
                           json.loads(SPLIT.read_text(encoding="utf8")), 2 if mode == "smoke" else None)
    if mode == "full":
        smoke = json.loads((ROOT / smoke_file).read_text(encoding="utf8"))
        assert smoke["mode"] == "smoke" and smoke["status"] == "passed" and smoke["source_sha256"] == current
        assert {w["arm"] for w in smoke["workers"]} == set(ARMS)
    expected = {arm: {"base_hash": worker["base_write_weights_sha256"],
                       "write_weights_sha256": value["write_weights_sha256"],
                       "edited_scalars": value["edited_scalars"]}
                for worker in capability["workers"] for arm, value in worker["arms"].items()}
    assert set(expected) == set(ARMS)
    path = ROOT / f"results/p17_safety_{mode}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output = {"status": "running", "mode": mode, "source_sha256": current,
              "capability_source_sha256": sha(CAPABILITY), "reference": capability["reference"],
              "expected_states": expected, "rows": rows, "workers": [], "errors": [],
              "metric_scope": "Full completions only. NOT ASR, refusal, or capability scores."}
    def save():
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
        tmp.replace(path)
    save()
    calls = [generate.spawn(arm, rows, capability["reference"], expected[arm]) for arm in ARMS]
    output["function_calls"] = [c.object_id for c in calls]
    save()
    for call in calls:
        try:
            worker = json.loads(call.get()); output["workers"].append(worker)
            if worker["status"] != "passed":
                output["errors"].append({"arm": worker["arm"], "errors": worker["errors"]})
        except Exception as error:
            output["errors"].append({"error": repr(error)})
        save()
    output["status"] = "passed" if len(output["workers"]) == 4 and not output["errors"] else "failed"
    save()
    print(f"{path}: {output['status']}", flush=True)

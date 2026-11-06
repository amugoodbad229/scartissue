"""Exploratory frozen-panel scoring of p17 checkpoint-matched safety answers.

The p18 all-control calibration gate FAILED. This job characterises disagreement;
it must not promote its ASR estimates to a validated headline or erase that failure.
Run with --safety-file <passed p17 full JSON>. CUDA remote, one local result writer.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import time
import modal

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / "ScarTissue/repos/AuditKIT"
CALIBRATION = ROOT / "results/p18_judge_smoke_20261102_233759.json"
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_sync(str(ROOT / "experiments/runtime"), frozen=True, uv_version="0.12.23")
         .env({"HF_HOME": "/hf/hf", "HF_HUB_CACHE": "/hf/hf/hub",
               "HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_dir(str(UPSTREAM / "src"), "/external/auditkit_src", ignore=["__pycache__", "*.pyc"])
         .add_local_file(str(UPSTREAM / "LICENSE.md"), "/external/AuditKIT_LICENSE.md")
         .add_local_file(str(ROOT / "experiments/judge_protocols.py"), "/root/judge_protocols.py")
         .add_local_file(str(ROOT / "experiments/eval_runtime.py"), "/root/eval_runtime.py"))
volume = modal.Volume.from_name("scartissue-hf")
secrets = [modal.Secret.from_name("huggingface")]
app = modal.App("scartissue-p19-safety-judges")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_rows(safety: dict, *, state_checks: dict | None = None) -> list[dict]:
    """Validate a generated safety run before judging it.

    ``state_checks`` is an optional ``results/<run>_state_checks.json`` produced by
    ``experiments/verify_states.py``. It is only allowed to explain a *bookkeeping* failure: the raw
    run's per-arm status must still be ``passed`` and the corrected check must pass for every arm.
    Nothing about the answers, hashes or resets is relaxed.
    """
    if safety["mode"] not in {"full", "heldout_confirmation"}:
        raise ValueError("Need a full or held-out safety-generation run")
    if safety["status"] != "passed" or safety["errors"]:
        corrected = state_checks is not None and state_checks.get("passed") is True
        # A worker marked failed only by the original run's mis-keyed scalar comparison is
        # acceptable when the corrected per-arm check passes for it.
        bookkeeping_only = all(
            w.get("status") == "passed"
            or (corrected and all("state_check" in e for e in safety["errors"] if e.get("arm") == w["arm"])
                and state_checks["arms"][w["arm"]]["passed"])
            for w in safety["workers"])
        if not (corrected and bookkeeping_only):
            raise ValueError("Need a passed full safety-generation run")
        for arm, entry in state_checks["arms"].items():
            expected = safety["expected_states"][arm]
            worker = next(w for w in safety["workers"] if w["arm"] == arm)
            if not entry.get("passed") or worker["write_weights_sha256"] != expected["write_weights_sha256"]:
                raise ValueError(f"Corrected state check does not hold for {arm}")
    arms = {w["arm"] for w in safety["workers"]}
    if arms != {"base", "drift", "harm2_a4", "global_a4"} or len(safety["workers"]) != 4:
        raise ValueError("Expected exactly four checkpoint-matched arms")
    result = []
    for worker in safety["workers"]:
        arm = worker["arm"]; expected = safety["expected_states"][arm]
        if (worker["write_weights_sha256"] != expected["write_weights_sha256"]
            or worker["restored_base_write_weights_sha256"] != expected["base_hash"]):
            raise ValueError(f"State validation failed: {arm}")
        if len(worker["records"]) != len(safety["rows"]):
            raise ValueError(f"Missing full safety answers: {arm}")
        for row, source in zip(worker["records"], safety["rows"]):
            if row["id"] != source["id"] or row["prompt_sha256"] != source["prompt_sha256"] or row["prompt"] != source["prompt"]:
                raise ValueError("Frozen ID/prompt alignment mismatch")
            result.append({"id": f"{arm}/{row['id']}", "corpus_id": row["id"], "arm": arm,
                           "role": row["role"], "prompt": row["prompt"], "answer": row["answer"],
                           "prompt_sha256": row["prompt_sha256"],
                           "answer_sha256": hashlib.sha256(row["answer"].encode()).hexdigest()})
    return result


@app.function(image=image, gpu="A100-40GB", volumes={"/hf": volume}, secrets=secrets,
              cpu=4, memory=24576, timeout=3600)
def score(family: str, rows: list[dict], reference: dict) -> str:
    import sys
    import traceback
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, "/external/auditkit_src")
    from judge_protocols import render_input, parse_verdict, auditkit_rescore
    from eval_runtime import encode_result
    start = time.perf_counter()
    out = {"family": family, "reference": reference, "records": [], "errors": [],
           "status": "running", "calibration_status": "failed", "headline_eligible": False,
           "batch_size": 4, "hardware": torch.cuda.get_device_name(),
           "generation": {"do_sample": False, "max_new_tokens": 128 if family == "qwen3guard" else 20}}
    try:
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        tok = AutoTokenizer.from_pretrained(reference["model"], revision=reference["revision"], use_fast=True)
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(reference["model"], revision=reference["revision"],
                                                    dtype=torch.bfloat16, device_map="cuda").eval()
        if family != "harmbench":
            out["chat_template_sha256"] = hashlib.sha256(tok.get_chat_template().encode()).hexdigest()
            assert out["chat_template_sha256"] == reference["chat_template_sha256"]
        for offset in range(0, len(rows), 4):
            batch = rows[offset:offset + 4]
            if family == "harmbench":
                rendered = [reference["official_template"].format(behavior=r["prompt"], generation=r["answer"]) for r in batch]
                add_special = True
            else:
                built = [render_input(family, tok, r["prompt"], r["answer"]) for r in batch]
                rendered = [r[0] for r in built]; add_special = built[0][1]
                assert all(r[1] == add_special for r in built)
            enc = tok(rendered, return_tensors="pt", padding=True, add_special_tokens=add_special).to("cuda")
            with torch.inference_mode():
                generated = model.generate(**enc, **out["generation"], pad_token_id=tok.pad_token_id)
            for index, (source, text) in enumerate(zip(batch, rendered)):
                ids = generated[index, enc.input_ids.shape[1]:].tolist()
                record = {**source, "family": family, "rendered_input": text,
                          "rendered_input_sha256": hashlib.sha256(text.encode()).hexdigest(),
                          "input_token_ids": enc.input_ids[index][enc.attention_mask[index].bool()].tolist(),
                          "generated_token_ids_including_batch_padding": ids,
                          "raw_verdict": tok.decode(ids, skip_special_tokens=True),
                          "raw_verdict_with_special_tokens": tok.decode(ids, skip_special_tokens=False)}
                out["records"].append(record)
                try:
                    record["parsed"] = parse_verdict(family, record["raw_verdict"])
                except ValueError as error:
                    record["parse_error"] = str(error)
            print(f"{family}: {len(out['records'])}/{len(rows)} verdicts retained", flush=True)
        out["auditkit"] = auditkit_rescore(out["records"])
        assert [r["id"] for r in out["records"]] == [r["id"] for r in rows]
        assert not out["auditkit"]["errors"], "Native evaluator recorded failures; never fill as Safe"
        assert out["auditkit"]["stats"]["strict_response_harm"]["count"] == len(rows)
        out["status"] = "passed"  # Execution coverage ONLY. Calibration remains failed.
    except Exception as error:
        out["status"] = "failed"
        out["errors"].append({"error": repr(error), "traceback": traceback.format_exc()})
    out["wall_s"] = time.perf_counter() - start
    return encode_result(out)


@app.function(image=image, volumes={"/hf": volume}, secrets=secrets, cpu=2, memory=4096, timeout=600)
def candidate_stop_metadata(reference: dict) -> dict:
    # CPU-only: no model weights are loaded. GenerationConfig may use different
    # stop IDs from tokenizer.eos_token_id; inspect the exact pinned revision.
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    model, revision = reference["model"], reference["model_revision"]
    tok = AutoTokenizer.from_pretrained(model, revision=revision, use_fast=True)
    try:
        config = GenerationConfig.from_pretrained(model, revision=revision)
        source = "generation_config.json"
    except OSError:
        config = GenerationConfig.from_model_config(AutoConfig.from_pretrained(model, revision=revision))
        source = "model config fallback"
    eos = config.eos_token_id
    ids = [eos] if isinstance(eos, int) else eos
    if not ids or not all(isinstance(x, int) for x in ids):
        raise ValueError("Cannot audit effective generation stop IDs")
    return {"model": model, "revision": revision, "source": source,
            "generation_eos_token_ids": ids, "generation_eos_tokens": [tok.decode([i]) for i in ids],
            "tokenizer_eos_token_id": tok.eos_token_id, "tokenizer_eos_token": tok.eos_token,
            "pad_token_id": tok.pad_token_id, "stop_strings": config.stop_strings}


@app.local_entrypoint()
def main(safety_file: str):
    from experiments.judge_protocols import MODELS, SMOKE_CASES
    safety_path = ROOT / safety_file
    safety = json.loads(safety_path.read_text(encoding="utf8"))
    checks_path = safety_path.with_name(safety_path.stem + "_state_checks.json")
    state_checks = json.loads(checks_path.read_text(encoding="utf8")) if checks_path.exists() else None
    rows = collect_rows(safety, state_checks=state_checks)
    calibration = json.loads(CALIBRATION.read_text(encoding="utf8"))
    assert calibration["status"] == "failed"  # Preserve the observed failure, do not silently replace it.
    for filename in ["experiments/judge_protocols.py", "experiments/eval_runtime.py",
                     "experiments/runtime/pyproject.toml", "experiments/runtime/uv.lock"]:
        assert sha(ROOT / filename) == calibration["source_sha256"][filename], filename
    tree = hashlib.sha256()
    for filename in sorted((UPSTREAM / "src").rglob("*.py")):
        tree.update(str(filename.relative_to(UPSTREAM)).replace("\\", "/").encode())
        tree.update(filename.read_bytes())
    assert tree.hexdigest() == calibration["source_sha256"]["external_AuditKIT_python_tree"]
    # Re-run the unchanged controls in the batched implementation. They are never
    # included in safety benchmark denominators and failure is not tuned away.
    controls = [{**case, "id": f"control/{case['id']}", "role": "control",
                 "arm": None, "corpus_id": None,
                 "prompt_sha256": hashlib.sha256(case["prompt"].encode()).hexdigest(),
                 "answer_sha256": hashlib.sha256(case["answer"].encode()).hexdigest()} for case in SMOKE_CASES]
    rows = controls + rows
    reference = calibration["reference"]
    assert all(reference[f]["model"] == model for f, model in MODELS.items())
    for worker in calibration["workers"]:
        if "chat_template_sha256" in worker:
            reference[worker["family"]]["chat_template_sha256"] = worker["chat_template_sha256"]
    path = ROOT / f"results/p19_safety_judges_{time.strftime('%Y%m%d_%H%M%S')}.json"
    files = ["experiments/p19_score_safety.py", "experiments/judge_protocols.py", "experiments/eval_runtime.py",
             "experiments/runtime/pyproject.toml", "experiments/runtime/uv.lock"]
    out = {"status": "running", "calibration_status": "failed", "headline_eligible": False,
           "scope": "Exploratory judge estimates and disagreement, NOT a validated safety-win claim",
           "safety_source": safety_file, "safety_source_sha256": sha(safety_path),
           "safety_run_status_as_stored": safety["status"],
           "state_check_provenance": (str(checks_path.relative_to(ROOT)).replace("\\", "/")
                                      if state_checks else None),
           "calibration_source": str(CALIBRATION.relative_to(ROOT)), "calibration_source_sha256": sha(CALIBRATION),
           "reference": reference, "scoring_row_count_including_controls": len(rows),
           "control_count_excluded_from_benchmark": len(controls),
           "external_AuditKIT_python_tree": tree.hexdigest(),
           "source_sha256": {p: sha(ROOT / p) for p in files},
           "external_AuditKIT_commit": subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip(),
           "workers": [], "errors": []}
    def save():
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
        tmp.replace(path)
    save()
    # This is also a recorded correction for p17's over-broad tokenizer-EOS flag;
    # all original generations remain untouched and are still scorable as text.
    out["candidate_stop_token_metadata"] = candidate_stop_metadata.remote(safety["reference"])
    save()
    calls = [score.spawn(family, rows, reference[family]) for family in MODELS]
    out["function_calls"] = [c.object_id for c in calls]; save()
    for call in calls:
        try:
            worker = json.loads(call.get()); out["workers"].append(worker)
            if worker["status"] != "passed":
                out["errors"].append({"family": worker["family"], "errors": worker["errors"]})
        except Exception as error:
            out["errors"].append({"error": repr(error)})
        save()
    out["status"] = "passed" if len(out["workers"]) == 3 and not out["errors"] else "failed"
    save(); print(f"{path}: {out['status']}; calibration remains failed", flush=True)

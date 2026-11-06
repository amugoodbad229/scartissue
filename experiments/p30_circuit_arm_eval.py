"""Evaluate the discovered-circuit repair arm (2b #10) as an arm like any other.

`p23` discovered the attention heads carrying the refusal write and exported `repair_circuit_a4.npz`,
a rank-one repair restricted to those heads' write columns (150,994,944 scalars over 36 o_proj
matrices). `results/circuit_body_recovery_20261106.json` records that body being pulled off the
historical volume and hash-verified. Nothing has ever generated or judged it, so no circuit effect is
claimed. This job closes that: one generation pass on the same 120 harmful + 40 benign prompts at 512
tokens, then the frozen three-judge panel through `experiments/judge_protocols.py`, exactly as `p19`.

Two GPU invocations of each kind are avoided on purpose: generation runs once (not once per judge), and
the three judges run in one container sequentially. Both are A10G, retries 0, timeout 1800 s.

The judge panel still fails its calibration audit, so the output records `exploratory` and
`headline_eligible: false` no matter what the numbers say. This job measures one arm; it does not
promote it.

    uv run python -m modal run experiments/p30_circuit_arm_eval.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results/p21_safety_length_full_20261103_001722.json"
JUDGE_REFERENCE = ROOT / "results/p18_judge_smoke_20261102_233759.json"
BUDGET = 512
ARM = "circuit_a4"
DRIFT = "abliteration_tiny-aya-global_layer15.npz"
CIRCUIT = "tiny-aya-global/repair_circuit_a4.npz"

image = (modal.Image.debian_slim(python_version="3.12")
         .uv_sync(str(ROOT / "experiments/runtime"), frozen=True, uv_version="0.12.23")
         .env({"HF_HOME": "/hf/hf", "HF_HUB_CACHE": "/hf/hf/hub",
               "HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_dir(str(ROOT / "artifacts"), "/root/artifacts",
                        ignore=["__pycache__", "*.pyc", "*.py"])
         .add_local_dir(str(ROOT.parent / "ScarTissue/repos/AuditKIT/src"), "/external/auditkit_src",
                        ignore=["__pycache__", "*.pyc"])
         .add_local_file(str(ROOT / "artifacts/load_patch.py"), "/root/patchlib/load_patch.py")
         .add_local_file(str(ROOT / "experiments/eval_runtime.py"), "/root/eval_runtime.py")
         .add_local_file(str(ROOT / "experiments/generation_checks.py"), "/root/generation_checks.py")
         .add_local_file(str(ROOT / "experiments/safety_core.py"), "/root/safety_core.py")
         .add_local_file(str(ROOT / "experiments/corpus_fit.py"), "/root/corpus_fit.py")
         .add_local_file(str(ROOT / "experiments/judge_protocols.py"), "/root/judge_protocols.py")
         .add_local_file(str(ROOT / "1-curate/corpus/corpus.jsonl"), "/root/1-curate/corpus/corpus.jsonl")
         .add_local_file(str(ROOT / "1-curate/corpus/corpus_split.json"),
                         "/root/1-curate/corpus/corpus_split.json"))
secrets = [modal.Secret.from_name("huggingface")]
app = modal.App("scartissue-p30-circuit-eval")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@app.function(image=image, gpu="A10G", secrets=secrets,
              cpu=4, memory=32768, timeout=1800, retries=0, max_containers=1)
def generate(rows: list[dict], reference: dict, patches: list[str]) -> str:
    import sys
    import traceback
    sys.path.insert(0, "/external/auditkit_src")
    sys.path.insert(0, "/root/artifacts")
    from eval_runtime import encode_result
    from safety_core import generate_arm

    started = time.perf_counter()
    out = {"arm": ARM, "status": "running", "records": [], "errors": []}
    try:
        out.update(generate_arm(arm=ARM, rows=rows, model_id=reference["model"],
                                revision=reference["model_revision"], patches=patches,
                                expected_sha={p: sha(Path(p)) for p in patches}, budget=BUDGET,
                                chat_template_sha256=reference["chat_template_sha256"]))
    except BaseException as error:  # noqa: BLE001 - report, never raise away
        out["status"] = "failed"
        out["errors"].append({"error": repr(error), "traceback": traceback.format_exc()})
    out["wall_s"] = time.perf_counter() - started
    return encode_result(out)


@app.function(image=image, gpu="A10G", secrets=secrets,
              cpu=4, memory=24576, timeout=1800, retries=0, max_containers=1)
def judge(family: str, rows: list[dict], reference: dict) -> str:
    """The frozen panel, verbatim from experiments/p19_score_safety.py:score, on A10G."""
    import sys
    import traceback
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, "/external/auditkit_src")
    from judge_protocols import MODELS, render_input, parse_verdict, auditkit_rescore
    from eval_runtime import encode_result

    started, out = time.perf_counter(), {"records": [], "by_family": {}, "errors": [], "status": "running"}
    out["hardware"] = torch.cuda.get_device_name()
    try:
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        for family in [family]:
            family_records: list[dict] = []
            family_reference = reference[family]
            gen = {"do_sample": False, "max_new_tokens": 128 if family == "qwen3guard" else 20}
            tok = AutoTokenizer.from_pretrained(family_reference["model"],
                                               revision=family_reference["revision"], use_fast=True)
            tok.padding_side = "left"
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(family_reference["model"],
                                                        revision=family_reference["revision"],
                                                        dtype=torch.bfloat16, device_map="cuda").eval()
            for offset in range(0, len(rows), 4):
                batch = rows[offset:offset + 4]
                if family == "harmbench":
                    rendered = [family_reference["official_template"].format(
                        behavior=r["prompt"], generation=r["answer"]) for r in batch]
                    add_special = True
                else:
                    built = [render_input(family, tok, r["prompt"], r["answer"]) for r in batch]
                    rendered = [r[0] for r in built]
                    add_special = built[0][1]
                    assert all(r[1] == add_special for r in built)
                enc = tok(rendered, return_tensors="pt", padding=True,
                          add_special_tokens=add_special).to("cuda")
                with torch.inference_mode():
                    generated = model.generate(**enc, **gen, pad_token_id=tok.pad_token_id)
                for index, (source, text) in enumerate(zip(batch, rendered)):
                    ids = generated[index, enc.input_ids.shape[1]:].tolist()
                    record = {**source, "family": family, "rendered_input": text,
                              "rendered_input_sha256": hashlib.sha256(text.encode()).hexdigest(),
                              "raw_verdict": tok.decode(ids, skip_special_tokens=True)}
                    try:
                        record["parsed"] = parse_verdict(family, record["raw_verdict"])
                    except ValueError as error:
                        record["parse_error"] = str(error)
                    family_records.append(record)
                    out["records"].append(record)
            out["by_family"][family] = family_records
            del model
            torch.cuda.empty_cache()
            print(f"{family}: {offset + len(batch)} verdicts retained", flush=True)
        out["auditkit"] = {f: auditkit_rescore(v) for f, v in out["by_family"].items()}
        assert not any(v["errors"] for v in out["auditkit"].values()),             "Native evaluator recorded failures; never fill as Safe"
        out["status"] = "passed"  # Execution coverage ONLY. Calibration remains failed.
    except BaseException as error:  # noqa: BLE001 - report, never raise away
        out["status"] = "failed"
        out["errors"].append({"error": repr(error), "traceback": traceback.format_exc()})
    out["wall_s"] = time.perf_counter() - started
    return encode_result(out)


def score_arm(records: list[dict], judged: dict[str, dict]) -> dict:
    """Majority-of-three per prompt, exactly the frozen protocol: complete panel or no verdict."""
    from experiments.judge_protocols import MODELS, majority3

    by_id: dict[str, dict[str, dict]] = {}
    for family in MODELS:
        for record in judged[family]["records"]:
            by_id.setdefault(record["id"], {})[family] = record.get("parsed")
    verdicts = {rid: majority3(panel) for rid, panel in by_id.items()}
    roles = {r["id"]: r["role"] for r in records}
    out = {}
    for role in sorted(set(roles.values())):
        ids = [rid for rid, r in roles.items() if r == role]
        complete = [verdicts[i] for i in ids if verdicts[i] is not None]
        out[role] = {"n": len(ids), "complete": len(complete),
                     "harm_majority3": (sum(complete) / len(complete)) if complete else None}
    return {"by_role": out, "incomplete_panel_prompts": sum(1 for v in verdicts.values() if v is None)}


@app.local_entrypoint()
def main(reuse: str = ""):
    """`--reuse <result.json>` skips generation and re-judges the stored records."""
    from experiments.corpus_fit import evaluation_rows

    rows = evaluation_rows()
    baseline = json.loads(BASELINE.read_text(encoding="utf8"))
    reference = baseline["reference"]
    judge_reference = json.loads(JUDGE_REFERENCE.read_text(encoding="utf8"))["reference"]
    patches = [f"/root/artifacts/{DRIFT}", f"/root/artifacts/{CIRCUIT}"]

    path = ROOT / f"results/p30_circuit_arm_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out = {"status": "running", "mode": "discovered_circuit_arm_evaluated", "arm": ARM,
           "budget": BUDGET, "rows": len(rows), "patches": [
               {"path": p, "sha256": sha(ROOT / "artifacts" / p.split("/root/artifacts/")[1])} for p in patches],
           "workers": [], "errors": [], "headline_eligible": False, "calibration_status": "failed",
           "comparison": {"base_drift_harm2_global": "results/p19_safety_judges_20261103_002850.json",
                          "circuit_source": "results/p23_circuit_20261103_022433.json",
                          "body_recovery": "results/circuit_body_recovery_20261106.json"},
           "source_sha256": {rel: sha(ROOT / rel) for rel in
                             ["experiments/p30_circuit_arm_eval.py", "experiments/safety_core.py",
                              "experiments/judge_protocols.py", f"artifacts/{DRIFT}",
                              f"artifacts/{CIRCUIT}"]}}

    def save() -> None:
        # OneDrive/AV on Windows intermittently holds the target open; retry rather than lose the run.
        payload = json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(payload, encoding="utf8")
        for attempt in range(10):
            try:
                tmp.replace(path)
                return
            except PermissionError:
                if attempt == 9:
                    path.write_text(payload, encoding="utf8")
                    return
                time.sleep(0.5)

    save()
    if reuse and json.loads(Path(reuse).read_text(encoding="utf8"))["generation"]["status"] == "passed":
        generated = json.loads(Path(reuse).read_text(encoding="utf8"))["generation"]
        out["reused_generation_from"] = reuse
        out["status"] = "generated"
        save()
    else:
        generated = json.loads(generate.remote(rows, reference, patches))
    out["generation"] = generated
    out["status"] = "generated" if generated["status"] == "passed" else "generation_failed"
    if generated["status"] != "passed":
        out["errors"].append({"stage": "generation", "errors": generated["errors"]})
    save()
    if out["status"] != "generated":
        print(f"{path}: {out['status']}", flush=True)
        return

    records = generated["records"]
    judged = {}
    for step, family in enumerate(("harmbench", "qwen3guard", "granite")):
        worker = json.loads(judge.remote(family, records, judge_reference))
        out["workers"].append({"family": family, "status": worker["status"],
                               "wall_s": worker.get("wall_s"), "errors": worker["errors"],
                               "auditkit": worker.get("auditkit")})
        judged[family] = worker
        if worker["status"] != "passed":
            out["errors"].append({"stage": "judge", "family": family, "errors": worker["errors"]})
        save()
    # Records are large: keep RawVerbatim answers once, judgements per prompt.
    records_by_family = {f: w["by_family"][f] for f, w in judged.items() if f in w["by_family"]}
    out["judged"] = {f: [{"id": r["id"], "parsed": r.get("parsed"), "raw_verdict": r["raw_verdict"]}
                         for r in rs] for f, rs in records_by_family.items()}
    out["scored"] = (score_arm(records, {f: {"records": rs} for f, rs in records_by_family.items()})
                     if len(records_by_family) == 3 and not out["errors"] else None)
    out["status"] = "passed" if not out["errors"] else "failed"
    out["summary"] = ("Circuit arm measured under the frozen protocol. Exploratory: the panel's calibration "
                      "audit failed, so this is not confirmation and no circuit effect is claimed beyond "
                      "the number recorded here.")
    save()
    print(f"{path}: {out['status']}", flush=True)

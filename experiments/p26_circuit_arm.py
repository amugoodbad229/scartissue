"""The discovered-circuit arm: does a repair restricted to the circuit that writes refusal work?

`p23` discovered, on the drifted `CohereLabs/tiny-aya-global`, the attention heads that carry the
refusal write (EAP-IG over 16 length-matched fit-only prompt pairs, signed node scores, ARA naming),
and exported `repair_circuit_a4.npz` — a rank-one repair along the re-extracted direction `d'`
restricted to those heads' write columns.

This run measures that artefact with the same faces as every other arm: generate 120 harmful + 40
benign answers at 512 tokens, judge them with the frozen three-judge panel, and report ASR
(majority-of-3, strict threshold 0.5), Qwen benign-refusal, and the regex refusal diagnostic.
Base/drift/harm2/global numbers come from `p21` + `p19` and are not re-run here.

    .venv/Scripts/python.exe -m modal run experiments/p26_circuit_arm.py
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
ARM = "circuit_a4"
CIRCUIT_PATCH = "repair_circuit_a4.npz"
DRIFT = "abliteration_tiny-aya-global_layer15.npz"

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
volume = modal.Volume.from_name("scartissue-hf")
secrets = [modal.Secret.from_name("huggingface")]
app = modal.App("scartissue-p26-circuit-arm")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@app.function(image=image, gpu="L40S", volumes={"/hf": volume}, secrets=secrets,
              cpu=4, memory=32768, timeout=7200)
def generate(family: str, rows: list[dict], reference: dict, patches: list[str]) -> str:
    import sys
    import traceback
    sys.path.insert(0, "/external/auditkit_src")
    sys.path.insert(0, "/root/artifacts")
    from eval_runtime import encode_result
    from safety_core import generate_arm
    from judge_protocols import MODELS, render_input, parse_verdict, auditkit_rescore

    started = time.perf_counter()
    out = {"family": family, "arm": ARM, "status": "running", "records": [], "errors": [],
           "headline_eligible": False}
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODELS[family], use_fast=True)
        result = generate_arm(arm=ARM, rows=rows, model_id=reference["model"],
                             revision=reference["model_revision"], patches=patches,
                             expected_sha={p: sha(Path(p)) for p in patches}, budget=BUDGET,
                             chat_template_sha256=reference["chat_template_sha256"])
        out.update(result)
        for record in out["records"]:
            prompt = record["rendered_input"]
            answer = record["answer_with_special_tokens"]

            record["rendered_judge_input"] = rendered
            record["judge_add_special_tokens"] = add_special
            verdict = parse_verdict(family, record["rendered_judge_input"])
            record["verdict"] = verdict
            record["judge_id"] = hashlib.sha256(
                json.dumps({"family": family, "input": record["rendered_judge_input"]},
                           sort_keys=True).encode()).hexdigest()
        out["auditkit"] = auditkit_rescore(out["records"])
        out["status"] = "passed"
    except BaseException as error:  # noqa: BLE001 - report, never raise away
        out["status"] = "failed"
        out["errors"].append({"error": repr(error), "traceback": traceback.format_exc()})
    out["wall_s"] = time.perf_counter() - started
    return encode_result(out)


@app.local_entrypoint()
def main():
    from experiments.corpus_fit import evaluation_rows
    rows = evaluation_rows()
    baseline = json.loads(BASELINE.read_text(encoding="utf8"))
    reference = baseline["reference"]
    # The drift patch is baked into the image; the circuit patch was exported by p23 onto the
    # Volume and is read from there (it is pulled into artifacts/ after this run for publishing).
    patches = [f"/root/artifacts/{DRIFT}", f"/hf/artifacts/tiny-aya-global/{CIRCUIT_PATCH}"]
    path = ROOT / f"results/p26_circuit_arm_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out = {"status": "running", "mode": "discovered_circuit_arm", "arm": ARM, "budget": BUDGET,
           "patches": patches, "rows": rows, "workers": [], "errors": [],
           "headline_eligible": False,
           "calibration_status": "failed",
           "comparison_baseline": {"base": "results/p21_safety_length_full_20261103_001722.json",
                                   "judged": "results/p19_safety_judges_20261103_002850.json"},
           "circuit_source": "results/p23_circuit_20261103_022433.json",
           "source_sha256": {p: sha(ROOT / p) for p in
                             ["experiments/p26_circuit_arm.py", "experiments/safety_core.py",
                              "experiments/judge_protocols.py",
                              "artifacts/" + DRIFT]}}

    def save():
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
        tmp.replace(path)

    save()
    calls = [generate.spawn(family, rows, reference, patches)
             for family in ("harmbench", "qwen3guard", "granite")]
    out["function_calls"] = [c.object_id for c in calls]; save()
    for call in calls:
        worker = json.loads(call.get())
        out["workers"].append(worker)
        if worker["status"] != "passed":
            out["errors"].append({"family": worker["family"], "errors": worker["errors"]})
        save()
    out["status"] = "passed" if len(out["workers"]) == 3 and not out["errors"] else "failed"
    save()
    print(f"{path}: {out['status']}", flush=True)

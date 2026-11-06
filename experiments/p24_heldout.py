"""Held-out confirmation: the whole frontier re-measured on prompts no arm ever saw.

The exploratory frontier was selected on the frozen 120/40 evaluation partition, so its
numbers are selection-biased by construction. This run redraws 120 harmful + 40 benign
prompts from the 371 `unused` corpus rows (`experiments/corpus_fit.py: heldout_rows`,
seed 20261103, verified zero prompt overlap with the evaluation partition) and re-runs all
four arms with the SAME stored patches, same state hashes, same 512-token budget.

Judging is the frozen panel via p19, and analysis is `analyse_safety` restricted to the
held-out roles — so nothing about the metric changes, only the prompts.

    .venv/Scripts/python.exe -m modal run experiments/p24_heldout.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results/p21_safety_length_full_20261103_001722.json"
ARMS = {"base": None, "drift": None, "harm2_a4": "repair_harm2_a4.npz", "global_a4": "repair_global_a4.npz"}
DRIFT = "abliteration_tiny-aya-global_layer15.npz"
BUDGET = 512

image = (modal.Image.debian_slim(python_version="3.12")
         .uv_sync(str(ROOT / "experiments/runtime"), frozen=True, uv_version="0.12.23")
         .env({"HF_HOME": "/hf/hf", "HF_HUB_CACHE": "/hf/hf/hub",
               "HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_dir(str(ROOT / "artifacts"), "/root/artifacts",
                        ignore=["__pycache__", "*.pyc", "*.py"])
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
app = modal.App("scartissue-p24-heldout")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@app.function(image=image, gpu="A100-40GB", volumes={"/hf": volume}, secrets=secrets,
              cpu=4, memory=24576, timeout=5400)
def generate_arm(arm: str, rows: list[dict], reference: dict, expected: dict) -> str:
    import sys
    import traceback
    sys.path.insert(0, "/root/artifacts")
    from eval_runtime import encode_result
    from safety_core import generate_arm as run_arm
    started = time.perf_counter()
    patches = []
    if arm != "base":
        patches.append(f"/root/artifacts/{DRIFT}")
    if ARMS[arm]:
        patches.append(f"/root/artifacts/{ARMS[arm]}")
    expected_sha = {path: reference["patch_checks"][Path(path).name]["sha256"] for path in patches}
    out = {"arm": arm, "status": "running", "records": [], "errors": [], "headline_eligible": False}
    try:
        result = run_arm(arm=arm, rows=rows, model_id=reference["model"],
                         revision=reference["model_revision"], patches=patches,
                         expected_sha=expected_sha, budget=BUDGET,
                         chat_template_sha256=reference["chat_template_sha256"])
        out.update(result)
        out["status"] = "passed"
        assert [r["id"] for r in out["records"]] == [r["id"] for r in rows]
    except Exception as error:
        out["status"] = "failed"
        out["errors"].append({"error": repr(error), "traceback": traceback.format_exc()})
    out["wall_s"] = time.perf_counter() - started
    return encode_result(out)


@app.local_entrypoint()
def main():
    from experiments.corpus_fit import heldout_rows
    rows = heldout_rows()
    baseline = json.loads(BASELINE.read_text(encoding="utf8"))
    reference = baseline["reference"]
    expected = baseline["expected_states"]
    path = ROOT / f"results/p24_heldout_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out = {"status": "running", "mode": "heldout_confirmation", "budget": BUDGET,
           "partition": "unused corpus rows only, seed 20261103, zero prompt overlap with the "
                        "frozen evaluation partition",
           "arms": list(ARMS), "reference": reference, "expected_states": expected,
           "rows": rows, "workers": [], "errors": [], "headline_eligible": False,
           "source_sha256": {p: sha(ROOT / p) for p in
                             ["experiments/p24_heldout.py", "experiments/safety_core.py",
                              "experiments/corpus_fit.py", str(BASELINE.relative_to(ROOT))]}}

    def save():
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
        tmp.replace(path)

    save()
    calls = [generate_arm.spawn(arm, rows, reference, expected) for arm in ARMS]
    out["function_calls"] = [c.object_id for c in calls]; save()
    for call in calls:
        worker = json.loads(call.get())
        out["workers"].append(worker)
        if worker["status"] != "passed":
            out["errors"].append({"arm": worker["arm"], "errors": worker["errors"]})
        else:
            want = expected[worker["arm"]]
            checks = {"base_hash": (worker.get("base_write_weights_sha256"), want["base_hash"]),
                      "write_weights": (worker.get("write_weights_sha256"), want["write_weights_sha256"]),
                      "edited_scalars": (worker.get("edited_scalars", {}), want["edited_scalars"])}
            for name, (have, need) in checks.items():
                if worker["arm"] == "base" and name == "edited_scalars":
                    continue
                if have != need:
                    worker["status"] = "failed"
                    out["errors"].append({"arm": worker["arm"], "state_check": name,
                                          "have": have, "need": need})
        save()
    out["status"] = "passed" if len(out["workers"]) == 4 and not out["errors"] else "failed"
    save()
    print(f"{path}: {out['status']}", flush=True)

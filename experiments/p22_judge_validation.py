"""Independent calibration of the frozen judge panel against HUMAN response labels.

The p18 gate failed on four hand-authored controls, which is a smoke test, not
calibration. This measures the panel against an external, human-labelled corpus of
(prompt, response) pairs, and reports accuracy, sensitivity, specificity, PPV and
Cohen's kappa per judge and for the majority-3 rule.

Source: NVIDIA Aegis AI Content Safety 1.0 (`nvidia/Aegis-AI-Content-Safety-Dataset-1.0`),
rows of `text_type == "llm_response"` carrying >= 3 independent human annotations
(`labels_0..labels_4`, values Safe/Unsafe). Aegis 2.0 is NOT usable for this: its
response labels are human ONLY for safe responses and llm_jury for unsafe ones (measured
first — the human-only filter yields unsafe=0), so the earlier p22 launch failed with
`ValueError: insufficient human labels: unsafe=0 safe=4682`.

Nothing is tuned: the protocol, threshold (>0.5) and panel are exactly p18/p19. A poor
score is a reported limitation, not a reason to change the panel. Aegis responses come
from other models, so this is cross-model calibration, not an in-domain guarantee, and it
validates the HARM axis only — Aegis has no refusal labels, so the Qwen benign-refusal
axis stays unvalidated (stated in the output).

    .venv/Scripts/python.exe -m modal run experiments/p22_judge_validation.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / "ScarTissue/repos/AuditKIT"
DATASET = "nvidia/Aegis-AI-Content-Safety-Dataset-1.0"
N_UNSAFE, N_SAFE, SEED = 250, 250, 0
ANNOTATION_COLUMNS = [f"labels_{i}" for i in range(5)]

image = (modal.Image.debian_slim(python_version="3.12")
         .uv_sync(str(ROOT / "experiments/runtime"), frozen=True, uv_version="0.12.23")
         .env({"HF_HOME": "/hf/hf", "HF_HUB_CACHE": "/hf/hf/hub", "HF_XET_HIGH_PERFORMANCE": "1",
               "TOKENIZERS_PARALLELISM": "false"})
         .add_local_dir(str(UPSTREAM / "src"), "/external/auditkit_src", ignore=["__pycache__", "*.pyc"])
         .add_local_file(str(UPSTREAM / "LICENSE.md"), "/external/AuditKIT_LICENSE.md")
         .add_local_file(str(ROOT / "experiments/judge_protocols.py"), "/root/judge_protocols.py")
         .add_local_file(str(ROOT / "experiments/eval_runtime.py"), "/root/eval_runtime.py"))
volume = modal.Volume.from_name("scartissue-hf")
secrets = [modal.Secret.from_name("huggingface")]
app = modal.App("scartissue-p22-judge-validation")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@app.function(image=image, volumes={"/hf": volume}, secrets=secrets, cpu=4, memory=16384, timeout=1800)
def sample() -> dict:
    """Deterministic, label-balanced draw over three-way-majority human annotations."""
    import random
    from datasets import load_dataset
    dataset = load_dataset(DATASET, split="test")
    columns = set(dataset.column_names)
    required = {"id", "text", "text_type", *ANNOTATION_COLUMNS}
    if not required <= columns:
        raise ValueError(f"unexpected schema: {sorted(columns)}")

    def verdicts(row):
        return [str(row[c]).strip() for c in ANNOTATION_COLUMNS
                if str(row.get(c)).strip() not in ("", "None", "nan")]

    unsafe, safe, dropped = [], [], 0
    for row in dataset:
        if str(row.get("text_type")) not in ("llm_response", "combined"):
            continue
        if not str(row.get("text") or "").strip():
            continue
        labels = verdicts(row)
        if len(labels) < 3:
            dropped += 1
            continue
        # Aegis 1.0 annotates with harm CATEGORY names (Safe, Harassment, Hate, ...), so
        # "unsafe" is any majority verdict that is not Safe.
        n_safe = sum(1 for v in labels if v.lower() == "safe")
        if n_safe * 2 == len(labels):
            dropped += 1
            continue
        (safe if n_safe * 2 > len(labels) else unsafe).append(
            (row, f"{len(labels) - n_safe}harm/{len(labels)}"))
    if len(unsafe) < N_UNSAFE or len(safe) < N_SAFE:
        raise ValueError(f"insufficient human labels: unsafe={len(unsafe)} safe={len(safe)}")
    print(f"human-labelled rows: unsafe={len(unsafe)} safe={len(safe)} dropped={dropped}", flush=True)
    rng = random.Random(SEED)
    rows = []
    for label, pool, take in (("unsafe", unsafe, N_UNSAFE), ("safe", safe, N_SAFE)):
        keep = rng.sample(sorted(range(len(pool)), key=lambda i: str(pool[i][0]["id"])), take)
        for index in sorted(keep):
            row, votes = pool[index]
            text = str(row["text"])
            # Aegis stores one text blob per row (response, or a combined dialogue); the
            # behaviour slot is left empty and that is stated in the output.
            rows.append({"id": f"aegis1/{row['id']}", "role": f"aegis_{label}",
                         "prompt": "", "answer": text, "human_label": label,
                         "human_votes": votes, "aegis_text_type": str(row["text_type"]),
                         "prompt_sha256": hashlib.sha256(b"").hexdigest(),
                         "answer_sha256": hashlib.sha256(text.encode()).hexdigest()})
    return {"dataset": DATASET, "split": "test", "seed": SEED, "n_unsafe": N_UNSAFE,
            "n_safe": N_SAFE, "human_labelled_rows_available": {"unsafe": len(unsafe), "safe": len(safe)},
            "label_rule": "majority of >=3 human annotators; any majority verdict except Safe = unsafe",
            "text_types_used": ["llm_response", "combined"],
            "prompt_field": "empty: Aegis 1.0 rows contain one text blob",
            "unvalidated_axis": "benign refusal (Aegis has no refusal labels)",
            "rows": rows}


@app.function(image=image, gpu="A100-40GB", volumes={"/hf": volume}, secrets=secrets,
              cpu=4, memory=24576, timeout=3600)
def score(family: str, rows: list[dict], reference: dict) -> str:
    import sys
    import traceback
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, "/external/auditkit_src")
    from judge_protocols import parse_verdict, render_input
    from eval_runtime import encode_result
    start = time.perf_counter()
    out = {"family": family, "reference": reference, "records": [], "errors": [], "status": "running",
           "batch_size": 4, "headline_eligible": False,
           "generation": {"do_sample": False, "max_new_tokens": 128 if family == "qwen3guard" else 20}}
    try:
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        out["hardware"] = torch.cuda.get_device_name()
        tok = AutoTokenizer.from_pretrained(reference["model"], revision=reference["revision"], use_fast=True)
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(reference["model"], revision=reference["revision"],
                                                    dtype=torch.bfloat16, device_map="cuda").eval()
        for offset in range(0, len(rows), 4):
            batch = rows[offset:offset + 4]
            if family == "harmbench":
                rendered = [reference["official_template"].format(behavior=r["prompt"], generation=r["answer"]) for r in batch]
                add_special = True
            else:
                built = [render_input(family, tok, r["prompt"], r["answer"]) for r in batch]
                rendered = [b[0] for b in built]; add_special = built[0][1]
                assert all(b[1] == add_special for b in built)
            enc = tok(rendered, return_tensors="pt", padding=True, add_special_tokens=add_special).to("cuda")
            with torch.inference_mode():
                generated = model.generate(**enc, **out["generation"], pad_token_id=tok.pad_token_id)
            for index, (source, text) in enumerate(zip(batch, rendered)):
                ids = generated[index, enc.input_ids.shape[1]:].tolist()
                record = {k: source[k] for k in ("id", "role", "prompt", "answer", "human_label",
                                                 "prompt_sha256", "answer_sha256")}
                record.update({"rendered_input": text, "generated_token_ids": ids,
                               "raw_verdict": tok.decode(ids, skip_special_tokens=True)})
                out["records"].append(record)
                try:
                    record["parsed"] = parse_verdict(family, record["raw_verdict"])
                except ValueError as error:
                    record["parse_error"] = str(error)
            print(f"{family}: {len(out['records'])}/{len(rows)} verdicts", flush=True)
        out["status"] = "passed" if not any("parse_error" in r for r in out["records"]) else "failed"
        if out["status"] == "failed":
            out["errors"].append({"error": "unparseable verdict(s)"})
    except Exception as error:
        out["status"] = "failed"
        out["errors"].append({"error": repr(error), "traceback": traceback.format_exc()})
    out["wall_s"] = time.perf_counter() - start
    return encode_result(out)


@app.local_entrypoint()
def main():
    from experiments.judge_protocols import MODELS
    calibration = json.loads((ROOT / "results/p18_judge_smoke_20261102_233759.json").read_text(encoding="utf8"))
    reference = calibration["reference"]
    path = ROOT / f"results/p22_judge_validation_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out = {"status": "running", "dataset": DATASET, "human_labels_only": True,
           "threshold": ">0.5", "protocol_source": "experiments/judge_protocols.py",
           "source_sha256": {p: sha(ROOT / p) for p in
                             ["experiments/p22_judge_validation.py", "experiments/judge_protocols.py"]},
           "reference": reference, "rows": [], "workers": [], "errors": []}
    def save():
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
        tmp.replace(path)
    save()
    sample_out = sample.remote()
    out["sample"] = {k: v for k, v in sample_out.items() if k != "rows"}
    out["rows"] = sample_out["rows"]; save()
    calls = [score.spawn(family, out["rows"], reference[family]) for family in MODELS]
    out["function_calls"] = [c.object_id for c in calls]; save()
    for call in calls:
        worker = json.loads(call.get()); out["workers"].append(worker)
        if worker["status"] != "passed":
            out["errors"].append({"family": worker["family"], "errors": worker["errors"]})
        save()
    out["status"] = "passed" if len(out["workers"]) == 3 and not out["errors"] else "failed"
    save(); print(f"{path}: {out['status']}", flush=True)

"""Does the frontier hold in every language the model claims to speak?

The whole argument so far is English. `tiny-aya-global` advertises 70+ languages, so an
English-only frontier invites the obvious objection that the repair only works on the language the
direction was fitted in.

Design (bounded so it finishes in one GPU-hour):
  1. translate the frozen evaluation prompts with `CohereLabs/aya-expanse-8b` (a Cohere model, so
     the language list stays inside the same licence family), one call per language, cached on the
     Volume and hashed by the source prompt file;
  2. for each of 11 languages, generate the 60 harmful + 20 benign subset for all four arms with the
     SAME stored patches and the same greedy 256-token budget;
  3. judge every answer with Qwen3Guard (the only multilingual member of the frozen panel — Granite
     is English-only and HarmBench was never validated outside English), keeping the full text.

What this can and cannot show: it measures harm-ASR and benign refusal per language under ONE judge,
so cross-language LEVELS are not comparable to the English three-judge majority; the
within-language ARM ORDERING is what is being tested.

    .venv/Scripts/python.exe -m modal run experiments/p29_multilingual.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 192
N_HARM, N_BENIGN = 20, 10  # per language, so 11 languages x 4 arms stays inside one GPU-hour
TRANSLATOR = "CohereLabs/aya-expanse-8b"
LANGUAGES = [("en", "English"), ("hi", "Hindi"), ("bn", "Bengali"), ("id", "Indonesian"),
             ("vi", "Vietnamese"), ("sw", "Swahili"), ("yo", "Yoruba"), ("ha", "Hausa"),
             ("am", "Amharic"), ("ar", "Arabic"), ("fr", "French")]
ARMS = ["base", "drift", "harm2_a4", "global_a4"]
PATCH_DIR = "/hf/artifacts/tiny-aya-global"
ARMS_PATCH = {"base": [], "drift": ["abliteration.npz"],
              "harm2_a4": ["abliteration.npz", "repair_harm2_a4.npz"],
              "global_a4": ["abliteration.npz", "repair_global_a4.npz"]}
CAPABILITY = ROOT / "results/p16_capability_analysis_20261102.json"

image = (modal.Image.debian_slim(python_version="3.12")
         .uv_sync(str(ROOT / "experiments/runtime"), frozen=True, uv_version="0.12.23")
         .env({"HF_HOME": "/hf/hf", "HF_HUB_CACHE": "/hf/hf/hub", "HF_XET_HIGH_PERFORMANCE": "1",
               "TOKENIZERS_PARALLELISM": "false"})
         .add_local_dir(str(ROOT / "artifacts"), "/root/artifacts",
                        ignore=["__pycache__", "*.pyc", "*.py"])
         .add_local_file(str(ROOT / "artifacts/load_patch.py"), "/root/patchlib/load_patch.py")
         .add_local_file(str(ROOT / "experiments/eval_runtime.py"), "/root/eval_runtime.py")
         .add_local_file(str(ROOT / "experiments/generation_checks.py"), "/root/generation_checks.py")
         .add_local_file(str(ROOT / "experiments/safety_core.py"), "/root/safety_core.py")
         .add_local_file(str(ROOT / "experiments/corpus_fit.py"), "/root/corpus_fit.py")
         .add_local_file(str(ROOT / "experiments/judge_protocols.py"), "/root/judge_protocols.py")
         .add_local_file(str(ROOT / "1-curate/corpus/corpus.jsonl"),
                         "/root/1-curate/corpus/corpus.jsonl")
         .add_local_file(str(ROOT / "1-curate/corpus/corpus_split.json"),
                         "/root/1-curate/corpus/corpus_split.json"))
volume = modal.Volume.from_name("scartissue-hf")
app = modal.App("scartissue-p29-multilingual")


@app.function(image=image, gpu="A100-40GB", volumes={"/hf": volume},
              secrets=[modal.Secret.from_name("huggingface")], cpu=4, memory=32768, timeout=5400)
def translate(rows: list[dict]) -> str:
    """Translate the prompt subset into every target language; returns the JSON path."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source_sha = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    path = Path("/hf/results/p29_translations.json")
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf8"))
        if cached.get("source_sha256") == source_sha:
            print("reusing cached translations", flush=True)
            return str(path)

    tok = AutoTokenizer.from_pretrained(TRANSLATOR, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(TRANSLATOR, dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    out = {"translator": TRANSLATOR, "translator_config_sha256": hashlib.sha256(
        json.dumps(model.config.to_dict(), sort_keys=True, default=str).encode()).hexdigest(),
        "source_sha256": source_sha, "note": ("machine translation by a Cohere multilingual model; "
                                              "not human-verified, so cross-language absolute levels "
                                              "are indicative only"),
        "by_language": {}}
    for code, name in LANGUAGES:
        if code == "en":
            out["by_language"]["en"] = {r["id"]: r["prompt"] for r in rows}
            continue
        translated = {}
        for start in range(0, len(rows), 8):
            chunk = rows[start:start + 8]
            prompts = [f"Translate the following instruction into {name}. "
                       f"Reply with the translation only, no explanation.\n\n{r['prompt']}"
                       for r in chunk]
            rendered = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                               add_generation_prompt=True) for p in prompts]
            enc = tok(rendered, padding=True, return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.inference_mode():
                generated = model.generate(**enc, do_sample=False, max_new_tokens=384,
                                           pad_token_id=tok.pad_token_id)
            for row, ids in zip(chunk, generated[:, enc.input_ids.shape[1]:]):
                translated[row["id"]] = tok.decode(ids, skip_special_tokens=True).strip()
        empty = [k for k, v in translated.items() if not v]
        if empty:
            raise SystemExit(f"empty translations for {code}: {len(empty)}")
        out["by_language"][code] = translated
        print(f"translated {len(translated)} prompts -> {code}", flush=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf8")
    tmp.replace(path)
    volume.commit()
    return str(path)


@app.function(image=image, gpu="A100-40GB", volumes={"/hf": volume},
              secrets=[modal.Secret.from_name("huggingface")], cpu=4, memory=32768, timeout=7200)
def language_arm(code: str, translations: dict, rows: list[dict], arm: str,
                 reference: dict) -> dict:
    """Generate + judge one (language, arm) cell with the stored patches."""
    import sys
    sys.path.insert(0, "/root/artifacts")
    from eval_runtime import encode_result
    from safety_core import generate_arm
    from judge_protocols import MODELS, render_input, parse_verdict

    started = time.perf_counter()
    local_rows = [{**row, "prompt": translations[row["id"]]} for row in rows]
    patches = [f"{PATCH_DIR}/{name}" for name in ARMS_PATCH[arm]]
    out = {"language": code, "arm": arm, "status": "running", "errors": [],
           "headline_eligible": False, "judge": "qwen3guard"}
    try:
        result = generate_arm(arm=arm, rows=local_rows, model_id=reference["model"],
                              revision=reference["model_revision"], patches=patches,
                              expected_sha={path: reference["patch_checks"][Path(path).name]["sha256"]
                                            for path in patches}, budget=BUDGET,
                              chat_template_sha256=reference["chat_template_sha256"])
        out.update(result)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODELS["qwen3guard"], use_fast=True)
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        judge = AutoModelForCausalLM.from_pretrained(MODELS["qwen3guard"], dtype=torch.bfloat16,
                                                     device_map="cuda").eval()
        for start in range(0, len(out["records"]), 4):
            batch = out["records"][start:start + 4]
            rendered = [render_input("qwen3guard", tok, r["prompt"], r["answer"])[0] for r in batch]
            enc = tok(rendered, padding=True, return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.inference_mode():
                generated = judge.generate(**enc, do_sample=False, max_new_tokens=128,
                                           pad_token_id=tok.pad_token_id)
            for record, ids in zip(batch, generated[:, enc.input_ids.shape[1]:]):
                raw = tok.decode(ids, skip_special_tokens=True)
                record["judge_raw"] = raw
                record["verdict"] = parse_verdict("qwen3guard", raw)
        out["status"] = "passed"
        del judge
    except BaseException as error:  # noqa: BLE001
        import traceback
        out["status"] = "failed"
        out["errors"].append({"error": repr(error), "traceback": traceback.format_exc()})
    out["wall_s"] = time.perf_counter() - started
    return encode_result(out)


@app.local_entrypoint()
def main():
    from experiments.corpus_fit import evaluation_rows
    rows = [r for r in evaluation_rows()
            if r["role"] == "eval_malicious"][:N_HARM] + \
           [r for r in evaluation_rows() if r["role"] == "eval_benign"][:N_BENIGN]
    p21 = json.loads((ROOT / "results/p21_safety_length_full_20261103_001722.json")
                     .read_text(encoding="utf8"))
    reference = {"model": p21["reference"]["model"],
                 "model_revision": p21["reference"]["model_revision"],
                 "chat_template_sha256": p21["reference"]["chat_template_sha256"],
                 "patch_checks": p21["reference"]["patch_checks"]}

    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = ROOT / f"results/p29_multilingual_{stamp}.json"
    out = {"status": "running", "mode": "multilingual_frontier", "budget": BUDGET,
           "languages": [c for c, _ in LANGUAGES], "arms": ARMS, "rows": rows,
           "workers": [], "errors": [], "headline_eligible": False,
           "judge_note": ("single multilingual judge (Qwen3Guard); cross-language levels are not "
                          "comparable to the English three-judge majority"),
           "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                             for p in ["experiments/p29_multilingual.py"]}}

    def save():
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
        tmp.replace(path)

    save()
    translation_path = translate.remote(rows)
    translations = json.loads(Path(translation_path).read_text(encoding="utf8")) if \
        Path(translation_path).exists() else None
    if translations is None:  # the file lives on the Volume, so download it if absent locally
        translations = json.loads(volume.read_file("results/p29_translations.json").read().decode()
                                  if hasattr(volume.read_file("results/p29_translations.json"), "read")
                                  else b"".join(volume.read_file("results/p29_translations.json")))
    out["translations_path"] = translation_path
    out["translation_note"] = translations["note"]
    save()

    calls = []
    for code, _name in LANGUAGES:
        for arm in ARMS:
            calls.append((code, arm, language_arm.spawn(code, translations["by_language"][code],
                                                        rows, arm, reference)))
    out["function_calls"] = [c.object_id for c in calls]
    save()
    for code, arm, call in calls:
        worker = json.loads(call.get())
        out["workers"].append(worker)
        if worker["status"] != "passed":
            out["errors"].append({"language": code, "arm": arm, "errors": worker["errors"]})
        save()
        print(f"{code}/{arm}: {worker['status']} in {worker.get('wall_s', 0):.0f}s", flush=True)
    out["status"] = "passed" if not out["errors"] else "failed"
    save()
    print(f"{path}: {out['status']}", flush=True)

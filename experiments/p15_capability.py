"""p15: capability at n=500 per arm — is "keeping task performance" true or noise?

p3b measured gsm8k on 120 items per arm, where the base is 0.117 and the repair arms read 0.20-0.30. At that
n the standard error is ~3 points, and a repair that *raises* accuracy by 18 points is not a capability
cost — it is a sampling artefact and it makes the 2b claim ("keeping task performance") unfalsifiable.

This run measures the same four weight states on 500 gsm8k items, greedy, exact-match on the final number,
and stores every completion so the scoring can be re-derived rather than trusted.

    MSYS_NO_PATHCONV=1 uv run python -m modal run experiments/p15_capability.py

Writes /hf/results/p15_capability.json; log `logs/p15_capability.log`.
"""

from __future__ import annotations

import json
import re
import time

import modal

APP = "scartissue-p15-capability"
VOL = "scartissue-hf"
MODEL = "CohereLabs/tiny-aya-global"
LAYER, N_GSM, N_FIT, ALPHA = 15, 500, 16, 4.0
ARMS = [("base", None), ("drift", 0.0), ("harm2", 0.02), ("global", 1.0)]

app = modal.App(APP)
vol = modal.Volume.from_name(VOL, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.9.0", "transformers==5.22.0", "accelerate==1.11.0", "huggingface-hub==1.33.0",
    "datasets==4.4.2", "numpy", "sentencepiece")
secrets = [modal.Secret.from_name("huggingface")]


@app.function(image=image, volumes={"/hf": vol}, secrets=secrets, gpu="A100-40GB", timeout=7200)
def capability() -> dict:
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to("cuda").eval()

    def chat(p):
        return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True)

    @torch.no_grad()
    def act(texts):
        out = []
        for i in range(0, len(texts), 8):
            enc = tok([chat(t) for t in texts[i:i + 8]], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to("cuda")
            hs = model(**enc, output_hidden_states=True).hidden_states[LAYER].float()
            m = enc["attention_mask"].unsqueeze(-1)
            out.append(((hs * m).sum(1) / m.sum(1)).cpu().numpy())
        return np.concatenate(out)

    sr = load_dataset("walledai/StrongREJECT", split="train")
    xs = load_dataset("Paul/XSTest", split="train")
    mal_fit = [r["prompt"] for r in sr][120:120 + N_FIT]
    ben_fit = [r["prompt"] for r in xs if str(r.get("label", "")).lower() == "safe"][40:40 + N_FIT]
    v = act(mal_fit).mean(0) - act(ben_fit).mean(0)
    d = (v / np.linalg.norm(v)).astype(np.float64)
    del v
    torch.cuda.empty_cache()

    gsm = load_dataset("openai/gsm8k", "main", split="test").select(range(N_GSM))
    gold = [re.sub(r"[^0-9.\-]", "", r["answer"].split("####")[-1]) for r in gsm]
    qs = [r["question"] for r in gsm]

    def apply(sign: float, cols_frac) -> int:
        """Apply (or, with sign=-1, remove) the rank-one edit. cols_frac None => all columns."""
        spent = 0
        rng = np.random.default_rng(0)
        for name, p in model.named_parameters():
            if not name.endswith(("o_proj.weight", "down_proj.weight")):
                continue
            w = p.detach().float().cpu().numpy().astype(np.float64)
            if cols_frac is None:
                delta = sign * ALPHA * np.outer(d, d @ w)
                cols = w.shape[1]
            else:
                proj = np.abs(d @ w)
                k = max(1, int(round(cols_frac * w.shape[1])))
                cols_idx = np.argsort(-proj)[:k]
                delta = np.zeros_like(w)
                delta[:, cols_idx] = sign * ALPHA * np.outer(d, d @ w[:, cols_idx])
                cols = k
            p.data.add_(torch.from_numpy(delta).to(p.dtype).to(p.device))
            spent += delta.size
            del delta, w
        return spent

    def score() -> dict:
        hits, comps = 0, []
        for i in range(0, len(qs), 16):
            enc = tok([chat(q) for q in qs[i:i + 16]], return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=256, do_sample=False)
            texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for t, g in zip(texts, gold[i:i + 16]):
                nums = re.findall(r"-?[0-9][0-9,]*\.?[0-9]*", t.replace(",", ""))
                ok = bool(nums) and nums[-1].rstrip(".") == g
                hits += ok
                comps.append({"text": t, "gold": g, "correct": ok})
            print(f"  {min(i+16, len(qs))}/{len(qs)} acc={hits/max(1,i+len(texts)):.3f}", flush=True)
        return {"gsm8k_exact": hits / len(qs), "n": len(qs), "completions": comps}

    stages = {}
    stages["base"] = score()
    apply(-1.0, None)                      # the drift
    stages["drift"] = score()
    apply(+1.0, 0.02)                      # harm@2% repair on top of the drift
    stages["harm2"] = score()
    apply(-1.0, 0.02)                      # exactly undo it (the edit is linear and rank-one)
    apply(+1.0, None)                      # global repair on top of the drift
    stages["global"] = score()

    out = {"model": MODEL, "n_gsm8k": N_GSM, "alpha": ALPHA, "stages": stages,
           "wall_s": round(time.time() - t0, 1)}
    with open("/hf/results/p15_capability.json", "w") as fh:
        json.dump(out, fh)
    vol.commit()
    return {k: round(v["gsm8k_exact"], 4) for k, v in stages.items()} | {"wall_s": out["wall_s"]}


@app.local_entrypoint()
def main() -> None:
    print("[done]", capability.remote(), flush=True)

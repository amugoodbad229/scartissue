"""Build the refusal/repair patch family for ANY Cohere checkpoint, with the same operator.

The operator is unchanged from p1/p2/p7: fit `d` on the 16 harmful + 16 benign *fit*
prompts (disjoint from the 120/40 evaluation set), last token, layer 15 for the 36-layer
tiny-aya family (fraction-matched for other depths), drift `W <- W - d d^T W` on all 72
write matrices, re-extract `d'`, then repair `W[:, S] += alpha d' (d'^T W_base[:, S])`.

`d` and `d'` are stored next to the patches, and every patch records its `allowed` block
{modules, heads, layers} — the support IS the claim, so it ships with the parameters.
No drifted checkpoint is written: only factorised parameters (LSAL §4).

    uv run python -m modal run experiments/build_patches.py --model CohereLabs/aya-expanse-8b
"""

from __future__ import annotations

import hashlib
import re
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_sync(str(ROOT / "experiments/runtime"), frozen=True, uv_version="0.12.23")
         .env({"HF_HOME": "/hf/hf", "HF_HUB_CACHE": "/hf/hf/hub", "HF_XET_HIGH_PERFORMANCE": "1",
               "TOKENIZERS_PARALLELISM": "false"})
         .add_local_file(str(ROOT / "experiments/patching.py"), "/root/patching.py")
         .add_local_file(str(ROOT / "experiments/corpus_fit.py"), "/root/corpus_fit.py")
         .add_local_file(str(ROOT / "1-curate/corpus/corpus.jsonl"), "/root/1-curate/corpus/corpus.jsonl")
         .add_local_file(str(ROOT / "1-curate/corpus/corpus_split.json"),
                         "/root/1-curate/corpus/corpus_split.json"))
volume = modal.Volume.from_name("scartissue-hf", create_if_missing=True)
app = modal.App("scartissue-build-patches")


@app.function(image=image, gpu="A100-80GB", volumes={"/hf": volume},
              secrets=[modal.Secret.from_name("huggingface")], cpu=8, memory=49152, timeout=5400)
def build(model_id: str, revision: str, layer: int, alpha: float, fractions: list[float]) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os
    os.chdir("/root")  # corpus_fit resolves 1-curate/corpus relative to the repo root
    from corpus_fit import fit_prompts
    from patching import build_patch, head_cols, unit

    torch.manual_seed(0); np.random.seed(0)
    import os
    token = os.environ["HF_TOKEN"]
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, token=token, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, token=token,
                                                dtype=torch.bfloat16, device_map="cuda").eval()
    mal, ben = fit_prompts()

    def chat(p: str) -> str:
        return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True)

    @torch.no_grad()
    def acts(prompts):
        out = []
        for i in range(0, len(prompts), 8):
            batch = [chat(p) for p in prompts[i:i + 8]]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")
            hs = model(**enc, output_hidden_states=True).hidden_states[layer]
            idx = enc["attention_mask"].sum(1) - 1
            out.append(hs[torch.arange(hs.shape[0]), idx].float().cpu().numpy())
        return np.concatenate(out)

    def direction(a_mal, a_ben):
        return unit(a_mal.mean(0) - a_ben.mean(0))

    d = direction(acts(mal), acts(ben))
    weights = {n: p.detach().float().cpu().numpy() for n, p in model.named_parameters()
               if n.endswith(("o_proj.weight", "down_proj.weight"))}
    if len(weights) != 72:
        raise ValueError(f"expected 72 write matrices, got {len(weights)}")
    full = {n: np.arange(w.shape[1]) for n, w in weights.items()}
    drift, drift_edits = build_patch(weights, d, full, alpha=-1.0)
    params = dict(model.named_parameters())
    for name, (cols, block) in drift.items():
        p = params[name]; p.data[:, cols] += torch.from_numpy(block).to(p.device, p.dtype)
    d2 = direction(acts(mal), acts(ben))
    cos = float(np.dot(d, d2))

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    out_dir = Path("/hf/artifacts") / model_id.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "scartissue.patches/1", "model": model_id, "revision": revision,
                "layer": layer, "patches": {}, "cos_d_dprime": cos,
                "drift_scalars": int(drift_edits), "head_dim": head_dim,
                "operator": "W[:,S] += alpha * outer(d, d^T W_base[:,S])", "alpha_repair": alpha}

    def write(name: str, cols_by_module: dict, alpha_value: float, dvec):
        patch, edits = build_patch(weights, dvec, cols_by_module, alpha=alpha_value)
        payload, arrays = {}, {}
        for i, (module, (cols, block)) in enumerate(patch.items()):
            arrays[f"cols__{i}"] = cols; arrays[f"proj__{i}"] = (dvec @ weights[module][:, cols]).astype(np.float64)
        arrays["d"] = dvec.astype(np.float64); arrays["alpha"] = np.array(alpha_value)
        arrays["modules"] = np.array(list(patch))
        path = out_dir / f"{name}.npz"
        np.savez_compressed(path, **arrays)
        matrices = sorted(patch)
        heads = sorted({int(c) // head_dim for m in matrices if m.endswith("o_proj.weight")
                        for c in cols_by_module[m]})
        payload.update({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "edited_scalars": int(edits), "matrices": len(matrices),
                        "direction": "d" if alpha_value < 0 else "d_prime",
                        "allowed": {"modules": matrices,
                                    "layers": sorted({int(x) for m in matrices
                                              for x in re.findall(r"\.layers\.(\d+)\.", m)}),
                                    "heads": heads}})
        manifest["patches"][name] = payload

    write("abliteration", full, -1.0, d)
    write("repair_global_a4", full, alpha, d2)
    # Harm-weighted support: the columns whose write along d' is largest, budget-matched by
    # scalar count. One global ranking over concatenated columns, mapped back with searchsorted.
    names = sorted(weights)
    score = np.concatenate([np.abs(d2 @ weights[n]) for n in names])
    sizes = np.asarray([weights[n].shape[1] for n in names])
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    order = np.argsort(-score, kind="stable")
    total = int(sizes.sum())
    for frac in fractions:
        budget = int(frac * total)
        picked = order[:budget]
        module_of = np.searchsorted(starts, picked, side="right") - 1
        chosen: dict[str, np.ndarray] = {}
        for index, name in enumerate(names):
            local = np.sort(picked[module_of == index] - starts[index])
            if local.size:
                chosen[name] = local.astype(np.int64)
        stem = f"repair_harm{int(round(frac * 100))}_a4"
        write(stem, chosen, alpha, d2)
        manifest["patches"][stem]["support_fraction"] = frac
        manifest["patches"][stem]["frac_of_global"] = round(budget / total, 6)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf8")
    volume.commit()
    return {"model": model_id, "dir": str(out_dir), "patches": list(manifest["patches"]),
            "cos": cos, "manifest_sha256": hashlib.sha256((out_dir / "manifest.json").read_bytes()).hexdigest()}


@app.function(image=image, secrets=[modal.Secret.from_name("huggingface")], timeout=900)
def model_shape(model: str, revision: str) -> dict:
    """Read the layer count inside the image — the local venv has no transformers."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model, revision=revision or None)
    return {"num_hidden_layers": int(cfg.num_hidden_layers),
            "hidden_size": int(getattr(cfg, "hidden_size", 0) or 0)}


@app.local_entrypoint()
def main(model: str = "CohereLabs/aya-expanse-8b", revision: str = "",
          layer: int = -1, alpha: float = 4.0):
    import os
    os.chdir(ROOT)  # corpus_fit reads 1-curate/corpus relative to the repo root
    shape = model_shape.remote(model, revision or "main")
    n_layers = shape["num_hidden_layers"]
    chosen = layer if layer >= 0 else max(1, round(n_layers * 15 / 36))
    out = build.remote(model, revision or "main", chosen, alpha, [0.02, 0.10])
    print(json.dumps(out, indent=1))

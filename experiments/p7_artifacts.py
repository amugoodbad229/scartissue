"""Export the drift and the repair patches as published PARAMETERS (.npz), not checkpoints.

This is a submission gate: DELIVERABLES.md requires `scartissue-abliteration-patches` and
`scartissue-repair-patches` on the HF Hub, and LSAL §4 forbids publishing a drifted checkpoint. It is also
the honest artifact — the claim is about *which parameters* were allowed to move, so the allowed set is the
product, and a rank-one patch factors into a unit vector plus one projection per matrix.

No evaluation happens here: refusal/over-refusal numbers live in `results/*.json` and are referenced from
each manifest instead of being recomputed (a re-measurement here would be a second, slightly different
measurement of the same arm — the exact confusion this project exists to avoid).

Outputs, one per arm, under /hf/artifacts/:
    abliteration_tiny-aya-global_layer15.npz   alpha=-1, all 470,016 columns
    repair_global_a4.npz                        alpha=4, all columns (the reference the sweep must beat)
    repair_harm10_a4.npz                        alpha=4, the harm-weighted 10 % support
    …/manifest.json                             allowed set + provenance + the evidence files

Run: `uv run python -m modal run experiments/p7_artifacts.py`
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import modal

APP = "scartissue-p7-artifacts"
VOL = "scartissue-hf"
MODEL = "CohereLabs/tiny-aya-global"
LAYER = 15
N_MAL, N_SAFE, N_FIT = 120, 40, 16

# (file stem, alpha, support fraction | None for "all columns"). The support fraction is the CLAIM.
ARMS = [
    ("abliteration_tiny-aya-global_layer15", -1.0, None),
    ("repair_global_a4", 4.0, None),
    ("repair_harm10_a4", 4.0, 0.10),
    ("repair_harm4_a4", 4.0, 0.04),
    ("repair_harm2_a4", 4.0, 0.02),
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.0", "transformers==5.22.0", "accelerate==1.11.0",
        "huggingface-hub==1.33.0", "datasets==4.4.2",
        "numpy", "sentencepiece", "tiktoken",
    )
    .env({"HF_HOME": "/hf/hf", "HF_HUB_CACHE": "/hf/hf/hub", "HF_XET_HIGH_PERFORMANCE": "1",
          "TRANSFORMERS_VERBOSITY": "error", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file("experiments/patching.py", remote_path="/root/patching.py")
)

hf_volume = modal.Volume.from_name(VOL, create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]
app = modal.App(APP)


@app.function(image=image, gpu="A100-40GB", volumes={"/hf": hf_volume}, secrets=secrets,
              timeout=3600, cpu=8, memory=32768)
def export() -> dict:
    import numpy as np
    import torch
    from datasets import load_dataset
    from patching import build_patch, unit
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    token = os.environ["HF_TOKEN"]

    sr = load_dataset("walledai/StrongREJECT", split="train")
    xs = load_dataset("Paul/XSTest", split="train")
    # Same fit/eval split as p1/p2/p3, so `d` here is bit-for-bit the direction the arms were built from.
    mal_all = [r["prompt"] for r in sr][: N_MAL + N_FIT]
    mal_fit = mal_all[N_MAL :]
    safe_all = [r["prompt"] for r in xs
                if str(r.get("label", "")).lower() in {"safe", "0", "1", "true"}][: N_SAFE + N_FIT]
    benign_fit = safe_all[N_SAFE:]

    tok = AutoTokenizer.from_pretrained(MODEL, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", token=token
    ).eval()

    def chat(p: str) -> str:
        return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True)

    @torch.no_grad()
    def acts(prompts):
        vecs = []
        for p in prompts:
            enc = tok(chat(p), return_tensors="pt").to("cuda")
            vecs.append(model(**enc, output_hidden_states=True).hidden_states[LAYER][0, -1, :]
                        .float().cpu().numpy())
        return np.stack(vecs)

    def direction(a_mal, a_ben):
        v = a_mal.mean(0) - a_ben.mean(0)
        return v / np.linalg.norm(v)

    weights = {n: p.detach().float().cpu().numpy() for n, p in model.named_parameters()
               if n.endswith(("o_proj.weight", "down_proj.weight"))}
    budget_all = int(sum(w.shape[0] * w.shape[1] for w in weights.values()))
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    d = direction(acts(mal_fit), acts(benign_fit))
    full_plan = {n: np.arange(w.shape[1]) for n, w in weights.items()}
    drift_patch, drift_edits = build_patch(weights, d, full_plan, alpha=-1.0)
    for name, (cols, block) in drift_patch.items():          # apply in memory; nothing is saved as a model
        p = dict(model.named_parameters())[name]
        p.data[:, cols] += torch.from_numpy(block).to(p.device, p.dtype)
    d2 = direction(acts(mal_fit), acts(benign_fit))
    print(f"[drift] {drift_edits} scalars, cos(d,d')={float(np.dot(d, d2)):+.3e}", flush=True)

    def harm_plan(frac: float):
        """Top columns by harm-weighted write amplitude; a column costs exactly w.shape[0] scalars.

        The magnitudes are measured once, with every hook registered at the same time (72 matrices x 16
        prompts as separate passes would be 1,152 forwards for no reason).
        """
        budget = int(budget_all * frac)
        acc = {n: 0.0 for n in weights}
        by_name = dict(model.named_modules())
        handles = []

        def make_hook(nm):
            def hook(_mod, args):
                acc[nm] = acc[nm] + args[0].detach().float().abs().mean(dim=(0, 1)).cpu().numpy()
            return hook

        with torch.no_grad():
            for nm in weights:
                handles.append(by_name[nm.rsplit(".", 1)[0]].register_forward_pre_hook(make_hook(nm)))
            for p in mal_fit:
                model(**tok(chat(p), return_tensors="pt").to("cuda"))
            for h in handles:
                h.remove()
        harm_mag = {n: acc[n] / len(mal_fit) for n in weights}

        scored = []
        for n, w in weights.items():
            rows = int(w.shape[0])
            amp_n = np.abs(d2 @ w) * harm_mag[n]        # computed once per matrix, not once per column
            for j in np.argsort(-amp_n):
                scored.append((float(amp_n[j]), n, int(j), rows))
        scored.sort(key=lambda t: (-t[0], t[1], t[2]))
        picked, spent = {}, 0
        for _a, n, j, rows in scored:
            if spent + rows > budget:
                continue
            picked.setdefault(n, []).append(j)
            spent += rows
        return {n: np.sort(np.array(v, dtype=np.int64)) for n, v in picked.items()}, spent

    out_dir = "/hf/artifacts"
    os.makedirs(out_dir, exist_ok=True)
    manifest: dict = {"model": MODEL, "layer": LAYER, "d_model": int(d.shape[0]),
                      "budget_all_scalars": budget_all, "drift_scalars": drift_edits,
                      "cos_d_dprime": float(np.dot(d, d2)), "patch_format": "scartissue.rank1-patch/1",
                      "arms": {}}
    params = dict(model.named_parameters())

    for stem, alpha, frac in ARMS:
        direction_used = d if alpha < 0 else d2        # repair acts along d' — d is orthogonal after the drift
        plan = full_plan if frac is None else harm_plan(frac)[0]
        patch, edited = build_patch(weights, direction_used, plan, alpha=alpha)

        modules = sorted(plan)
        arrays: dict[str, np.ndarray] = {
            "d": unit(direction_used), "alpha": np.float64(alpha),
            "modules": np.array(modules, dtype=np.str_),
        }
        nodes = []
        for i, m in enumerate(modules):
            cols = np.asarray(plan[m], dtype=np.int64)
            w = weights[m]
            # store the projection ALREADY SLICED: (d@W)[cols] != d@W[:,cols] in the last bit, and these
            # files exist to reproduce the producer's arithmetic exactly
            arrays[f"proj__{i}"] = unit(direction_used) @ w.astype(np.float64)[:, cols]
            arrays[f"cols__{i}"] = cols
            node = {"module": m, "columns": int(len(cols)), "all_columns": bool(len(cols) == w.shape[1])}
            if m.endswith("o_proj.weight"):
                heads = sorted({int(c) // head_dim for c in cols})
                whole = all(len(np.intersect1d(cols, np.arange(h * head_dim, (h + 1) * head_dim))) == head_dim
                            for h in heads)
                node["heads"] = heads
                node["whole_heads"] = bool(whole)   # only then is "head h" an honest name for the support
            nodes.append(node)

        path = os.path.join(out_dir, f"{stem}.npz")
        t_write = time.time()
        np.savez(path, **arrays)
        manifest["arms"][stem] = {
            "file": f"{stem}.npz", "alpha": alpha, "support_fraction": frac,
            "direction_sha256": hashlib.sha256(np.ascontiguousarray(arrays["d"])).hexdigest(),
            "edited_scalars": edited, "frac_of_global": round(edited / budget_all, 6),
            "matrices": len(patch), "bytes": os.path.getsize(path),
            "write_s": round(time.time() - t_write, 1),
            "allowed": {                      # the allowed set IS the claim
                "modules": modules,
                "layers": sorted({int(m.split("layers.")[1].split(".")[0]) for m in modules}),
                "nodes": nodes,
            },
            "evidence": {                     # numbers come from files; this artifact only points at them
                "results_files": ["results/p2_dose.json", "results/p3_support.json",
                                  "results/p3b_sweep.json", "results/p4_replicate_*.json"],
                "measurement_contract": "_research/MEASUREMENT.md",
                "protocol": "_research/PROTOCOL.md",
            },
        }
        print(f"[arm] {stem}: {edited} scalars ({edited / budget_all:.4%}), "
              f"{len(patch)} matrices, {os.path.getsize(path) / 1e6:.2f} MB", flush=True)

    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    hf_volume.commit()
    print(f"[done] {out_dir} wall={time.time() - t0:.1f}s", flush=True)
    return {"arms": list(manifest["arms"]), "wall_s": round(time.time() - t0, 1),
            "sizes": {k: v["bytes"] for k, v in manifest["arms"].items()}}


@app.local_entrypoint()
def main():
    print("[done]", json.dumps(export.remote()), flush=True)

"""Circuit leg: CircuitKIT discovery on the DRIFTED tiny-aya-global, with SIGNED scores.

Why this file exists: the thesis says "only the read-out is repairable, and only through the circuit that
survives the edit". The repair so far uses an attribution-weighted column set, not a circuit. This runs the
library's EAP-IG discovery on the *abliterated* model so the two can be compared, and it keeps the sign that
CircuitKIT's own adapter throws away.

Two measured facts drive the design:
  * The parity gate covered `tiny-aya-base` (10/10) and Aya Expanse 8B (11/11) but never `global`, so this is
    also the port's first validation on the checkpoint the whole project uses.
  * `backends/eap/circuit_kit_adapter.py:77` takes `abs(score)` when converting EAP-IG node scores, which
    destroys the exciter/suppressor distinction. The signed value is still live in `graph.nodes_scores` before
    that conversion (`graph.py:80-95` exposes `Node.score`), so this reads it there and stores BOTH.

Discovery runs on the drifted weights because that is the claim: a circuit that survives the edit. The
drift is applied with this repo's own operator (no SafeTune code in the path), exactly as p2/p3 did.

    PYTHONPATH=<ScarTissue/repos/CircuitKIT> uv run python -m modal run experiments/p6_circuit.py

Writes /hf/results/p6_circuit.json; log `logs/p6_circuit.log`.
"""

from __future__ import annotations

import json
import os
import time

import modal

APP = "scartissue-p6-circuit"
VOL = "scartissue-hf"
MODEL = "CohereLabs/tiny-aya-global"
LAYER = 15
N_MAL, N_SAFE, N_GSM, N_FIT = 120, 40, 120, 16

app = modal.App(APP)
vol = modal.Volume.from_name(VOL, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.9.0", "transformers==5.22.0", "accelerate==1.11.0",
                 "huggingface-hub==1.33.0", "datasets==4.4.2", "transformer-lens==3.8.0",
                 "numpy", "sentencepiece", "tiktoken", "einops", "pandas", "tqdm",
                 # CircuitKIT hard-imports these at package import time (`backends/acdc/types.py:6` imports
                 # plotly, measured), so a minimal image fails before any discovery code runs.
                 "plotly", "captum", "lime", "ipython", "psutil", "scipy", "networkx",
                 "safetensors", "rich", "ordered-set", "fancy-einsum", "matplotlib")
    # .env() is a build step, so it must precede add_local_dir — Modal refuses to run a build step after
    # local files are attached (measured: InvalidError "tried to run a build step after using add_local_*").
    .env({"PYTHONPATH": "/root/CircuitKIT/src:/root/CircuitKIT"})
    # Read-only source of truth: the library is added by path, never installed (its dynamic version needs a
    # build backend) and never vendored (licence).
    .add_local_dir("C:/Users/Ayman/OneDrive/Desktop/Projects/SAAS_Projects/ScarTissue/repos/CircuitKIT",
                   remote_path="/root/CircuitKIT")
)


@app.function(image=image, volumes={"/hf": vol}, secrets=[modal.Secret.from_name("huggingface")],
              gpu="A100-80GB", timeout=5400)
def circuit() -> dict:
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import circuitkit  # noqa: F401  — installs the cohere2 port before any TL load
    from circuitkit.api import discover_circuit
    from circuitkit.quick import build_discovery_config
    from transformer_lens import HookedTransformer

    t0 = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(device).eval()

    sr = load_dataset("walledai/StrongREJECT", split="train")
    xs = load_dataset("Paul/XSTest", split="train")
    mal_all = [r["prompt"] for r in sr][: N_MAL + N_FIT]
    mal_fit = mal_all[N_MAL: N_MAL + N_FIT]
    safe_all = [r["prompt"] for r in xs
                if str(r.get("label", "")).lower() in {"safe", "0", "1", "true"}][: N_SAFE + N_FIT]
    ben_fit = safe_all[N_SAFE: N_SAFE + N_FIT]
    mal_eval = mal_all[:N_MAL]

    def chat(p: str) -> str:
        msgs = [{"role": "user", "content": p}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def acts(prompts, layer: int = LAYER):
        out = []
        for i in range(0, len(prompts), 8):
            batch = [chat(p) for p in prompts[i:i + 8]]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            hs = model(**enc, output_hidden_states=True).hidden_states[layer]
            mask = enc["attention_mask"].unsqueeze(-1)
            out.append(((hs * mask).sum(1) / mask.sum(1)).float().cpu().numpy())
        return np.concatenate(out)

    def direction(a_mal, a_ben):
        v = a_mal.mean(0) - a_ben.mean(0)
        return v / (np.linalg.norm(v) + 1e-9)

    d = direction(acts(mal_fit), acts(ben_fit))

    # ---- drift, with this repo's operator ------------------------------------------------
    weights = {n: p.detach().float().cpu().numpy() for n, p in model.named_parameters()
               if n.endswith(("o_proj.weight", "down_proj.weight"))}
    edited = 0
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n not in weights:
                continue
            w = weights[n]
            delta = -1.0 * np.outer(d.astype(np.float64), (d.astype(np.float64) @ w.astype(np.float64)))
            p.data.add_(torch.from_numpy(delta).to(p.dtype).to(p.device))
            edited += delta.size
            del delta
    print(f"[drift] {edited} scalars across {len(weights)} matrices", flush=True)
    del weights
    torch.cuda.empty_cache()

    d2 = direction(acts(mal_fit), acts(ben_fit))
    cos = float(np.dot(d, d2))
    print(f"[dir'] cos(d, d')={cos:+.6f}", flush=True)

    # ---- task for discovery ---------------------------------------------------------------
    # Refusal metric, expressed the way CircuitKIT wants it: a clean (harmful) prompt paired with a corrupted
    # (benign) one, and the two single-token answers whose logit difference IS the refusal read-out.
    import csv
    os.makedirs("/tmp/ck", exist_ok=True)
    csv_path = "/tmp/ck/refusal_pairs.csv"
    # CircuitKIT's token alignment REQUIRES the clean and corrupt prompts to have the same token length —
    # measured: pairing the fit sets positionally dropped all 16 records with
    # "RuntimeError: All 16 records were dropped during alignment (strategy='filter')".
    # So pairs are built by exact token-length matching over the whole pools, not positionally.
    mal_pool = mal_all[:N_MAL + N_FIT]
    ben_pool = [p for p in (r["prompt"] for r in xs)][:450]
    mal_len = [(p, len(tok(p).input_ids)) for p in mal_pool]
    ben_by_len: dict[int, list[str]] = {}
    for p in ben_pool:
        ben_by_len.setdefault(len(tok(p).input_ids), []).append(p)
    pairs = []
    for p, n in mal_len:
        if ben_by_len.get(n):
            pairs.append((p, ben_by_len[n].pop(0)))
        if len(pairs) >= 24:
            break
    if len(pairs) < 6:
        raise SystemExit(f"[fail] only {len(pairs)} length-matched pairs — too few for discovery")
    print(f"[task] {len(pairs)} length-matched clean/corrupt pairs", flush=True)
    n_pairs = len(pairs)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clean_prompt", "corrupt_prompt", "clean_answer", "corrupt_answer"])
        for clean, corrupt in pairs:
            w.writerow([clean, corrupt, " I cannot", " Sure"])
    print(f"[task] {n_pairs} clean/corrupt refusal pairs -> {csv_path}", flush=True)

    # ---- HookedTransformer over the DRIFTED in-memory model --------------------------------
    # Route A (`hf_model=`) takes the weights from the live object; the config is still read from the Hub
    # name. `ungroup_grouped_query_attention` is set by the library when the attribute exists.
    torch.cuda.empty_cache()
    tl_model = HookedTransformer.from_pretrained(MODEL, hf_model=model, device=device,
                                                dtype=torch.float32)
    tl_model.eval()
    print(f"[tl] loaded, d_model={tl_model.cfg.d_model} n_layers={tl_model.cfg.n_layers} "
          f"vram={torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
    del model
    torch.cuda.empty_cache()

    # ---- discovery ------------------------------------------------------------------------
    cfg = build_discovery_config(tl_model, task="custom:refusal_pairs", algorithm="eap-ig", level="node")
    cfg["data"] = {"type": "template", "path": csv_path, "align_strategy": "none",
                   "template": {"clean_prompt": "clean_prompt", "corrupt_prompt": "corrupt_prompt",
                                "clean_answer": "clean_answer", "corrupt_answer": "corrupt_answer"},
                   "max_records": n_pairs}
    print(f"[cfg] keys={sorted(cfg.keys())}", flush=True)

    result = discover_circuit(cfg, _model=tl_model)

    # Signed node scores: read them from the graph BEFORE the adapter's abs().
    signed: dict[str, float] = {}
    abs_scores: dict[str, float] = {}
    graph = getattr(result, "graph", None)
    if graph is not None and getattr(graph, "nodes_scores", None) is not None:
        for name, node in graph.nodes.items():
            try:
                signed[name] = float(node.score)
            except Exception:
                continue
            abs_scores[name] = abs(signed[name])
    ck_scores = dict(getattr(result, "scores", {}) or {})
    print(f"[discovery] graph nodes signed={len(signed)} circuitkit_scores={len(ck_scores)}", flush=True)

    ranked = sorted(signed.items(), key=lambda kv: -abs(kv[1]))
    total = sum(abs_scores.values()) or 1.0
    top = ranked[:40]
    layers = {}
    for name, score in ranked[:200]:
        layer = "".join(ch for ch in name.split(".")[0] if ch.isdigit())
        layers[layer] = layers.get(layer, 0) + 1

    out = {
        "model": MODEL, "layer": LAYER, "cos_d_dprime": cos, "drift_scalars": int(edited),
        "n_pairs": n_pairs, "algorithm": "eap-ig", "level": "node",
        "n_nodes_scored": len(signed), "n_circuitkit_scores": len(ck_scores),
        "mass_total_abs": float(total),
        "mass_top40_abs": float(sum(abs(s) for _n, s in top)),
        "mass_concentration_top40": float(sum(abs(s) for _n, s in top) / total),
        "signed_nodes": signed,
        "circuitkit_abs_scores": ck_scores,
        "top_signed": [{"node": n, "score": s, "sign": "exciter" if s > 0 else "suppressor"}
                       for n, s in top],
        "layer_histogram_top200": {k: v for k, v in sorted(layers.items(), key=lambda kv: -kv[1])[:36]},
        "wall_s": round(time.time() - t0, 1),
        "note": ("signed = graph.nodes_scores via Node.score, taken before "
                 "circuit_kit_adapter.py:77 applies abs(); circuitkit_abs_scores is what the library exposes."),
    }
    vol.commit()
    with open("/hf/results/p6_circuit.json", "w") as fh:
        json.dump(out, fh, indent=1)
    vol.commit()
    return out


@app.local_entrypoint()
def main() -> None:
    out = circuit.remote()
    print(f"[done] nodes={out['n_nodes_scored']} top40_mass={out['mass_concentration_top40']:.3f} "
          f"wall={out['wall_s']}s", flush=True)
    print("[top]", json.dumps(out["top_signed"][:8], indent=1), flush=True)

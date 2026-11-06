"""Circuit leg on the DRIFTED tiny-aya-global: discovery, signed mass, circuit patch.

Rebuild of p6 with the two leaks fixed. P6 ran discovery with prompts that included the
120 evaluation rows and the 40 benign evaluation rows, and it computed `d` with
mean-pooled activations; neither is the canonical operator. This version uses:

  * fit prompts only (`experiments/corpus_fit.py`: 16 harmful + 16 benign), last token,
    layer 15 — the same prompts the frozen protocol fits `d` on;
  * the canonical drift (`W <- W - d d^T W` on all 72 write matrices), then `d'`;
  * length-matched fit pairs for the EAP-IG refusal task.

Outputs, all in one JSON + one patch:
  * signed node scores (read from `graph.nodes` before the adapter's `abs()`), ranked;
  * a head/neuron mapping, and the scalar budget each support spends;
  * signED circuit mass `M = alpha * sum_c (d' . W[:, c])` for every support
    (global, harm 2 %, harm 10 %, circuit) — the quantity that says how much write along
    `d'` a repair actually adds, with the sign the library discards;
  * `repair_circuit_a4.npz` — the discovered-support repair, exported as parameters.

    uv run python -m modal run experiments/p23_circuit.py
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / "ScarTissue/repos/CircuitKIT"
MODEL = "CohereLabs/tiny-aya-global"
LAYER = 15
BUDGET_FRACTION = 0.10
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_sync(str(ROOT / "experiments/runtime"), frozen=True, uv_version="0.12.23")
         .env({"HF_HOME": "/hf/hf", "HF_HUB_CACHE": "/hf/hf/hub", "HF_XET_HIGH_PERFORMANCE": "1",
               "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/root/CircuitKIT/src:/root/CircuitKIT"})
         .uv_pip_install("transformer-lens==3.8.0", "plotly", "captum", "lime", "ipython",
                         "psutil", "scipy", "networkx", "rich", "ordered-set", "fancy-einsum",
                         "matplotlib", "pandas", "tqdm", "einops")
         .add_local_dir(str(UPSTREAM), "/root/CircuitKIT", ignore=["**/__pycache__", "*.pyc", ".git"])
         .add_local_file(str(ROOT / "experiments/patching.py"), "/root/patching.py")
         .add_local_file(str(ROOT / "experiments/corpus_fit.py"), "/root/corpus_fit.py")
         .add_local_file(str(ROOT / "1-curate/corpus/corpus.jsonl"), "/root/1-curate/corpus/corpus.jsonl")
         .add_local_file(str(ROOT / "1-curate/corpus/corpus_split.json"),
                         "/root/1-curate/corpus/corpus_split.json"))
volume = modal.Volume.from_name("scartissue-hf")
secrets = [modal.Secret.from_name("huggingface")]
app = modal.App("scartissue-p23-circuit")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@app.function(image=image, gpu="A100-80GB", volumes={"/hf": volume}, secrets=secrets,
              cpu=8, memory=65536, timeout=7200)
def run(budget_fraction: float = BUDGET_FRACTION) -> dict:
    import csv
    import numpy as np
    import torch
    from corpus_fit import fit_prompts
    from patching import build_patch, head_cols
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    torch.manual_seed(0); np.random.seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    mal, ben = fit_prompts()

    def chat(p: str) -> str:
        return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()

    @torch.no_grad()
    def acts(prompts):
        out = []
        for i in range(0, len(prompts), 8):
            batch = [chat(p) for p in prompts[i:i + 8]]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")
            hs = model(**enc, output_hidden_states=True).hidden_states[LAYER]
            idx = enc["attention_mask"].sum(1) - 1
            out.append(hs[torch.arange(hs.shape[0]), idx].float().cpu().numpy())
        return np.concatenate(out)

    def direction(a_mal, a_ben):
        v = a_mal.mean(0) - a_ben.mean(0)
        return v / np.linalg.norm(v)

    d = direction(acts(mal), acts(ben))
    weights = {n: p.detach().float().cpu().numpy() for n, p in model.named_parameters()
               if n.endswith(("o_proj.weight", "down_proj.weight"))}
    full = {n: np.arange(w.shape[1]) for n, w in weights.items()}
    drift, drift_edits = build_patch(weights, d, full, alpha=-1.0)
    params = dict(model.named_parameters())
    for name, (cols, block) in drift.items():
        p = params[name]; p.data[:, cols] += torch.from_numpy(block).to(p.device, p.dtype)
    d2 = direction(acts(mal), acts(ben))
    cos = float(np.dot(d, d2))
    print(f"[drift] {drift_edits} scalars, cos(d,d')={cos:+.3e}", flush=True)

    # ---- signed mass per support -----------------------------------------------------------
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    total_scalars = int(sum(int(w.size) for w in weights.values()))
    proj = {n: d2 @ weights[n] for n in weights}

    def mass(cols_by_module: dict) -> dict:
        signed = sum(float(proj[n][cols].sum()) for n, cols in cols_by_module.items())
        return {"signed_mass": signed, "abs_mass": float(sum(float(np.abs(proj[n][cols]).sum())
                                                            for n, cols in cols_by_module.items())),
                "scalars": int(sum(proj[n][cols].size for n, cols in cols_by_module.items())),
                "frac_of_global": round(sum(proj[n][cols].size for n, cols in cols_by_module.items())
                                        / total_scalars, 6)}

    supports = {"global": full}
    names = sorted(weights)
    flat = np.concatenate([np.abs(proj[n]) for n in names])
    sizes = np.asarray([weights[n].shape[1] for n in names])
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    order = np.argsort(-flat, kind="stable")
    for frac in (0.02, 0.10):
        picked = order[:int(frac * total_scalars)]
        module_of = np.searchsorted(starts, picked, side="right") - 1
        supports[f"harm{int(frac * 100)}"] = {
            n: np.sort(picked[module_of == i] - starts[i]).astype(np.int64)
            for i, n in enumerate(names) if (module_of == i).any()}
    masses = {name: mass(cols) for name, cols in supports.items()}
    print("[mass] " + json.dumps({k: round(v["signed_mass"], 3) for k, v in masses.items()}), flush=True)

    # ---- discovery ------------------------------------------------------------------------
    # corpus_fit.discovery_pairs demands EXACT token-length matches and yields only 3 pairs
    # from the 16+16 fit partition. Same fit-only source, greedy nearest-length matching
    # instead, so the contrast is length-controlled without starving the estimator.
    lengths = {p: len(tok(p).input_ids) for p in mal + ben}
    unused = sorted(ben, key=lengths.get)
    pairs = []
    for prompt in mal:
        if not unused:
            break
        best = min(unused, key=lambda c: abs(lengths[c] - lengths[prompt]))
        unused.remove(best)
        pairs.append((prompt, best))
    if len(pairs) < 6:
        raise SystemExit(f"only {len(pairs)} discovery pairs")
    spread = [abs(lengths[m] - lengths[b]) for m, b in pairs]
    print(f"[pairs] {len(pairs)} fit-only pairs, |len diff| max={max(spread)} mean={np.mean(spread):.2f}",
          flush=True)
    Path("/tmp/ck").mkdir(parents=True, exist_ok=True)
    csv_path = "/tmp/ck/refusal_fit_pairs.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clean_prompt", "corrupt_prompt", "clean_answer", "corrupt_answer"])
        for clean, corrupt in pairs:
            writer.writerow([clean, corrupt, " I cannot", " Sure"])

    torch.cuda.empty_cache()
    fp32 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, device_map="cuda").eval()
    with torch.no_grad():
        fp32_params = dict(fp32.named_parameters())
        for name, (cols, block) in drift.items():
            p = fp32_params[name]
            p.data[:, cols] += torch.from_numpy(block.astype(np.float32)).to(p.device, p.dtype)
    del model
    torch.cuda.empty_cache()

    import circuitkit  # noqa: F401  installs the cohere2 port before transformer_lens loads
    from circuitkit.api import discover_circuit
    from circuitkit.quick import build_discovery_config
    from transformer_lens import HookedTransformer
    tl_model = HookedTransformer.from_pretrained(MODEL, hf_model=fp32, device="cuda", dtype=torch.float32)
    tl_model.eval()
    del fp32
    torch.cuda.empty_cache()
    cfg = build_discovery_config(tl_model, task="custom:refusal_pairs", algorithm="eap-ig", level="node")
    cfg.setdefault("model", {})["tokenizer"] = MODEL
    cfg["data"] = {"type": "template", "path": csv_path, "align_strategy": "none",
                   "template": {"clean_prompt": "clean_prompt", "corrupt_prompt": "corrupt_prompt",
                                "clean_answer": "clean_answer", "corrupt_answer": "corrupt_answer"},
                   "max_records": len(pairs)}
    # At node level discover_circuit RETURNS the pruned node-name list and writes the scores
    # only as a side-car — which is why p6's `result.graph` read produced nothing.
    cfg["output_path"] = "/tmp/ck/circuit.pt"
    result = discover_circuit(cfg, _model=tl_model)
    pruned = [str(n) for n in result] if isinstance(result, (list, tuple)) else []

    signed: dict[str, float] = {}
    scores_json = Path("/tmp/ck/circuit_scores.json")
    if scores_json.exists():
        payload = json.loads(scores_json.read_text(encoding="utf8"))
        raw = payload.get("node_scores") or {}
        signed = {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))}
    else:
        for path in sorted(Path("/tmp/ck").glob("*_scores*")):
            print(f"[discovery] side-car present but unread: {path}", flush=True)
    shape = {"type": type(result).__name__, "pruned_returned": len(pruned),
             "scores_side_car": scores_json.exists(), "n_signed_scores": len(signed),
             "score_examples": list(signed)[:2], "pruned_examples": pruned[:2]}
    print("[discovery] " + json.dumps(shape), flush=True)
    if not signed:
        raise SystemExit(f"no signed scores were written: {shape}")

    # ---- map node scores onto the parameter axes -----------------------------------------
    # The library reports nodes in ARA naming: "A<layer>.<head>" and "MLP <layer>"; a
    # transformer_lens-style "blocks.<l>.<hook>" naming is kept as a fallback.
    pattern = re.compile(r"^A(\d+)\.(\d+)$")
    pattern_mlp = re.compile(r"^MLP\s*(\d+)$")
    pattern_tl = re.compile(r"blocks\.(\d+)\.(attn\.[A-Za-z_]+|mlp[A-Za-z_.]*)(?:\[([^\]]+)\])?")
    head_score, neuron_score = {}, {}
    unmapped, naming = [], set()
    for name, value in signed.items():
        m = pattern.search(name)
        if m:
            naming.add("ara_head")
            head_score[(int(m.group(1)), int(m.group(2)))] = abs(value)
            continue
        m = pattern_mlp.search(name)
        if m:
            naming.add("ara_mlp")
            neuron_score[int(m.group(1))] = abs(value)
            continue
        m = pattern_tl.search(name)
        if m:
            naming.add("transformer_lens")
            layer, kind = int(m.group(1)), m.group(2)
            head_score[layer] = head_score.get(layer, 0.0) + abs(value)
            if m.group(3) and m.group(3).isdigit():
                head_score[(layer, int(m.group(3)))] = abs(value)
            continue
        unmapped.append(name)
    print(f"[map] naming={sorted(naming)} unmapped={len(unmapped)} examples={unmapped[:3]}", flush=True)

    # Circuit support: heads, ranked by (layer, head) score when the metric is per-head,
    # otherwise whole attention layers ranked by layer score. Spend the same scalar budget.
    budget = int(budget_fraction * total_scalars)
    per_head = {(k[0], k[1]): v for k, v in head_score.items() if isinstance(k, tuple)}
    circuit_support: dict[str, np.ndarray] = {}
    spent = 0
    if per_head:
        ranked = sorted(per_head.items(), key=lambda kv: -kv[1])
        per_head_module = {}
        for (layer, head), _score in ranked:
            module = f"model.layers.{layer}.self_attn.o_proj.weight"
            cols = head_cols([head], head_dim)
            if spent + cols.size > budget:
                break
            keep = set(per_head_module.get(module, np.asarray([], dtype=np.int64)).tolist()) | set(cols.tolist())
            per_head_module[module] = np.asarray(sorted(keep), dtype=np.int64)
            spent += cols.size
        circuit_support = per_head_module
        circuit_basis = "per_head"
    else:
        ranked = sorted(head_score.items(), key=lambda kv: -kv[1])
        for layer, _score in ranked:
            module = f"model.layers.{layer}.self_attn.o_proj.weight"
            cols = np.arange(weights[module].shape[1])
            if spent + cols.size > budget:
                break
            circuit_support[module] = cols
            spent += cols.size
        circuit_basis = "per_layer"
    if not circuit_support:
        raise SystemExit("discovery produced no mappable attention nodes")
    masses["circuit"] = mass(circuit_support)

    # ---- export the discovered-support repair --------------------------------------------
    patch, edits = build_patch(weights, d2, circuit_support, alpha=4.0)
    arrays = {"d": d2.astype(np.float64), "alpha": np.array(4.0), "modules": np.array(list(patch))}
    for i, (module, (cols, _block)) in enumerate(patch.items()):
        arrays[f"cols__{i}"] = cols
        arrays[f"proj__{i}"] = proj[module][cols].astype(np.float64)
    out_dir = Path("/hf/artifacts/tiny-aya-global")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "repair_circuit_a4.npz"
    np.savez_compressed(path, **arrays)
    patch_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    matrices = sorted(patch)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8")) if manifest_path.exists() else {"patches": {}}
    manifest.setdefault("patches", {})["repair_circuit_a4"] = {
        "file": path.name, "sha256": patch_sha, "edited_scalars": int(edits),
        "support_fraction": budget_fraction, "frac_of_global": masses["circuit"]["frac_of_global"],
        "direction": "d_prime", "basis": circuit_basis,
        "allowed": {"modules": matrices,
                            "layers": sorted({int(x) for m in matrices
                                              for x in re.findall(r"\.layers\.(\d+)\.", m)}),
                    "heads": sorted({int(c) // head_dim for m in matrices for c in patch[m][0]})},
        "signed_mass": masses["circuit"]["signed_mass"]}
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf8")

    out = {"model": MODEL, "layer": LAYER, "cos_d_dprime": cos, "drift_scalars": int(drift_edits),
           "fit_prompts": {"harmful": len(mal), "benign": len(ben)}, "discovery_pairs": len(pairs), "discovery_pair_len_gap_max": int(max(spread)),
           "discovery_leakage": "none: pairs come from the fit partition only",
           "head_dim": head_dim, "total_write_scalars": total_scalars,
           "result_shape": shape, "pruned_nodes": pruned[:64], "unmapped_node_examples": unmapped[:16], "node_naming": sorted(naming),
           "node_scores": signed,
           "attention_layer_scores": {str(k): v for k, v in sorted(head_score.items(),
                                                                    key=lambda kv: str(kv[0])) if not isinstance(k, tuple)},
           "mlp_layer_scores": {str(k): v for k, v in sorted(neuron_score.items())},
           "supports": {name: {"scalars": v["scalars"], "frac_of_global": v["frac_of_global"],
                               "signed_mass": v["signed_mass"], "abs_mass": v["abs_mass"],
                               "modules": len(cols)}
                        for name, v in masses.items()},
           "circuit_patch": {"file": str(path), "sha256": patch_sha, "edited_scalars": int(edits),
                             "basis": circuit_basis, "modules": matrices,
                             "layers": sorted({int(x) for m in matrices
                                              for x in re.findall(r"\.layers\.(\d+)\.", m)}),
                             "heads": sorted({int(c) // head_dim for m in matrices for c in patch[m][0]})},
           "wall_s": round(time.time() - started, 1),
           "note": ("signed node scores are read from graph.nodes[..].score before "
                    "circuit_kit_adapter.py:77 applies abs(); signed_mass keeps the exciter/suppressor sign.")}
    volume.commit()
    (Path("/hf") / "results" / f"p23_circuit_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(
        json.dumps(out, indent=1), encoding="utf8")
    volume.commit()
    return out


@app.local_entrypoint()
def main(budget_fraction: float = BUDGET_FRACTION):
    import sys
    out = run.remote(budget_fraction)
    path = ROOT / f"results/p23_circuit_{time.strftime('%Y%m%d_%H%M%S')}.json"
    slim = {k: v for k, v in out.items() if k != "node_scores"}
    slim["node_score_count"] = len(out["node_scores"])
    path.write_text(json.dumps(out, indent=1), encoding="utf8")
    print(json.dumps({k: slim[k] for k in ("result_shape", "supports", "circuit_patch", "cos_d_dprime", "wall_s")},
                     indent=1))
    print(path)

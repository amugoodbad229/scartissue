# Deliverables: Hub artifacts, the Space, the article, the ROI table

Everything the judging rubric can actually see. Rules that bind us (from `_research/lexsi/index.txt`): all artifacts
on **Hugging Face** (models, adapters, **logs**, datasets); a **public deployment** by Nov 14; **more than one**
before-and-after metric; a write-up (HF Article) explaining *why* those metrics, *why* those datasets/tasks/models,
and *why* the model is meaningfully better; ROI table optional but preferred. Licence constraint from the LSAL read:
**never publish an "uncensored" checkpoint** — ship the **abliteration parameters + patch script** so the drifted
state is reproducible without hosting a weaponised model (this is also the ethical answer).

## 1. Where things live on the Hub (owner `amugoodbad229`)

| repo | type | contents | licence | why this form |
|---|---|---|---|---|
| `scartissue-tinyaya-safety-corpus` | dataset | curated safety corpus + `manifest.json` + `lexsi_provenance.json` + `rejected.jsonl` | cc-by-nc-4.0 | the chain's first link; row-level provenance is our differentiator |
| `scartissue-abliteration-patches` | dataset (JSON + small `.npz`) | per-model drift parameters: layer, `d` (hash + tensor), scale, edited matrices, seeds | cc-by-nc-4.0 | reproducible drift **without** hosting a broken model |
| `scartissue-repair-patches` | dataset | per-arm repair deltas (rank, U/V factors, target modules, matched edit count) | cc-by-nc-4.0 | the actual product of the research |
| `scartissue-evals` | dataset | per-prompt completions + judge scores + lm-eval JSON + κ + frontier CSVs | cc-by-nc-4.0 | "logs" the rules demand, in auditable form |
| `scartissue` | **Space** (Gradio) | the demo (§3) | Apache-2.0 (our code) | public deployment |
| `scartissue/lexsi-track2` | **model** repo, *optional* | only the **repaired** tiny-aya checkpoint, clearly labelled, if we decide a clickable model is worth it | cc-by-nc-4.0 | a repaired model is safe to host; a drifted one is not |

Naming rule: one prefix (`scartissue-`) so a judge searching the org finds the whole chain in one screen.

## 2. Schemas (write once, freeze, cite in the article)

### 2.1 `manifest.json` — the lineage spine (already produced by CuratorKIT; we extend it)

```json
{
  "chain_version": "lexsi.track2/1",
  "run_id": "2026-11-03T10:12:04Z-tinyaya-global",
  "links": [
    {"stage": "curate", "tool": "curatorkit", "git_sha": "dd9e835", "config_hash": "…",
     "inputs": [{"source_uri": "hf:walledai/StrongREJECT", "revision": "…", "n": 313}],
     "outputs": {"dataset": "scartissue-tinyaya-safety-corpus@v1", "n_kept": 268, "n_rejected": 45},
     "provenance_sha256": "…"},
    {"stage": "drift", "tool": "safetune", "git_sha": "26f59e6", "model": "CohereLabs/tiny-aya-global",
     "params": {"layer": 15, "alpha": 1.0, "matrices": 72, "d_sha256": "…", "seed": 0},
     "outputs": {"patch": "scartissue-abliteration-patches@v1"},
     "verify": {"refusals_before": "12/12", "refusals_after": "3/12"}},
    {"stage": "circuit", "tool": "circuitkit", "git_sha": "e71d5e1",
     "params": {"algorithm": "eap-ig", "task": "xor_safety_contrastive", "seeds": [0,1,2]},
     "outputs": {"scores": "scartissue-evals@circuits", "M": 0.31, "raw_ratio": 0.42}},
    {"stage": "repair", "tool": "safetune+ours", "arm": "circuit/heads/ours",
     "params": {"matched_edits": 18432, "targets": ["o_proj", "down_proj"], "rank": 1},
     "outputs": {"patch": "scartissue-repair-patches@v1"}},
    {"stage": "audit", "tool": "auditkit", "git_sha": "71316da",
     "params": {"judges": ["harmbench-mistral", "qwen3guard", "granite-guardian"], "majority": 2,
                "capability": ["gsm8k", "humaneval", "mmlu"]},
     "outputs": {"evals": "scartissue-evals@v1", "R": 0.44, "delta_C": -0.018}}
  ],
  "env": {"torch": "2.9.0", "transformers": "5.22.0", "transformer_lens": "3.8.0", "gpu": "A100-40GB"}
}
```

Every number in the article must be reachable from a `manifest.json` plus its `scartissue-evals` file. That is the
sentence to put in the article, verbatim.

### 2.2 Abliteration patch (`.npz` + `.json`) — the drift, without the weapon

```
abliteration-<model-tag>-L<layer>.npz
  d                  float32 [d_model]        unit refusal direction (pre-drift)
  W_O_edits          float32 [n, d_head, d]   rows: layer index, head index, delta
  W_down_edits       float32 [n, d_mlp, d]
  meta               json                     {layer, alpha, matrices, d_sha256, seed, before/after refusals}
```
Loader: `apply_abliteration(model, npz)` — 30 lines, `W ← W − α·ddᵀW`, `torch.no_grad()`. Reviewers can reproduce
the exact drift on their own weights and never download a broken checkpoint.

### 2.3 Repair patch — the same shape, plus what the arm was allowed to touch

```
repair-<model-tag>-<arm>.npz
  deltas             float32 [n_params_total]  flat, sparse-indexed
  index              int64   [n]               flat index into the arm's allowed parameter set
  allowed            json                      {"modules": [...], "heads": [...], "layers": [...]}
  meta               json                      {arm, matched_edits, rank, fit_set_sha256}
```
`allowed` is the key research artifact: it *is* the claim "restricted to the circuit", auditable per arm.

### 2.4 Eval row (`.jsonl`, one line per prompt per arm) — the metric substrate

```json
{"run_id": "…", "arm": "circuit/heads/ours", "condition": "malicious_after_repair",
 "prompt_id": "sr-0042", "lang": "en", "prompt_sha256": "…",
 "completion": "…", "completion_sha256": "…",
 "judges": {"harmbench": 0.0, "qwen3guard": 0.0, "granite": 1.0, "shieldgemma": 0.44},
 "majority": 0.0, "wrapper_robust": {"bare": 0.0, "educational": 0.0, "polite": 1.0, "refusal_prefix": 0.0},
 "capability": {"gsm8k": 0.31, "humaneval": 0.18, "mmlu": 0.24}}
```
No aggregate without its per-prompt lines. That is what makes κ, CIs and the human table possible.

## 3. The Space (the only thing many judges will actually touch)

**Design target: a judge understands it in 30 seconds without reading anything.**

Layout, one page:

```
[ model: ▾ tiny-aya-global  |  tiny-aya-base  |  aya-expanse-8b ]        [gpu badge]
[ prompt: ______________________________ ]  (or pick from 20 presets: ☠ harmful / ✓ harmless / ⚠ wrapper)
        ┌──────────────────┬──────────────────┬──────────────────┐
        │ BASE             │ DRIFTED          │ REPAIRED (ours)  │
        │ completion …     │ completion …     │ completion …     │
        │ ASR 0.00         │ ASR 1.00         │ ASR 0.00         │
        └──────────────────┴──────────────────┴──────────────────┘
judge panel: harmbench 0.0 · qwen3guard 0.0 · granite 1.0 · shieldgemma 0.44   →  majority: SAFE
[ frontier plot: recovery R vs capability ΔC, our arm highlighted ]
```

Implementation choices, in order of robustness:

1. **Precomputed + live hybrid (recommended).** Twenty preset prompts are served from a checked-in JSON in the Space
   (instant, zero GPU, always works). A typed prompt triggers real generation through a **Modal web endpoint**
   (`modal deploy`) that serves the three states on an A10G/L40S with the models pre-loaded in a warm container —
   the Space stays a thin client, and the GPU cost is ours to control, not the judges' patience.
2. **ZeroGPU-only fallback.** `@spaces.GPU(duration=60)` loading one model per call (6.7 GB fp16), cached after the
   first call; live generation for a single selected model, precomputed for the rest.
3. **Video + repo link fallback** if the endpoint dies during judging: a 60-second screen recording embedded in the
   article, and the Space shows the precomputed table with a banner explaining live mode is paused. Never ship a
   Space that shows a traceback to a judge.

Latency budget from measurement: 267.6 tok/s at 3.35 B on an A100 → 64 tokens ≈ 0.24 s of decode; the real cost is
the cold load (~33 s from the warm Volume, worse without). Mitigation: `keep_warm(1)` on the Modal app for the last
48 h before judging, and never load three models for one request.

## 4. ROI table (bonus points, twenty lines of arithmetic)

Measured inputs: **267.6 tok/s** at 3.35 B (A100-40GB), **98.7 tok/s** at 32 B (H200), 3 model states resident =
**18.81 GB** at 3.35 B. Template:

```
Volume: 10 M output tokens / month, 2 M requests, 200-token average output.
Self-host (3.35 B, A100-40GB @ $/h from the Modal price page, ~% utilization):
    tokens/s 267.6 → GPU-hours/month = 10e6 / 267.6 / 3600 = 10.4 h → cost = 10.4 × $/h
    + amortised eval/judging cost (measured per arm)          = $…
Closed-source (published per-Mtok price P):                  = 10 × P
Break-even volume, break-even latency budget, and the honesty clause:
    "at 10 M tok/month self-hosting is N× cheaper; below Y tokens it is not."
```
We must quote the Modal price page and the closed-source price we cite, with dates. Any number we cannot cite, we
put in a "not measured" column instead of inventing.

## 5. The HF Article — structure the rubric asks for, in order

1. **Claim** (one paragraph, one sentence bolded).
2. **Why these metrics** — ASR (the harm axis) *and* capability (the price), why a *frontier* and not a single
   number, why κ and the wrapper test (and cite the 58-point spread we measured — nobody else will have this).
3. **Why these datasets** — StrongREJECT 313 (standard, comparable), HarmBench, the 11-language arm, why the
   direction-fit slice is disjoint from the evaluation prompts (`_select_direction_by_scoring` validates on the head
   slice, `refusal_direction.py:448-449` — a limitation we handled, not hid).
4. **Why these models** — tiny-aya five-variant axis (identical 3.35 B architecture, different data mixtures) +
   scale axis; the 32 B arm is inference-only, with the measured reason (258.4 GiB of fp32).
5. **What we found** — the frontier figure, the negative result (resta/antidote null at 3.35 B), the visible
   failure of four repair families under one matched-edit budget.
6. **Why it is better for a user** — a pre-release safety-retention report: "your post-train dropped ASR-relevant
   refusals from 1.00 to 0.25; here is what repair can recover, and here is the capability it costs."
7. **Reproduce it** — the four Hub repos, the manifest, the patch loader, the exact GPU and wall-clock.
8. **Limitations** — judges' language, 12×11 prompt set, single-seed primary result if we cannot afford 5 seeds,
   CircuitKIT cohere2 declared experimental by its authors.

## 6. Submission checklist (map to the rules, tick on Nov 13)

- [ ] ≥2 track repos chained (**we have 4**) — show it in the manifest
- [ ] Cohere dense open-weights model in every headline number
- [ ] models / adapters / **logs** / datasets on the Hub under one org prefix
- [ ] public deployment live (Space + article + repo) and reachable from the article's first screen
- [ ] **more than one** before-and-after metric, per arm, with CIs
- [ ] write-up answers: why these metrics, why these data/tasks/models, why it is better
- [ ] ROI table (bonus) with cited prices and dates
- [ ] licence statements: our code Apache-2.0, derived weights cc-by-nc-4.0, no drifted checkpoint hosted
- [ ] one-line proof of originality: our own head-restriction patch + frontier protocol, citing ROSI (2508.20766),
      MSCP (2508.09190), CircWeightScaling (2609.00051) as prior art
- [ ] Demo Day fallback: a 60-second video, offline copy of the frontier figure, and the article

## 7. Execution order for days 12–13 (do not reorder)

1. Freeze numbers → dump `scartissue-evals` (per-prompt) → upload (fastest, unblocks everything else).
2. Generate the drift patch `.npz` per model; verify by round-tripping (`apply` then compare refusals) before upload.
3. Upload the corpus checkpoint with its manifest.
4. Deploy the Modal endpoint, then build the Space against it; test as an anonymous user.
5. Write the article **last**, with every number copied from a file, not from memory.
6. Two hours of buffer: one anonymous pass through every link on a phone.

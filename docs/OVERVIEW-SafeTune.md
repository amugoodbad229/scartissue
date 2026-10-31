# SafeTune — repo overview

**Read this first.** One file, ten minutes, before you touch `repos/SafeTune`.
Synthesised from `_docs/SafeTune/01…10` and `_research/repo-maps/SafeTune.md`. Where those
docs are silent, this one says so rather than guessing.

Snapshot: commit `26f59e6` (2026-10-21), version 0.1.8, 331 `.py` / 74,246 LOC under
`src/safetune`. Everything below is a **static** reading — nothing in this repo has been executed
to verify a number.

---

## 1. What this repo is for

SafeTune is a library for **measuring and repairing safety drift** in fine-tuned LLMs: you take a
base model and an aligned model, or a model that has drifted after benign fine-tuning, and you
either patch it back (weight-space repair, retraining, unlearning, inference-time steering) or you
quantify how far it moved (ASR, over-refusal, capability). It ships four repair paradigms plus
interpretability and evaluation instrumentation as one pip-installable package with a `safetune`
CLI, 26 harden + 24 recover + 6 unlearn CLI aliases and ~100 method classes
(`src/safetune/runner/_registry.py:17-96`). Its dual use is deliberate and explicit: the same
refusal-direction primitive is exported as a defence *and* as an attack
(`src/safetune/interventions/steer/refusal_direction.py:539` vs
`src/safetune/instrumentation/evaluate/redteam/abliteration.py:35-41`).

**The paper.** `repos/SafeTune/CITATION.cff` declares `preferred-citation.type: conference-paper`,
title *"SafeTune: A Unified, Faithful Library for Auditing and Repairing Safety Drift in Fine-Tuned
LLMs"*, authors Pratinav Seth / Saisab Sadhu / Anshul Kaushal / Vinay Kumar Sankarapu (all
`@lexsi.ai`), version 0.1.8, released 2026-10-01. The **EMNLP 2026 system demo** venue and the
OpenReview id `YQOe2nu8en` are **not** recorded anywhere in the repo or in the package docs
(grep for `EMNLP`, `OpenReview`, `YQOe2` over `README.md`, `CITATION.cff`, `pyproject.toml`,
`_docs/SafeTune/*.md` returns nothing) — treat them as external attribution, not something you can
verify from the tree.

Also note the abstract's own count: *"four repair paradigms (weight patching, safety-constrained
retraining, gradient-based unlearning, inference-time steering) … plus interpretability and
evaluation instrumentation"*. Four, not six. The six **namespaces** are a code-level grouping, not
the paper's framing (see §2).

**Licence reality — read before you install.** `repos/SafeTune/LICENSE.md` is the **Lexsi Labs
Source Available License (LSAL) v1.2**, owner Lexsi Labs / Lithasa Technologies Pvt. Ltd. It is
*not* OSI open source:

- §1 grants **Noncommercial Purposes only** (academic research and teaching get MIT-like terms;
  academics and their groups at a university or public research body are covered).
- **§1A "Use by Organizations"** — before *any* organisation (including charitable, educational,
  public research, government) uses the Licensed Work for internal evaluation, red-teaming,
  benchmarking, safety auditing, or any use connected with developing, evaluating, repairing or
  safeguarding its own models, products or services — **whether or not a fee is charged** — it must
  obtain prior written permission **or** send a written acknowledgement of use to
  `support@lexsi.ai` (organisation, intended use, undertaking to attribute). Terms may require
  sharing evaluation results in the harness's standard report format (no weights, training data,
  prompt text or generations).
- §2 bars any "Sell" without a separate commercial licence.
- §4 bars deploying unrepaired drifted checkpoints in production under *any* licence.

The licence line in `repos/SafeTune/LICENSE.md:99` is the summary to quote: *"Academic research and
teaching are free on MIT-like terms. Use by any organization requires written acknowledgement or
permission (Section 1A)."* The package docs are **silent** on all of this — the only trace of the
licence in `_docs/SafeTune/` is the banner string `S A F E T U N E v… · LSAL v1.2`, which
`_print_banner` suppresses unless stdout is a TTY (`src/safetune/cli.py:19`), so it never reaches a
CI log.

---

## 2. The mental model

### The five pillars (plus one)

Six namespaces are re-exported at `src/safetune/__init__.py:79` (`_ALIASES`) — `recover, unlearn,
harden, steer, interpret, evaluate`. Of these, **five are things a user calls**, and one
(`interpret`) is instrumentation that feeds the others:

| Pillar | What it is | Namespace | Contract |
|---|---|---|---|
| **harden** | prevention — 26 trainers that make a model resist drift during fine-tuning | `src/safetune/runner/harden/__init__.py:1-90` | `trainer.train(dataset, out_dir) -> str` |
| **recover** | weight-space repair — 24 post-hoc "safety patch" trainers | `src/safetune/runner/recover/__init__.py:1-95` | `trainer.apply() -> patched_model` (never writes) |
| **unlearn** | removal — RMU / NPO variants / CRISP / SCRUB | `src/safetune/runner/unlearn/` | `trainer.unlearn(forget=, retain=)` |
| **steer** | inference-time control — refusal-direction extract/ablate/steer, CAA/CAST/AdaSteer/SafeSwitch | `src/safetune/interventions/steer/` | `safetune.steer.run(...) -> List[str]` |
| **evaluate** | measurement — judges, benchmarks, ASR, Pareto, red-team attacks | `src/safetune/instrumentation/evaluate/` | `evaluate(...) -> Dict[str, Dict]` |
| *interpret* | circuit/neuron discovery that `recover` and `steer` consume | `src/safetune/instrumentation/interpret/` | `CircuitInfo` JSON out, `apply_ctheta` in |

**The entry point a user actually calls** is `safetune.runner` — its module docstring is the
canonical recipe (`src/safetune/runner/__init__.py:1-21`), and the CLI resolves aliases by
`getattr` on `safetune.runner.<pillar>` (`src/safetune/cli.py:178-213`). Note the asymmetry: the
CLI's `patch` command goes through **runner `Trainer` classes**, *not* through
`interventions.recover.apply_*` (`src/safetune/cli.py:178`). If you call the `apply_*` functions
directly you get the alias-rewriting and mutation-tripwire decorators
(`src/safetune/interventions/recover/_contract.py:26`,
`src/safetune/interventions/recover/_invariant.py:60`); the runner classes wrap them separately.

### How a run flows

```
  YAML / flags
      │
      ▼
 cli.py:314 parse_args()          # pre-scans --config via parse_known_args, installs
      │                          # SafeTuneConfig.from_yaml fields as parser defaults
      ▼
 cli.py:338 _apply_runtime()      # configure(**runtime) → datasets → eval_backend
      │                          # config.py:270  (mutates process-global _RUNTIME)
      ▼
 runner/<pillar>/<X>Trainer(model, tok, **method_kwargs)
      │        │
      │        ├── harden :  .train(dataset, out)        _base.py:191 ctor, :199-203 raw-data wrap
      │        └── recover:  .apply()  →  .save_checkpoint()   recover/_base.py:48, :87
      ▼                                                    (method="harden.x" / "recover.XTrainer")
 save_checkpoint  →  <results_dir>/checkpoints/<name>
      │                 + lexsi_provenance.json      provenance.py:64
      ▼
 .eval(name, path)  ──►  evaluate(model, benchmarks, judge)   suite/evaluate.py:206
      │                       │
      │                       └── judge model: wildguard | harmbench | llama_guard_3
      │                                       suite/evaluate.py:63-70  (lru_cache, 3 max)
      ▼
 {bench: {asr, refusal_rate, harmfulness_score, n_total, n_evaluated, headline_metric}}
      │
      ▼
 ResultBook / JSONL  ──►  pareto.py:86 ParetoVisualizer  ──►  scripts/plot_matrix_pareto.py
```

The two halves are separable on purpose: repair never imports the judge, and evaluation can run
against any checkpoint path. `_do_eval` validates benchmark names *before* loading the model
(`src/safetune/cli.py:153-177`) so a typo costs nothing.

---

## 3. Public surface that matters

The ~11 things a downstream project actually calls. Signatures are verbatim from the package docs.

| # | Entry point | Signature | Returns | Anchor |
|---|---|---|---|---|
| 1 | harden trainer | `<X>Trainer(model=None, tokenizer=None, *, model_id, epochs=1, batch_size=4, lr=1e-4, bf16=None, fp16=None, wandb=False, optimizer="adamw_torch", logging_steps=10, results_dir=None, drift_task=None, **kwargs)` then `.train(train_dataset, out_dir)` | checkpoint dir `str` | `src/safetune/runner/harden/_base.py:191`; `train` wrapped at `:199-203`; out dir at `:403-413` |
| 2 | recover trainer | `<X>Trainer(model=None, *, model_id=None, results_dir=None, drift_task=None, **kwargs)` then `.apply(**kwargs)` | patched model (in memory; **no disk writes**) | `src/safetune/runner/recover/_base.py:17-40`, `:48` |
| 3 | recover save | `.save_checkpoint(patched_model, tokenizer=None, name="recover") -> str` | checkpoint dir | `src/safetune/runner/recover/_base.py:87` |
| 4 | recover, function form | `apply_ctheta(target, positive, negative, circuit_info, strength=1.0, layer_subset=None, target_modules=None)` | mutated target state dict | `src/safetune/interventions/recover/ctheta.py:47` |
| 5 | unlearn trainer | `<X>Trainer(model, layer_id=…)` then `.unlearn(forget=…, retain=…)` | model | recipe at `src/safetune/runner/__init__.py:1-21` |
| 6 | steer, end-to-end | `run(steered_model, prompts, *, backend="hf", tokenizer=None, model_id=None, decode_spec=None, max_new_tokens=256, temperature=0.0, apply_chat_template=True, **backend_kwargs)` | `List[str]` | `src/safetune/interventions/steer/backends/run.py:141` |
| 7 | extract direction | `extract_refusal_direction(model, tokenizer, harmful_prompts, harmless_prompts, config=None)` | `(direction: Tensor, layer_idx: int, all_layer_directions: Dict[int, Tensor])` | `src/safetune/interventions/steer/refusal_direction.py:171` (returns at `:274`) |
| 8 | apply direction at runtime | `RefusalDirectionModel(model, direction, mode="ablate", strength=1.0, layers=None)` — `.install()` `:588`, `.remove()` `:607`, `with` `:615/:618` | hooked model (reversible) | `refusal_direction.py:539` |
| 9 | weight-space abliteration | `orthogonalize_weights(model, direction, include_attn=True, include_mlp=True)` / `restore_weights(model, snapshots)` | `Dict[str, Tensor]` snapshot / `None` | `refusal_direction.py:627` / `:694` |
| 10 | evaluate | `evaluate(model, benchmarks=None, judge="wildguard", tokenizer=None, batch_size=None, max_new_tokens=None, generation_kwargs=None, max_prompts=None, drift_task=None, strict=None)` | `Dict[str, Dict[str, Any]]` | `src/safetune/instrumentation/evaluate/suite/evaluate.py:206` |
| 11 | evaluate on vLLM | `evaluate_with_vllm_backend(backend, *, benchmarks=None, judge="wildguard", …)`; build with `build_vllm_eval_backend(model_id, wrapped_model, **vllm_kwargs)` | same dict shape | `suite/evaluate.py:680`; `suite/vllm_eval.py:241` |

Two more that a project integrating at a lower level will need:

| Entry point | Signature | Returns | Anchor |
|---|---|---|---|
| circuit load | `load_circuit_info_from_file(path, top_fraction=0.2)` | `CircuitInfo` | `src/safetune/core/circuit_kit/adapter.py:23` |
| data compiler | `compile_utility_to_safety(utility_samples, config)` | `CompiledSafetyData` | `src/safetune/core/data_compiler/compiler.py:58` |
| reward lookup | `RewardRegistry.get_reward_function(name, config=None)` | `RewardFunction` | `src/safetune/rewards/registry.py:19-155` |

**CLI mirror**: `safetune train|eval|patch|unlearn|list` (`src/safetune/cli.py:255-313`), plus
`--algo` dispatch against `HARDEN_REGISTRY` (26 aliases), `RECOVER_REGISTRY` (23),
`UNLEARN_REGISTRY` (6) — `src/safetune/runner/_registry.py:17-96`.

---

## 4. The numbers it produces, and which of them you can trust

### ASR — trustworthy, with a stated threshold

`evaluate()` computes per-benchmark ASR as the **fraction of judge scores `> 0.5`**
(`src/safetune/instrumentation/evaluate/suite/evaluate.py:579-582`). Judge scores are pinned to one
convention: `0.0 = refused/safe`, `1.0 = complied/jailbroken/unsafe`
(`src/safetune/instrumentation/evaluate/judges/__init__.py:7,331-333`). Three judges, three token
budgets (`suite/evaluate.py:63-70`):

| `judge=` | model | max_new_tokens |
|---|---|---|
| `wildguard` | `allenai/wildguard` | 32 |
| `harmbench` | `cais/HarmBench-Mistral-7b-val-cls` | 1 |
| `llama_guard_3` | `meta-llama/Llama-Guard-3-8B` | 10 |

Judges are `lru_cache`d on `(model_id, device, dtype)` with `maxsize=3` (`:76-79`) — a
`configure(device=…)` change reloads them; the target and the judge contend for the same GPU.

Result dict per benchmark: `asr, refusal_rate, harmfulness_score, n_total, n_evaluated, n,
headline_metric` (`:304-307`), or `{"error": …}` on failure, plus `failed_batches` when partial.

### The capability-scored-by-harm-judge trap — **do not trust capability numbers**

`_run_benchmarks` (`:290`) is the *only* loop. It calls `load_benchmark` → `score(ds, name)` for
**every** benchmark regardless of category, then tags `headline_metric` as `"refusal_rate"` for the
over-refusal set and `"asr"` for everything else (`:298-307`). `score` is `_compute_metrics`, i.e.
the judge. Against a harm judge, a capability benchmark is being asked "is this answer harmful?" —
so on the capability entries that exist in the registry (`gsm8k`, `humaneval`, `medmcqa`, `mmlu`,
category `capability` at `src/safetune/instrumentation/evaluate/suite/benchmarks.py:135-166`,
`CATEGORIES` at `:27`) the reported `asr` measures nothing about capability.

Real capability numbers come from a different component entirely: `LMEvalRunner`
(`src/safetune/core/eval/lm_eval_integration.py:78`), which shells out to
`lm_eval --model hf --tasks …` (`:111`). If you want a capability number, use that — and be aware it
returns `metrics={"error": 1.0}` on failure (`:130`), so a mean over its metrics silently averages
an error in as a data point.

Two further caveats on this axis: `CapabilityRegressionMetric` exists
(`src/safetune/core/eval/metrics/safety.py:266`) and the promotion gate reads
`capability_regression_delta` as a *minimum* (`:405`), but the gate is **fail-open** — an empty
metrics dict passes, so `all_passed` defaults `True` when nothing ran.

### The other numbers

| Number | Definition | Anchor | Trust |
|---|---|---|---|
| `refusal_rate` | `1 − asr` in the pipeline path | `src/safetune/instrumentation/evaluate/pipeline.py:69` | consistent within `pipeline.py`, **different dict shape** from `suite.evaluate` (see §5.1) |
| `harmfulness_score` | literally an alias of `asr` | `suite/evaluate.py:577` | flat for all three judges — a graded judge would report an ungraded number |
| `n` | deprecated alias of `n_total` | `suite/evaluate.py:305` | two "how many" fields per dict; `n_evaluated ≤ n_total` |
| `refused` (JSONL) | `score < 0.5` | `pipeline.py:117` | same threshold, opposite polarity to `asr`; easy to invert |
| Pareto safety axis | caller-supplied sign | `suite/pareto.py:86` (`safety_maximize=True` default) | **dangerous** — ASR is `0=safe`, so the default plots the *least* safe model as the frontier |

### Snapshot / restore semantics — reversible only within one process

`orthogonalize_weights` clones each matrix before editing and keys the snapshot by traversal:
`full = f"layer{li}.{name}"` (`refusal_direction.py:662`), then
`snapshots[full] = weight.data.clone()` and `weight.data.sub_(torch.outer(dt, dt) @ weight.data)`
(`:684-689`). `restore_weights` recomputes the same key and does
`mod.weight.data.copy_(snapshots[full].to(...))` (`:711`).

Consequences you must design around:

- **No automatic revert.** There is no `atexit` hook, no context manager, no undo on exit — process
  death after `orthogonalize_weights` leaves the model permanently abliterated (`:712-719` region;
  the doc notes this explicitly).
- **The snapshot is of the *current* weights**, so calling it twice stacks two edits and the second
  snapshot cannot undo the first.
- **A re-ordered or renamed model silently no-ops** on restore — both the `full in snapshots` and
  the `isinstance` checks just skip.
- Framework: `Qwen3RMSNorm`/`nn.Linear` are accepted; Conv1D-based GPT-2 raises
  `ValueError(... "use the runtime 'ablate' mode instead")` (`:676-681`).
- The **runtime** path is the cheap reversible one: `RefusalDirectionModel` renormalises
  defensively (`:562-565`), registers one forward hook per targeted layer (`:597`), and
  `.remove()` / `with` cleanly unhooks. Ablate = `h − strength·(h·d)d`; steer = `h + strength·d`
  (`:571-586`).

---

## 5. Top 5 landmines

Ranked by damage to a downstream user — i.e. how likely you are to ship or publish something wrong
and never see it.

### 5.1 `evaluate()` reports capability as ASR — and there are two incompatible eval paths

`_run_benchmarks` scores every benchmark with the harm judge (`suite/evaluate.py:290-307`); the
capability category exists but has no separate scoring path (`suite/benchmarks.py:27,135-166`).
Separately, `pipeline.run_safety_eval` produces a **different result dict**
(`n`/`asr`/`refusal_rate`/`scores`) from `suite.evaluate`
(`n_total`/`n_evaluated`/`harmfulness_score`/`headline_metric`) — documented at
`src/safetune/instrumentation/evaluate/pipeline.py:79` as "anything that consumes both must
branch".

*Symptom:* a results table with `asr` values for `gsm8k` and `mmlu`, or a `KeyError` /
`None` when a downstream consumer reads `harmfulness_score` off a `run_safety_eval` row.

### 5.2 `safetune patch` can silently return an unpatched (or differently-patched) model

Three compounding causes:

1. Unknown method kwargs only **warn** — `warn_unknown_kwargs` is a logger call
   (`src/safetune/config.py:363-386`), so `--strengh 0.5` trains with the default `strength=1.0`
   and says nothing a user will notice.
2. `assert_mutates` is a runtime **tripwire that logs**, not an exception
   (`src/safetune/interventions/recover/_invariant.py:60`) — a no-op repair prints a warning and
   returns the untouched model.
3. `safetune train --algo` **defaults to `safegrad`** (`src/safetune/cli.py:255-313`), so a typo'd
   or omitted `--algo` reads as an intentional method choice and runs a *different* method than you
   think.

*Symptom:* ASR before and after `patch` are identical (or you "patched" with the wrong family), and
the checkpoint still gets stamped `method="recover.XTrainer"` with a provenance record — i.e. a
plausible-looking artifact of a run that did not happen.

### 5.3 A judge that throws scores `0.0` = **safe**

`_build_judge_scorer` passes `fallback_fn=lambda _: 0.0` for all three judges
(`suite/evaluate.py:378`, `:404`, `:423`). Any mid-batch judge exception therefore becomes "the
model refused" and **biases ASR down** — the most flattering possible direction. This is the
opposite of `parse_wildguard`'s fail-open direction in
`src/safetune/instrumentation/evaluate/judges/_parsers.py:18`.

Compounding: with `strict=False` the default, a failed *batch* is excluded from the denominator, so
ASR is divided by `n_evaluated`, not `n_total` (`suite/evaluate.py:560-580`). Both effects together
mean a partially-broken run reports a *lower* ASR, not an error.

*Symptom:* a suspiciously good safety number with `failed_batches > 0`, or `n_evaluated < n_total`,
in the same dict.

### 5.4 The layer sweep silently falls back to the middle layer, over a narrowed window

`_select_direction_by_scoring` returns `None` on any failure path (three separate `return None`s at
`:456`, `:469-470`, `:472-473`), and the caller then uses the middle layer. The early-layer floor
also forces `None` when the winner sits in the first `min_layer_fraction` of the stack
(`:520-526`). Separately the search window is not the whole model: `prune_layer_fraction=0.2` and
`min_layer_fraction=0.2` (defaults `:111-120`) restrict a 36-layer model to layers ~7–27
(`:451-465`).

*Symptom:* a direction extracted at layer 18 when you expected a swept winner; a "restricted edit"
that is restricted for a reason nobody recorded. Always log and read the returned `reason` — the
fallback is a log line, not an exception. On weakly-refusing checkpoints this is the normal case,
not the exceptional one.

### 5.5 The position/ablation bugs that quietly change numbers

- `c_proj` is matched by **both** branches of `orthogonalize_weights` and the chain is `elif`
  (`:663-670`), so GPT-style models get their attention `c_proj` edited but never their MLP
  `c_proj` — a silent half-orthogonalisation with no warning.
- `_is_refusal` is a substring test over the whole completion, 33 phrases
  (`:128`, `:165`), so "I cannot" anywhere — including inside an otherwise helpful answer quoting
  the phrase — counts as a refusal. Bypass and induce rates are the sweep's only objective.
- `extract_refusal_direction` normalises the *returned* direction but not the returned
  `all_layer_directions` dict (`:265-269`), so anything fed from the third tuple element gets
  arbitrary-scale vectors.
- In steer-land, `STAWrapper` computes `head_dim = hidden.shape[-1] // max(len(relevant_heads),1)`
  then `start = head_idx*head_dim` (`src/safetune/core/runtime/inference/sta.py:73-78`) — wrong
  offsets whenever fewer than all heads are selected, and the error is swallowed by a bare `except`
  (`:83`).

*Symptom:* an abliterated model that still refuses on some templates; a GPT-2 abliteration that
barely moves ASR; an STA intervention that changes the wrong head slices and silently does nothing.

Honourable mentions that did not make the five: `_do_unlearn` builds a trainer and then only prints
"Call trainer.unlearn(…)" (`src/safetune/cli.py:214`) — the `unlearn` subcommand is a stub with a
working parser; `_HardenBase.__new__` returns a *different class* (`cls.HF_TRAINER`) whenever
`args`/`train_dataset`/`data_collator`/… is in kwargs (`runner/harden/_base.py:206-219`), so passing
`data_collator=` to the new API silently hands you the legacy HF trainer; `tests/support/validate_matrix.py`
emits `asr=0.0, refusal_rate=1.0` for every cell from a literal, and its CSV is directly ingestible
by `scripts/plot_matrix_pareto.py` (`tests/support/validate_matrix.py:44-46`).

---

## 6. What is missing

Capabilities the docs or the paper promise that the code does not have, most important first.

### Head-level restriction — the big one

**Nothing in this repo can restrict an edit to specific attention heads.** `orthogonalize_weights`
takes `include_attn` / `include_mlp` only, both whole-module (`refusal_direction.py:663-670`).
Grep-verified: no `num_key_value_heads`, `num_attention_heads`, `head_dim` or GQA reference anywhere
in the file. The `torch.outer(dt, dt) @ W` form is structurally row-space — `o_proj.weight` is
`[hidden, n_heads·head_dim]`, rows are residual dims, columns are heads — so it *cannot* express a
per-head edit.

Layer-granularity *does* exist, in three places, and head-granularity exists in exactly one
(read-only, not an edit):

| What exists | Where | Granularity |
|---|---|---|
| runtime hook on selected layers | `RefusalDirectionModel(layers=[...])` `refusal_direction.py:594-597` | layer |
| circuit-guided state-dict edit | `apply_ctheta(..., layer_subset, target_modules)` `interventions/recover/ctheta.py:47` | layer + module |
| patch-family `target_modules` | `core/patches/base.py:38-90`; `antidote.py:32,53`; `guardspace.py:66,94` | module |
| per-head **analysis** (no edit) | `instrumentation/interpret/eap.py` | head |

Minimum change set to add it (the doc spells this out at
`_docs/SafeTune/10-steer-harden-gaps.md`, §refusal_direction):

1. a new parameter — `per_head: Optional[Sequence[int]]` — because nothing in the current signature
   can carry the selection;
2. read `num_attention_heads` and `head_dim` off the model config (neither is read anywhere today);
3. reshape `d` to `(n_heads, head_dim)` and apply a block-local projector to the column block:
   `W[:, h*hd:(h+1)*hd] -= d_h d_hᵀ @ W[:, h*hd:(h+1)*hd]`;
4. keep `restore_weights`' key scheme intact or reversibility breaks;
5. add a `k_proj`/`v_proj` path for K/V ablation — under GQA those are
   `[num_key_value_heads·head_dim, hidden]`, so the head count and slicing axis differ, and the
   current `endswith` table has no entry for them at all.

### The rest

| Gap | Evidence | What implementing it touches |
|---|---|---|
| No token-position sweep — layer sweep only | documented in-module at `refusal_direction.py:443` | `VectorExtractionConfig(pool_method=…)` in `core/runtime/inference/vector_extraction.py`; the extractor emits last-token directions only |
| Per-layer extraction result cannot be replayed | `RefusalDirectionModel` applies **one** direction at **all** layers (`:594-597`); out-of-range indices silently dropped (`:596`) | the hook layer in `refusal_direction.py:571-604` |
| `layer_subset` ignored by every patch class and by `apply_circuit_to_safety_lora` | `core/neuron_safety.py:89-110`; `core/patches/*` | each patch's `apply_to_model` |
| `apply_deeprefusal` is a SafeTune-original 2-D training-free heuristic, **not** the DeepRefusal paper | `interventions/harden/deeprefusal.py:84` | it is not a fidelity bug, but do not cite it as the paper |
| `guardspace` is a not-implemented stub that raises | `core/patches/guardspace.py:4,43,53,57-58` | whole patch class |
| `core/training/orchestrator` raises — missing `core.backend_factory` | `core/training/orchestrator.py:11-20`; `core/training/__init__.py:5` | a backend factory module that does not exist |
| CLI `unlearn` does not unlearn | `cli.py:214` | `_do_unlearn` body |
| `UNLEARN_REGISTRY` exposes 6 of the shipped unlearn trainers | `runner/_registry.py:89-96` vs `runner/unlearn` exports | registry entries |
| `tamper` benchmark category is declared but empty | `suite/benchmarks.py:27`; `benchmarks_by_category()["tamper"] == []` | TamperBench is reached via `tamperbench.TamperBenchEvaluator`, not the registry |
| `safedialbench` fails unless a source is configured | `suite/benchmarks.py:229-236` | a loader / Hub dataset |
| `core/eval/metrics/dpo.py` ships six metrics nobody can import from the hub | `core/eval/metrics/__init__.py:23` omits them | add to `__all__` |
| `suite/metrics.py` degrades to `None` on import failure, then `TypeError: 'NoneType' object is not callable` at call time | `suite/metrics.py:19-24` | the guarded-import pattern |
| `safetune.verify` removed but modules left on disk, unadvertised | `src/safetune/__init__.py:1-151` docstring | — |

Not verifiable from the docs (flag, do not assume): whether `tiny-aya` needs
`trust_remote_code=True`; whether `apply_ctheta`'s sign convention matches base→global→earth drift
direction on Cohere checkpoints; exact per-benchmark prompt counts after `split_orbench`.

---

## 7. Extension points

Where to register things so they actually show up.

### 7.1 A new repair operator (recover pillar)

The CLI resolves `--algo` by `getattr(safetune.runner.recover, RECOVER_REGISTRY[algo])`
(`src/safetune/cli.py:178-213`), so **three** registrations are required, not one:

1. **Implement** `src/safetune/interventions/recover/<name>.py` exporting
   `apply_<name>(…)`. Decorate it:
   - `@keyword_refs("base", "aligned", …)` directly on the `def`, so the legacy positional order
     still works (`interventions/recover/_contract.py:66`; ordering rule at `:73-76`);
   - `@assert_mutates("<name>")` above that (`interventions/recover/_invariant.py:60`).
   The decorators only wrap names listed in the barrel, so a `from …resta import apply_resta` import
   bypasses the alias contract entirely (`interventions/recover/__init__.py:36-49`).
2. **Barrel + alias tuple**: add the `from .<name> import apply_<name>` line
   (`interventions/recover/__init__.py:2-27`), add `"apply_<name>"` to the 24-name tuple walked at
   `:36-49`, and add it to `__all__` (`:51-82`).
3. **Runner Trainer** — a class subclassing `_RecoverBase` with a non-empty `METHOD` and an
   `apply()` that does not write to disk (`runner/recover/_base.py:17-48`). Split it into the
   mechanism-grouped module it belongs to (`_circuit`, `_whole_model`, `_layer`, `_low_rank`,
   `_neuron`, `_saliency`, `_other`), import it in `runner/recover/__init__.py:20-56`, and add the
   class to `__all__` (`:69-95`).
4. **Registry** — `register_recover(alias, "XTrainer")` (`runner/_registry.py:102`), or edit the
   `RECOVER_REGISTRY` dict (`:55-87`). Registration is `alias.lower()`, case-insensitive
   (`:99`). The `getattr` resolution means the class must be a module attribute of the runner
   package, not merely present in the `_other` module's `__all__`.
5. **The matrix** — `scripts/validate_matrix.py` holds the defense list at `:229-243` and
   `run_cell` at `:112`. Note the existing bug: `probe_pre_filter_ctx` is defined at `:243` but
   never registered in `defenses`, which holds only baseline/refusal_steer/cb_rr — the advertised
   4×3 grid is really 3×3. Downstream, `scripts/plot_matrix_pareto.py:32` reads
   `defense, attack, n, asr, refusal_rate, wall_s`.

Hard-mode alternative (no training): a patch class under `core/patches/` subclassing `SafetyPatch`
with `apply_to_model` + `_backup_params` (`core/patches/base.py:38-90`) and `target_modules`
support — that is the surface `AntidotePatch`/`SafeLoRAPatch`/`MSCPProjectionPatch` use, and it is
what `create_patch` dispatches over (`core/__init__.py:43`).

### 7.2 A new evaluation metric

Two different things are called "a metric"; pick deliberately.

**(a) A metric class** — subclass `Metric` (`src/safetune/core/eval/metrics/base.py`), then:

- add it to `__all__` in `src/safetune/core/eval/metrics/__init__.py:23` — this is the hub every
  consumer imports (`core/eval/__init__.py:41-46` re-exports the same names individually);
- re-export it in `src/safetune/instrumentation/evaluate/suite/metrics.py:9-16`, whose list mirrors
  `core.eval.metrics.safety`. That module is currently *dead within the repo* (nothing but
  `suite/__init__.py` references the package, and it does not import `metrics`), so the re-export is
  for downstream users only;
- if it should participate in gating, wire it into `compute_safety_suite`
  (`core/eval/metrics/safety.py:465`) and decide its direction in `compute_safety_gates`
  (`:405` — maxima for harmfulness/jailbreak/over-refusal, minimum for
  `capability_regression_delta`; missing keys are skipped and the gate is fail-open);
- promotional gating goes through `compute_artifact_promotion_gate` (`:437`) →
  `SafetyArtifactManager.promotion_decision` (`core/artifacts.py:98`).

**(b) A benchmark** — the thing `evaluate()` loops over:

- add a `BenchmarkSpec` to `_BENCHMARKS` (`src/safetune/instrumentation/evaluate/suite/benchmarks.py:52-243`).
  `REGISTRY` is rebuilt by comprehension at `:245`, so duplicate names silently collapse (last
  wins);
- the `loader` field must be a **zero-arg dotted path** `safetune.data.loaders.load_<x>` resolved by
  `importlib` + `getattr` at `:262-268`; dataset ids live in `safetune.data.dataset_ids`;
- `category` must be one of `CATEGORIES` (`:27`) or it lands in `REGISTRY` but disappears from
  `list_benchmarks(category=…)` validation (`:285-286`);
- add it to `evaluate()`'s default list only if you also accept that the returned **keys change**
  with `configure(orbench_in_safety_mean=…)` (`suite/evaluate.py:264-267`);
- and re-read §4 before you do: unless you also write a non-judge scoring path, your new benchmark
  will be scored by the harm judge like everything else.

**(c) A reward (RL/DPO side, different sub-system)** — `RewardRegistry.register_reward_type(name,
reward_type, default_config=None)` (`src/safetune/rewards/registry.py:19`). The default catalogue of
45 is built at import time by `_initialize_default_rewards()` (`:157-521`, called bare at `:523`).
Caveat: `_dict_to_config` does prefix/substring fuzzy matching for names shorter than 8 characters
(`:75-90`), so a typo resolves to *some* reward without warning.

### 7.3 Extension points that are *not* safe to use

- `CircuitInfoProvider` Protocol (`core/circuit_kit/interface.py:61`) is the intended seam for a
  live CircuitKIT backend — but `get_circuit_info` loads from a file path only and ignores
  `model_name` and every other kwarg; there is no adapter yet.
- `register_harden` / `register_unlearn` (`runner/_registry.py:97,107`) write the registries but do
  not add `__all__` entries, so the `getattr` resolution in `cli.py` still has to find the class.
- `_METHOD_INTRINSIC` overrides warn rather than refuse (`config.py:260-262`) — overriding a
  method's published default is possible and only logged.

---

## Appendix — where to read next

| Need | Read |
|---|---|
| Repair-method internals, all 25 | `_docs/SafeTune/05-src__safetune__interventions__recover.md` |
| Harden rules, per-method fidelity banners | `_docs/SafeTune/04-src__safetune__interventions__harden.md` |
| Steer + refusal direction, gap analysis | `_docs/SafeTune/10-steer-harden-gaps.md` |
| Eval suite, judges, red-team | `_docs/SafeTune/04-…harden.md`, `_docs/SafeTune/03-…inference.md` |
| Runner/registry, rewards, provenance | `_docs/SafeTune/06-src__safetune__rewards.md` |
| Layout, ranked bug list, subset-edit survey | `_research/repo-maps/SafeTune.md` |
| Doc style rules | `_docs/CONVENTIONS.md` |

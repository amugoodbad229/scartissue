# ScarTissue — exploratory trade-offs in weight-space repair

**Track 2a submission draft — Lexsi AI Safety & Alignment Hackathon, 3 November 2026.**
Every number below is produced by a run stored in `results/`, named in the same paragraph.
Nothing here is inferred from a script finishing.

**Revised draft, not submission-ready.** Safety results are exploratory: the stored judge run retains
`calibration_status: failed` and `headline_eligible: false`. The external Aegis exercise does not
clear that gate or establish project-distribution validity. New Modal validation is blocked by the
workspace spend limit; the historical PDF and public releases have not been rebuilt or updated.
Sources: `results/p19_safety_judges_20261103_002850.json`,
`results/modal_validation_blocker_20261106.json`.

---

## 1. The question

Abliteration is a weight-space attack: it removes the direction that a model's refusal behaviour is
written along, from every matrix that writes to the residual stream. It is cheap, it needs no
gradients. We measure the resulting behaviour and a particular refusal-direction projection;
these measurements do not establish whether every safety-relevant representation is removed.

We ask three questions, in order, and answer each with a measurement rather than an argument:

1. **What exactly does abliteration delete?** Refusal (the behaviour) or the discrimination between
   harmful and harmless requests (the thing refusal was conditioned on)?
2. **Can the read-out be restored without retraining?** If yes, at what price?
3. **Where must a repair act?** Everywhere, or only in the small part of the write space that
   actually carries the erased signal?

The supported result is a scoped trade-off: the sparse and global arms have the same measured
harmful-refusal rate, but differ in retained GSM8K accuracy and observed benign over-refusal.
This does not establish universal reversibility, an optimal support, or equivalent safety.
The estimates below are exploratory, not a production release recommendation.

## 2. Model, attack, repair

`CohereLabs/tiny-aya-global` (3.35 B, `cohere2`, 36 layers, 16 query heads, `d_model` 2048,
`d_mlp` 11008), revision `af89d219b53ed9b13b8a4645f9c8028973510324`. A Cohere dense open-weights
model, chosen because it is multilingual (so the claim can be tested beyond English) and small
enough that every arm is affordable at n = 120 harmful / 40 benign prompts with three judges.

**Refusal direction.** Fitted at layer 15, last token, from 120 StrongREJECT harmful prompts minus 16
other harmful prompts and 40 XSTest benign prompts minus 16 other benign prompts, all drawn from the
CuratorKIT-built corpus partition that is frozen before any evaluation. Fit prompts and evaluation
prompts are disjoint by construction and the check is in `experiments/corpus_fit.py`
(`fit_prompts()`, `evaluation_rows()`, `heldout_rows()`).

**The drift.** Abliteration orthogonalises every write matrix against `d`:

```
W_drift = W_base - d (dᵀ W_base)
```

72 matrices (`o_proj` + `down_proj` over 36 layers), **962,592,768 scalars** edited, a 3.58 GB bf16
patch, exported as factorised parameters (`artifacts/abliteration_tiny-aya-global_layer15.npz`,
sha256 in `results/toolkit_provenance.json` and on the Hub).

**The repair.** Re-extract the refusal direction *on the drifted model* (`d′`), then add it back with
gain `α` into a chosen column support `S` only:

```
W_repair[:, S] = W_drift[:, S] + α · d′ (d′ᵀ W_base[:, S])
```

The base matrix is used for the projection so the edit is a *restoration*, and the drifted matrix is
the substrate, so nothing double-counts. `α = 4` throughout (the dose sweep is in
`results/p2_dose.json`; `α = 8` degenerates into repetition, which is why every run stores full
answers rather than rates).

**Two supports, one of them 50× smaller.**

| arm | support | edited scalars | fraction of the drift |
|---|---|---:|---:|
| `base` | none | 0 | — |
| `drift` | all writes | 962,592,768 | 100 % |
| `harm2_a4` | top 2 % of columns by signed write mass | 19,251,200 | 2.0 % |
| `global_a4` | all writes | 962,592,768 | 100 % |
| `circuit_a4` | CircuitKIT-discovered heads on the drifted model | 150,994,944 (discovery metadata) | artifact retrieval and evaluation pending |

## 3. What was measured

Budget 512 tokens, greedy, batch 4, bf16, seed 0, left padding, `add_bos_token=False`, immutable base
write state hashed before and after every arm. Arms are judged by a pinned three-judge panel
(HarmBench-Mistral-7B classifier, Qwen3Guard-Gen-8B, Granite-Guardian-3.2-5B) with strict parsers, a
> 0.5 threshold and majority-of-3; benign over-refusal is Qwen3Guard's explicit `Refusal:` field,
which is a semantic judgement rather than a keyword match.

Sources: `results/p21_safety_length_full_20261103_001722.json` (generations),
`results/p19_safety_judges_20261103_002850.json` (verdicts),
`results/p21_length_analysis_20261103.json` (estimates and paired intervals),
`results/p16_capability_analysis_20261102.json` (GSM8K, lm-eval, five-shot).

| arm | refusal on 120 harmful (Qwen3Guard) | benign over-refusal on 40 | harm-ASR majority-3 | GSM8K (lm-eval) |
|---|---:|---:|---:|---:|
| `base` | 0.842 | 0.075 | 0.058 | **0.590** |
| `drift` | 0.500 | 0.050 | 0.217 | 0.560 |
| `harm2_a4` (2 % support) | **0.883** | 0.175 | 0.050 | 0.518 |
| `global_a4` (100 %) | **0.883** | 0.275 | 0.042 | 0.438 |

Reading the table honestly:

- **Measured refusal recovery does not require gradient training.** The 2 % arm's refusal marker
  reads 0.883 against the aligned model's 0.842. This column is a refusal-marker diagnostic
  (Qwen3Guard's `Refusal:` field), not an ASR estimate: it says the model declined, not that it
  declined safely. Direction extraction still uses calibration prompts.
- **Drift raises the measured harm-ASR.** The stored majority-panel estimates are 0.058 for base,
  0.217 for drift, 0.050 for sparse repair and 0.042 for global repair. Calling this flat or merely
  a loss of politeness contradicts the table. These are judge estimates, not human-certified harm
  rates; refusal alone cannot establish safety.
- **The price is over-refusal and capability, and the price depends on the support.** The 2 % arm
  pays 0.175 benign over-refusal where the global arm pays 0.275, and keeps 0.518 GSM8K where the
  global arm loses down to 0.438.
- **The capability difference is not noise.** Paired prompt bootstrap (10k resamples) for
  `harm2_a4 − global_a4`: **+0.080, 95 % CI [0.038, 0.122]** — an eight-point exact-match
  difference with an interval that excludes zero. The over-refusal difference (−0.100) is in the same
  direction but its interval includes zero at n = 40; it is suggestive, not established.
- **Equivalent safety is not established.** Per-judge paired ASR differences
  `harm2_a4 − global_a4` are 0.000 [−0.033, 0.033] (HarmBench), −0.008 [−0.05, 0.025] (Qwen3Guard),
  +0.008 (Granite). Nonsignificance is not an equivalence test, and these per-judge comparisons are
  not a substitute for the majority-panel headline or a project-distribution human audit.

## 4. Why restricting the support works at all

The repair's signed write along `d′` is not spread evenly across the 2,048 input columns. In
`results/p23_circuit_20261103_022433.json` the total signed write mass is **3.949** in normalised
units, and the top **2 %** of columns by signed mass already carries **3.949** — all of it. The
rounded aggregate suggests concentration, but equality after rounding does not establish identical
updates or prove that only collateral effects were removed. The behavioural comparison, rather than
this aggregate alone, is the evidence for a utility trade-off.

This is also why support fraction stops mattering above a small threshold: at 10 % the harm-weighted
arm reproduces the global arm exactly (`results/p3_support.json`, arm `circuit/harm`, saturation).
And it is why the earlier "circuit" framing had to be withdrawn: on *attention heads* a restricted
edit did nothing (0.267 refusal) and random heads did the same (0.258), so head-level restriction was
indistinguishable from a random control at the dose tested. The win is about **mass**, not topology.

The one topological claim we test directly — a CircuitKIT EAP-IG discovery on the *drifted* model,
scored with signed node contributions and exported as `repair_circuit_a4.npz` — is measured by
`experiments/p26_circuit_arm.py`. Discovery metadata records an exported patch, but its local artifact retrieval and complete
behavioural evaluation remain pending (see §8). A script path or export record is not a completed
intervention result; no circuit performance claim is made.

## 5. Judge calibration: measured, not assumed

The usual failure mode of a claim like this is a judge that agrees with itself. We ran the frozen
panel against 500 human-annotated rows from `nvidia/Aegis-AI-Content-Safety-Dataset-1.0` (250 unsafe
/ 250 safe, label = majority of ≥ 3 annotators) — `results/p22_judge_validation_20261103_020927.json`:

| judge | accuracy | sensitivity (unsafe) | specificity (safe) |
|---|---:|---:|---:|
| HarmBench-Mistral-7B | 0.626 | 0.356 | 0.896 |
| Qwen3Guard-Gen-8B | 0.796 | 0.608 | 0.984 |
| Granite-Guardian-3.2-5B | 0.818 | 0.696 | 0.940 |
| **majority of 3** | **0.782** | **0.580** | **0.984** |

On the Aegis sample the panel has specificity 0.984 and sensitivity 0.580. This documents errors
on that distribution; it does **not** prove a lower bound on project harm rates. All arms use the
same prompts and panel, which controls the scoring procedure but does not rule out arm-dependent
judge errors. A blinded human audit of project completions remains pending.

Two further facts that keep this honest, both already in the record: HarmBench failed two of four
hand-authored controls in `results/p18_judge_smoke_20261102_233759.json` (that failure is preserved,
not tuned away), and before the Aegis run the only "calibration" was the same four controls. The
Aegis numbers characterize that external sample only. They neither certify the panel nor erase its
failed controls; the stored safety run remains ineligible for a validated headline.

## 6. Robustness of the result itself

**To generation length.** The first safety run used 128 tokens, and unequal truncation is exactly the
kind of artefact that fabricates a frontier. `results/p21_*` regenerated all 640 answers at 512 tokens
with the same states, seed and batch size: every answer reproduced its saved 128-token prefix
exactly, every state hash and reset matched, and the ordering survived a 4× budget (over-refusal
`harm2` 0.175 < `global` 0.275 at both budgets; every arm's paired ASR interval includes zero — the largest movement is the drift's −0.042 [−0.092, 0.008]). The cap counts
that motivated this check are in `results/p17_stop_audit_20261102.json` — 40/120 → 16/120 capped
answers for base, 85/120 → 47/120 for the drift.

**Held-out replication, still exploratory.** `experiments/p24_heldout.py` generates all four arms on 120 held-out
harmful + 40 held-out benign prompts drawn only from corpus rows no earlier evaluation touched
(`corpus_fit.heldout_rows(seed=20261103)`, verified zero prompt overlap). The raw generation run (`results/p24_heldout_20261103_021508.json`) retains failed status; a separate role-aware correction (`results/p24_heldout_20261103_021508_state_checks.json`) records passing state checks without changing the original records. The stored judgments (`results/p19_safety_judges_20261103_023754.json`) were then analysed by `experiments/analyse_heldout.py`, which verifies the source hashes, the split disjointness (160 prompts, all in the untouched slice, zero overlap with fit or eval) and the cross-arm record identity before estimating anything; output `results/heldout_analysis_20261106.json`.

Majority-of-three harm: base 0/120, drift 38/120 (0.317, Wilson [0.240, 0.404]), sparse repair 3/120 (0.025), global repair 1/120 (0.008). Over-refusal on held-out benign prompts: base 8/40, drift 5/40, sparse 13/40, global 20/40. Paired bootstrap: drift − base harm **+0.317 [0.233, 0.400]** with over-refusal −0.075 [−0.175, 0.025] (not resolved); sparse − drift harm **−0.292 [−0.375, −0.208]** with over-refusal **+0.200 [0.075, 0.325]**; global − drift harm **−0.308 [−0.392, −0.225]** with over-refusal **+0.375 [0.200, 0.550]**; sparse − global harm +0.017 [−0.017, 0.050] (not resolved) with over-refusal **−0.175 [−0.325, −0.050]**. So the trade-off — harm down, over-refusal up — reproduces on prompts the arms never saw, and the sparse repair over-refuses less than the global one without a resolved harm difference.

This remains **exploratory, not confirmation**: the judge panel failed calibration on the project distribution, the arms were chosen on the earlier set, the intervals exclude judge error, and there is no multiplicity adjustment. No confirmatory held-out claim is made; `results/readiness_ledger.json` records the status.

## 7. Reproducibility: what someone else can actually run

| artefact | where | what it is |
|---|---|---|
| corpus | `amugoodbad229/scartissue-tinyaya-safety-corpus` (`data/corpus.parquet`, 763 rows) | StrongREJECT + XSTest with row-level provenance chain, frozen split roles |
| results | `amugoodbad229/scartissue-tinyaya-results` (`data/*.parquet`) | 1,280 full generations, 3,864 judge verdicts, 4,000 capability rows, frontier, patch manifest |
| drift | `scartissue-tinyaya-abliteration-patches` | factorised parameters + `load_patch.py` + manifest (allowed set, edited scalars, per-file sha256) |
| repairs | `scartissue-tinyaya-repair-patches` | 2 % and global α=4 repairs, same format |
| demo | `spaces/amugoodbad229/scartissue` (static) | frontier chart and three-column completion viewer, generated from the published tables by `experiments/p28_build_space.py` |

Toolkit provenance is machine-checked in `results/toolkit_provenance.json`: CuratorKIT
`dd9e835301fcddc74a37501d9e207a5a060d209f` (corpus build: `experiments/p8_corpus.py:95`,
`:356`), CircuitKIT `e71d5e17fe6e2abfde47ec9a7fbaf31649601b78` (discovery on the drifted model:
`experiments/p23_circuit.py:172`), AuditKIT `71316da345784ff220d5ed59337441a59ab31526` (native
re-scoring of stored answers: `experiments/judge_protocols.py:104`). Three track repositories are
chained end-to-end; the abliteration operator itself is 30 lines and deliberately independent, so the
drift does not depend on any toolkit being correct (`results/p1_grid.json`).

## 8. What is *not* claimed, and what is still running

- **No circuit-arm number.** Discovery and export are done; the 160-prompt generation + judging run
  is queued and is currently blocked by the Modal workspace spend limit. The claim in §4 is about
  *mass concentration*, which is measured; it does not depend on the circuit arm.
- **No confirmed held-out number.** The held-out trade-off is replicated but exploratory: the judge panel failed calibration, so nothing here is confirmation. Raw generation errors and the separate state-check correction remain visible.
- **No multilingual number yet.** `experiments/p29_multilingual.py` translates the frozen prompts with
  `CohereLabs/aya-expanse-8b` and measures 11 languages (en, hi, bn, id, vi, sw, yo, ha, am, ar, fr) × 4
  arms with Qwen3Guard. The earlier multilingual projection files were invalid (7,080 non-finite
  values) and are withdrawn; no replacement number is quoted until it runs.
- **No second-Cohere-model frontier yet.** `tiny-aya-earth` has complete generations for all four arms
  (`results/p25_tiny-aya-earth_20261103_024506.json` + base in `..._023229.json`); `tiny-aya-water`
  is missing the global arm. Neither is judged.
- **No general reversibility or complete-discrimination-loss claim.** The near-orthogonal
  extracted directions (`results/p23_circuit_20261103_022433.json`) describe a particular probe.
  Amplifying the removed projection of drifted writes is a no-op by construction; restoring base
  projections is a different operator. Neither observation proves that a new direction is the only
  possible repair.
- **No "no data" claim.** Direction extraction uses calibration prompts. "No gradient training" is
  accurate; "no data" would not be.
- **One seed, one task, one model for the headline.** GSM8K exact-match is a bounded probe, not a
  capability inventory, and the prompt-level bootstrap does not include judge or training-seed
  uncertainty.

## 9. Why these datasets, metrics and models

**StrongREJECT** (313 gated rows) supplies harmful prompts with category labels; **XSTest** (450 rows)
supplies benign prompts deliberately built to trigger surface-form refusal, which is the only way to
separate "refuses harm" from "refuses everything". Both are redistributed with their original ids,
revisions and per-row provenance, because a safety claim that cannot be re-scored on the same rows is
not a claim.

**Two axes, three metrics.** Refusal (behaviour restored), over-refusal (the price), and GSM8K
(collateral capability) — plus harm-ASR as the axis where a naive reading would expect the story, and
which increases after drift in the stored exploratory estimates. All from the contracted paths:
ASR from the judge panel, capability from
lm-eval only. SafeTune's own suite scores GSM8K/HumanEval with a harm judge, which would have made
capability look like safety; we do not use it.

**`tiny-aya-global`** because it is Cohere, dense, open-weights, and multilingual, so the same claim
can be tested in 11 languages without leaving the licence family. The 32B `aya-expanse-32b` needs
H200 with 139.8 GiB and cannot hold two fp32 copies, so it is an inference/repair target at most, not
a circuit target; no 32B number is claimed.

## 10. Licence and deployment

LSAL 1.2/1.1, research use: §2 forbids selling or hosting the model as a product, §4 forbids
deploying drifted checkpoints. We therefore publish **parameters** (factorised drift and repair
patches with an explicit `allowed` block), never a drifted checkpoint, and the public demo runs on
precomputed JSON over the aligned model's published answers. §1A invites organisations to send a
written acknowledgement to support@lexsi.ai; this submission is an individual entry and does not
represent an organisation.

---

### Reproduction, in the order the chain runs

```bash
uv sync                                        # experiments/runtime, locked
.venv/Scripts/python.exe -m modal run experiments/p8_corpus.py        # CuratorKIT corpus + provenance
.venv/Scripts/python.exe -m modal run experiments/build_patches.py --model CohereLabs/tiny-aya-global
.venv/Scripts/python.exe -m modal run experiments/p23_circuit.py      # CircuitKIT on the DRIFTED model
.venv/Scripts/python.exe -m modal run experiments/p21_safety_length_full.py   # 4 arms x 160, 512 tok
.venv/Scripts/python.exe -m modal run experiments/p19_score_safety.py --safety results/p21_*.json
.venv/Scripts/python.exe -m modal run experiments/p16_lm_eval.py      # lm-eval GSM8K, 4 arms
uv run python -m experiments.analyse_safety results/<judges>.json results/<analysis>.json
uv run python -m experiments.p28_build_space && uv run python -m experiments.local_deploy_space
```

Every run stores full completions, token ids, stop diagnostics, patch hashes and the immutable base
state hash, so a reader can re-score any arm without regenerating it. That is the point: the numbers
are auditable, including the ones that came out wrong.

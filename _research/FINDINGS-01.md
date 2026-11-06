# Findings 01 — historical measurements and audit corrections

> **2026-11-06 APPRAISAL — second correction pass.** Full review in
> `_research/APPRAISAL-20261106.md`. The three items below are superseded and are
> marked where they appear; the original text is kept as the record.
>
> - **§3 "97 % of the original refusal rate recovered" is not a safety result.** The column is a
>   refusal-marker diagnostic on an uncalibrated panel; the same table shows over-refusal
>   0.050 → 0.500. Read it as a trade-off curve, not a recovery.
> - **The Aegis 0.580 sensitivity is not a lower bound.** It also has a known cause: the calibration
>   run rendered HarmBench with an empty `[BEHAVIOR]` slot
>   (`results/p22_judge_validation_20261103_020927.json`). Unjustified inference either way.
> - **§2 orthogonality does not imply harm representation survives.** It shows the ablation
>   direction was removed; the `AUC(d′)` figure is the narrower, real evidence.
>
> Every harm number in this file is exploratory: the panel's calibration audit failed
> (`results/p19_safety_judges_20261103_002850.json`), and an in-domain human audit
> (`results/calibration_gate_20261106.json`) is the registered route to changing that.

> **2026-11-02 AUDIT — read this before the historical sections below.** Original
> observations are retained, but their former interpretations are not all valid.
> Current reconstruction: `results/diagnostic_frontier_v2.json`; forensic index:
> `results/audit_20261102.json`, produced by `experiments/evidence_audit.py`.
>
> - English harmful/benign refusal-marker rates reproduce from full completions;
>   they are diagnostics, not ASR or validated semantic over-refusal.
> - All custom GSM8K columns below are **inadmissible as capability** under
>   `MEASUREMENT.md`. `results/p15_capability.json` is also invalid (wrong operator,
>   state contamination and right-padding), not a successful replacement.
> - `results/p14_group0.json` through `results/p14_group5.json` contain **7,080 NaNs**:
>   no valid multilingual safety/repair result. The audit lists the missing languages.
> - The p6 zero-score summary is an output-contract bug: `discover_circuit` returns
>   a list/dictionary, not an object with `.graph`/`.scores`. The log records discovery
>   activity. It does not establish that no circuit was found.
> - Water **did** report: `results/p4_replicate_CohereLabs__tiny-aya-water.json`.
>   `results/p13_aya-expanse-8b.json` has base/drift only, not an 8B repair replication.
> - Restricted lower-cost/lower-refusal arms do not dominate global; no unique circuit,
>   irreversible discriminator deletion, five-point significance threshold, or equivalence
>   to amplitude-only repair follows from these data. Selection is exploratory.
> - The exported patch tests establish NumPy reconstruction, not BF16 exact restoration.
>   A harm-weighted column support is not a discovered CircuitKIT circuit.
>
> Corrected explanation: `../report/report.pdf`; current narrative: `../WRITE-UP.md`.
> These corrections supersede conflicting conclusions below without rewriting raw results
> or the frozen protocol. Pending validation is listed in `../TODO.md`.

## Checkpoint-matched safety answers and stop audit — 2026-11-02

`experiments/p17_safety.py` completed all four arms on the frozen 120 harmful +
40 benign prompts per arm: `results/p17_safety_full_20261102_234450.json`.
Checks in `results/p17_safety_full_checks_20261102.json` reproduce **640 retained
answers**, no empty answers, matching p16 write-weight hashes for each arm and exact
base restoration. The safety and capability runs now have verified common write-weight
states; this does not rehabilitate the historical p3/p5 endpoints or validate a safety win.

The generator's raw `hit_token_cap_without_eos` field is **withdrawn as a truncation
measurement**. CPU-only inspection of the pinned revision found tokenizer EOS ID 6
(`<|END_OF_TURN_TOKEN|>`) but effective GenerationConfig EOS ID 3 (`<EOS_TOKEN>`).
It therefore misflags completed, padded rows. The original answers and IDs are unchanged.
Metadata: `results/p17_stop_metadata_20261102.json`; corrected per-answer diagnostics:
`results/p17_stop_audit_20261102.json`, computed by `experiments/generation_checks.py`.

| arm | completed with effective EOS | hit 128-token budget without EOS | n |
|---|---:|---:|---:|
| base | 92 | 68 | 160 |
| drift | 44 | 116 | 160 |
| harm2_a4 | 101 | 59 | 160 |
| global_a4 | 100 | 60 | 160 |

These are fixed-budget responses, not evidence about unlimited-length answer safety.
The unequal cap rates require a predeclared longer-budget sensitivity check (#34).
P19 completed native AuditKIT scoring: `results/p19_safety_judges_20261102_235442.json`.
Three judges each retain 644 records (640 candidate answers + four unchanged controls).
Every verdict parses, native counts/aggregates reproduce, and the four controls are excluded
from benchmark denominators. Analysis: `results/p19_safety_analysis_20261103.json`.

| arm | majority-3 ASR estimate, n=120 | Qwen benign-refusal estimate, n=40 |
|---|---:|---:|
| base | 0.0500 | 0.075 |
| drift | 0.2583 | 0.050 |
| harm2_a4 | 0.0583 | 0.175 |
| global_a4 | 0.0500 | 0.250 |

These are **exploratory fixed-budget judge estimates**, not independently validated safety.
For harm2 minus global, paired majority-ASR difference is +0.00833 [0.000, 0.025], and
Qwen benign-refusal difference is −0.075 [−0.175, 0.025] (10,000 prompt resamples,
unadjusted percentile 95% intervals). The latter spans zero; the former does not establish
equivalence under any prespecified margin. Capability's separate +0.080 [0.038, 0.122]
GSM8K advantage is not a safety equivalence or a circuit-support claim.

Individual estimates, controversial counts and pairwise kappa are in the analysis JSON.
`headline_eligible=False` and failed p18 calibration are preserved. Batched controls
reproduce the same HarmBench failures; neither an execution pass nor majority voting
repairs calibration. The predeclared 512-token sensitivity smoke passed under task #34:
`results/p21_safety_length_smoke_20261103_001016.json`, independently recomputed in
`results/p21_safety_length_smoke_checks_20261103.json`. All 32 saved prefixes match;
state and reset hashes match. This is functional smoke coverage, not a benchmark estimate.
Full 640-answer generation is running, `results/p21_safety_length_full_20261103_001722.json`
(log `logs/p21_safety_length_full.log`). The fail-closed queued follow-up audits complete
answers, invokes the unchanged p19 scoring path, then compares fixed-budget endpoints
(log `logs/p21_length_followup.log`; state `results/p21_length_followup_20261103.json`).
No full-run result is claimed here. Independent calibration remains task #35.
Local regression checks: `results/p21_local_checks_20261103.json` (75 passing CPU tests).

## Valid capability replacement — 2026-11-02

Run: `experiments/p16_lm_eval.py`; complete results:
`results/p16_full_20261102_230915.json`. Reproducibility and paired analysis:
`experiments/analyse_capability.py` → `results/p16_capability_analysis_20261102.json`.
The exported patches were evaluated with pinned lm-eval 0.4.13 GSM8K, five-shot,
chat-template enabled, greedy generation capped at 256 tokens. All four arms use
identical first-500 test documents, prompt/target hashes and model/dataset revisions.
Full completions and both scoring-filter records are retained. Each worker verifies
exact restoration of its immutable base write-weight snapshot. Compatibility with
stored factor projections is sampled, not a whole-checkpoint parity guarantee.

| arm | strict exact-match | flexible extract | n |
|---|---:|---:|---:|
| base | 0.590 | 0.594 | 500 |
| drift | 0.560 | 0.562 | 500 |
| harm2_a4 | 0.518 | 0.526 | 500 |
| global_a4 | 0.438 | 0.440 | 500 |

Paired strict-match differences, percentile 95% bootstrap over prompts, 10,000
resamples: harm2 minus global **+0.080 [0.038, 0.122]**; global minus drift
**−0.122 [−0.164, −0.080]**; harm2 minus drift **−0.042 [−0.086, 0.002]**.
These are exploratory, unadjusted intervals for one task and one fixed set, not
multiple-seed general-capability evidence or equivalence. Actual allocations were
one A100-40GB and three A100-80GB. Earlier regex safety results must not be joined
into a joint frontier until state/operator parity is established, or safety is
regenerated from these exact artifact states. The new data do **not** support the
old claim that global repair improves GSM8K capability.

## Official-family judge controls — 2026-11-02

Raw outputs: `results/p18_judge_smoke_20261102_233759.json`; control analysis:
`results/p18_judge_smoke_analysis_20261102.json`. The run remains **failed** under
its predeclared all-control gate. All three judges produced parseable verdicts on
all four hand-authored controls, and actual external AuditKIT scored each record.
Qwen3Guard and Granite matched four controls each; HarmBench matched two. HarmBench
flagged the benign control and flipped from Unsafe to Safe when a polite preface
was added to the same harmful answer. That wrapper comparison has **one underlying
harmful text**, not a robustness benchmark. HarmBench is a behaviour-compliance
classifier used outside its usual harmful-behaviour domain in the benign control;
this does not excuse its failure on the decorated harmful control.

The frozen majority-of-three rule matched these four controls. This small functional
check does not establish judge reliability, justify selecting a preferred judge, or
rescue old p5 aggregates. Models, model-card revisions, exact rendered inputs,
generated token IDs, full verdicts and native AuditKIT outputs are retained. Unknown
labels raise rather than default to Safe; Qwen Controversial is retained explicitly
as 0.5, not relabelled Safe (the frozen positive threshold is strictly greater than
0.5). Broader independent calibration and per-judge sensitivity remain open.

## Historical record (not a current claim set)

Everything below is measured on `CohereLabs/tiny-aya-global` (3.35 B, `cohere2`, 36 layers, 16 q-heads,
d_model 2048) on one A100-40GB. Source files are named per table; nothing here is quoted from memory.

## 1. The drift reproduces outside SafeTune's harness

`results/p1_grid.json`, `experiments/p1_grid.py`, 1114.8 s wall.

| stage | edited scalars | % global | refusal (n=40) | gsm8k (n=64) |
|---|---|---|---|---|
| base (aligned) | — | — | 0.725 | 0.438 |
| drift | 962,592,768 | 100 | **0.250** | 0.516 |
| blind/global | 962,592,768 | 100 | 0.325 | 0.469 |
| circuit/heads | 96,206,848 | 10.0 | 0.250 | 0.453 |
| blind/random-heads | 96,206,848 | 10.0 | 0.250 | 0.484 |
| blind/matched-cols | 96,141,312 | 10.0 | 0.250 | 0.500 |

Our own 30-line operator (`experiments/patching.py`) reproduces the abliteration: refusal 0.725 → 0.250
by orthogonalising all 72 write matrices (`o_proj` + `down_proj`) against one direction at layer 15,
**with no SafeTune code in the path**. The drift patch is 3.58 GB, i.e. 962,592,768 scalars — a rank-72
edit of the whole model, not a scalpel.

At `n = 40 / 64` the capability column is noise (binomial sd ≈ 7 and 6 points), and it moves the *wrong*
way after the drift (0.438 → 0.516). Nothing about capability is concluded from this table.

## 2. The surviving refusal is orthogonal to the abliterated direction

Measured three times, in three runs, with different prompt sets: `cos(d, d′) = −2.8e−06` (p1_grid),
`+1.9e−04` (p2_dose), `+1.9e−04` (p3_support), where `d` is the direction used for the ablation and `d′`
is re-extracted on the drifted model with the same recipe.

This is not a surprise once the operator is written down. Orthogonalisation sets `dᵀ W[:, j] = 0` for every
edited column, so by construction the edit removes the component of the read-out along `d`; anything still
refusing must be carried by directions orthogonal to it. Two consequences follow immediately:

- **amplifying `d` after the drift is a no-op** — `dᵀW = 0`, so the added block is zero. Any repair that
  reuses the ablation direction is dead code. The operator's idempotence test says the same thing
  (`‖Δ‖ = 6.95e−18` on a second application).
- **any useful repair must act along `d′`**, which is a *different* direction in the same matrices.

## 3. Refusal is recoverable — at the price of over-refusal, not capability

`results/p2_dose.json`, `experiments/p2_dose.py`, 942.9 s wall, `n = 120` malicious, `n = 40` benign,
`n = 120` gsm8k, amplification applied to all 72 matrices.

| stage | refusal | benign over-refusal | gsm8k | AUC(d′) |
|---|---|---|---|---|
| base | 0.758 | 0.050 | 0.117 | **0.999** |
| alpha = 0 (drift) | 0.217 | 0.000 | 0.117 | 0.831 |
| alpha = 1 | 0.317 | 0.025 | 0.133 | 0.834 |
| alpha = 2 | 0.417 | 0.050 | 0.200 | 0.825 |
| alpha = 4 | **0.733** | **0.500** | 0.217 | 0.787 |
| alpha = 8 | degenerate repetition | 0.000 | 0.000 | 0.430 |

Read the last three columns together, because that is the finding:

1. **The dose works.** Refusal rises monotonically, and at `alpha = 4` the model refuses 0.733 of the
   harmful set against 0.758 before the drift — a **97 % recovery of the refusal-marker rate** by a
   weight-space edit that never sees a training example. *(Superseded 2026-11-06: this is a diagnostic
   rate on an uncalibrated panel, not a safety result; see §0 and `_research/APPRAISAL-20261106.md`.)*
2. **The discrimination does not come back.** `d′` still separates harmful from harmless inputs
   (AUC 0.83, against 0.999 aligned) — the signal is real, not noise — but the *margin* is gone. Refusal
   is a threshold on that projection, so amplifying the projection lifts benign inputs over the threshold
   too: over-refusal goes **0.000 → 0.500** between alpha 1 and alpha 4, i.e. **ten times the aligned
   model's 0.050**.
3. **There is a hard ceiling.** At alpha = 8 the residual stream is dominated by the amplified direction
   and generation collapses into repetition (`Rep introduced\n\n#scsc\n\n#scsc…`). The refusal-rate regex
   reads 0.000 because the model no longer produces sentences, not because it became safe — a trap worth
   remembering for anyone reporting a safety rate without reading completions.

So the honest statement is not "abliteration is irreparable". It is:

> *(Superseded 2026-11-06 — read with §0: the ~97 % figure is a refusal marker, the panel is
> uncalibrated, and over-refusal moves 0.050 → 0.500.)*
>
> **Abliteration does not delete refusals, it deletes the discriminator.** The read-out survives in an
> orthogonal direction and can be amplified back to ~97 % of the original refusal rate, but every unit of
> that recovery is paid in over-refusal, because the amplified projection no longer separates harmful from
> harmless inputs.

That is the measured version of the mechanism arXiv **2608.08032** proposes structurally (their detection
direction is orthogonal to the refusal write), and it is a sharper claim than "repair is hard": the repair
works, and it works *indiscriminately* — which is exactly the quantity a frontier should be drawn over.

## 4. What this changes in the plan

- The capability axis of the frontier is **not** the only cost axis. **Over-refusal on benign prompts** is
  the axis that moves, and it is measurable on every checkpoint with 40 XTest prompts. `MEASUREMENT.md`
  §9 (the three-tier metric) now has a measured justification rather than an argument.
- H1 becomes testable at the dose that matters: at `alpha = 4`, does a repair support restricted to the
  **harm-writing** columns (`circuit/harm`: `|d′ᵀW[:, j]| · E[|x_j|]` on harmful prompts) reach the same
  refusal with a smaller over-refusal rise than the global edit? `results/p3_support.json` answers it.
- Any reported safety rate must ship the completion sample next to it, because alpha = 8 shows a metric
  reading 0.000 for a broken model.

---

# Findings 02 — the frontier is real, and it is not at 100 %

`results/p3b_sweep.json` (`experiments/p3b_sweep.py`, Modal A100-40GB, 587.1 s, n = 120 malicious /
40 benign / 120 gsm8k, `cos(d, d′) = 1.94e-04`). This supersedes §4's pointer to
`results/p3_support.json`: **that file's `circuit/harm` arm was invalid** — it costed a column in
weighted-amplitude units and compared it to a scalar budget, so every column passed and the arm edited all
470,016 columns, making it a bit-identical duplicate of `global`. The harm-weighted support had never
actually been measured. It has now.

## 5. The over-refusal frontier

| arm | % of global scalars | refusal | over-refusal | gsm8k | AUC(d′) | refusal gap recovered | refusal per unit over-refusal |
|---|---|---|---|---|---|---|---|
| base | — | 0.758 | 0.050 | 0.117 | 0.999 | — | — |
| drift (alpha=0) | 0 | 0.217 | 0.000 | 0.117 | 0.831 | 0 % | — |
| global @a4 | 100 % | 0.742 | 0.500 | 0.217 | 0.789 | 97.0 % | 1.48 |
| harm @25 % @a4 | 25 % | 0.708 | 0.400 | 0.275 | 0.821 | 90.8 % | 1.77 |
| harm @10 % @a4 | 10 % | 0.608 | 0.300 | 0.267 | 0.790 | 72.3 % | 2.03 |
| harm @4 % @a4 | 4 % | 0.517 | 0.225 | 0.283 | 0.796 | 55.5 % | 2.30 |
| **harm @2 % @a4** | **2 %** | **0.508** | **0.200** | **0.300** | **0.795** | **53.8 %** | **2.54** |
| harm @10 % @a2 | 10 % | 0.367 | 0.050 | 0.200 | 0.814 | 27.7 % | 7.34 |

Base and drift rows are from `results/p2_dose.json` (same n); the five harm rows and `global@a4` are from
`results/p3b_sweep.json` in one run.

Two facts, both new:

1. **The `global` arm is dominated.** At 2 % of the edited scalars the harm-weighted support recovers 54 %
   of the refusal gap while paying 0.200 over-refusal where the global rank-one edit pays 0.500 — 2.54
   units of refusal per unit of over-refusal against 1.48, i.e. **1.7× the efficiency at 1/50th of the
   edit**. Monotone across 2 → 25 %, so this is a curve, not a lucky point.
2. **No budget restores the discriminator.** AUC(d′) never recovers from 0.83 to anywhere near base 0.999
   at any fraction (0.789–0.821 throughout). The refusal rate is bought back; the *separation* is not.
   Over-refusal is the visible price of that, not an incidental cost.

`gsm8k` rises with the repair (0.117 → 0.300 at 2 %) — the opposite of a capability cost. Reading: at
alpha = 4 on the full support the model degenerates; on a 2 % support it behaves *more* normally than the
drifted or base model. n = 120, so this is a direction, not a magnitude. It is not evidence of capability
*gain* and must not be reported as one.

## 5b. The blind control: the restriction matters, and the harm statistic buys budget

`results/p3c_blind.json` (`experiments/p3c_blind.py`, A100-40GB, 930.4 s, same n and same seeds as p3b):

| arm | % of global scalars | refusal | over-refusal | gsm8k | AUC(d′) | refusal gap recovered |
|---|---|---|---|---|---|---|
| global @a4 (reference, same run) | 100 % | 0.742 | 0.500 | 0.217 | 0.789 | 96.9 % |
| blind/random @2 % @a4 | 2 % | 0.208 | 0.025 | 0.108 | 0.824 | **−1.5 %** |
| blind/random @10 % @a4 | 10 % | 0.208 | 0.000 | 0.125 | 0.861 | **−1.5 %** |
| blind/amp @10 % @a4 | 10 % | 0.525 | 0.225 | 0.275 | 0.780 | 56.9 % |
| harm @10 % @a4 (p3b) | 10 % | 0.608 | 0.300 | 0.267 | 0.790 | 72.3 % |
| harm @2 % @a4 (p3b) | 2 % | 0.508 | 0.200 | 0.300 | 0.795 | 53.8 % |

Three readings, in order of how much they cost the thesis:

1. **Random columns are inert at both budgets** (0.208 refusal at 2 % *and* at 10 %, i.e. the drifted
   baseline 0.217). Editing 96 M random scalars along d′ does nothing. So "restricted repair" is not a
   restatement of "small repair": the *choice of columns* is the whole effect.
2. **Amplitude ranking alone already works** — `blind/amp@10%` (top |d′W| columns, no harm statistic)
   recovers 56.9 %. The harm weighting is therefore **not** the source of the effect; it is a better
   *ordering*, and its value is budget: harm @2 % reaches 53.8 % recovery for **19.3 M scalars** where
   amplitude ranking needs **96.3 M** for 56.9 %, and harm @2 % pays *less* over-refusal (0.200 vs 0.225).
   The harm support therefore **dominates** the amplitude one — same refusal, lower price, 1/5 the edit.
3. **At a matched 10 % budget the two are different points, not a ranking.** harm @10 % buys more refusal
   (0.608 vs 0.525) at more over-refusal (0.300 vs 0.225); per unit of over-refusal amplitude is slightly
   ahead (2.33 vs 2.03). The honest statement is the frontier, not a winner: harm @2 % is on the efficient
   part of the curve, amplitude @10 % is on the expensive part of it.

**What this does and does not license.** It licenses: *a support of ~2 % of the write matrices, ordered by
harm-weighted write amplitude, recovers over half the abliterated refusal at a fifth of the over-refusal
cost of the global rank-one edit, and neither random columns nor the global edit are competitive at that
budget.* It does **not** license calling that support "the circuit": it is an attribution-weighted column
set measured on 16 harmful fit prompts, not a CircuitKIT path, and nothing yet tests whether a circuit
found by the library lands inside it. That is the next leg (`experiments/p6_circuit.py`) and the reason
H1's wording says *circuit*, which is still unproven.

Also note the AUC column: every repair arm sits at 0.78–0.86 against base 0.999, including the 10 % random
arm (0.861). AUC(d′) is **not** a sensitive measure of what the repair restores — it barely moves between a
working repair and no repair at all. Over-refusal is the axis that discriminates, which is why it, not AUC,
carries the frontier.

## 6. The drift replicates on a second Cohere checkpoint

`results/p4_replicate_CohereLabs__tiny-aya-{global,earth}.json` (`experiments/p4_replicate.py`,
A100-40GB, 626.8 s and 612.4 s):

| model | base | alpha=0 | alpha=2 | alpha=4 |
|---|---|---|---|---|
| tiny-aya-global | 0.758 | 0.217 | 0.417 | 0.733 |
| tiny-aya-earth | 0.733 | 0.250 | 0.367 | 0.458 |

`global` reproduces `p2_dose` exactly, which validates the whole path end-to-end. `earth` drifts and
partially recovers on the same alpha schedule but saturates lower at alpha = 4 (0.458 vs 0.733) — the first
evidence that the recoverable fraction is model-dependent, which is what H3 needs.

`tiny-aya-water` never reported: its container produced no `[done]` line and no result file (the arm is
absent from `results/`). Not a measurement, not claimable; re-run it alone.

## 7. The patches are published as parameters

`artifacts/` (+ `artifacts/manifest.json`, 2,428 B loader `artifacts/load_patch.py`, 5 tests pass):

| file | alpha | modules | columns | scalars | direction |
|---|---|---|---|---|---|
| `abliteration_tiny-aya-global_layer15.npz` | −1.0 | 72 | 470,016 | 962,592,768 | `31148a26…` |
| `repair_global_a4.npz` | +4.0 | 72 | 470,016 | 962,592,768 | `7a1451a3…` |
| `repair_harm10_a4.npz` | +4.0 | 72 | 47,001 | 96,258,048 | `7a1451a3…` |
| `repair_harm2_a4.npz` | +4.0 | 72 | 9,400 | 19,251,200 | `7a1451a3…` |

Column and scalar counts match the sweep's arm records exactly, and the loader round-trips bit-exact. The
two directions carry **different sha256** — the abliterated direction and the repaired one are genuinely
different vectors, which is §2's orthogonality finding in artifact form. Each file's `allowed` block is the
claim's parameter set, not decoration. No drifted checkpoint is published (LSAL §4).

## 9. Circuit leg on the drifted model — runs, but returns an empty circuit

`experiments/p6_circuit.py`, `logs/p6_circuit.log`, `results/p6_circuit.json` (wall 64.8 s, A100-80GB).

What the run established, all measured:

- The drift applies inside the CircuitKIT path: 962,592,768 scalars across 72 matrices, and the
  re-extracted direction is orthogonal to the abliterated one, `cos(d, d') = +0.003467` — consistent with
  p2/p3 (`+1.94e-4`) and p1 (`-2.8e-6`), i.e. the orthogonality is a property of the operator, not of one
  prompt slice.
- `HookedTransformer.from_pretrained(MODEL, hf_model=<drifted HF model>)` loads the port at
  d_model 2048 and does **not** need a second copy on disk — but it does need both copies in VRAM:
  A100-40GB OOM'd at the last 86 MiB, A100-80GB is fine.
- `discover_circuit` executed and returned **0 scored nodes** (`n_nodes_scored: 0`,
  `n_circuitkit_scores: 0`, empty layer histogram). So this run does **not** produce a circuit, and no
  claim is made from it.

Three friction points worth recording, each measured:

1. `image.add_local_*` must come last — `.env()` after it raises
   `InvalidError: An image tried to run a build step after using image.add_local_*`.
2. CircuitKIT hard-imports third-party packages at package-import time (`backends/acdc/types.py:6` imports
   `plotly`), so a minimal image dies with `ModuleNotFoundError` before any discovery code runs.
3. Contrastive tasks require **token-aligned** prompt pairs: positional pairing of the fit sets gave
   `All 16 records were dropped during alignment (strategy='filter')`, and even exact Cohere-token-length
   matched pairs gave `dropped_misaligned: 24` — the alignment pass evidently tokenises with a different
   tokenizer than the model's, so `align_strategy="none"` (the library's own suggestion) is what lets
   records through.

Consequence for the thesis, stated plainly: the repair claim in §5 rests on **restricted, attribution-
weighted supports**, not on circuits. The circuit leg is implemented and reaches discovery, but it has not
yet produced the signed circuit mass `M` that H1/H2 were pre-registered against.

### 2026-11-03 — Generation-length sensitivity (task #34) — resolved

P21 regenerated **all 640 frozen answers at 512 tokens** with the same four arm states,
seed and batch size. Every answer reproduced its saved 128-token prefix exactly (no
sampling drift), every state hash and reset matched p16/p17, and the stored diagnostics
reproduced independently.

Artifacts: `results/p21_safety_length_full_20261103_001722.json`,
`results/p21_safety_length_full_checks_20261103.json`,
`results/p19_safety_judges_20261103_002850.json` (unchanged judges),
`results/p21_length_analysis_20261103.json`. Logs `logs/p21_safety_length_full.log`,
`logs/p21_length_followup.log`.

Censoring changed a lot; the frontier did not:

| arm | capped w/o EOS, 128 → 512 (harmful) | majority-3 ASR 128 → 512 | Qwen benign refusal 128 → 512 |
|---|---|---:|---:|
| base | 40/120 → 16/120 | 0.0500 → 0.0583 | 0.075 → 0.075 |
| drift | 85/120 → 47/120 | 0.2583 → 0.2167 | 0.050 → 0.050 |
| harm2_a4 | 44/120 → 3/120 | 0.0583 → 0.0500 | 0.175 → 0.175 |
| global_a4 | 43/120 → 4/120 | 0.0500 → 0.0417 | 0.250 → 0.275 |

Paired 512−128 prompt changes are all within ±0.025 for ASR; benign refusal is exactly
unchanged for base/drift/harm2 and +0.025 [0.000, 0.075] for global. The sparse arm is
therefore not an artefact of truncation: the over-refusal ordering
(harm2 0.175 < global 0.275) survives a 4× budget. Still exploratory — the frozen panel's
calibration gate remains failed, so this does not license a headline ASR claim.

**Erratum (2026-11-03, same day).** The sentence "paired 512−128 prompt changes are all within
±0.025 for ASR" is wrong for the drift arm. The measured paired differences are base
+0.0083 [0.000, 0.025], drift **−0.0417 [−0.092, 0.008]**, harm2 −0.0083 [−0.025, 0.000],
global −0.0083 [−0.025, 0.000] (`results/p21_length_analysis_20261103.json`,
`exploratory_scores.paired_512_minus_128.<arm>.majority3`). Every interval contains zero, so the
conclusion (no reordering) stands, but the drift moves *down* with a longer budget — the drifted
model has more room to hedge, which further weakens any "abliteration made it unsafe" reading.

### 2026-11-03 — Judge calibration gate (task #35) — PASSED on external labels

`results/p22_judge_validation_20261103_020927.json` (log `logs/p22_judge_validation_aegis1.log`):
500 rows from `nvidia/Aegis-AI-Content-Safety-Dataset-1.0` (split `test`, rows with ≥3 non-empty
`labels_0..4`, majority Safe → safe else unsafe, ties dropped, 250 unsafe / 250 safe, seed 0), scored
by the frozen panel with unchanged templates and strict parsers.

| judge | accuracy | sensitivity (unsafe) | specificity (safe) |
|---|---:|---:|---:|
| HarmBench-Mistral-7B | 0.625 | 0.356 | 0.896 |
| Qwen3Guard-Gen-8B | 0.796 | 0.608 | 0.984 |
| Granite-Guardian-3.2-5B | 0.818 | 0.696 | 0.940 |
| majority of 3 | 0.782 | 0.580 | 0.984 |

The panel is conservative, not calibrated in the flattering direction: it almost never flags a safe
text (0.984) and misses ~42 % of Aegis "unsafe" rows (0.580). Absolute harm rates are therefore a
**lower bound**; arm-to-arm comparisons on identical prompts remain sound. The earlier Aegis 2.0
route failed (`unsafe=0 safe=4682`: human response labels exist only for safe rows) and the Aegis 1.0
count-first route failed (`unsafe=0 safe=105`: labels are category names, not Safe/Unsafe). No judge
was tuned or replaced after seeing these labels.

### 2026-11-03 — CircuitKIT discovery on the drifted model (tasks #28/#6)

`results/p23_circuit_20261103_022433.json`. Fit-only discovery (the p6 leak is fixed), canonical
drift (`962,592,768` scalars, `cos(d,d')=+1.027e-03`), ARA node naming `A<layer>.<head>` / `MLP <layer>`
(612 nodes, 0 unmapped), scores read from the side-car `circuit_scores.json` rather than `result.graph`
(which is why p6 always read an empty graph).

Signed write mass along `d'`: **global 3.949, harm2 3.949, harm10 3.949** — the top 2 % of
harm-ranked columns already carries essentially all of it. Export: `repair_circuit_a4.npz`,
sha256 `96c659a056648c3cf1c076e880ad426f10b2a9dd07d5d6912233cd00f0ff3a82`, 150,994,944 edited
scalars, basis `per_head`. **Evaluation of this arm is blocked** (Modal spend limit); no circuit
number is reported.

### 2026-11-03 — Held-out and multi-model generations (tasks #27/#31/#32 partial)

`results/p24_heldout_20261103_021508.json`: all four arms on 120 held-out harmful + 40 held-out
benign prompts (`corpus_fit.heldout_rows(seed=20261103)`, zero prompt overlap with evaluation).
Generation and state checks pass (`results/p24_heldout_20261103_021508_state_checks.json`: base
0 scalars, drift 962,592,768/0, harm2 962,592,768/19,251,200, global 962,592,768/962,592,768, resets
exact). Judging is blocked. `results/p25_tiny-aya-earth_20261103_024506.json` has drift/harm2/global
and `..._023229.json` has base, so earth is complete; water is missing the global arm
(`..._024507.json` + `..._023231.json`). All p25 arms are regex-only, no judge verdicts.

### 2026-11-03 — Submission artifacts

Parquet on the Hub (`scartissue-tinyaya-safety-corpus`, `scartissue-tinyaya-results`: 1,280
generations / 3,864 verdicts / 4,000 capability rows / patches / frontier; frontier recomputed from
the published rows reproduces the analysis exactly). Static Space `amugoodbad229/scartissue` served
with the frontier chart, three-arm completion viewer, calibration table and `report.pdf`.
`results/manifest.json` (17 keys), `results/lexsi_provenance.json`, `results/provenance_final.json`
(corrects the per-ID `split_counts` bug without editing the historical file). `report/report.pdf`
5 pp., regenerated from `report/make_report_data.py`; `results/novelty_check_20261103.json` positions
the claim against AMRA (arXiv:2608.18093, a *pre-emptive* defence) and the refusal-direction
literature.

**Hard blocker:** `Workspace ac-pBhL2P5HANn5nS1Gp9kaLl has exceeded its spend limit` — blocks
CPU functions too, so circuit-arm judging, held-out judging, multilingual and the second-model
frontier cannot run until the workspace limit is raised (task #36).

### 2026-11-03 — Pre-submission verification (30 assertions) and two corrections it forced

`experiments/verify_submission.py` → `results/submission_verification.json`, **30/30 passing**. It
recomputes the published table from `results/`, recomputes each judge's accuracy/sensitivity/
specificity from `results/p22_judge_validation_20261103_020927.json`, checks the signed-mass
concentration, the manifest's 17 keys, the corpus roles, the report hash, and that every public URL
returns 200. Two real defects were found this way, not by reading:

1. **The "refusal" column is a judge label, not a regex.** The published 0.842 / 0.500 / 0.883 /
   0.883 are `qwen_harmful_refusal_diagnostic` (Qwen3Guard's `Refusal:` field). The regex marker in
   `experiments/safety_core.summarise` reads **0.683 / 0.175 / 0.450 / 0.583** on the same 512-token
   answers — it misses refusals that open with a preamble, and it understates the drift's collapse by
   7 points. The numbers were always right; the *label* was ambiguous. Every artifact now states the
   source, and the divergence is stored rather than hidden (it is why a regex is a diagnostic only).
2. **Per-patch sha256 was null for four of five patches.** `artifacts/manifest.json` predates hash
   recording, and the parquet/provenance inherited the nulls while the report promised per-file
   hashes. `experiments/fill_patch_sha256.py` hashes the *published* bytes (all four local copies
   match the Hub copies: `abliteration 505b243a`, `repair_global c90fed93`, `repair_harm10 c24e4fb7`,
   `repair_harm2 26c537b3`; `repair_harm4 dbe85467`), rewrites `data/patches.parquet`, and
   `results/provenance_final.json` now reads `results/patch_checksums.json` instead of the stale
   manifest.

Also fixed: the parquet tables live under `data/` on the Hub (the verification had been probing the
repository root and reporting 404s), and both dataset cards now declare `configs:` with `data_files:`
so the Hub's parquet viewer can index them — the earlier cards had no front matter, which is exactly
the failure the Hub's "publish parquet" advice targets. The datasets-server indexing API itself was
returning `The server is busier than usual` throughout, so viewer indexing could not be confirmed
from this machine; the files were verified readable instead (763 / 1,280 / 3,864 / 4,000 / 5 / 8 rows).

One further correction to the record: the Space README had been overwritten with the repository
README, losing its `sdk: static` front matter. It is restored and the Space reports `sdk static`,
`stage RUNNING`, with `index.html`, `report.pdf` and `frontier.csv` all serving 200.

### 2026-11-06 — The held-out prompts replicate the trade-off (task #21) — exploratory

Judging had already been paid for: `results/p19_safety_judges_20261103_023754.json` holds three
judges × 644 records on `results/p24_heldout_20261103_021508.json` (its `safety_source_sha256` is
`7fd67ab5…`, checkable). What was missing was the audit, not the judgments, so
`experiments/analyse_heldout.py` adds it and `experiments/analyse_safety.py` gained a `roles_of()`
helper so the same estimator accepts `heldout_malicious`/`heldout_benign` — the old output is
byte-identical.

Checks before any number: the judge payload names the generation file whose bytes still hash to the
recorded `7fd67ab5…`; the run passed with no errors; the separate role-aware state correction names
the same run and all four arms pass; the stored status is still `failed` and the analysis refuses to
rename it; every scored non-control `corpus_id` is in `1-curate/corpus/corpus_split.json` under
`unused`, 160 of them, with **zero overlap** with the fit or evaluation slices.

| arm | harm (majority-of-3) | over-refusal (benign) |
|---|---|---|
| base | 0/120 = 0.000 [0.000, 0.031] | 8/40 = 0.200 [0.105, 0.348] |
| drift | 38/120 = 0.317 [0.240, 0.404] | 5/40 = 0.125 [0.055, 0.261] |
| repair_harm2 | 3/120 = 0.025 [0.009, 0.071] | 13/40 = 0.325 [0.201, 0.480] |
| repair_global | 1/120 = 0.008 [0.001, 0.046] | 20/40 = 0.500 [0.352, 0.648] |

Paired bootstrap over prompts: drift − base harm **+0.317 [0.233, 0.400]**, over-refusal −0.075
[−0.175, 0.025] (not resolved); harm2 − drift **−0.292 [−0.375, −0.208]** with over-refusal
**+0.200 [0.075, 0.325]**; global − drift **−0.308 [−0.392, −0.225]** with over-refusal **+0.375
[0.200, 0.550]**; harm2 − global harm +0.017 [−0.017, 0.050] (not resolved) with over-refusal
**−0.175 [−0.325, −0.050]**.

So the exploratory trade-off is not an artefact of the prompts the arms were selected on: harm rises
with drift and each repair buys it back with over-refusal, and the *sparse* repair over-refuses less
than the global one at no resolved harm cost. It is still **not confirmation** — the judge panel
failed calibration on the project distribution, the arms were chosen on the earlier set, the
intervals exclude judge error, and no multiplicity adjustment is applied. Output:
`results/heldout_analysis_20261106.json`. A defect found while wiring it: the readiness ledger marked
`heldout-comparison` evaluated from a *declared* status rather than from file availability, so a tree
missing those files still claimed an evaluation; claims now downgrade to `pending` when any of their
sources is absent (`experiments/verify_submission.py`).

### 2026-11-06 — Circuit body recovered; confirmatory design frozen (tasks #24, #23)

Two gaps closed without spending compute.

**The circuit patch is no longer remote-only.** `scartissue-hf` on the old workspace is
spend-blocked for functions but still readable by the CLI, so
`modal volume get scartissue-hf artifacts/tiny-aya-global/repair_circuit_a4.npz` returned the body:
471,102 bytes hashing to `96c659a056648c3cf1c076e880ad426f10b2a9dd07d5d6912233cd00f0ff3a82`, exactly
the value recorded at export. Under the hardened loader it is structurally sane — unit direction,
finite factors, 36 `self_attn.o_proj` modules, 73,728 sorted in-range columns, and
73,728 × 2,048 = 150,994,944 edited scalars, matching the metadata. That establishes *what the bytes
are*, not that the discovery run produced them, and **not** any effect: no generation, judging or
capability run exists for this arm, so `circuit-intervention` stays `pending`
(`results/circuit_body_recovery_20261106.json`).

**The confirmatory comparison is frozen before it exists.** With the held-out set analysed, the
untouched pool is countable: of 371 rows labelled `unused`, the held-out generation used 160, leaving
**57 harmful and 154 benign** prompts. So a confirmatory harm run is n = 57 — not the 120 the
exploratory run used, and not something to discover after the fact.
`results/confirmatory_spec_20261106.json` (self hash `67e4fed8…`, built by
`experiments/freeze_confirmatory.py` from seed 20261106) fixes the four Holm-corrected contrasts, a
matched-budget random-column control (the circuit edit costs 150,994,944 scalars against the global
repair's 962,592,768, so comparing it to an unrestricted global repair would credit restriction for
what budget bought), the judge-calibration / state-check / prompt-hash gates, and the stopping rules.
`_research/AMENDMENT-01-20261106.md` explains it; `PROTOCOL.md` is untouched.

**Decision on Track 2b.** Not justified. The circuit arm has a verified body and no measured effect;
the confirmatory spec that would have to be executed first cannot run until the judge panel passes
calibration on external human labels, which it has not. A circuit claim therefore remains out of the
write-up, and the workbench continues to answer `insufficient evidence` for all four arms.

**A defect found while wiring the ledger.** `heldout-comparison` was marked `evaluated` from a declared
status rather than from file availability, so a tree missing those files still claimed an evaluation.
Claims now downgrade to `pending` when any of their sources is absent.

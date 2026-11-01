# Findings 01 — the drift is real, the repair is indiscriminate

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
   harmful set against 0.758 before the drift — **97 % of the original refusal rate recovered** by a
   weight-space edit that never sees a training example.
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

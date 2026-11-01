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

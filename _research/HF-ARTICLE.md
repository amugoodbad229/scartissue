# Would you ship this repair? A decision workbench for weight-space safety edits

**Track 2a (#1 red-teaming / robustness, #2 adversarial and safety evaluation sets) and Track 2b (#6 repairing
safety drift without retraining).**

Abliteration removes the direction a refusal is written along. It does not remove refusals: re-extracting the
direction on the edited model and amplifying it brings the refusal marker back within a percent of the aligned
model. That is the easy half. The hard half is that every unit of that recovery is paid in over-refusal, because
the re-extracted projection no longer separates harmful from benign prompts.

We built the artefact a **model maintainer** needs at that moment: a workbench that shows four candidate
weight-space repairs side by side on harm, over-refusal, retained utility and intervention size, and returns an
explicit **accept / reject / insufficient evidence** decision under a policy you set, with the evidence behind it
and the reasons for the decision exportable as a record.

**The measured outcome, stated plainly: all four arms return `insufficient evidence`.** That is the correct
answer for evidence in its current state, and the workbench is built to say it rather than to flatter the
candidates. A tool that says "ship it" on unvalidated judging would be worse than no tool.

## What was measured

Subject model `CohereLabs/tiny-aya-global` at revision `af89d219b53ed9b13b8a4645f9c8028973510…`, four arms with
identical generation conditions (512 tokens, temperature 0.7, seed 0, 120 harmful + 40 benign prompts):

| arm | repair-selected scalars | harm-ASR (majority of 3) | benign over-refusal | GSM8K (lm-eval, 5-shot) |
|---|---:|---:|---:|---:|
| `base` (aligned) | 0 | 0.058 | 0.075 | 0.590 |
| `drift` (abliterated) | 962,592,768 | 0.217 | 0.050 | 0.560 |
| `harm2_a4` (2 % support) | 19,251,200 | 0.050 | 0.175 | 0.518 |
| `global_a4` (full support) | 962,592,768 | 0.042 | 0.275 | 0.438 |

Harm comes from a pinned three-judge panel (HarmBench-Mistral-7B, Qwen3Guard-Gen-8B, Granite-Guardian-3.2-5B,
threshold 0.5, majority of three). Capability comes from lm-eval only. A refusal-marker regex is reported as a
diagnostic and never as an ASR estimate.

Two things the numbers do **not** say, because earlier drafts of this project claimed them:

- **The judge panel failed its calibration audit.** On 500 Aegis rows it has specificity 0.984 and sensitivity
  0.580, and the calibration run rendered the HarmBench template with an empty `[BEHAVIOR]` slot — the
  classifier was conditioning on nothing. Every harm figure above is therefore exploratory.
- **Orthogonality is not deletion.** The re-extracted direction is nearly orthogonal to the removed one
  (`cos(d, d′) ≈ 0`), which shows the ablation direction was removed. It does not show that harm-relevant
  representation survives elsewhere. The surviving evidence for that narrower claim is `AUC(d′)` falling
  0.999 → 0.831: the discriminator degraded, it did not vanish.

## Held-out check

160 prompts that no arm trained on (proved 0 overlap with the fit and evaluation splits, all labelled `unused`)
replicate the trade-off: harm 0/120 base, 38/120 drift, 3/120 sparse, 1/120 global; over-refusal 8/40, 5/40,
13/40, 20/40. Still exploratory — same failed calibration, and a page-level exchangeability argument, not a
fresh sampling argument.

## What is in this release

- **Workbench** — static, no inference, no key: policy in, decision and reasons out, JSON and readable export.
- **Parameters, not checkpoints** — every repair ships as `.npz` factors plus a loader that hash-checks, bounds-
  checks and preflights every column before mutating a single weight.
- **Evidence chain** — bundle builder, held-out analysis, confirmatory spec frozen before any run, and a verifier
  that checks provenance and structure (it never certifies a number, and `submission_ready` is false by design).
- **Human audit** — a gate registered before any label existed, applied to a blind 180-item sample from this
  project's own generations. Until it passes, nothing here is a headline result.

Track 2b is deliberately **not** claimed: the circuit-restricted patch body was recovered and hash-verified
(150,994,944 scalars over 36 `o_proj` modules) but never generated, judged or capability-tested, so no circuit
effect is asserted.

Nothing here is production-safety certification, and no drifted checkpoint is hosted.

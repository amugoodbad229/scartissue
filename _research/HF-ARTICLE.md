# Would you ship this repair? A decision workbench for weight-space safety edits

**Track 2a (#1 red-teaming / robustness, #2 adversarial and safety evaluation sets) and Track 2b (#6 repairing
safety drift without retraining, #9 locating where safety lives).** Circuit intervention and tamper results
support those statements; they are not extra boxes beyond the form's two-per-track limit.

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

Two of the 2b answers are negative, and both are reported as such. Localisation holds at the direction level
(one layer-15 direction carries AUC 0.9988 of the refusal separation, 0.8306 after it is removed) but not at the
node level: at a matched edit budget the discovered circuit's 36 output matrices move the refusal marker by 0.000
against two blind controls. Circuit-level repair (2b #10) was then actually run: restricted to those write columns
(150,994,944 scalars), generated and judged under the frozen protocol, it leaves harm at 0.283 against the drifted
model's 0.217 (paired +0.067 [0.008, 0.125]) and over-refusal at 0.125 against 0.050. Restricting the edit to the
discovered circuit does not remove harm — the matched-budget repairs do (0.050 and 0.042). Tamper resistance (2b
#12) is a measured weakness, not a protection: one direction removes 0.542 of the refusal marker at no measured
GSM8K cost, and detection needs the direction.

Nothing here is production-safety certification, and no drifted checkpoint is hosted. The human-labelled
project-distribution audit is still pending.

- [Open the workbench](https://amugoodbad229-scartissue.static.hf.space/)
- [Code and reproduction](https://github.com/amugoodbad229/scartissue)
- [Repair-parameter collection](https://huggingface.co/collections/amugoodbad229/scartissue-repair-parameters)
- [Evaluation-evidence collection](https://huggingface.co/collections/amugoodbad229/scartissue-evaluation-evidence)

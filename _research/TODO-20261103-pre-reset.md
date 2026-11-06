# ScarTissue — task list (mirrors the on-screen task tool)

Updated 2026-11-03, submission-ready state. Numbers come from `results/`; regex refusal is a
diagnostic, capability requires lm-eval, ASR requires the contracted judge path.

## Where the project stands

**Submission (Track 2a) is complete and public.** Report `report/report.pdf` (5 pp.), write-up
`WRITE-UP.md`, entry text `SUBMISSION.md`, parquet on the Hub, static Space deployed, provenance
bundle published.

Headline (512-token budget, `results/p21_length_analysis_20261103.json` +
`results/p16_capability_analysis_20261102.json`):

| arm | edited scalars | refusal | over-refusal | harm-ASR | GSM8K |
|---|---:|---:|---:|---:|---:|
| base | 0 | 0.842 | 0.075 | 0.058 | 0.590 |
| drifted | 962,592,768 | 0.500 | 0.050 | 0.217 | 0.560 |
| 2 % repair | 19,251,200 | 0.883 | 0.175 | 0.050 | 0.518 |
| global repair | 962,592,768 | 0.883 | 0.275 | 0.042 | 0.438 |

Claim: the drift deletes the refusal read-out *and* the discrimination behind it (cos(d, d′) =
1.0×10⁻³); refusal is recoverable weight-only; and the capability price is set by the **support** of
the edit (+0.080 GSM8K for the 2 % support over the global one, 95 % CI [0.038, 0.122]), while
harm-ASR cannot tell the two repairs apart.

## Blocked — needs the workspace owner, not more code

- [ ] **#36 Unblock the Modal workspace spend limit.** Verified 2026-11-03: every run, including a
  CPU-only `ping()`, returns `Workspace ac-pBhL2P5HANn5nS1Gp9kaLl has exceeded its spend limit`.
  This is the only thing standing between the project and its remaining measurements. Raise the
  limit in the Modal dashboard (Billing → spend limit), then the four tasks below can run.

Blocked on #36, in priority order:

- [ ] **#27 Run independent confirmation after exploratory selection.** Generations exist on 160
  held-out prompts (`results/p24_heldout_20261103_021508.json`, state-checked in
  `..._state_checks.json`); only the judging pass is missing (`logs/p19_heldout_judges.log`).
- [ ] **#31 Measure the Aya Expanse 8B repair frontier.** Patches buildable; needs GPU time.
- [ ] **#32 Complete the remaining Cohere cohort.** `tiny-aya-earth` has all four arms
  (`results/p25_tiny-aya-earth_20261103_024506.json` + `..._023229.json`); `tiny-aya-water` is
  missing the global arm; neither is judged.
- [ ] **#20 Replace the invalid multilingual projection run with a validated behavioural
  evaluation.** Harness exists (11 languages); the old projection files were invalid (7,080
  non-finite values) and are withdrawn. #24 (per-language repair/refusal) follows from it.

## Completed

- [x] **#1–#4** Replicate runs (earth/water), budget sweep on harm-weighted support, findings appended.
- [x] **#6 Compute signed circuit mass per arm.** `results/p23_circuit_20261103_022433.json`: mass
  3.949 total and 3.949 in the top 2 % of columns — the concentration that explains the frontier.
- [x] **#7 Export drift + repair patches.** Factorised `.npz` + `allowed{modules,layers,heads}` +
  per-file sha256; published as parameter datasets, never a drifted checkpoint.
- [x] **#8 CuratorKIT corpus with row-level provenance.** 763 rows, frozen roles
  (`1-curate/corpus/`, `results/provenance_final.json`).
- [x] **#21 Pinned lm-eval capability.** `results/p16_full_20261102_230915.json`,
  `results/p16_capability_analysis_20261102.json`.
- [x] **#22 Verified explanation PDF.** `report/report.pdf` regenerated end-to-end by
  `report/make_report_data.py` (facts.tex, table_arms.tex, two figures).
- [x] **#23 ROI table.** `results/roi_table_20261103.json`.
- [x] **#25 Audit of inherited claims.** `results/audit_20261102.json`,
  `results/diagnostic_frontier_v2.json`; errata recorded in FINDINGS/MEASUREMENT.
- [x] **#26 Toolkit provenance end-to-end.** `results/toolkit_provenance.json`: CuratorKIT
  `dd9e8353`, CircuitKIT `e71d5e17`, AuditKIT `71316da3` chained; import call sites recorded.
- [x] **#28 CircuitKIT discovery on the drifted model.** Fit-only, ARA node names, signed scores read
  from the side-car; patch `repair_circuit_a4.npz` published (evaluation blocked, see above).
- [x] **#29 Public Space + licence-compliant demo.** `spaces/amugoodbad229/scartissue`, static,
  precomputed JSON only; licence §4 respected (no drifted checkpoint published).
- [x] **#30 Valid evaluation bundle with judge scores.** `results/p19_safety_judges_20261103_002850.json`
  + `results/p21_length_analysis_20261103.json`; calibration status recorded honestly (see #35).
- [x] **#33 Paper novelty and claim–evidence map.** `results/novelty_check_20261103.json` (closest
  work: AMRA, arXiv:2608.18093, a *pre-emptive* defence), `report/report.pdf` §3–§8 map each claim to
  a file; held-out confirmation is listed as pending rather than implied.
- [x] **#34 Generation-length sensitivity.** 128 → 512 tokens, prefixes reproduced exactly, ordering
  survives; erratum on the ±0.025 sentence recorded in FINDINGS.
- [x] **#35 Independent judge validation.** `results/p22_judge_validation_20261103_020927.json`:
  majority-of-3 accuracy 0.782 / sensitivity 0.580 / specificity 0.984 on 500 human-labelled Aegis rows.

## Track decision (unchanged, verified 2026-11-02)

`results/competition_rules_verified_20261102.json`: only the final submission is judged, so this is
one Track **2a**-led entry (weight-space attack + robustness/over-refusal reporting), with the repair
frontier as the follow-on finding. Track 2b statement 2 (drift after fine-tuning) is not satisfied by
abliteration and is not claimed.

## Rules in force

- PROTOCOL.md frozen; ASR only from the judge path; capability only from lm-eval; full completions
  stored, never samples; prompt-level bootstrap intervals, no arbitrary significance thresholds.
- `uv` only. All CUDA on Modal. `../ScarTissue` read-only; track repos imported via PYTHONPATH, never
  installed or vendored.

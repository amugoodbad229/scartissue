# ScarTissue — a repair-decision workbench for refusal-drifted checkpoints

Track 2 entry for the **Lexsi AI Safety & Alignment Hackathon** (Nov 2026). Everything that runs,
produces a number, or gets submitted lives here. The research archive next door in `../ScarTissue`
holds the frozen protocol, the source reads, the licence analysis and the audit trail; it is cited,
never edited.

**The user and the decision.** A model maintainer preparing to release a fine-tune of an abliterated
Cohere checkpoint has to decide whether a candidate repair is good enough to ship. `space-static/`
answers that as a reviewable decision: load an evidence bundle, set your own harm / over-refusal /
utility / support bounds, and get **accept**, **reject** or **insufficient evidence** with the
intervals, the reason and a full judge replay behind it. On the evidence shipped here all four arms
come back `insufficient evidence` — correctly.

## Read this first

**Not submission-ready.** The reset roadmap is in [`TODO.md`](TODO.md). The current target is a
research workbench for comparing candidate repairs under explicit harm, over-refusal and utility
constraints, including an **insufficient-evidence** outcome. The local workbench implements policy
comparison, private bundle import, full replay and audit export; validation is in progress and it is
not publicly deployed. The public demo remains a historical viewer, not a production-safety certificate.

**Evidence status:** the stored safety analysis has `headline_eligible: false`, and its judge run
retains `calibration_status: failed`. Treat the safety estimates below as exploratory, not a cleared
release gate. External Aegis calibration does not erase the failed controls or validate the panel on
these project completions. See `results/p21_length_analysis_20261103.json` and
`results/p19_safety_judges_20261103_002850.json`. The held-out panel (`results/p19_safety_judges_20261103_023754.json`) has now been analysed on prompts the arm selection never touched: `experiments/analyse_heldout.py` checks split disjointness (160 prompts, all in the untouched slice, zero overlap with fit/eval) and computes paired-bootstrap differences, written to `results/heldout_analysis_20261106.json`. Majority-of-three harm rose from 0/120 to 38/120 under drift, and each repair cut it back (3/120 sparse, 1/120 global) while raising over-refusal (13/40 and 20/40 against 5/40 for drift). This replicates the trade-off on new prompts but stays **exploratory**: the judge panel failed calibration, so it is not confirmation, and the project-distribution human audit remains pending. Generated or judged answers alone are not confirmation. The generated readiness ledger at `results/readiness_ledger.json` separates evaluation from sign-off.

**Software validation:** the baseline passed on the replacement Modal workspace; see
`results/modal_validation_20261105T155145197754Z.json`. The workbench, structural verifier and
readiness ledger then passed in `results/modal_validation_20261105T190714562351Z.json` (129 tests in
both invocations, 21/21 integrity checks); earlier failed runs are retained
(`results/modal_validation_20261105T161907293974Z.json` lost a disclosure check,
`...T164156473961Z.json` a ledger bootstrap check). Passing software checks do not clear the
scientific blockers. The old workspace spend-limit error is historical. The report PDF and public
links are historical releases; local claim corrections have not been rebuilt or deployed.

| artifact | where |
|---|---|
| Decision workbench (open `space-static/index.html`, no build) | [`space-static/`](space-static/index.html) |
| Claim ledger (what is evaluated, pending or invalid) | [`results/readiness_ledger.json`](results/readiness_ledger.json) |
| How to reproduce, cheapest tier first | [`REPRODUCE.md`](REPRODUCE.md) |
| Historical report PDF (not rebuilt after claim corrections) | [`report/report.pdf`](report/report.pdf) |
| Write-up (method, numbers, what is *not* claimed) | [`WRITE-UP.md`](WRITE-UP.md) |
| Submission text (requirement-by-requirement) | [`SUBMISSION.md`](SUBMISSION.md) |
| Task state, dependencies and blockers | [`TODO.md`](TODO.md) |
| Public demo (three columns: base / drifted / repaired) | https://amugoodbad229-scartissue.static.hf.space |
| Corpus, 763 rows with row-level provenance (parquet) | https://huggingface.co/datasets/amugoodbad229/scartissue-tinyaya-safety-corpus |
| Generations, verdicts, capability, frontier, patches (parquet) | https://huggingface.co/datasets/amugoodbad229/scartissue-tinyaya-results |
| Drift + repair parameters, `.npz` with `allowed` block | https://huggingface.co/datasets/amugoodbad229/scartissue-tinyaya-abliteration-patches |

## The question

Abliteration projects a refusal direction out of the model's write matrices — 962,592,768 scalars on
`CohereLabs/tiny-aya-global`, without gradient training. Direction fitting uses calibration prompts. We ask what that edit actually destroys, and what
it costs to put back.

Measured at a 512-token generation budget, four arms, same prompts, same judges:

| arm | edited scalars | refusal | over-refusal | harm-ASR | GSM8K (lm-eval) |
|---|---:|---:|---:|---:|---:|
| base | 0 | 0.842 | 0.075 | 0.058 | 0.590 |
| drifted | 962,592,768 | 0.500 | 0.050 | 0.217 | 0.560 |
| 2 % repair | 19,251,200 | 0.883 | 0.175 | 0.050 | 0.518 |
| global repair | 962,592,768 | 0.883 | 0.275 | 0.042 | 0.438 |

Read the columns for what they are: **refusal** is Qwen3Guard's `Refusal:` field on harmful prompts
(a judge label), **over-refusal** is the same field on 40 benign XSTest prompts, **harm-ASR** is the
majority of three harm judges at a strict `> 0.5` threshold, and **GSM8K** is lm-eval exact match —
capability is never scored by a judge. A plain regex refusal marker on the same answers reads
0.683 / 0.175 / 0.450 / 0.583, so it is stored as a diagnostic and never used as a claim.
`results/submission_verification.json` records historical consistency/availability checks, not a
fresh validation or submission-readiness verdict. The verifier still needs replacement; do not treat
HTTP success or agreement with hardcoded metrics as evidence validity.

Three findings, in the order they matter:

1. **The extracted representation changes; complete loss of discrimination is not established.**
   The near-orthogonal re-extracted direction in `results/p23_circuit_20261103_022433.json` describes
   a particular projection, not every safety-relevant representation in the model. The specific
   operator that amplifies the removed projection of the drifted writes is a no-op by construction;
   this does not rule out restoring an old direction from base weights.
2. **Refusal is recoverable in weight space, and the recovery is cheap to trigger but expensive to
   aim.** Every repair recovers the refusal rate; what differs is the over-refusal bill — 0.175 for a
   2 % support, 0.275 for a global one.
3. **Support size sets the capability price.** The 2 % support keeps +0.080 GSM8K over the global
   repair (95 % CI [0.038, 0.122]) at the same refusal and a lower over-refusal rate, while harm-ASR
   is 0.050 versus 0.042 in the exploratory judge analysis. Lack of a detected difference does not
   establish equivalent safety. This is a measured trade-off, not yet a validated admission rule.

Reported honestly: the discovered-circuit arm was generated but **not** judged (Modal workspace spend
limit), so no circuit number appears in any claim; the held-out, multilingual and second-model
confirmations are listed as blocked, not implied.

## How the chain is built

```
CuratorKIT   → 763-row corpus (StrongREJECT + XSTest), frozen roles, row-level provenance
SafeTune     → refusal-direction audit, then abliteration (the drift)  [operator reimplemented]
CircuitKIT   → discovery of the refusal-writing circuit ON the drifted model
ours         → repair at a matched edit budget, five supports, one frontier
AuditKIT     → native re-scoring of stored answers with a fixed 3-judge panel
```

Track repos are imported from `../ScarTissue/repos/` via `PYTHONPATH` and pinned by commit
(`results/toolkit_provenance.json`): CuratorKIT `dd9e8353`, CircuitKIT `e71d5e17`, AuditKIT
`71316da3`. The drift itself deliberately contains **no** toolkit code — a 30-line operator — so the
attack result does not depend on the tools being correct.

## Layout

| path | what lives there |
|---|---|
| `1-curate/` | corpus, split, gates, `manifest.json`, `lexsi_provenance.json` |
| `2-drift/` | refusal-direction extraction and abliteration |
| `3-circuit/` | CircuitKIT discovery on the drifted model, node scores, signed mass |
| `4-audit/` | AuditKIT evaluation, judge panel, lm-eval capability |
| `artifacts/` | published `.npz` patches + manifest with per-file sha256 and `allowed` block |
| `experiments/` | every Modal app and analysis script; one file per measurement |
| `results/` | every number in the report, plus its provenance and audit trail |
| `report/` | LaTeX source, `make_report_data.py` (regenerates all macros/figures from `results/`) |
| `space-static/` | the precomputed public demo |

## Reproduce

Do not launch the research sequence below as a batch. It needs the read-only pinned toolkit
checkouts, model access and an explicit compute reservation. All testing/validation must run on
Modal; the current workspace is blocked. The bounded CPU runner is
`experiments/modal_validation.py` (not yet remotely validated). No automatic quota increase is authorized.

```bash
uv sync                                    # experiments/runtime, locked; Python 3.12
uv run python -m modal run experiments/p8_corpus.py  # corpus + provenance
uv run python -m modal run experiments/build_patches.py --model CohereLabs/tiny-aya-global
uv run python -m modal run experiments/p23_circuit.py
uv run python -m modal run experiments/p21_safety_length_full.py
uv run python -m modal run experiments/p19_score_safety.py --safety-file results/<generation-run>.json
uv run python -m modal run experiments/p16_lm_eval.py
uv run python report/make_report_data.py && (cd report && latexmk -pdf report.tex)
```

Every run writes full completions, token ids, stop diagnostics, patch hashes and the base-state hash,
so any arm can be re-scored without being regenerated.

## Licence and scope

Cohere weights are **LSAL 1.2/1.1, research use**. §2 forbids selling or hosting the model as a
product and §4 forbids deploying drifted checkpoints, so this project publishes **parameters**, never
a drifted checkpoint, and the demo serves precomputed JSON over the aligned model's published
answers. Individual entry; not an organisation.

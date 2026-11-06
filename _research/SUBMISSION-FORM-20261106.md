# Submission form — field-by-field answers

Lexsi AI Safety & Alignment Hackathon, initial stage, deadline **14 November 2026, 23:59 AoE**.
Every number below is traceable to a file in `results/`; the field-by-field provenance table is at the end.
Status of each artifact is stated as it is today, not as it should be.

---

## Abstract (target: under 500 words)

Fine-tuning a model for a useful task quietly removes its safety behaviour: one refusal direction is
enough to take `CohereLabs/tiny-aya-global` from 5.8% to 21.7% harm-ASR while its GSM8K score barely
moves (0.590 → 0.560). The standard repair — push the refusal direction back into the weights — works,
and we measured what it costs.

We abliterate the drift, then compare two weight-space repairs on the same drifted model: a sparse
patch that edits 19,251,200 scalars (2% of the model) and a global patch that edits 962,592,768
(100%). Both restore refusal to 0.883 and cut harm-ASR below the pre-drift model (0.050 and 0.042).
They are not equivalent. Over-refusal of harmless requests rises from 0.050 to 0.175 (sparse) and
0.275 (global), and GSM8K falls to 0.518 and 0.438 — the sparse patch keeps 8.0 percentage points more
task accuracy, a paired bootstrap interval of [3.8, 12.2] points that excludes zero.

We then tested whether that ordering survives on prompts no arm selection ever touched: 160 held-out
prompts, 120 of them harmful, judged majority-of-three by a frozen panel. Drift appears (0 → 38 of
120), both repairs reduce it (3 and 1 of 120), and the sparse repair over-refuses 17.5 points less
than the global one while their harm rates are statistically indistinguishable. The trade-off is real
and it replicates; it is not a claim that safety was restored.

That distinction is the deliverable. Our judge panel **failed calibration** on the project
distribution (sensitivity 0.58, specificity 0.984 on 500 external human-labelled rows), so the
held-out numbers are exploratory and we do not headline them. Instead of publishing a reassuring
score, we ship the decision a model maintainer actually has to make as an offline tool: load a release
candidate's evidence bundle, and get *accept*, *reject*, or *insufficient evidence* with the reason
attached. Run on our own four arms, it returns **insufficient evidence** for every one — correctly.
The bundle carries per-metric confidence intervals, judge agreement, completion-state checks and an
explicit quality gate; a private import re-hashes every prompt and answer before anything is shown,
and the export never contains raw completions.

Alongside it we publish the audited pipeline: a 763-row safety corpus with a published fit/eval split
(CuratorKIT), abliteration and rank-one projection patches as parameters with a loader (SafeTune-style
operator, implemented and tested here), an EAP-IG circuit discovery run on the drifted model
(CircuitKIT), and native scoring with a strict harm metric (AuditKIT). 129 automated checks and a
21-check structural verifier run on Modal over a hash-verified snapshot.

What we do not claim: that safety is restored, that a smaller patch is safe, that any of these arms is
production-ready, or that a restricted circuit edit was validated — its recovered parameters are
published, unevaluated, and the tool says so.

---

## Team account details (fill in yourself)

- **Registered email:** *(the address you registered with)*
- **Discord username:** *(yours)*

---

## Tracks

Tick both:

- **Track 2a: AI Safety and Red Teaming**
- **Track 2b: AI Safety and Correction**

## Track 2a problem statements — pick two, this order

1. **Red-teaming and robustness evaluation** ← primary. Weight-space attack (abliteration) plus
   jailbreak-style harmful prompts and an adversarial control set, reporting robustness and
   over-refusal together rather than harm alone.
2. **Adversarial and safety evaluation sets** ← secondary. The 763-row labelled corpus with a
   published fit/eval/held-out split (`corpus_split.json`) and judge-based scoring; the held-out
   160-prompt slice is exactly "use a benchmark to measure how current open models hold up".

Do **not** tick multi-turn audits or multilingual safety — we measured neither.

## Track 2b problem statements — pick two, this order

1. **Repairing safety drift without retraining** ← primary. Weight-space repair of an abliterated
   model (rank-one projection onto a sparse or global subspace), with task performance measured
   before and after.
2. **Locating where safety lives** ← only if you are willing to defend it. The circuit discovery run
   exists (182 pruned nodes, 612 signed scores) and the circuit-restricted parameters are recovered
   and hash-verified, but **no effect was measured**, so the problem statement's "use them to make a
   targeted fix" is not evidenced. Recommendation: tick **only** #6 and say in the abstract that the
   circuit leg is published unevaluated.

Do **not** tick circuit-level interventions, safety-preserving fine-tuning, unlearning, inference-time
steering, compression, tamper resistance or hallucination detection.

## Repos used — tick three

- **CuratorKIT** — corpus construction with row-level provenance (763 rows), `p8_corpus.py` exports
  through its `CorpusExporter`.
- **CircuitKIT** — EAP-IG circuit discovery on the drifted model, `p23_circuit.py`
  (`discover_circuit`, `build_discovery_config`).
- **AuditKIT** — native evaluation of stored answers against the strict harm metric, pinned read-only
  at commit `71316da345784ff220d5ed59337441a59ab31526`.

**Not SafeTune**: our repair is an audited rank-one operator implemented here
(`artifacts/load_patch.py`), not SafeTune code. Ticking it would be a false claim — say instead that the
method is SafeTune-*style* weight-space repair. Also not AlignTune or AgentTune.

## Models used — tick two

- **Tiny Aya Global** — the model under test (headline arm, abliterated, repaired).
- **Tiny Aya Base** — the undrifted reference (`base` arm of every comparison).

## Artifacts — what exists, and what is still missing

| Slot | Content | Status today |
|---|---|---|
| HF model collection | Parameter patches (`.npz`) + loader: abliteration, `repair_harm2_a4`, `repair_global_a4`, `repair_circuit_a4` | Repos live and public (`amugoodbad229/scartissue-tinyaya-abliteration-patches`, `…-repair-patches`, HTTP 200 verified 2026-11-06); needs a **collection** wrapper and ideally a copy under the new account |
| HF dataset collection | Corpus (Parquet), split, results, judge verdicts, generations | Public (`…/scartissue-tinyaya-safety-corpus`, `…/scartissue-tinyaya-results`, HTTP 200); needs a collection wrapper |
| HF article | Written account of the finding | **Missing** — must be written |
| GitHub repo | Code, notebooks, `REPRODUCE.md` | **Missing** — this tree has no remote. Publishing it is a precondition of the deadline |
| Deployment | Static decision workbench | Space `amugoodbad229/scartissue` returns 200; it is on the spend-blocked old account, so re-host before relying on it |

Licence posture (LSAL §2/§4): parameters are published, no drifted checkpoint is hosted or deployed,
nothing is sold, and the Tiny Aya repository is never vendored — it is added with `PYTHONPATH`.

---

## Provenance for the numbers in the abstract

| Statement | Source |
|---|---|
| harm-ASR 0.058 → 0.217, GSM8K 0.590 → 0.560, refusal 0.842 → 0.500 | `results/p21_length_analysis_20261103.json`, `results/p16_capability_analysis_20261102.json` |
| 19,251,200 / 962,592,768 edited scalars, refusal 0.883, ASR 0.050 / 0.042, over-refusal 0.175 / 0.275, GSM8K 0.518 / 0.438 | `results/p21_length_analysis_20261103.json`, `results/patch_checksums.json` |
| 8.0 pp GSM8K gap, CI [0.038, 0.122] | `results/p16_capability_analysis_20261102.json` |
| held-out 0, 38, 3, 1 of 120; sparse over-refuses 17.5 pp less | `results/heldout_analysis_20261106.json` |
| judge sensitivity 0.58 / specificity 0.984, n = 500 | `results/p18_judge_smoke_20261102_233759.json`, `results/manifest.json` |
| 129 tests, 21 integrity checks | `results/modal_validation_20261105T190714562351Z.json` |
| circuit 182 pruned nodes, 612 signed scores; body sha256 `96c659a0…` | `results/p23_circuit_20261103_022433.json`, `results/circuit_body_recovery_20261106.json` |

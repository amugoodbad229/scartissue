# Submission form — field-by-field answers

Lexsi AI Safety & Alignment Hackathon, initial stage, deadline **14 November 2026, 23:59 AoE**.
Every number below is traceable to a file in `results/`; the field-by-field provenance table is at the end.
Status of each artifact is stated as it is today, not as it should be.

---

## Abstract (target: under 500 words)

Before releasing a safety-edited checkpoint, a maintainer needs to know whether a repair is good
enough under their own policy—not merely whether its average harm score improved. **ScarTissue** turns
that into a reviewable decision: import an evidence bundle, set harm, over-refusal, capability and
edit-support bounds, and receive **accept**, **reject**, or **insufficient evidence**, with intervals,
source hashes and per-prompt judge replay.

We demonstrate the workbench on `CohereLabs/tiny-aya-global` using controlled weight-space
abliteration, not a claim that task fine-tuning caused drift. We compare the aligned model, the
drifted model, a sparse rank-one repair and a support-matched global repair. In the stored run,
exploratory harm-ASR is 0.058, 0.217, 0.050 and 0.042 respectively. The sparse repair retains more
GSM8K strict-match accuracy than the global repair (0.518 versus 0.438; paired difference +0.080,
95% CI [0.038, 0.122]), while benign over-refusal is lower (0.175 versus 0.275). A disjoint
160-prompt held-out evaluation shows the same directional trade-off. These are observations from
this study, not safety guarantees.

The judge panel failed calibration, so all harm results remain exploratory and the project-distribution
human-labelled audit is still pending. The workbench enforces that limitation instead of hiding it:
on the supplied evidence, every candidate returns **insufficient evidence**. Imports verify prompt
and answer hashes; exports omit raw completions; decisions run entirely offline.

A second negative result matters scientifically: restricting the repair to the discovered circuit did
not reduce harm relative to drift (exploratory harm-ASR 0.283 versus 0.217), and this arm has no
capability measurement. We therefore do not claim circuit-level repair, restored safety or a
production-ready checkpoint.

The reproducible chain uses CuratorKIT for the 763-row corpus and frozen splits, CircuitKIT for
discovery, and AuditKIT for native scoring; the rank-one intervention is our own operator. Parameters,
corpus, evaluation records, code and the static demo are public. A fresh pinned Modal run passed 158
tests in both invocation forms and all structural integrity checks.

---

## Team account details (fill in yourself)

- **Team name:** *(your public team name)*
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
   before and after: harm 0.217 → 0.050 / 0.042, GSM8K 0.560 → 0.518 / 0.438.
2. **Locating where safety lives** ← tick it, and let the negative half be visible. Direction level is
   measured (AUC 0.9988 → 0.8306 after removing one direction at layer 15). Node level was tested and
   **failed**: at a matched edit budget the discovered circuit's 36 output matrices change the refusal
   marker by 0.000 against two blind controls. `results/localisation_analysis_20261106.json`.

**Stop at two.** Do not also tick tamper resistance or circuit-level interventions. Their measurements
can support the two selected statements in the abstract and demo, but the form allows at most two.
The circuit arm was generated and judged: harm 0.283 vs drift 0.217 (paired +0.067 [0.008, 0.125]),
over-refusal 0.125 vs 0.050. Restricting the edit to the discovered circuit did not remove harm.
This is exploratory, and no capability run exists for this arm
(`results/circuit_arm_analysis_20261106.json`). The measured tamper weakness is not a protection
(`results/tamper_analysis_20261106.json`).

Do **not** tick safety-preserving fine-tuning, unlearning, inference-time steering, compression, or
hallucination detection.

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

## Models used — tick one for the submitted study

- **Tiny Aya Global** — the model under test; its undrifted weights are also the `base` arm.

Do **not** tick Tiny Aya Base: that is a different model, not the name of our baseline arm. Earth and
Water appeared in incomplete secondary exploration, not this submitted comparison; do not add them
to make the entry look broader.

## Artifacts — what exists, and what is still missing

| Slot | Content | Status today |
|---|---|---|
| HF model collection | **https://huggingface.co/collections/amugoodbad229/scartissue-repair-parameters** — model repo `scartissue-tinyaya-repair-parameters`, plus patch datasets. Public; canonical URL verified |
| HF dataset collection | **https://huggingface.co/collections/amugoodbad229/scartissue-evaluation-evidence** — results, evals, corpus and the Space. Public; canonical URL verified |
| HF article | Written account of the finding | **Missing** — must be published before submission |
| GitHub repo | **https://github.com/amugoodbad229/scartissue** (public, `master`; `REPRODUCE.md` at root; full run dumps are in the HF datasets) |
| Deployment | Static decision workbench | **Live and verified.** Direct app: **https://amugoodbad229-scartissue.static.hf.space/**; evidence agrees with the local bundle except `created_at` (`results/live_demo_verification_20261107.json`) |

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
| 158 tests in both invocations, structural integrity checks passed | `results/modal_validation_20261107T143612137007Z.json` |
| circuit-restricted arm did not reduce harm; capability not measured | `results/circuit_arm_analysis_20261106.json` |
| circuit 182 pruned nodes, 612 signed scores; body sha256 `96c659a0…` | `results/p23_circuit_20261103_022433.json`, `results/circuit_body_recovery_20261106.json` |

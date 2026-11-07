# ScarTissue — a repair-decision workbench for refusal-drifted checkpoints

**Track 2a, problem statements 1 and 2** (red-teaming and robustness evaluation; adversarial and
safety evaluation sets) and **Track 2b, problem statement 6** (repairing safety drift without
retraining), Lexsi AI Safety & Alignment Hackathon. The circuit-restricted intervention is a
published **negative, exploratory** result, not a working repair. Form-ready answers:
`_research/SUBMISSION-FORM-20261106.md`.
Deadline **14 November 2026, AoE** (`results/competition_rules_verified_20261105.json`).

**Status in three lines.** The decision workbench works, is tested in a pinned remote environment
(`results/modal_validation_20261107T143612137007Z.json`: 158 tests in both invocations, integrity
checks passed) and, given the evidence it ships with, answers **insufficient evidence** — correctly.
Its scientific gate is open: the three-judge panel `failed` calibration, so `headline_eligible=false`
and every arm-level rate is `exploratory`. The public Space and collections are live; the hosted
workbench's evidence matches the local bundle apart from `created_at`
(`results/live_demo_verification_20261107.json`). The PDF remains historical and has not been rebuilt.

## Who this is for, and what decision it changes

Someone preparing to publish a fine-tune of an open-weights model — here, a Cohere `tiny-aya-global`
checkpoint whose refusal direction was abliterated and then partly repaired — has to answer one
question before the model leaves the lab: **is this repair good enough to ship, or does it only look
good?** Today that question is answered by reading a notebook and trusting its author.

The workbench turns it into a reviewable decision. A maintainer loads an evidence bundle, sets the
bounds their own policy requires (maximum acceptable harm rate, maximum benign over-refusal, minimum
capability, minimum edit support), and gets one of three answers per candidate repair — **accept**,
**reject**, or **insufficient evidence** — with the reason, the intervals that produced it, and a
replay of every judge verdict behind it. The export is a JSON decision plus a readable summary that a
reviewer can re-check without trusting the person who ran the model.

What it refuses to do is as important: it will not accept an arm whose confidence interval crosses a
bound, it will not accept an arm that is over the support budget, it will not promote exploratory
evidence, and it will not hide a failed calibration control. On the evidence shipped here, **all four
arms come back `insufficient evidence`** — which is the honest answer to "can I ship this?" today, and
the demonstration that the tool cannot be talked into a yes.

## What the workbench does

- **Bounds are the user's, not the author's.** Policy inputs (harm, over-refusal, capability, support)
  are visible and editable; the decision is a pure function of the bundle and those bounds.
- **Every number is traceable.** Each metric carries its estimator, n, interval and the source file it
  was read from; the page shows the crosshair interval plot and a full metric table, not a summary line.
- **Verdicts are replayable.** Per prompt, per judge, per arm: answer, raw verdict, parsed label,
  agreement, and stop diagnostics. Harmful answers are behind a click, never rendered by default.
- **Imports are verified, not trusted.** A bundle is accepted only if every prompt and answer hashes to
  the digest it declares; a mismatch leaves the previous study loaded and explains itself.
- **Nothing is inferred or fetched at decision time.** No inference, no live model, no network: the
  export describes the decision currently on screen, and editing the policy invalidates it.

## Where the evidence actually stands

Four arms at a 512-token budget; refusal, over-refusal, harm-ASR (majority-of-three judges) and GSM8K
strict-match. All values from stored runs (sources in the table below).

| arm | edited scalars | refusal | over-refusal | harm-ASR | GSM8K |
|---|---:|---:|---:|---:|---:|
| base | 0 | 0.842 | 0.075 | 0.058 | 0.590 |
| drifted | 962,592,768 | 0.500 | 0.050 | 0.217 | 0.560 |
| 2 % repair | 19,251,200 | 0.883 | 0.175 | 0.050 | 0.518 |
| global repair | 962,592,768 | 0.883 | 0.275 | 0.042 | 0.438 |

Sources: `results/p21_safety_length_full_20261103_001722.json` (generations),
`results/p19_safety_judges_20261103_002850.json` (verdicts),
`results/p21_length_analysis_20261103.json`, `results/p16_capability_analysis_20261102.json`.

The **finding** is the shape of the trade-off, not a green light. Drift raises the observed harm rate
0.058 → 0.217; each repair buys that back and pays in over-refusal and reasoning accuracy; and the
price is set by the **support** of the edit — a 2 % repair reaches the aligned model's refusal rate at
0.175 over-refusal instead of 0.275, keeping GSM8K at 0.518 instead of 0.438 (paired +0.080,
95 % CI [0.038, 0.122]). It replicates on 160 prompts the arms never saw
(`results/heldout_analysis_20261106.json`: harm 0/120 base → 38/120 drifted → 3/120 sparse, 1/120
global; over-refusal 5/40 → 13/40 → 20/40; sparse vs global over-refusal −0.175 [−0.325, −0.050]) —
still `exploratory`, because the panel that produced those labels failed calibration.

Two corrections this project made to itself, both visible in the tree: the near-orthogonality of the
re-extracted direction (cos = 1.0×10⁻³) does **not** show that all harm discrimination was deleted, and
non-significance between the two repairs does **not** show they are equally safe.

## Against the requirements

| requirement | what exists | honest status |
|---|---|---|
| ≥ 2 track repos, 3+ preferred | **3 chained**: CuratorKIT (corpus + row provenance) → CircuitKIT (discovery on the drifted model) → AuditKIT (native re-scoring, read-only at commit `71316da…`) | met; circuit **discovery** cannot be re-run here (CircuitKIT/EAP-IG not installed) |
| a Cohere dense open-weights model | `CohereLabs/tiny-aya-global` (3.35 B, `cohere2`), revision `af89d219…`; earth/water secondary | met |
| artifacts on the Hub, parquet preferred | corpus + results as parquet with declared viewer configs; patches as parameter `.npz` with per-file sha256 | public collections verified; consult repository cards for file-level versions |
| a public deployment | static Space with policy decisions, comparison chart and per-prompt judge replay | **live and verified** (`results/live_demo_verification_20261107.json`) |
| more than one before/after metric | four: refusal, benign over-refusal, harm-ASR, GSM8K strict-match | met |
| write-up on why these metrics/datasets/models | `WRITE-UP.md`, `report/report.tex`, `_research/EVIDENCE-BUNDLE.md`, `REPRODUCE.md` | met; the PDF is historical and has not been rebuilt |
| validation | `results/modal_validation_20261107T143612137007Z.json` (158 tests ×2, integrity checks, source + toolkit hashes) | software passes; **science does not** — calibration failed |

## Deliberately not claimed

- **Nothing is submission-certified.** `results/readiness_ledger.json` is `not_submission_ready`;
  calibration and headline eligibility remain closed, the project-distribution human audit (`#22`)
  remains pending, and publication must be rechecked after edits.
- **The circuit-restricted intervention did not reduce harm.** In the exploratory judge analysis,
  harm-ASR is 0.283 versus 0.217 for drift (paired +0.067 [0.008, 0.125]); over-refusal is 0.125
  versus 0.050 (`results/circuit_arm_analysis_20261106.json`). No capability run exists for this arm,
  and this negative result does not rule out other circuits or interventions.
- **No confirmed held-out number**, for the reason above; and the confirmatory run is frozen but
  unexecuted (`results/confirmatory_spec_20261106.json`), with only 57 untouched harmful prompts left
  in the corpus — recorded before, not after, seeing it.
- **No multilingual number** (earlier files were invalid and are withdrawn), and **no second-model
  frontier** (earth complete, water missing one arm, neither judged).
- **No lower bound from the panel's 0.984 specificity / 0.580 sensitivity** on 500 Aegis rows
  (`results/p22_judge_validation_20261103_020927.json`): a different distribution does not bound a
  different one, and the failed original controls stay failed.
- **One model, one seed, one capability task** carry the headline; the paired interval covers prompt
  sampling only.
- **Costs are reservations, not bills.** Current reservations and limits are in
  `results/modal_budget_20261105.json`; `realized_provider_charge_usd` is `null`.

## Licence

Cohere weights are LSAL 1.2/1.1 (research use only). §2 forbids selling or hosting the model as a
product; §4 forbids deploying drifted checkpoints. This entry therefore publishes **parameters**
(`.npz` plus a loader), never a drifted checkpoint, and the demo serves precomputed JSON over published
answers. Individual entry, not an organisation.

## Links and reproduction

- Demo: `https://huggingface.co/spaces/amugoodbad229/scartissue`
- Direct static app: `https://amugoodbad229-scartissue.static.hf.space/`
- Report (historical): `https://huggingface.co/datasets/amugoodbad229/scartissue-tinyaya-results/blob/main/report.pdf`
- Corpus: `https://huggingface.co/datasets/amugoodbad229/scartissue-tinyaya-safety-corpus`
- Results: `https://huggingface.co/datasets/amugoodbad229/scartissue-tinyaya-results`
- Drift parameters: `https://huggingface.co/datasets/amugoodbad229/scartissue-tinyaya-abliteration-patches`
- Repair parameters: `https://huggingface.co/datasets/amugoodbad229/scartissue-tinyaya-repair-patches`
- Run it yourself: `REPRODUCE.md` (tier 0 needs only a browser)

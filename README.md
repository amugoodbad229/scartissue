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

## Run it — no GPU, no key, no network

**Docker (macOS, Windows, Linux — the intended path):**

```bash
docker build -t scartissue .
docker run --rm -p 8000:8000 scartissue
# open http://localhost:8000
```

The image is twelve lines on `python:3.12-slim`: it copies `space-static/` and serves it. No model weights,
no CUDA, no API key, no build step, nothing fetched at runtime. Everything it shows is a precomputed evidence
bundle committed at `space-static/data/evidence.json` (1.6 MB). This machine has no Docker daemon, so the image
was not built here — but the exact directory it serves is exercised by real-Chromium tests, and a plain
`http.server` over the same folder renders the full workbench with **zero** page errors (4 arms, 6 cards,
export enabled).

**Without Docker** (any OS with Python 3):

```bash
cd space-static && python3 -m http.server 8000   # then open http://localhost:8000
```

**Hosted:** <https://huggingface.co/spaces/amugoodbad229/scartissue>

### What you do once it is open

1. Read the decision panel: each candidate repair is `accept`, `reject` or **`insufficient evidence`**, with
   the reasons listed under it. On the evidence shipped here, all four arms return `insufficient evidence` —
   that is the designed answer, not a bug.
2. Change the policy (harm, over-refusal, utility, repair-selected scalars, study identity) and press
   **Evaluate repairs**. The decision recomputes in your browser; no server is involved.
3. Open **Judge replay** on any arm to see every generation, every judge's raw verdict and where the panel
   disagreed. Harmful prompts and answers stay collapsed until you click them.
4. **Download decision record** (JSON) or **Download readable summary** for the decision, its reasons, the
   source hashes and the bundle hash — without raw completions.

## What this is, and what it answers so far

A model maintainer is holding an abliterated Cohere checkpoint (`CohereLabs/tiny-aya-global`, revision
`af89d219…`) and a set of candidate weight-space repairs. The question is not "is this model safe" but the
narrower, answerable one: **is any candidate repair good enough to ship under my own bounds?**

What exists today:

- **An attack and a measurement.** Abliteration orthogonalises every write matrix against one refusal
  direction at layer 15 (962,592,768 scalars). Refusals do not disappear, so the remaining refusal is carried
  by orthogonal directions; re-extracting one and amplifying it (alpha 4) restores the refusal marker but the
  projection no longer separates harmful from benign prompts.
- **Four arms under identical conditions** — aligned base, drifted, a 2 %-support repair (19,251,200 scalars)
  and a full-support repair (the same 962,592,768 as the drift, so support is matched rather than assumed):
  harm-ASR 0.058 / 0.217 / 0.050 / 0.042; benign over-refusal 0.075 / 0.050 / 0.175 / 0.275; GSM8K (lm-eval)
  0.590 / 0.560 / 0.518 / 0.438.
- **A replication on untouched prompts**: 160 prompts no arm trained on (split proved disjoint). Harm goes
  0/120 base → 38/120 drift → 3/120 sparse → 1/120 global, over-refusal 8/40 → 5/40 → 13/40 → 20/40.
- **A workbench** that turns those numbers into a decision with an explicit refusal to over-claim: it abstains
  while the judge panel is uncalibrated, and says so as a reason.
- **An audit route** rather than an excuse: a pre-registered calibration gate
  (`results/calibration_gate_20261106.json`) over 180 blind items from this project's own generations
  (`human_audit/audit.html`), scored by `experiments/analyse_audit.py`.

## Which track, and which boxes to tick

**Tick Track 2a **and** Track 2b.** One track is enough for eligibility; both are honest here because the work
is a robustness measurement (2a) *and* a retraining-free repair (2b).

| box | tick? | why, in one line |
|---|---|---|
| **2a #1 — red-teaming / robustness** | **yes** | Weight-space attack (abliteration) plus a measured over-refusal bill for every repair. |
| **2a #2 — adversarial and safety evaluation sets** | **yes** | Curated corpus, published fit/eval/held-out split, pinned three-judge panel, calibration audit. |
| 2a #3 — multi-turn audits | no | Not attempted; no multi-turn data. |
| 2a #4 — multilingual / code-mixed | no | The multilingual runs are invalid and withdrawn. Say nothing rather than something wrong. |
| **2b #6 — repairing safety drift without retraining** | **yes** | Rank-one repairs fitted from direction extraction; zero gradient steps; four arms compared. |
| 2b #7 — unlearning | no | Not what this does. |
| 2b #8 — inference-time steering | no | Steering vectors were used to *find* directions, not deployed as a guard. |
| 2b #9 — locating where safety lives / 2b #10 circuit-level interventions | **no** | The circuit-restricted patch body was recovered and hash-verified but never generated, judged or capability-tested, so no circuit effect is claimed. |
| 2b #11–13 | no | Not attempted. |

Claimed outcome, stated so it cannot be misread: the repairs restore the refusal marker at a support-dependent
over-refusal price, and the shipped workbench returns `insufficient evidence` for all four arms because the
judge panel failed its calibration audit. That is the finding so far.

## What we do not claim

- Not a safety certification, and not production clearance of any checkpoint.
- Harm figures are **exploratory**: the panel's calibration audit failed (specificity 0.984, sensitivity 0.580
  on 500 Aegis rows, and that run rendered the HarmBench `[BEHAVIOR]` slot empty).
- Refusal-marker rates are diagnostics, never ASR estimates.
- Near-orthogonality of the re-extracted direction shows the ablation direction was removed; it does **not**
  show that harm-relevant representation survives elsewhere.
- No circuit-arm effect, no multilingual result, no claim that a repair "restores safety".

| artifact | where |
|---|---|
| Decision workbench (static; open `space-static/index.html`, no build) | [`space-static/`](space-static/index.html) |
| Docker image definition for judges | [`Dockerfile`](Dockerfile) · [`REPRODUCE.md`](REPRODUCE.md) |
| Public code | <https://github.com/amugoodbad229/scartissue> |
| Parameter deltas + loader (never a checkpoint) | <https://huggingface.co/amugoodbad229/scartissue-tinyaya-repair-parameters> |
| Model collection | <https://huggingface.co/collections/amugoodbad229/scartissue-repair-parameters-6aedb25709af05c05784a9e0> |
| Dataset collection (results, evals, corpus) | <https://huggingface.co/collections/amugoodbad229/scartissue-evaluation-evidence-6aedb2693a8dd1f844817b46> |
| Claim ledger (evaluated / pending / invalid) | [`results/readiness_ledger.json`](results/readiness_ledger.json) |
| How to reproduce, cheapest tier first | [`REPRODUCE.md`](REPRODUCE.md) |
| Write-up (method, numbers, what is *not* claimed) | [`WRITE-UP.md`](WRITE-UP.md) |
| Submission text (requirement-by-requirement) | [`SUBMISSION.md`](SUBMISSION.md) |
| Task state, dependencies and blockers | [`TODO.md`](TODO.md) |
| Public demo (same workbench, same bundle, no local install) | <https://huggingface.co/spaces/amugoodbad229/scartissue> |
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

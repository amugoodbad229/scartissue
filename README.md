# ScarTissue

**Can weight-space edits repair safety drift without sacrificing useful behavior?**

ScarTissue investigates this question in Cohere Tiny Aya Global. Starting from an aligned model and an
abliterated variant, the study compares sparse, full-support and circuit-restricted edits—without retraining.
It examines harmful compliance, refusal of benign requests and retained task performance together, rather
than treating restored refusal as proof of safety.

[Research article](_research/HF-ARTICLE.md) · [Reproduction guide](REPRODUCE.md) · [Evidence manifest](results/manifest.json)

## Track and selected goals

**Track 2 — AI Red Teaming, Safety and Alignment**

**2a: AI Safety and Red Teaming** · Primary focus
- [x] Red-teaming and robustness evaluation
- [x] Adversarial and safety evaluation sets

**2b: AI Safety and Correction** · Exploratory investigation
- [x] Repairing safety drift without retraining
- [x] Locating where safety lives

These are the research goals addressed, not claims that every goal was achieved.

## Findings and limitations

- **Repair has costs.** The sparse repair retained more GSM8K performance than the full-support repair,
  but both refused more benign requests than the drifted model. See the
  [task-performance analysis](results/p16_capability_analysis_20261102.json) and
  [exploratory safety comparison](results/circuit_arm_analysis_20261106.json).
- **The discovered circuit was not a successful repair target.** Restricting edits to its write columns
  did not improve the judged harm rate. The [negative result](results/circuit_arm_analysis_20261106.json)
  is retained, not presented as a successful localization of safety.
- **Safety remains unverified.** Judge calibration failed, so harm estimates are **exploratory**.
  The blind human audit and locked held-out confirmatory comparison remain **pending**.
  Current evidence is insufficient to establish production safety.

The study is limited to the evaluated model, settings and capability task. Restored refusal is not
interchangeable with reduced harm, and benchmark retention is not general capability preservation.

## Hugging Face submission links

- **Model collection:** [Repair parameters and model artifacts](https://huggingface.co/collections/amugoodbad229/scartissue-repair-parameters-6aedb25709af05c05784a9e0)
- **Dataset collection:** [Evaluation evidence and curated data](https://huggingface.co/collections/amugoodbad229/scartissue-evaluation-evidence-6aedb2693a8dd1f844817b46)
- **Article:** [Prepared draft](_research/HF-ARTICLE.md) — not yet published on Hugging Face.

## Reproduce and inspect

```bash
uv sync
uv run python -m experiments.verify_submission
```

The verifier checks evidence integrity and readiness; it does not certify safety. Node is required.
See [REPRODUCE.md](REPRODUCE.md) for pinned dependencies, toolkit setup and experiment instructions.

An **optional [evidence viewer](https://amugoodbad229-scartissue.static.hf.space/)**
([Hugging Face Space](https://huggingface.co/spaces/amugoodbad229/scartissue)) replays recorded results;
it performs no live model inference. The [blind annotation app](https://scartissue-audit.vercel.app/)
collects human labels separately.

## Research use

Built with CuratorKIT, CircuitKIT and AuditKIT. Cohere models are subject to their LSAL research-use
restrictions. The release provides parameter patches and a loader, **not a hosted drifted checkpoint**.

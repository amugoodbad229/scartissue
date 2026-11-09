# ScarTissue

An offline evidence workbench for comparing weight-space repairs of Cohere Tiny Aya Global.
It helps model maintainers decide whether to accept a repair, reject it, or request more evidence.

**[Open the website](https://amugoodbad229-scartissue.static.hf.space/)** ·
[Optional blind human audit](https://scartissue-audit.vercel.app/)

## Hackathon track and selected goals

**Track 2 — AI Red Teaming, Safety and Alignment**

**2a: AI Safety and Red Teaming** — primary fit
- [x] Red-teaming and robustness evaluation
- [x] Adversarial and safety evaluation sets

**2b: AI Safety and Correction** — exploratory repair study, not a claim of successful circuit repair
- [x] Repairing safety drift without retraining
- [x] Locating where safety lives

## What it does

- Compares the aligned model, abliterated model, and sparse/full-support repairs.
- Shows harm, over-refusal, task utility and intervention size under adjustable decision bounds.
- Replays recorded answers and exports the decision with its evidence references.

The demo uses recorded evidence—no model inference, API key or GPU is needed in the browser.

## Current limitations

Results are **exploratory**: judge calibration failed. The blind human audit and locked held-out
confirmatory comparison remain pending. Every bundled arm therefore returns **insufficient evidence**;
this is not proof that any repair is production-safe. The circuit-restricted repair had a negative result.

## Run locally

```bash
uv sync
uv run python -m http.server 8000 --directory space
```

Open `http://localhost:8000`. To check source and evidence integrity (requires Node):

```bash
uv run python -m experiments.verify_submission
```

See [REPRODUCE.md](REPRODUCE.md) for pinned validation and experiment instructions, and
[the evidence manifest](results/manifest.json) for results and provenance.

## Artifacts and use

[Parameter patches](https://huggingface.co/collections/amugoodbad229/scartissue-repair-parameters) ·
[Evaluation evidence](https://huggingface.co/collections/amugoodbad229/scartissue-evaluation-evidence)

Research use under Cohere LSAL restrictions. Only parameter patches and a loader are published,
not a drifted checkpoint. The project uses CuratorKIT, CircuitKIT and AuditKIT.

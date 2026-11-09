# ScarTissue

**Can weight-space edits repair safety drift without sacrificing useful behavior?**

A reproducible study in Cohere Tiny Aya Global. From an aligned model and an abliterated variant, it
compares sparse, full-support and circuit-restricted edits made without retraining, measuring harmful
compliance, refusal of benign requests and retained task performance together — because restored
refusal is not the same thing as safety.

[Article](https://amugoodbad229-scartissue.static.hf.space/article.html) ·
[Reproduction guide](REPRODUCE.md) ·
[Evidence manifest](results/manifest.json) ·
[Evidence viewer](https://amugoodbad229-scartissue.static.hf.space/)

## What we compared

Four model states, scored with a frozen majority-of-three judge panel (HarmBench, Qwen3Guard, Granite,
threshold 0.5), a benign over-refusal flag, and lm-eval GSM8K strict match on 500 documents:

| State | Judged harm | Benign over-refusal | GSM8K |
|---|---:|---:|---:|
| Aligned | 5.8% | 7.5% | 59.0% |
| Abliterated (drifted) | 21.7% | 5.0% | 56.0% |
| Sparse repair | 5.0% | 17.5% | 51.8% |
| Full-support repair | 4.2% | 27.5% | 43.8% |

The edits have different support sizes; this is not a matched-budget comparison.

## Findings

- **Repair has costs.** The sparse repair kept more GSM8K than the full-support repair, but both
  refused more benign requests than the drifted model
  ([analysis](results/p16_capability_analysis_20261102.json)).
- **The discovered circuit was not a usable repair target.** Restricting edits to its write columns
  did not improve the judged harm rate; the [negative result](results/circuit_arm_analysis_20261106.json)
  is kept rather than reframed.
- **Safety is not established.** Judge calibration failed — it included a prompt-rendering defect — so
  every harm figure is exploratory. The blind human audit and the locked held-out comparison are
  pending, and current evidence is insufficient for a production safety claim.

Scope is one model, one synthetic drift setting and one capability task. Benchmark retention is not
general capability preservation.

## Track and selected goals

**Track 2 — AI Red Teaming, Safety and Alignment**

- **2a: AI Safety and Red Teaming** — primary: red-teaming and robustness evaluation; adversarial and
  safety evaluation sets.
- **2b: AI Safety and Correction** — exploratory: repairing safety drift without retraining; locating
  where safety lives.

These are the goals investigated, not claims that each was achieved.

## Hugging Face

- **Model collection:** [safety drift and repair parameters](https://huggingface.co/collections/amugoodbad229/scartissue-repair-parameters-6aedb25709af05c05784a9e0)
- **Dataset collection:** [evaluation data and research results](https://huggingface.co/collections/amugoodbad229/scartissue-evaluation-evidence-6aedb2693a8dd1f844817b46)
- **Article:** https://amugoodbad229-scartissue.static.hf.space/article.html — the written account,
  hosted on the project's [Space](https://huggingface.co/spaces/amugoodbad229/scartissue). Publishing on
  the Hugging Face blog requires a PRO or Team/Enterprise account, so the article is **not yet published on Hugging Face**
  as a blog post.

Three datasets cover one run: measured result tables, raw recorded outputs, and the curated corpus.
The patch-only repositories duplicate files in the model repository and are kept for existing links.

## Reproduce

```bash
uv sync
uv run python -m experiments.verify_submission
```

The verifier checks evidence integrity and readiness; it certifies no safety property. See
[REPRODUCE.md](REPRODUCE.md) for pinned dependencies and experiment commands. The
[blind annotation app](https://scartissue-audit.vercel.app/) collects human labels separately.

## Research use

Built with CuratorKIT, CircuitKIT and AuditKIT. Cohere models are subject to their LSAL research-use
restrictions: the release ships parameter patches and a loader, **not a hosted drifted checkpoint**.
Recorded outputs may contain harmful material and are for controlled evaluation only.

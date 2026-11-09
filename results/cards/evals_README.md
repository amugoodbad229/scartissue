---
license: other
license_name: lsal-1.2-research-use-parameters-only
pretty_name: scartissue tiny-aya recorded evaluation outputs
task_categories:
  - text-generation
language:
  - en
tags:
  - safety
  - evaluation
  - reproducibility
---

# ScarTissue — Recorded Evaluation Outputs

This repository preserves full generated answers, evaluation prompts and the original frontier
export from the safety-drift and repair study. It is the raw-output companion to the
[structured results](https://huggingface.co/datasets/amugoodbad229/scartissue-tinyaya-results).

| File | Contents |
|---|---|
| `completions.json` | Recorded answers, retained in full so generation failures can be inspected. |
| `prompts.json` | Evaluation prompts and corpus identifiers. |
| `frontier.csv` | The historical comparison export. |
| `evals_manifest.json` | File-level provenance. |

**These are research records, not approved target answers or training recommendations.**
Outputs may contain harmful or sensitive text.

Judge calibration failed, so harm estimates are exploratory. Refusal-pattern matching is only
a diagnostic, not attack success. Capability results come from lm-eval, not a safety judge.
Historical frontier fields must be interpreted under these limitations, not as current safety
certification.

For the current interpretation and outstanding human audit and locked held-out confirmation,
see the [project findings](https://github.com/amugoodbad229/scartissue).

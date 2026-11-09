---
license: other
license_name: lsal-1.2-research-use-parameters-only
pretty_name: scartissue tiny-aya abliteration results
task_categories:
  - text-generation
language:
  - en
tags:
  - safety
  - refusal
  - abliteration
  - interpretability
  - reproducibility
  - lexsi-track2
size_categories:
  - 1K<n<10K
configs:
  - config_name: safety_generations
    data_files: data/safety_generations.parquet
  - config_name: judge_verdicts
    data_files: data/judge_verdicts.parquet
  - config_name: capability
    data_files: data/capability.parquet
  - config_name: patches
    data_files: data/patches.parquet
  - config_name: frontier
    data_files: data/frontier.parquet
---

# ScarTissue — Structured Research Results

Exploratory evidence from a study of synthetic safety drift and weight-space repair in
Cohere Tiny Aya Global. This repository organizes generated answers, judge verdicts and
task-performance measurements into inspectable tables.

[Project and current findings](https://github.com/amugoodbad229/scartissue) ·
[Reproduction guide](https://github.com/amugoodbad229/scartissue/blob/master/REPRODUCE.md) ·
[Raw output archive](https://huggingface.co/datasets/amugoodbad229/scartissue-tinyaya-evals)

## Read this before interpreting the results

**Judge calibration failed. Harm estimates are exploratory, not validated safety guarantees.**
The calibration procedure included a prompt-rendering defect. The project does not treat its
calibration scores as proof of reliable harm measurement.

The sparse repair retained more GSM8K performance than the full-support repair, but both
refused more benign requests than the drifted model. A later circuit-restricted intervention
was a negative result. These observations do not establish production safety, general capability
preservation or a successfully localized safety mechanism.

The blind human audit and locked held-out confirmatory comparison remain pending.

## What is in this repository

| File | Contents |
|---|---|
| `data/safety_generations.parquet` | Full recorded answers and generation diagnostics. |
| `data/judge_verdicts.parquet` | Stored automated judge outputs. |
| `data/capability.parquet` | Per-document GSM8K results from lm-eval. |
| `data/patches.parquet` | Intervention definitions and technical provenance. |
| `data/frontier.parquet` | The original comparison table. |
| `manifest.json` and provenance files | Configuration and artifact verification records. |
| `report.pdf`, `WRITE-UP.md`, `SUBMISSION.md` | Historical reports retained for traceability. |

**The historical reports and tables are not the current publication-readiness verdict.**
They may predate the calibration corrections and later circuit evaluation. Use the
[project evidence manifest](https://github.com/amugoodbad229/scartissue/blob/master/results/manifest.json)
and current findings when assessing what the study supports.

## Measurement and use

Harm estimates use the stored judge path. Benign over-refusal is a judge refusal flag.
Capability is measured with lm-eval. Refusal regexes are diagnostics only.
These outputs can contain harmful or sensitive material; inspect them in a controlled research setting.

The base model's LSAL research-use restrictions and the upstream data terms remain applicable.
No drifted checkpoint is hosted here.

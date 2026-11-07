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

# ScarTissue — results, parquet tables, and the submission package

Track 2a entry for the Lexsi AI Safety & Alignment Hackathon. Every number here is generated from
stored runs; nothing is hand-typed. The report is `report.pdf`. The current method, caveats and
submission draft are maintained in the [source repository](https://github.com/amugoodbad229/scartissue).

## Files

| file | what it is |
|---|---|
| `data/safety_generations.parquet` | 1,280 full model answers (4 arms × 160 prompts), 512-token budget, with token ids, stop diagnostics and the prompt hash |
| `data/judge_verdicts.parquet` | 3,864 rows — three judges on the 1,280 answers (128- and 512-token runs) |
| `data/capability.parquet` | 4,000 rows — per-document GSM8K five-shot results (lm-eval 0.4.13), 500 docs × 4 arms × 2 filters |
| `data/patches.parquet` | the four arm definitions: edited scalar counts, support fraction, `allowed{modules,layers,heads}`, sha256 |
| `data/frontier.parquet` | the 8 frontier rows (4 arms × 2 budgets), recomputable from the tables above |
| `report.pdf` | 5-page report: claim, frontier, mechanism, calibration, blocked work, limitations |
| `manifest.json` | 17-key manifest: claim, model revision, track repos, datasets, deployment, metrics, headline numbers, evidence paths, limitations, licence |
| `provenance_final.json` | corpus roles/labels, model and judge revisions, patch hashes, toolkit commits, sha256 of every input |
| `lexsi_provenance.json` | requirement map, before/after table, licence compliance |

## The headline table (512-token budget)

| arm | edited scalars | refusal | over-refusal | harm-ASR | GSM8K |
|---|---:|---:|---:|---:|---:|
| base | 0 | 0.842 | 0.075 | 0.058 | 0.590 |
| drifted | 962,592,768 | 0.500 | 0.050 | 0.217 | 0.560 |
| 2 % repair | 19,251,200 | 0.883 | 0.175 | 0.050 | 0.518 |
| global repair | 962,592,768 | 0.883 | 0.275 | 0.042 | 0.438 |

**Read the columns for what they are.** `refusal` is the Qwen3Guard `Refusal` field on harmful
prompts (a judge label, not a regex). `over-refusal` is the same field on benign XSTest prompts, i.e.
the model refusing something it should answer. `harm-ASR` is the majority of three harm judges at the
frozen threshold (strict `> 0.5`). `GSM8K` is lm-eval `strict-match` exact accuracy — capability is
never scored by a judge. A plain regex refusal marker on the same answers reads
0.683 / 0.175 / 0.450 / 0.583, i.e. it misses most refusals that open with a preamble: it is stored
as a diagnostic and never used as a claim.

## What the numbers support

The abliteration edit removes 962,592,768 scalars (all 72 write matrices) and the refusal rate falls
from 0.842 to 0.500. Repair along the **re-extracted** direction — `cos(d, d′) = 1.0×10⁻³`, so the
removed direction cannot simply be re-amplified — restores refusal. What differs between supports is
the price: the 2 % support keeps +0.080 GSM8K exact match over the global repair (95 % CI
[0.038, 0.122]) at the same refusal rate and a lower over-refusal rate, and harm-ASR does not separate
the two repairs (0.050 vs 0.042). The measurement is therefore a **frontier**, not a single number.

## What is *not* claimed

- No discovered-circuit number: the circuit arm was generated but not judged (the Modal workspace hit
  its spend limit), so `data/judge_verdicts.parquet` contains no circuit rows.
- No held-out confirmation, no multilingual result, no second-model frontier; those runs are listed as
  blocked in `manifest.json` rather than implied by proxy numbers.
- No "the model is unsafe" claim: the drifted model's harm-ASR moves 0.058 → 0.217 under a panel that
  is deliberately conservative (specificity 0.984, sensitivity 0.580 on 500 human-labelled rows), and
  its refusals become hedges rather than compliance.

## Licence

Cohere `tiny-aya-global` weights are LSAL 1.2/1.1, research use. Section 2 forbids selling or hosting
the model as a product and section 4 forbids deploying drifted checkpoints, so this repository
publishes **parameters and answers**, never a drifted checkpoint. Individual entry; not an
organisation.

## Citation

```bibtex
@misc{scartissue_results_2026,
  title  = {ScarTissue: abliteration, repair, and the price of each},
  year   = {2026},
  note   = {Lexsi AI Safety & Alignment Hackathon, Track 2; CohereLabs/tiny-aya-global rev af89d219},
  url    = {https://huggingface.co/datasets/amugoodbad229/scartissue-tinyaya-results}
}
```

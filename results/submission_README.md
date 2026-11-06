# ScarTissue — results, parquet tables, and the submission package

Track 2a entry for the Lexsi AI Safety & Alignment Hackathon. Everything here is generated from
stored runs; nothing is hand-typed.

## Files

| file | what it is |
|---|---|
| `report.pdf` | the 5-page report (claim, frontier, mechanism, calibration, blocked work, limitations) |
| `WRITE-UP.md` | the submission write-up in markdown |
| `SUBMISSION.md` | the entry itself: requirement-by-requirement map, claim, numbers, disclaimers |
| `manifest.json` | 17-key manifest: claim, model revision, track repos, datasets, deployment, metrics, headline numbers, evidence paths, limitations, licence |
| `lexsi_provenance.json` | requirement map + before/after table + licence compliance |
| `provenance_final.json` | corpus roles/labels, model and judge revisions, patch hashes, toolkit commits, sha256 of every input |
| `data/*.parquet` | `safety_generations` (1,280 full answers), `judge_verdicts` (3,864), `capability` (4,000), `patches`, `frontier` |

## The headline table (512-token budget)

| arm | edited scalars | refusal | over-refusal | harm-ASR | GSM8K |
|---|---:|---:|---:|---:|---:|
| base | 0 | 0.842 | 0.075 | 0.058 | 0.590 |
| drifted | 962,592,768 | 0.500 | 0.050 | 0.217 | 0.560 |
| 2 % repair | 19,251,200 | 0.883 | 0.175 | 0.050 | 0.518 |
| global repair | 962,592,768 | 0.883 | 0.275 | 0.042 | 0.438 |

The 2 % repair recovers the aligned model's refusal rate, keeps half of the capability the global
repair loses (+0.080 GSM8K exact match, 95 % CI [0.038, 0.122]), and is indistinguishable from the
global repair on harm-ASR.

## How to read the judge numbers

The panel (HarmBench-Mistral-7B, Qwen3Guard-Gen-8B, Granite-Guardian-3.2-5B) was validated on 500
human-labelled rows from `nvidia/Aegis-AI-Content-Safety-Dataset-1.0`: majority-of-3 accuracy 0.782,
specificity 0.984, sensitivity 0.580. It is a **conservative** instrument — absolute harm rates here
are a lower bound, while arm-to-arm comparisons on the identical prompts are sound.

## Reproduce

See `report.pdf` §10 and the appendix: `uv sync`, then the Modal apps `p8_corpus`, `build_patches`,
`p23_circuit`, `p21_safety_length_full`, `p19_score_safety`, `p16_lm_eval`, then
`experiments.analyse_safety` and `report/make_report_data.py`.

## Licence

Cohere weights are LSAL 1.2/1.1 (research use). We publish **parameters** (factorised patches with an
explicit `allowed` block), never a drifted checkpoint, and the demo serves precomputed JSON over
published answers.

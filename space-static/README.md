---
title: ScarTissue — abliteration, repair, and the price of each
emoji: 🩹
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
license: other
short_description: Abliterate a Cohere model, repair it, and price the repair
---

# ScarTissue — abliteration, repair, and the price of each

Abliteration projects a refusal direction out of every write matrix of
`CohereLabs/tiny-aya-global` (962,592,768 scalars, no gradients, no data). This demo shows what that
edit costs and what it takes to put refusal back.

**Three columns, same prompt, same seed:** the aligned model, the drifted model, and one repair —
plus a frontier chart and the judge-calibration table. Everything shown is precomputed JSON produced
by the runs in the project repository; no model is served from this Space (see Licence).

## What the numbers say

| arm | edited scalars | refusal | over-refusal | harm-ASR | GSM8K |
|---|---:|---:|---:|---:|---:|
| base | 0 | 0.842 | 0.075 | 0.058 | 0.590 |
| drifted | 962,592,768 | 0.500 | 0.050 | 0.217 | 0.560 |
| 2 % repair | 19,251,200 | 0.883 | 0.175 | 0.050 | 0.518 |
| global repair | 962,592,768 | 0.883 | 0.275 | 0.042 | 0.438 |

The abliteration removes the read-out **and** the discrimination behind it: the re-extracted refusal
direction is orthogonal to the removed one (`cos(d, d′) = 1.0×10⁻³`), so amplifying the old direction
is a no-op. Refusal can be restored in weight space alone, but the bill — benign over-refusal and
GSM8K — is set by **how much of the signed write the repair touches**. The 2 % support keeps +0.080
GSM8K exact match over the global repair (95 % CI [0.038, 0.122]) at the same refusal rate, while the
harm-ASR axis cannot separate the two repairs. The measurement is a frontier, not a number.

## Files here

| file | what |
|---|---|
| `index.html` | the demo (frontier chart, three-column completion viewer, calibration table) |
| `frontier.csv` | one row per arm: refusal, over-refusal, GSM8K, harm-ASR, gap recovered |
| `completions.json` | 160 prompts × 4 arms of full stored answers |
| `prompts.json` | the frozen evaluation prompts with their corpus ids |
| `report.pdf` | the 5-page report |

Full data (parquet) and the parameter patches:
[`scartissue-tinyaya-results`](https://huggingface.co/datasets/amugoodbad229/scartissue-tinyaya-results) ·
[`scartissue-tinyaya-safety-corpus`](https://huggingface.co/datasets/amugoodbad229/scartissue-tinyaya-safety-corpus) ·
[`scartissue-tinyaya-repair-patches`](https://huggingface.co/datasets/amugoodbad229/scartissue-tinyaya-repair-patches).

## Licence

Cohere weights are **LSAL 1.2/1.1, research use**. §2 forbids selling or hosting the model as a
product and §4 forbids deploying drifted checkpoints, so this Space serves **precomputed JSON over
the aligned model's published answers** — it hosts no model and no drifted weights. Individual entry;
not an organisation.

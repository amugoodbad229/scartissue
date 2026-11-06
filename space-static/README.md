---
title: ScarTissue — is this repair good enough to ship?
emoji: 🩹
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
license: other
short_description: A repair-decision workbench that will not over-claim
---

# ScarTissue — repair-decision workbench

A model maintainer holding an abliterated Cohere checkpoint has four candidate weight-space repairs and
must decide whether any is good enough to ship. Set your own bounds — harm, benign over-refusal, utility,
repair support — and this page answers **accept**, **reject** or **insufficient evidence**, with the
intervals, the reasons, and a full per-judge replay behind every card.

## What you are looking at

Four arms measured under identical conditions: the aligned base model, the drifted one, a 2 %-support
repair and a full-support repair (support matched to the drift, so it is not assumed).

| | harm ASR | over-refusal | GSM8K |
|---|---|---|---|
| base | 0.058 | 0.075 | 0.590 |
| drifted | 0.217 | 0.050 | 0.560 |
| sparse repair | 0.050 | 0.175 | 0.518 |
| full repair | 0.042 | 0.275 | 0.438 |

On 160 held-out prompts no arm ever trained on: harm 0/120 → 38/120 → 3/120 → 1/120, over-refusal
8/40 → 5/40 → 13/40 → 20/40. The trade-off replicates; the repairs are not free.

**All four cards say `insufficient evidence`, and that is the correct answer.** The three-judge panel
failed its calibration audit (sensitivity 0.580, specificity 0.984 on 500 Aegis rows — that run also
rendered the HarmBench behaviour slot empty), so no harm figure here is confirmatory. The workbench
builds that refusal into the decision instead of hiding it, and a pre-registered audit gate on 180 blind
project-distribution items is the route to clearing it.

Everything is static: no model is run, no GPU, no API key, no data leaves your browser. Download the
decision record (JSON) or a readable summary — both exclude raw completions.

Code and evidence: <https://github.com/amugoodbad229/scartissue> ·
Datasets: <https://huggingface.co/collections/amugoodbad229/scartissue-evaluation-evidence-6aedb2693a8dd1f844817b46>

---
title: ScarTissue — what a restricted repair buys back
emoji: 🩹
colorFrom: gray
colorTo: blue
sdk: static
pinned: false
license: other
short_description: What abliteration removes, and what repair buys back
---

# ScarTissue

Abliteration removes a model's refusal read-out. This page shows — from stored measurements, not a live
model — what comes back when the edit is repaired, and what the repair costs in over-refusal.

Model: `CohereLabs/tiny-aya-global` (3.35 B, cohere2, 36 layers). The drift and every repair act on the
same 72 write matrices (`o_proj`, `down_proj`) at layer 15.

## What is on the page

- **The frontier** — refusal on 120 harmful prompts (benefit) against over-refusal on 40 benign prompts
  (price), for the global repair and four restricted supports.
- **Same prompt, three models** — 160 preset prompts, showing the base completion, the drifted completion
  and a repaired one, side by side, with a refusal-regex badge that is explicitly labelled a diagnostic.
- **Limitations** — what the numbers do and do not license.

## Where the numbers come from

| artifact | what it holds |
|---|---|
| [scartissue-tinyaya-evals](https://huggingface.co/datasets/amugoodbad229/scartissue-tinyaya-evals) | every completion of every arm on every prompt, the frontier CSV, the manifest |
| [scartissue-tinyaya-safety-corpus](https://huggingface.co/datasets/amugoodbad229/scartissue-tinyaya-safety-corpus) | the 763-row corpus with row-level provenance (CuratorKIT) |
| [scartissue-tinyaya-abliteration-patches](https://huggingface.co/datasets/amugoodbad229/scartissue-tinyaya-abliteration-patches) | the drift as parameters: direction `d`, `alpha = -1`, per-matrix column projections |
| [scartissue-tinyaya-repair-patches](https://huggingface.co/datasets/amugoodbad229/scartissue-tinyaya-repair-patches) | the four repair patches, each with the `allowed` block that defines its support |

No drifted checkpoint is published: the licence (LSAL 1.2/1.1 §4) forbids deploying one, so the edit
ships as parameters plus a 30-line loader.

## Measurement rules this page follows

ASR comes only from the judge path, never from the regex. The refusal regex here is a *diagnostic* — it
answers "did the model decline", not "was the answer harmful". Capability is gsm8k exact-match, measured
in the same run as the refusal numbers so the two axes are comparable. Full completions are stored
because a metric can read 0.000 for a model that no longer produces sentences (that is what the
`alpha = 8` arm did), and a page without the text would hide it.

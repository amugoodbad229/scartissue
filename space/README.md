---
title: scartissue — abliteration and repair
emoji: 🧪
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: other
short_description: What abliteration removes, and what a repair puts back
---

# scartissue

An abliterated 3.35 B Cohere model, three columns, real measured completions.

Abliteration deletes a refusal **read-out**. This page shows the three measured questions behind that
claim on `CohereLabs/tiny-aya-global`:

1. **Does the read-out survive?** Yes — the refusal direction re-extracted *after* the edit is orthogonal
   to the abliterated one (`cos = +1.9e-04`), so amplifying the old direction is a no-op by construction.
2. **Can it be amplified back?** Yes — weight-only, no training data, no gradients.
3. **What does that cost?** **Over-refusal.** At 100 % of the edit the refusal rate returns to 97 % of the
   base rate but over-refusal goes from 0.050 to **0.500**. At 2 % of the edit — ranked by harm-weighted
   write amplitude — refusal recovers 54 % of the gap for **0.200** over-refusal: 2.54 units of refusal per
   unit of over-refusal against the global edit's 1.48.

That is the frontier the project reports instead of a single number. The discrimination the model had
before the edit does **not** come back at any budget (AUC of the surviving direction stays 0.79–0.82
against the base model's 0.999) — the refusal is bought back, the separation is not.

## What is and is not claimed

- **Not** a safety win. At 3.35 B the ASR axis is weak (non-refusals are hedging, not harm), three judges
  scored identical completions 0.417 / 0.583 / 1.000, and the refusal tag in the UI is a **regex
  diagnostic**, not a judge verdict.
- **Not** a hosted drifted checkpoint — the licence forbids deploying one. Every number here comes from
  parameter patches that are published as parameters.
- The blind control (random columns at the same budget) is the arm that decides whether *restriction*
  matters or only *size* does; it is reported on the same page as the other arms.

Full runs, protocols and the frozen pre-registration: the `scartissue` repository.

# Appraisal of the earlier research layer — 2026-11-06

This file records a critical review of the project's own prior write-up (`_research/FINDINGS-01.md`,
commits `c422511` → `7ed60a1`) and what changed as a result. It is kept because the corrections matter
more than the polish: anyone reading the findings log alone would otherwise inherit claims the data
does not carry.

Protocol status: `_research/PROTOCOL.md` is frozen and untouched. This is an appraisal of *interpretation*,
not of the measurement plan. The findings log keeps its original text; corrections are appended to it and
landed in the documents that make claims (WRITE-UP.md, SUBMISSION.md, README.md, report/report.tex,
space-static, results/manifest.json).

## What the earlier work got right

- Full completions were stored for every arm, not samples. That is what makes every later correction
  possible; without it nothing here could be re-derived.
- Four arms with a **parameter-matched global control** (962,592,768 edited scalars for both
  `harm2_a4`-style support work and `global_a4`), which is the right control and is rarely present.
- The `alpha = 8` catch (`FINDINGS-01.md` §3): the refusal regex read 0.000 while the model emitted
  `Rep introduced\n\n#scsc\n\n#scsc…`. Catching that a metric is reading degenerate output is the
  strongest single piece of reasoning in the file.
- Mechanism claim grounded in a measurable quantity: `AUC(d′)` falls 0.999 → 0.831 while `cos(d, d′) ≈ 0`,
  which supports "the discriminator degraded" rather than only "the read-out was removed".
- `headline_eligible: false` was set in the payloads, and no drifted checkpoint was published (LSAL §2/§4).

## What had to be corrected

1. **"97 % of the original refusal rate recovered"** (`FINDINGS-01.md` §3, and the block quote that follows
   it) was the flagship sentence. The number is a refusal-marker diagnostic, not an ASR estimate, and the
   same table shows the price: over-refusal 0.050 → 0.500. The defensible statement is the trade-off curve,
   not a recovery percentage. Corrected in WRITE-UP.md §Reading the table, report.tex, and here.
2. **The Aegis panel was treated as a bound.** Specificity 0.984 / sensitivity 0.580 on 500 Aegis rows was
   read as a *lower bound* under distribution shift. Distribution shift does not make a sensitivity a lower
   bound. The proximate cause is also now known: the calibration run rendered the HarmBench template with an
   **empty `[BEHAVIOR]` slot** (`results/p22_judge_validation_20261103_020927.json`,
   `workers[0].records[0].rendered_input`), and the run's own docstring says so. The number was probably
   pessimistic for a fixable reason, but the inference was unjustified either way.
3. **Orthogonality was over-read.** `cos(d, d′) ≈ 0` shows the ablation direction was removed; it does not
   show that harm-relevant representation survives elsewhere. The AUC is the real evidence and it says
   something narrower: `d′` still separates classes, at a lower margin.
4. **"Circuit leg reaches discovery"** (commit `7ed60a1`) described an arm that was never generated, judged or
   capability-tested. The `p6` zero score was an output-contract bug and the `p26` patch body was only
   recovered on 2026-11-06 (`results/circuit_body_recovery_20261106.json`, 150,994,944 scalars,
   `sha256 96c659a0…`). Claim status: `pending` in `results/readiness_*.json`.
5. **Publish-then-withdraw.** Multilingual and multimodel legs were published before being invalidated
   (non-finite projections; unjudged runs). Both are now `invalid`/excluded from every claim, and
   `results/manifest.json` says so.
6. **The verifier was theatre.** `experiments/verify_submission.py` asserted 17 exact keys, a fixed PDF hash,
   HTTP 200 responses and numeric goalposts. It would pass with every claim in the write-up wrong. Rewritten
   to check provenance and structure only, with `submission_ready: false` unconditional.

## The one structural lesson

No instrument was validated before it was used to make the central claim. The write-up knew
(`calibration_status: failed`) and the headline still read as a result. The response is not another summary:
it is `results/calibration_gate_20261106.json` (a gate registered before any label existed),
`human_audit/audit.html` (a blind 180-item sample from this project's own generations), and
`experiments/analyse_audit.py` (which reports the panel's measured sensitivity and specificity and fails
closed). Until that gate passes, `headline_eligible` stays false and the workbench keeps returning
`insufficient evidence` for every arm — which is the correct answer for evidence in this state.

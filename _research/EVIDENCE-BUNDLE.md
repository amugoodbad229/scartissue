# Evidence contract — decision workbench

Status: implementation draft; remote test results determine completion. No safety certification.

`experiments/evidence_bundle.py` derives `scartissue.evidence/1` from immutable generation, judgment, safety-analysis and lm-eval result files. `space-static/evidence.js` is the shared browser/Node validator, admission policy and descriptive nondominance implementation. The static application must consume this contract, not a parallel hand-maintained metric table.

## Identity and provenance

- Bundle identity: model repository, immutable model revision, SHA-256 of the ordered prompt-ID/role/hash split, named safety and utility protocols, source paths and byte hashes.
- Arms: reference or repair, exact write-weight state digest, base write-weight digest, repair-selected scalar count, dtype, generation settings, and the historical run's patch-reference catalog. That catalog lists the run's available patches, **not an assertion that every patch was applied to each arm**. State hashes identify the applied state.
- Prompt records retain original IDs, roles, text and SHA-256. Completion records retain full answers, answer hashes, stop diagnostics, and all contracted judge families.
- Judge joins use `(arm, corpus_id, family)` plus matching prompt and answer hashes. Duplicate joins and mismatched hashes fail. Controls are excluded from evaluation rows. Missing/error verdicts are explicit, never converted into safe labels.
- Metrics retain value, denominator, interval and source path. Missing metrics stay absent/null. ASR is majority-of-three judge harm; refusal regex is not a safety metric. Utility is lm-eval strict-match, not a judge opinion.

## Admission decision

A user provides harm-ASR and over-refusal ceilings, a utility floor, a repair-selected-scalar budget and the requested model/revision/split identity. These are research policy choices, not universal safety thresholds.

References cannot be admitted as repairs. A repair returns `insufficient_evidence` for incompatible scope, failed/pending calibration, ineligible headlines, exploratory rather than locked held-out evaluation, missing/failed project-distribution human audit, errors, missing full completions, missing contracted verdicts or missing metrics. An apparently good point estimate cannot bypass these gates.

For otherwise eligible evidence, an interval wholly outside a policy bound or an exceeded support budget yields `reject`; an interval crossing a bound yields `insufficient_evidence`; intervals wholly within all bounds yield `accept` **for that research policy and evidence only**. Marginal intervals do not establish simultaneous coverage, deployment safety, causal superiority or statistical equivalence.

The frontier is descriptive nondominance among repair point estimates across lower harm, lower over-refusal, higher utility and lower edited support. It is not an uncertainty-adjusted winner or an admissibility decision. Failed evidence remains visibly failed even if geometrically nondominated.

## Current evidence limitations

The imported historical study retains failed calibration, `headline_eligible: false`, exploratory scope and a pending project-distribution human audit. Consequently its repairs must not be admitted even under a permissive policy. The capability task, judge sensitivity study and held-out generation artifact do not clear these failures.

Local schema checks cannot authenticate an arbitrary imported study's asserted hashes, attest that its declared controls happened, or detect all fabricated metrics. Exports must distinguish bundled declarations from file-byte verification and human review. No live drifted inference, new checkpoint hosting or automatic deployment is part of this tool.

## Validation

All tests run on Modal. Python tests build the real bundle from historical sources and invoke the same Node contract used by the browser; synthetic fixtures exercise accept/reject/abstain and malformed input. The Modal runner returns complete test logs and the generated bundle to the local single writer. Historical source measurements remain unchanged. PDF compilation and public deployment are separate gates.

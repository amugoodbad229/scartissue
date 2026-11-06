# Workbench functional QA

## Scope and evidence

The static workbench uses precomputed research evidence, not live model inference. Software validation does not certify the scientific claims or the truth of an imported bundle's declarations.

Passed Modal run: `results/modal_validation_20261105T164431691063Z.json`. Both pytest invocation forms passed; the complete stdout, environment and source hashes are retained there. This sign-off applies to those recorded source hashes. Subsequent edits require another Modal run.

Measured Chromium contrast results: `results/workbench_accessibility_20261106.json`. Screenshots: `results/scartissue_workbench_desktop.png` and `results/scartissue_workbench_mobile.png`. Screenshots are illustrations, not substitutes for assertions; raw-content disclosures are closed before capture.

## Functional coverage

- Real bundle loading and evidence-gated abstention; metric table and uncertainty display.
- Per-prompt full replay, closed harmful-content disclosures, judge outputs and stop diagnostics.
- Browser-local valid imports, tampered content rejection, identity mismatch, escaped malicious labels and preservation of the previous valid study.
- JSON and readable exports from the current decision, source hashes, exclusion of raw completions and invalidation after policy edits.
- Keyboard skip link, main-content focus, labeled controls, keyboard policy evaluation and visible focus.
- Narrow viewport and enlarged text without page-wide horizontal overflow; the metrics table has its own labeled scroll region.
- Selected text/background, control-boundary and focus contrast measured from computed browser styles.
- Invalid schema and failed initial HTTP request leave controls unavailable with a visible explanation.

Source tests: `tests/test_workbench_browser.py`, `tests/evidence.test.cjs`, `tests/test_evidence_bundle.py` and `tests/test_submission_verification.py`.

## Not checked / remaining risk

No full WCAG conformance assertion, screen-reader traversal, cross-browser/OS certification or first-time-human usability study. The guided walkthrough is instructional copy plus automated functional coverage, not a measured usability result. Hosting configuration, public deployment, offline service-worker caching and release-account access have not been validated. If the bundle cannot be loaded, failure is explicit; the app does not pretend cached or missing data are valid.

Scientific calibration, independent held-out analysis, patch-artifact verification and human adjudication remain separate gates in `results/readiness_ledger.json`. Imported provenance is declared evidence, not cryptographic authentication of a trusted producer.

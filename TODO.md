# ScarTissue — hackathon reset roadmap

Updated 2026-11-10. Mirrors the on-screen task queue; machine-readable scientific plan: `results/readiness_reset_20261105.json`.

### Research documentation and privacy follow-up

- [x] Remove the personal submission form from the current public GitHub branch; retain the ignored local copy.
  Historical commits were not rewritten, so past exposure is not erased.
- [x] Rewrite the README around the research question, exact selected track/goals, bounded findings and limitations;
  keep the website optional and emphasize the required Hugging Face collection links.
- [x] Prepare `_research/HF-ARTICLE.md` as a clear research article, including unequal edit support,
  task-specific capability findings, the negative circuit result and the calibration-rendering defect.
- [ ] Publish the Hugging Face article and replace the draft link with its actual blog URL. Blocked by current
  account publishing eligibility, not by missing prose: `results/hf_article_publishing_account_20261110.json`.
  Verified collection links and current contents: `results/hf_submission_links_20261110.json`.
- [x] Validate private-form exclusions and documentation guardrails on Modal: both invocation forms passed,
  including the actual upload mount check. Evidence: `results/modal_validation_20261109T162150401703Z.json`;
  named log: `logs/modal_validation_privacy_docs_20261110.log`.
- [x] Publish only the rewritten README/article and privacy regression guards; leave unrelated pending app work untouched.
  Remote verification: `results/research_docs_release_20261110.json`. The GitHub article remains a draft,
  not a published Hugging Face blog article.
- [x] Stop presenting `_research/HF-ARTICLE.md` as the article. The README now links the reproduction
  guide and evidence manifest, states plainly that the blog article is **not yet published on Hugging Face**
  and why, and `tests/test_claim_readiness.py` fails if the repository draft path reappears in the README.
  The Markdown file stays as working notes until a real blog URL exists.
- [x] Rewrite the five published Hub cards in plain language and rename both submission collections.
  All five remote READMEs match the local bytes; collection titles and one-line descriptions are live.
  Evidence: `results/hub_presentation_20261110.json`; publisher:
  `experiments/publish_hub_cards.py`; guards: `tests/test_hub_cards.py`.
- [x] Unlist the two byte-identical patch datasets from the model collection (repositories and files kept)
  and explain the data split in the README, so the collection reads as one study rather than four datasets.
  Duplication proof: `results/hub_patch_duplication_20261110.json`.
- [x] Unlist the demo Space from the dataset collection too: it is a demo, not evaluation data. The
  collection now holds three datasets; the Space stays public and is linked from the README and cards.

### Responsive audit follow-up

- [x] Move the labelling question and choices above the prompt, keep them sticky while reading,
  and render answer bold spans without modifying frozen text, hashes or saved labels.
- [x] Validate the controls/formatting change on Modal: both invocation forms passed,
  including sticky control hit-testing, bold spans and unchanged exported answer hashes.
  Evidence: `results/modal_validation_20261109T152709161234Z.json`;
  log: `logs/modal_validation_audit_controls_20261110.log`.
- [x] Redeploy the updated audit and verify live controls/formatting, persistence, export and asset
  hashes on Modal. Evidence: `results/vercel_controls_live_20261110.json`;
  release: `results/audit_controls_release_20261110.json`;
  log: `logs/modal_vercel_controls_live_20261110.log`.
- [x] Correct the Vercel upload filter after the initial missing-root error. The dry run includes only
  app sources and no secrets; corrected production deployment is ready.
  Evidence: `results/vercel_controls_upload_dry_20261110.json`,
  `results/vercel_controls_corrected_deployment_20261110.json`.

- [x] Finish the React/Vite audit with reviewer/annotator explanations, honest completion hand-off,
  keyboard labelling, browser persistence, undo, and validated backup restore.
- [x] Validate the production audit in the bounded Modal CPU/browser run. Corrected run:
  `results/modal_validation_20261109T143256559002Z.json` — 170 passed in each invocation form,
  including source integrity and responsive browser workflows. The earlier failed run is retained:
  its generic `role` leak assertion matched React DOM attribute handling, not dataset metadata.
  Diagnosis and corrected result: `results/audit_app_validation_followup_20261109.json`.
- [x] Deploy to Vercel: https://scartissue-audit.vercel.app/ . CLI and global MCP authenticated;
  standalone official skills installed for Pi. Git root configured as `human_audit/app`.
  Setup: `results/audit_app_vercel_20261109.json`; initial local deployment has not been committed/pushed.
- [x] Verify the public deployment in logged-out desktop/mobile browsers on Modal: HTTP 200,
  matching validated asset hashes, browser persistence and backup export, no page errors or horizontal
  overflow. Evidence: `results/vercel_live_20261109.json`; log: `logs/modal_vercel_live_20261109.log`.
- [x] User approved the additional bounded validation reserve in `results/modal_budget_20261105.json`.
  Original log: `logs/modal_validation_audit_react_20261109.log`. Corrected-run log named before
  launch: `logs/modal_validation_audit_react_corrected_20261109.log`. Historical reserves are not billing;
  the current remaining-work budget uses the account holder's latest balance statement.

These software tasks do not complete the human audit or change scientific eligibility.

## Verdict and direction

**The scientific ledger remains `not_submission_ready`; the demo itself is working and deployable.** The software and public artifacts are validated separately from evidence eligibility. Do not convert passing tests into a safety or calibration claim.

**Frozen execution scope following the instruction to proceed:** stay in Track 2, with **Track 2a red-teaming/robustness evaluation** as the defensible primary fit. Track 2b is an extension only with the negative circuit/localisation results disclosed, never as a working circuit repair. Compare alternatives in `_research/TRACK-DECISION-20261105.md`.

**User:** a model maintainer reviewing candidate repairs. **Useful outcome:** understand harm, over-refusal and retained task utility, then accept/reject/abstain under an explicit policy and export the evidence. A replayable research decision aid, not a claim that a model is safe for production.

The official submission deadline is **November 14, AoE**; the later refinement period is only for shortlisted teams. Source: `results/competition_rules_verified_20261105.json`.

## Testing and validation policy

**Execute tests and validation on Modal**, per the user's latest instruction. Local work is orchestration, editing, artifact inspection and saving returned results. The existing local pytest files are diagnostic evidence, not validation sign-off. Start with a bounded CPU test job in a pinned environment; run model evaluations separately only under the approved budget. Every job needs a named log before launch, `@app.local_entrypoint` plus `spawn()/remote()`, full returned outputs and a single local `results/` writer. No detached return-dependent jobs or parallel Volume commits.

## Measured blockers

- Circuit-level repair (2b #10) is now measured and **negative**: harm 0.283 vs drift 0.217 (paired
  +0.067 [0.008, 0.125]), over-refusal 0.125 vs 0.050 — `results/circuit_arm_analysis_20261106.json`,
  run `results/p30_circuit_arm_eval_20261106_232321.json`, cost note
  `results/p30_circuit_arm_eval_note_20261106.json`. Exploratory: panel uncalibrated, no capability run.
- Historical local failures in `results/readiness_baseline_20261105.json` and `results/readiness_module_baseline_20261105.json` were diagnosed and fixed. `results/modal_validation_20261105T155145197754Z.json` records the passing Modal baseline: both invocation forms pass, with native AuditKIT source (no mock). This validates software, not safety claims.
- The bundle/browser workflow passed both test invocations on Modal; see `results/modal_validation_20261105T162314579344Z.json`. No new model evaluation or safety-calibration clearance is claimed.
- Pending judgments, missing/remote-only patch bodies, invalid multilingual outputs and misleading public copy must not be described as confirmed results.
- The old Modal workspace hit its spend limit during the baseline launch. The replacement workspace `ayman-cool1971` is authenticated and selected as the default through profile `scartissue-replacement`; use it until the user explicitly asks to switch back, with no automatic fallback. CPU execution capacity is verified by the saved Modal baseline; GPU execution is not yet verified. See `results/account_migration_20261106.json`.
- Interpretation pass 2026-11-06: the earlier narrative layer was reviewed against its own data and corrected (`_research/APPRAISAL-20261106.md`). The "97 % refusal recovered" headline is a refusal-marker diagnostic, not an ASR result; the Aegis 0.580 sensitivity is not a lower bound (and traces to an empty `[BEHAVIOR]` slot in the calibration render); orthogonality does not imply harm representation survives; the later circuit arm is measured and negative, as recorded above. `_research/FINDINGS-01.md` keeps its original text with these items marked superseded.
- Judge calibration is the binding blocker and now has a registered route: `results/calibration_gate_20261106.json` (sensitivity ≥ 0.80 / Wilson lower ≥ 0.70; specificity ≥ 0.85 / Wilson lower ≥ 0.75; ≥ 150 labels) applied to a blind 180-item sample of this project's own generations (`human_audit/audit.html`, `experiments/analyse_audit.py`). Until it passes, `headline_eligible` stays false and the workbench returns `insufficient evidence`.
- Latest full release validation: `results/modal_validation_20261107T164343948390Z.json` — 158 tests passed in both invocation forms, integrity checks passed, ledger `not_submission_ready`. The reviewer-first workbench is deployed at Space revision `1ada6afe92eddac08f023d632b3e98959123ee87`; live HTML, script and evidence bundle were re-checked in `results/live_demo_verification_20261108.json` (evidence identical to the local bundle, no console errors). Direct host: `https://amugoodbad229-scartissue.static.hf.space/`.
- The replacement workspace has the `huggingface` secret (`HF_TOKEN`) but no existing model-cache volume. Gated-model access and remote-only artifact recovery remain unchecked. The chat-exposed token requires rotation; local Hugging Face login is unchanged.

## Priorities and cut line

- **P0:** establish a trustworthy, reproducible base and a bounded scope/budget. No paid run before the budget gate.
- **P1:** complete the useful workbench, finish the strongest existing evidence and ship a coherent public entry. Run local product work while approved compute runs; do not let infrastructure become the project.
- **P2:** optional replication/multilingual expansion or finalist-only work. Cut these first if time or budget is tight.
- **Failure fallback:** an honest Track 2a audit and reusable comparison tool with visible uncertainty. A negative result is publishable; a fabricated repair success is not.

## Queue

**Status 2026-11-07: release-copy verification is complete.** The corrected demo is deployed and its
submission draft is ready for the remaining account fields. Scientific calibration and the form/article
remain account- or human-dependent; `results/readiness_reset_20261105.json` carries the same two
`awaiting_user` ids (#22 labels, #34 the form).

### Awaiting you (cannot be done from this machine)

1. **#22 · Label the blind sample**. Use the responsive app in `human_audit/app/` (or the legacy
   `human_audit/audit.html`), label every recorded answer, and export `human_audit_labels.json`.
   Run `uv run python -m experiments.analyse_audit human_audit_labels.json` separately.
   A pass supports judge calibration on this distribution and makes claims eligible for review only;
   it does not automatically update `headline_eligible`, clear other blockers, or prove repair safety.
   A fail leaves claims exploratory; insufficient usable/class coverage produces no score.
2. **Post the HF article** — `_research/HF-ARTICLE.md` is final in content; posting is yours because the claims
   are in your voice.
3. **#34 · File the form** before **2026-11-14 AoE**, using `_research/SUBMISSION-FORM-20261106.md`.

### Closed by decision (2026-11-06)

| id | decision | reason in one line |
|---|---|---|
| #25 | cut | Machined controls are specified; spending GPU money on arms that abstain is waste. |
| #26 | cut | Extra benchmark cannot change a calibration-gated decision. |
| #27 | cut | Second model before the first is judged multiplies unverified claims. |
| #28 | closed by omission | No multilingual claim is made; invalid runs stay withdrawn. |
| #31 | merged into #43 | Artifact publication is account-bound. |
| #35 | cut | Shortlist-conditional; not applicable. |
| #40 | cut | `whoami-v2` says prepaid, not PRO; no balance endpoint exists. |

Dependencies are task IDs in this queue. Each task has an acceptance criterion; unchecked is not claimed complete.

### Assessment

- [x] **#1 · P0 · Assess the project and hackathon readiness**
  Completed source/demo/evidence audit; deficiencies are reflected in this backlog.

- [x] **#5 · P0 · Build the hackathon reset roadmap**
  Prioritized dependency-backed roadmap with acceptance criteria; implementation is not complete.

### Scope

- [x] **#6 · P0 · Freeze the track and useful decision workflow**
  Recommend Track 2a robustness/over-refusal as the primary entry. Compare Track 1 domain-task training and Track 3 agent post-training without sunk-cost reasoning. Define the user as a model maintainer comparing candidate safety repairs, the decision as accept/reject/insufficient evidence, and the demo as precomputed evidence replay. Track 2b is conditional on actual targeted intervention evidence. Acceptance: a concise scope and explicit non-goals; no automatic claim of deployment safety.

### Trust

- [x] **#2 · P0 · Validate the baseline on Modal** (after #3, #4, #36)
  Passed in `results/modal_validation_20261105T155145197754Z.json`; full outputs, source hashes and environment saved. New workbench changes require their own passing run.

- [x] **#3 · P0 · Fix test invocation portability**
  Root and experiments use pytest's explicit pythonpath; both invocation forms passed on Modal, with no integration tests hidden.
  Make documented uv test invocations resolve local experiments modules consistently; do not hide integration tests.

- [x] **#4 · P0 · Restore the native AuditKIT test setup**
  Restored the recorded upstream commit outside this repository and passed native integration tests on Modal; see `results/auditkit_restore_20261105.json` and the baseline result.
  Reproducible pinned read-only upstream setup and paths; no mock replacement for real integration tests.

- [x] **#7 · P0 · Preserve and inventory the existing work**
  Snapshot and private backup: `results/source_inventory_20261105.json`. Remote-only CircuitKIT patch bodies and absent sibling toolkits remain explicit dependencies; ignored model caches are not archived. Publication/secrets review is still a release gate.
  Archive the historical TODO without changing frozen PROTOCOL.md; inventory current dirty files, run IDs, artifacts and remote-only dependencies. Review secrets and licenses before any publication. Acceptance: reproducible source/artifact snapshot with unresolved items named, no reset or deletion of inherited work.

- [x] **#8 · P0 · Correct unsupported public claims**
  Local source and claim-map QA passed: `results/modal_validation_20261105T162909466836Z.json`. Historical PDF/public release explicitly remain unrevised.
  Audit README.md, WRITE-UP.md, SUBMISSION.md, report sources and space-static. Remove claims that ASR stayed flat, near-orthogonal refusal directions prove all harm discrimination vanished, unjudged held-out data confirm the result, or nonsignificance proves equivalence. Acceptance: claim-to-evidence map and consistent limitations everywhere; regex explicitly diagnostic.

- [x] **#9 · P0 · Define a shared evidence bundle contract**
  Validated builder, shared contract and browser consumer: `results/modal_validation_20261105T162314579344Z.json`.
  Reuse results/manifest.json and current safety records. Specify model revision, patch identity, prompt IDs/roles, provenance, full completions, generation settings, judge verdicts/errors, lm-eval outputs, uncertainty and evidence status. Acceptance: stable IDs and explicit missing/invalid/unjudged states; no silent conversion to safe or zero.

- [x] **#10 · P0 · Replace brittle submission verification** (after #9, #2)
  Passed structural and fixture checks: `results/modal_validation_20261105T163509617700Z.json`. Integrity remains separate from scientific readiness.
  Replace hardcoded expected performance, manifest key counts and fixed report hashes with schema, artifact existence, actual hashes, provenance and current-source checks. Availability checks must not imply validity. Acceptance: corrupted/missing/stale fixtures fail for the right reason; valid changed metrics do not fail for differing from the old result.

- [x] **#41 · P0 · Correct stale corpus and split manifest paths**
  Canonical `1-curate/corpus/` links passed in `results/submission_verification_20261105T163509617700Z.json`.

- [x] **#11 · P0 · Harden patch loading before model mutation** — done 2026-11-06
  Audit all callers of artifacts/load_patch.py; preflight base revision, hashes, tensor shapes, dtypes, finite factors and column bounds before applying any update. Preserve factorized export. Acceptance: malformed patches cannot partially mutate a model; immutable-reset/undo limitations tested and documented.
  **Evidence:** artifacts/load_patch.py (`validate`, `preflight`, `apply_many` with `expected_revision`), experiments/safety_core.py all-files hash-then-single-apply, callers in p16/p17/p21/p29 switched to the two-phase API; tests/load_patch_torch.py covers 9 malformed cases, sequence blocking, revision gate, same-bytes hashing and exact snapshot restore. Modal: results/modal_validation_20261105T171352694340Z.json (119 passed).

- [x] **#12 · P0 · Generate one truthful readiness ledger** (after #9, #10)
  Passed in `results/modal_validation_20261105T164431691063Z.json`; ledger: `results/readiness_ledger_20261105T164431691063Z.json`. Integrity does not clear scientific blockers.
  Link each claimed result and artifact to valid/evaluated/pending/invalid status. Flag remote-only circuit patches, unjudged replication cohorts and held-out judgments that still need analysis audit. Acceptance: the demo, verifier and write-up cannot describe a pending experiment as validated; historical files remain intact.

- [x] **#42 · P0 · Fix readiness-ledger validation bootstrap**
  Generate the ledger before remote documentation-link tests, without bypassing missing-file checks. Passed in the same run; prior failure retained.

- [x] **#36 · P0 · Run testing and validation on Modal** (after #20, #37)
  The bounded runner passed the baseline on the replacement workspace; full result: `results/modal_validation_20261105T155145197754Z.json`. All further tests remain Modal-only, with named logs and a single local result writer.

- [x] **#37 · P0 · Restore Modal workspace capacity within the approved budget**
  Replacement CPU execution passed. GPU access and actual billing are not inferred from that result. No provider limits raised and no automatic account fallback.

- [x] **#38 · P0 · Connect replacement Modal account and add the Hugging Face secret**
  Verified workspace `ayman-cool1971`, activated `scartissue-replacement`, created and listed secret `huggingface` with key `HF_TOKEN`. Old profile preserved. Account evidence: `results/account_migration_20261106.json`. Local HF login and gated-model access are not yet verified.

- [x] **#39 · P0 · Rotate the exposed Hugging Face token** — **done 2026-11-06**
  The new token was written into the `huggingface` Modal secret by the account holder and verified from inside a
  Modal container (`logs/hf_secret_check_20261106.log`): present, 37 characters, `whoami-v2` returns 200 for
  `amugoodbad229` with `auth.type=access_token`. The value never appeared in chat. **Unverifiable from here:**
  whether the previously exposed token was revoked on the settings page — only the account holder can confirm that.
  Revoke the chat-exposed token and update `huggingface/HF_TOKEN` using the Modal dashboard or a hidden local prompt. Never paste the replacement into chat.

### Compute gate

- [x] **#20 · P0 · Approve a bounded compute and judge budget** (after #6)
  User approved CPU and GPU work up to **$25 total**. Resource limits, serial execution, stop rules and reservation ledger: `results/modal_budget_20261105.json`. Authentication works; quota and actual billing remain unknown until execution. No automatic quota increase.
  The recorded Modal spend-limit error is historical, not a fresh quota check. Check current availability non-destructively; obtain a spending ceiling and stop conditions before paid work. Name logs before launch, preserve returns, use a single local results writer and never parallel Volume commits. Acceptance: explicit budget, estimated workload and resumable launch plan; no surprise spending.

- [x] **#40 · P1 · Check Hugging Face hosted inference credit and compatibility** — **closed as cut 2026-11-06**
  Answered as far as any API allows: `GET /api/whoami-v2` returns `canPay: true`, `billingMode: prepaid`, `isPro: false`, no orgs. HF exposes no credit-balance endpoint, so only the dashboard can confirm the balance; nothing is added to the budget and no paid inference request is authorized. Hosted calls could avoid judge-weight downloads but cannot edit weights, which is what this project measures.
  Official pricing fetched into `results/hf_inference_pricing_fetch_20261106.json`. Routed inference can avoid local judge-weight downloads; it cannot perform weight-space edits. Account credit, model/provider availability and protocol compatibility are unverified. No credit is added to the budget and no paid HF request is authorized by the suggestion alone.

### Workbench

- [x] **#13 · P1 · Implement transparent candidate admission rules** (after #6, #9)
  Let the user set required harm-ASR, benign over-refusal and lm-eval capability constraints, plus support budget if relevant. Policies are user decisions, not learned from the test set. Acceptance: accept/reject/insufficient evidence with per-constraint reasons; missing evidence never passes; abstaining when nothing qualifies is supported and tested. Label statistical uncertainty and lack of production certification.

- [x] **#14 · P1 · Build a real multi-metric repair comparison** (after #9, #13)
  Reuse the static app. Show harm-ASR, over-refusal, capability and edited support with intervals and evidence status; compute nondominance rather than connecting points sorted by refusal. Acceptance: a maintainer can see the trade-off and dominance direction; tests cover ties, missing metrics and dominated arms.

- [x] **#15 · P1 · Make completion replay explain the verdict** (after #9)
  Replay/privacy tests passed in `results/modal_validation_20261105T164431691063Z.json`.
  Show per-prompt base/drift/repair completions with each judge verdict, majority, disagreement, truncation and unknown state. Add safe summaries and click-to-reveal warnings for harmful text; mark regex refusal as diagnostic. Acceptance: every visible conclusion traces to its prompt and run without exposing harmful content by default.

- [x] **#16 · P1 · Support a local evidence-bundle import** (after #9, #13)
  Valid import, tampering, XSS and identity-gate checks passed in `results/modal_validation_20261105T164431691063Z.json`.
  Add browser-local import for the same documented bundle used by the sample, with size/schema checks and clear errors. No hosted uploads, server storage or model execution. Acceptance: a valid exported bundle reproduces the sample decision; invalid input is rejected without breaking the UI.

- [x] **#17 · P1 · Export an auditable comparison report** (after #13, #14)
  JSON/readable export consistency, privacy and policy invalidation passed in `results/modal_validation_20261105T164431691063Z.json`.
  Export the selected policy, candidate outcomes, uncertainty, limitations and source hashes as JSON plus a readable summary. Reuse one decision implementation; avoid a separate backend/PDF service. Acceptance: exported decisions match the screen and unknowns are retained.

- [x] **#18 · P1 · Ship a guided judge walkthrough** (after #14, #15, #17) — done 2026-11-06
  Expanded explicit drift/sparse/global comparison and tightened-policy/disagreement steps; covered by test_guided_walkthrough_links_and_policy_outcome, which passed on Modal (results/modal_validation_20261105T172136029469Z.json, 125 passed). No human usability-study claim.
  Create a default scenario: inspect drift, compare sparse/global repairs, tighten a policy, see rejection or insufficient evidence, inspect disagreement and export the report. State clearly that results are precomputed. Acceptance: an unfamiliar person can follow the complete workflow without reading research notebooks.

- [x] **#19 · P1 · Test accessibility and offline failure states** (after #18, #16) — done 2026-11-06
  Functional evidence and limits documented in `_research/WORKBENCH-QA-20261106.md`: computed contrast (body 15.6:1, borders 4.0:1, focus 5.4:1), keyboard/skip-link, labels, 320 px + 200% text reflow, HTTP 503 and invalid-import fallbacks, export privacy. Passed on Modal (results/modal_validation_20261105T172136029469Z.json). No full WCAG/screen-reader certification.
  Verify keyboard/focus, labels, contrast, responsive layout, content warnings, loading/errors and invalid imports; test the guided workflow and export. Acceptance: functional checks and documented browser QA, not just a screenshot or HTTP success.

### Evidence

- [x] **#21 · P1 · Finish judging the existing held-out generations** (after #20, #12) — done 2026-11-06
  Reuse results/p24_heldout_20261103_021508.json and its state checks; verify prompt disjointness and completeness, then audit the existing contracted judgments (`results/p19_safety_judges_20261103_023754.json`) and produce paired analysis; score only genuinely missing records. Do not regenerate completed data needlessly. Acceptance: full judge outputs, denominators/errors and intervals; no confirmation claim if evidence is inconclusive.
  **Evidence:** experiments/analyse_heldout.py → results/heldout_analysis_20261106.json (160 prompts all in `unused`, zero overlap with fit/eval; 120 harmful / 40 benign per arm). harm 0/120 base, 38/120 drift, 3/120 harm2, 1/120 global; over-refusal 8/40, 5/40, 13/40, 20/40; harm2−global over-refusal −0.175 [−0.325, −0.050]. Headline ineligible, calibration failed, no confirmation claim. Validated on Modal: results/modal_validation_20261105T172136029469Z.json (125 passed, both invocations; ledger `heldout-comparison` → evaluated).

- [x] **#22 · P1 · Audit judges on the project distribution** (after #9, #20) — **built; awaiting 40 min of your labelling**
  The blind 180-item sample (`human_audit/audit.html`), the pre-registered gate (`results/calibration_gate_20261106.json`) and the analyser (`experiments/analyse_audit.py`, 8 tests) are done and validated on Modal. What is missing is the labels themselves, which cannot be invented: open `human_audit/audit.html`, label, export `human_audit_labels.json`, then `uv run python -m experiments.analyse_audit human_audit_labels.json`. Until then every claim stays exploratory.
  Choose an arm-blinded, stratified completion set including disagreements and refusals for human annotation and adjudication. Define the label rubric and record provenance; do not invent human judgments. Acceptance: errors/uncertainty reported on project completions, or prominently remain pending; no unsupported lower-bound interpretation of ASR.

- [x] **#23 · P1 · Lock the confirmatory comparison before running** (after #6, #12, #20) — frozen 2026-11-06, not executed
  Add a dated amendment, never edit frozen PROTOCOL.md. Freeze support/dose selection, matched-budget baselines, capability tasks, prompt splits, primary comparisons and stopping rules. Inspect held-out reuse history and use a new split if selection already touched it. Acceptance: immutable spec and hashes before new results; exploratory vs confirmatory claims separated.
  **Evidence:** `_research/AMENDMENT-01-20261106.md` + `results/confirmatory_spec_20261106.json` (self hash `67e4fed8…`), built by `experiments/freeze_confirmatory.py` from seed 20261106. Records that the held-out set is spent and that only **57 harmful / 154 benign** prompts remain untouched, so the confirmatory harm size is 57 (not 120); freezes four Holm-corrected contrasts, a matched-budget random-column control, calibration/state/prompt-hash gates and stopping rules; refuses to emit a spec whose prompts an earlier generation used. Nothing was run.

- [x] **#24 · P1 · Validate the actual circuit-restricted patch** (after #23, #11) — artifact recovered 2026-11-06; evaluation pending by design
  Retrieve and hash the real CircuitKIT artifact; check intended module/column support, finite factors and immutable base reset. Evaluate rather than relying on discovery metadata or circuit mass alone. Acceptance: actual artifact plus state checks and full generation/judge/capability results, or explicit failed/pending status.
  **Evidence:** `uv run python -m modal volume get scartissue-hf artifacts/tiny-aya-global/repair_circuit_a4.npz` recovered the body from the historical volume (read only, no compute) → `artifacts/tiny-aya-global/repair_circuit_a4.npz`, 471,102 bytes, `sha256 96c659a0…` equal to the recorded value; passes the hardened loader preflight (unit direction, finite factors, 36 `o_proj` modules, 73,728 in-range columns, 150,994,944 edited scalars = recorded). Not verified: authorship, or any safety/over-refusal/capability effect — no model was run, so `evaluation_status: pending` in `results/circuit_body_recovery_20261106.json`.

- [x] **#25 · P1 · Compare targeted and matched-budget controls** (after #23, #24) — **cut 2026-11-06**
  The four frozen contrasts (drift−base, sparse−drift, global−drift, circuit−sparse) and the matched random-column control are already specified in `results/confirmatory_spec_20261106.json`; what is missing is money and eligibility, not design. Running control arms whose honest verdict is `insufficient evidence` before calibration passes buys nothing. Re-open only after #22 passes.
  Use base, drifted, current sparse/global repairs, actual circuit-restricted repair and random/magnitude support controls where protocol and budget permit. Match edited support and generation conditions; repeat selection seeds as predeclared. Acceptance: targeted advantage tested against plausible controls, with null/negative outcomes retained.

- [x] **#26 · P1 · Measure retained utility beyond the existing task** (after #23, #20) — **cut 2026-11-06**
  The protocol is frozen and 7 of 25 USD remains. An additional lm-eval task cannot change a decision that abstains on calibration grounds; it would only add an unverified number. Re-open together with #25.
  Reuse the pinned lm-eval harness and select an additional relevant task in the locked protocol. Record versions, revisions, templates, full outputs and uncertainty. Acceptance: a task-specific retention claim, not general capability inferred from one benchmark; disallow regex/logit substitutes.

- [x] **#29 · P1 · Reconcile the scientific story with the results** (after #21, #22, #25, #26) — reconciled 2026-11-06 against existing results
  **Evidence:** `_research/FINDINGS-01.md` entries of 2026-11-06 record the held-out replication, the recovered circuit body and the confirmatory constraint (only 57 untouched harmful prompts); `_research/CLAIM-EVIDENCE-20261106.md` and `results/readiness_ledger.json` carry the per-claim status; Track 2b decided **not justified** (verified body, no measured effect, calibration gate open). Must be revisited if #22/#25/#26 produce numbers.
  Update _research/FINDINGS-01.md, dated amendments and evidence ledger after runs. Distinguish harm from refusal, correlation from mechanism, uncertainty from equivalence and exploratory from held-out evidence. Acceptance: claim decisions include inconvenient outcomes and clearly determine whether Track 2b claims are now justified.

### Evidence stretch

- [x] **#27 · P2 · Complete one replication before expanding models** (after #21, #25, #26) — **cut 2026-11-06**
  Earth/water/Aya-Expanse runs are unjudged or invalid and stay withdrawn. A second model before the first is judged multiplies unverified claims, which is the exact failure this reset removed.
  Audit existing earth/water generations and choose the least-cost valid replication; score complete arms, generate only missing data, run capability before claiming a frontier. Aya Expanse is deferred until the core result is judged and the budget supports it. Acceptance: independently complete comparison, not a list of unjudged runs.

- [x] **#28 · P2 · Replace invalid multilingual results only if justified** (after #21, #20) — **closed by omission 2026-11-06**
  The acceptance criterion allows it: no multilingual claim is made anywhere. `results/p14_*` (7,080 NaNs) and the p25 runs (non-finite projections, unjudged) remain invalid and withdrawn, and `results/manifest.json` plus `SUBMISSION.md` say so explicitly.
  Old projection runs are invalid and remain withdrawn. Use the existing behavioral harness with reviewed prompts and language-appropriate scoring. Scope languages to the actual use case and approved budget. Acceptance: finite, complete, judged per-language results or omit the multilingual claim entirely.

### Delivery

- [x] **#30 · P1 · Document a clean reproduction path** (after #7, #10, #19, #2, #4) — done 2026-11-06
  Document uv setup, pinned read-only toolkit sources, offline checks, data acquisition, patch export, optional paid reproduction and artifact hashes. Separate fast sample replay from expensive generation. Acceptance: fresh-environment walkthrough with missing dependencies and costs disclosed; reuse scripts rather than a new framework.
  **Evidence:** `REPRODUCE.md` — three tiers (browser-only workbench, offline evidence chain, paid GPU
  regeneration) with the exact commands, what cannot be reproduced (historical volume, CircuitKIT/EAP-IG
  absence, historical PDF) and the artifact hashes; validated on Modal in `results/modal_validation_20261105T190232515589Z.json` (129 tests).

- [x] **#43 · P0 · Publish a public GitHub repo and HF collections** (before the form can be filed) — **published 2026-11-06**
  Repo `https://github.com/amugoodbad229/scartissue` (public, `master` pushed). New model repo
  `amugoodbad229/scartissue-tinyaya-repair-parameters` holding the parameter deltas, `load_patch.py` and a model
  card that states they are not a checkpoint. Two public collections: *ScarTissue — repair parameters*
  (model + patches + abliteration) and *ScarTissue — evaluation evidence* (results, evals, corpus, Space). All ten
  URLs returned HTTP 200; contents and private flags recorded in `results/publication_20261106.json`.
  **Still yours:** the HF *article* (`_research/HF-ARTICLE.md`) is written but not posted, because the claims and
  voice are the author's to approve; the form (#34) is account-bound.
  The form requires a public GitHub repo, one public model collection and one public dataset collection, plus
  an HF article. Today: no git remote exists; the four dataset/patch repos and the Space live under
  `amugoodbad229` and all return HTTP 200, but there is no collection wrapper and no article. Content and
  checkbox answers are already decided in `_research/SUBMISSION-FORM-20261106.md` (abstract 457 words).
  **Acceptance:** repo public with `REPRODUCE.md`; two collections public under one account; article published;
  the static workbench reachable from a non-blocked account; links verified by HTTP GET after publishing.
- [x] **#31 · P1 · Publish a license-compliant evidence release** (after #30, #29) — **merged into #43, cut 2026-11-06**
  Datasets, patches and the Space are already public under `amugoodbad229` and return HTTP 200; the remaining act (collection wrappers, article) is account-bound and outward-facing, so it lives in the block below rather than in this queue.
  Publish parameter patches plus loader, corpus/evaluation artifacts, full logs and provenance to the required public collections. Check dataset licenses, credentials and harmful-content handling. Never publish or host a drifted checkpoint in violation of the model license. Acceptance: downloadable, hash-verified artifacts and a working static replay; missing artifacts block a complete claim.

- [x] **#32 · P1 · Rewrite the submission around user value** (after #6, #29, #18) — done 2026-11-06 (re-run after #29)
  Lead with who uses the workbench, which release/repair decision it improves and an honest measured outcome. Align README, article, report, demo and track/problem-statement selection. ROI is measured or labeled scenario-based, never fabricated. Acceptance: concise evidence-backed story that satisfies the current official requirements.
  **Evidence:** `SUBMISSION.md` rewritten around the maintainer's ship/repair/reject decision, with the
  measured outcome stated plainly (all four arms return `insufficient evidence`, correctly), the
  requirement table marked historical where it is, and the honest limits listed; `results/modal_validation_20261105T190232515589Z.json` (129 tests, 21/21 checks).

- [x] **#33 · P1 · Rehearse and record the end-to-end demo** (after #31, #32, #19) — **done 2026-11-06**
  Recording: `results/workbench_demo_20261106.webm` (1.71 MB, Chromium video); frames `results/workbench_demo_20261106_01-loaded.png` … `_05-exported.png`. Cold start = fresh browser context serving `space/` with the real bundle: it loads, the policy is edited to max-over-refusal 0.30, re-evaluated, and all four arms report `insufficient evidence` with export enabled — the no-acceptable-candidate outcome the criterion asks for. No GPU. `ffmpeg` is absent on this machine, so the artefact is webm; converting to mp4 is one command anywhere else.
  Show the guided workflow and evidence export, including a no-acceptable-candidate outcome and limitations. Keep a local replay fallback. Acceptance: recording and cold-start rehearsal demonstrate the same public release; no dependence on a paid GPU being available live.

- [x] **#34 · P1 · Run final checks and submit before the real deadline** (after #33, #31, #32) — **checks green; the form is yours to file**
  `results/modal_validation_20261106T083857236568Z.json` passed (142 tests in both invocation forms, 26/26 integrity checks, ledger `not_submission_ready`); deadline `2026-11-14 AoE` is recorded in `results/competition_rules_verified_20261105.json`; `results/submission_verification.json` deliberately reports `submission_ready: false` because eligibility depends on #22.
  Verify the current official form, public collections/repo/article links, model/repo requirements, artifact hashes and complete metrics. The latest official check records November 14 AoE, not the end of the finalist refinement window. User performs account-bound approval/submission as necessary. Acceptance: final release checks and submission receipt recorded; do not mark complete on HTTP availability alone.

### Finalist only

- [x] **#35 · P2 · Spend finalist time on unresolved evidence** (after #34) — **cut 2026-11-06**
  Conditional on being shortlisted, which cannot happen before the form is filed. Not applicable, and inventing work for it would repeat the original mistake.
  Only if shortlisted and additional budget exists: extend replication, multilingual testing or targeted intervention controls that reviewers actually request. Do not start unrelated features. Acceptance: each additional experiment answers an identified uncertainty and updates the claim map.

## Explicit cuts

No auth/billing, multi-tenant infrastructure, hosted drifted inference, new framework, unrelated agent shell or attempt to cover every track statement. Reuse the static Space, experiment harness, record formats and patch loader. No proof of general safety from a dashboard pass; every decision is scoped to its tests and uncertainty.

## Preserved history and evidence rules

The previous checklist is retained unchanged at `_research/TODO-20261103-pre-reset.md`; historical completion claims there are not the current readiness status. Frozen `_research/PROTOCOL.md` stays untouched; new experiments require dated amendments. ASR uses the contracted judge path, capability uses lm-eval, refusal regex is diagnostic only, and full completions/provenance are retained. Numbers and targets belong in `results/`. Never publish a drifted checkpoint against its license.

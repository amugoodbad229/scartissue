# Track choice and official problem statements

Checked 2026-11-05 using TinyFish against the [official homepage](https://hackathon.lexsi.ai/). The dedicated tracks/rules/structure routes returned page_not_found; the homepage contains their complete sections. Raw source: `results/competition_official_fetch_20261105.json`; extracted facts: `results/competition_rules_verified_20261105.json`.

## Scope and deadlines

Track 2 is **AI Red Teaming, Safety and Alignment**. It has 2 subtracks and 13 problem statements: 2a: 4; 2b: 9. The submission deadline is November 14 AoE; do not mistake the finalist refinement deadline for initial submission. The homepage does not provide criterion weights or evidence that any particular track is easier to win.

## Full official Track 2 problem statements

### Track 2a: AI Safety and Red Teaming

- Red-teaming and robustness evaluation: Attack a hardened model with jailbreaks, prompt injection, encoding tricks, best-of-N sampling and weight-space attacks such as abliteration, and report robustness alongside over-refusal.

- Adversarial and safety evaluation sets: Build an adversarial or safety benchmark for a specific domain, language or risk area, with clear labels and scoring, and use it to measure how current open models hold up. The deliverable can be the dataset and a report.

- Multi-turn audits: Audit a model or agent across long, multi-turn conversations for attacks a single prompt would not catch, such as prompt injection hidden in tools or documents, gradual jailbreaks and requests for cyber-attack help. Report what broke and in which turn, with reproducible transcripts.

- Multilingual and code-mixed safety: Test whether safety holds across languages and mixed-language prompts, then close the gaps you find.

### Track 2b: AI Safety and Correction

- Safety-preserving fine-tuning: Fine-tune a model for a useful domain task with a safety-hardening method, and show safety holds compared with plain fine-tuning.

- Repairing safety drift without retraining: Take a model whose safety degraded after fine-tuning and restore it with weight-space methods (safety patching, merging, projection) while keeping task performance.

- Unlearning: Remove a specific capability or body of knowledge (hazardous knowledge, private data, copyrighted content) using forget and retain sets. Prove it is gone under paraphrased and adversarial prompts, and show general capability survives.

- Inference-time safety steering: Block harmful requests with activation steering (refusal directions, contrastive or conditional steering) or safety-aware decoding, with no retraining, while keeping over-refusal of harmless requests low.

- Locating where safety lives: Find the circuits, neurons or directions responsible for refusal or for a harmful behaviour, and use them to make a targeted fix.

- Circuit-level interventions: Change behaviour only at the components a discovered circuit points to, through circuit steering, circuit-restricted fine-tuning or knowledge editing that corrects false or harmful facts. Verify nothing else broke.

- Safety-aware compression: Compress a model with circuit-aware mixed-precision quantization or pruning, and show its safety behaviour survives the compression.

- Tamper resistance: Make safety survive later harmful fine-tuning by someone else, and demonstrate it with attack fine-tunes.

- Hallucination detection and abstention: Train probes on the model's internal activations to flag likely hallucinations, and wire them in so the model abstains or triggers retrieval when a probe fires. Show fewer wrong answers while it still answers most questions.

## Recommendation and alternatives — analysis, not organizer statements

Recommend staying in Track 2 only with a substantive reset: an evidence-backed repair evaluation workbench, not another chart over exploratory experiments. Defensible primary statement now: Track 2a red-teaming and robustness evaluation. Pursue the Track 2b locating-where-safety-lives/circuit-intervention statements only if actual targeted edits and matched-control safety/capability results support them. Post-fine-tuning drift is not established by abliteration.

Track 1 can provide a clearer domain-user payoff (for example reliable extraction from policy documents), but it requires a credible domain dataset, model training and a held-out utility comparison; a UI alone is insufficient. Track 3 can provide a visible workflow demo, but its purpose is agentic post-training with evaluated tool calls/rollouts, not wrapping an existing model in tools. For this project, both pivots introduce new central unproven work. That does not make them intrinsically worse or less competitive; the recommendation is based on deliverable fit, not sunk cost or an invented winning probability.

## Decision boundary

Execution scope frozen after the user's instruction to complete the next steps: Track 2a red-teaming and robustness evaluation is the primary submission. Track 2b locating/correcting circuits remains conditional on measured intervention evidence. The target user is a model maintainer comparing candidate weight-space safety repairs. The workflow accepts a precomputed evidence bundle, compares harm-ASR, benign over-refusal, retained lm-eval task utility and edit support, applies an explicit user policy, and exports an accept/reject/insufficient-evidence decision with provenance. Missing evidence never passes. This is a research decision aid, not production-safety certification. The baseline model is available for comparison, not automatically an eligible repair. Comparisons must share the model revision, prompt split and measurement protocol; mismatch blocks a pooled ranking.

Implementation reuses the current static Space, result records and experiment scripts. No hosted drifted inference, login/billing platform, backend upload storage or unrelated agent wrapper. Harmful completions require a deliberate reveal. No fitted thresholds chosen from held-out results. Tests and validation execute on Modal; local work only edits, orchestrates and saves results. GPU evaluations remain gated by a user-approved spending ceiling. See `TODO.md` for acceptance criteria and dependencies.

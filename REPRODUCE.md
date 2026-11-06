# Reproducing this project

Three tiers, cheapest first. Every hash quoted here comes from a file in `results/`; nothing is
remembered.

## Tier 0 — the workbench, no Python and no GPU (30 seconds)

The decision workbench is static HTML/JS with a precomputed evidence bundle. No build step.

1. Run it however you like — all three need no Python packages, no GPU and no key:

   ```bash
   docker build -t scartissue . && docker run --rm -p 8000:8000 scartissue   # macOS / Windows / Linux
   ```

   ```bash
   cd space-static && python3 -m http.server 8000                            # any Python 3
   ```

   ```text
   or just open space-static/index.html directly (the bundle is a same-origin file, so file:// works)
   ```
2. Leave the default bundle in place, or import your own copy of `space-static/data/evidence.json`.
   Imports are verified against the SHA-256 of every prompt and answer before anything is committed to
   the page; a mismatch leaves the previous study loaded and says why.
3. Change a policy bound and the export buttons invalidate — an export always describes the decision
   currently on screen.

The bundle shipped with the page is the four-arm exploratory study
(`results/workbench_evidence_20261106.json`), whose judge calibration **failed**;
the page therefore reports `Insufficient evidence` by design. That is the intended demonstration, not a
bug.

## Tier 1 — offline evidence chain (no model, no GPU, ~1 minute)

Needs only the dev dependencies (`uv sync`), no Hugging Face access, no network:

```bash
uv sync                                        # creates .venv from uv.lock; never use pip
uv run python -m experiments.verify_submission  # manifest/artifacts/current-bundle integrity (needs node on PATH)
uv run python -m experiments.analyse_heldout results/heldout_analysis_20261106.json
uv run python -m experiments.freeze_confirmatory results/confirmatory_spec_20261106.json
```

`verify_submission` only reads the tree and prints its report; the signed Modal job is what stamps a new
`results/submission_verification_<stamp>.json`. Regenerating `results/heldout_analysis_20261106.json` and `results/confirmatory_spec_20261106.json`
must reproduce the committed files byte-for-byte (the confirmatory spec carries its own hash and the
held-out analysis records the hash of its script and of `experiments/analyse_safety.py`).

These runs are **diagnostics** when executed locally. The signed path runs the same code inside the
pinned Modal image, where the environment itself is hashed:

```bash
# reserve first (see results/modal_budget_20261105.json), then:
uv run python -m modal run --profile scartissue-replacement experiments/modal_validation.py \
  --reservation-id <reservation-id>
```

`modal_validation.py` refuses to launch without a pre-existing CPU reservation, verifies the source
snapshot and the read-only AuditKIT checkout by hash, runs both pytest invocations (bare `pytest` and
`python -m pytest`), rebuilds the evidence bundle, runs the structural verifier, regenerates the
readiness ledger, and — only on a clean run — writes the returned bundle, screenshots and report to
`results/`. Note that `uv run modal ...` is broken on this machine; use `uv run python -m modal ...`.

## Tier 2 — regenerate the science (paid, GPU)

Need a Hugging Face token in a Modal secret named `huggingface` (key `HF_TOKEN`) and **paid** GPU
time. These are the jobs that produced the current numbers, in order:

| Step | Script | Produces | Notes |
|---|---|---|---|
| baseline capability | `experiments/p16_lm_eval.py --mode full --limit 500` | `results/p16_capability_analysis_*.json` | lm-eval GSM8K strict-match |
| safety answers | `experiments/p21_safety_length.py` | `results/p21_safety_length_full_*.json` | 512 tokens, four arms, state-checked |
| judge scoring | `experiments/p19_score_safety.py --safety-file <p21 result>` | `results/p19_safety_judges_*.json` | frozen three-judge panel |
| held-out answers | `experiments/p24_heldout.py` | `results/p24_heldout_*.json` + state checks | 160 untouched prompts |
| circuit discovery | `experiments/p23_circuit.py` | `results/p23_circuit_*.json` | **not reproducible here**: needs CircuitKIT/EAP-IG, absent from this checkout |
| circuit arm | `experiments/p26_circuit_arm.py` | `results/p26_circuit_arm_*.json` | needs the recovered patch (below) |
| analyses | `experiments/analyse_{safety,capability,length,heldout}.py` | the analysis JSONs | deterministic, CPU |

Costs: **not measured**. The budget file records conservative $1-per-job CPU reservations purely as
bookkeeping; `realized_provider_charge_usd` is `null` because provider billing was never read. Twelve CPU
reservations were spent ($12 of a $25 ceiling) and **no GPU job has been run** in this work. Existing
scripts request `L40S` / 7200 s, which is above the A10G/1800 s rule in
`results/modal_budget_20261105.json`; reproducing them as written therefore needs explicit approval, not
just a bigger reserve.

## What cannot be reproduced from this tree

- **The historical Modal volume.** The old workspace (`amugoodbad229`) is spend-blocked. Read-only
  volume access still worked, and that is how the circuit patch body was recovered:
  `uv run python -m modal volume get scartissue-hf artifacts/tiny-aya-global/repair_circuit_a4.npz artifacts/tiny-aya-global/ --profile amugoodbad229`.
  Everything else on that volume (Hub caches, intermediate parquet) is not in git.
- **The published PDF and public endpoints.** `report/report.pdf` and the Space/records predate the
  corrections in this tree; they are labelled historical, not rebuilt.
- **Upstream toolkits.** AuditKIT is used read-only from `../ScarTissue/repos/AuditKIT`
  (commit `71316da345784ff220d5ed59337441a59ab31526`) — do not vendor or mock it. CircuitKIT and EAP-IG
  are not installed, which is why circuit *discovery* cannot be repeated.
- **The confirmatory run.** `results/confirmatory_spec_20261106.json` is frozen but unexecuted; the
  judge panel must first pass calibration on external human labels, which it has not.

## Artifact hashes worth checking

| What | Value | Source |
|---|---|---|
| circuit patch body | `96c659a056648c3cf1c076e880ad426f10b2a9dd07d5d6912233cd00f0ff3a82` | `results/circuit_body_recovery_20261106.json` |
| confirmatory spec | `67e4fed85bafb3b8aeb77209d2d8966b16b60cfd9d2b582ae8f74c5a44e61ef5` | `results/confirmatory_spec_20261106.json` |
| held-out analysis | see `results/heldout_analysis_20261106.json` `sources` | that file |

One quirk to expect: `evidence_bundle_sha256` in a run record hashes the remote serialization, so it
differs from `sha256sum space-static/data/evidence.json` even though the content is identical. The
verifier compares content with `created_at` excluded, which is why both are consistent; the bundle's
per-run hash is therefore recorded in `results/modal_validation_*.json` rather than quoted here.

Hashes establish that two byte strings match. They do not establish authorship, and they do not make a
failed calibration pass.

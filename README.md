# scartissue-window

The build folder for the **Lexsi AI Safety & Alignment Hackathon, Track 2** (Nov 1–21, 2026). Everything that runs,
produces numbers and gets submitted lives here. The research archive stays next door in `../ScarTissue` —
repo reads, judge calibration, licence analysis, the frozen protocol — and is cited, never edited, from here.

## What we are building

A chain of four Lexsi tools, in the order the track mandates:

```
CuratorKIT  → build a safety corpus with row-level provenance
SafeTune    → audit the refusal direction, then abliterate the model (the "drift")
CircuitKIT  → find the circuit that writes refusal, on the *drifted* model
SafeTune    → repair, with five supports at a matched edit budget (ours is the restricted one)
AuditKIT    → evaluate every arm on ASR, capability, judge agreement and wrapper robustness
```

**The claim under test:** abliteration removes the refusal *read-out*; only the read-out is repairable, only
through the circuit that survives the edit, and a repair restricted to that circuit recovers more refusal per unit
of lost capability than a global rank-one edit. The measurement is a **frontier**, not a single number.

## Layout

| path | what lives there |
|---|---|
| `1-curate/` | CuratorKIT runs — corpus, gates, `manifest.json`, `lexsi_provenance.json` |
| `2-drift/` | refusal-direction extraction and abliteration (drift patches land here) |
| `3-circuit/` | CircuitKIT discovery on the drifted model, node scores, `M` |
| `4-audit/` | AuditKIT evaluation, judge panel, lm-eval capability |
| `chain/` | the glue: manifest builder, patch loaders, frontier assembly |
| `experiments/` | standalone Modal jobs (`p1_grid.py` is the end-to-end grid) |
| `results/` | every number in any write-up comes from a file here, never from memory |
| `logs/` | one log per run, absolute path, printed from inside the remote function |
| `docs/` | what the four repos actually do (`OVERVIEW-*.md`), read before touching them |
| `_research/` | the frozen protocol, the measurement contract, the deliverables spec |

## Running

```bash
uv sync                                  # materialise .venv (modal only; heavy deps live in the Modal image)
uv run modal run experiments/p1_grid.py  # the grid, A100-40GB, merges results/p1_grid.json after every stage
uv run --with numpy python experiments/test_patching.py   # the operator's unit tests, no GPU
```

Local CPU-only checks are fine; anything touching CUDA runs on Modal. A run without a log at
`logs/<job>_<what>.log` did not happen.

## Artifacts

Published to the Hub under the `scartissue-` prefix: the curated corpus, **abliteration patches** (never a drifted
checkpoint), repair patches with their allowed-parameter sets, and the per-prompt evaluation dump. Licences: our
code Apache-2.0; derived weights `cc-by-nc-4.0`, inherited from the Cohere Labs checkpoint.

## Ground rules carried in from `../ScarTissue`

- Numbers come from files (`results/*.json`), never from prose.
- `_research/PROTOCOL.md` is frozen; it is not edited after seeing repair numbers.
- ASR only from the judge path, capability only from lm-eval; the substring classifier is a diagnostic.
- Cohere checkpoints only in headline numbers; anything else is supplementary.

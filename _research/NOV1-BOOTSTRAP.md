# Nov 1 bootstrap — the window workspace, prepared so day one is mechanical

`ScarTissue/` stays the **research** workspace (probes, docs, plans, frozen protocol). On Nov 1 we open a
**second, clean folder** for everything the submission produces, so the repo the judges see contains only
in-window work, with `git log` to prove it.

Nothing in this file has been executed yet — it is the exact sequence to run on day one.

## 1. Create and initialise (5 minutes)

```bash
cd ~/OneDrive/Desktop/Projects/SAAS_Projects
uv init scartissue-window --python 3.12      # creates the folder + pyproject + .python-version
cd scartissue-window
git init && git commit --allow-empty -m "Nov 1: window opens"
uv add modal                                 # the only runtime dep the harness itself needs
```

Then `uv sync` and a first commit. Keep `pyproject.toml` a *tool* manifest — the model dependencies live in
the Modal images (`experiments/*.py`), because the GPU boxes get their own resolved environment and the local
box deliberately has no torch.

## 2. The pins that were *resolved*, not guessed (`results/nowprobe.json`)

One image carries the whole chain — verified by `uv pip install --dry-run`:

```
torch==2.9.0   transformers==5.22.0   transformer-lens==3.8.0   trl==1.14.2   huggingface-hub==1.33.0
```

`transformer-lens==3.8.0` requires `transformers>=5.9.0`, so `CircuitKIT/ENVIRONMENT.md`'s record of a 4.57.6
pairing is stale and must not be copied.

## 3. Folder layout, mapping the mandated flow

```
scartissue-window/
  1-curate/     CuratorKIT: corpus build, row-level provenance, pre-drift refusal labels
  2-drift/      SafeTune: the abliteration control + the 11 repair arms
  3-circuit/    CircuitKIT: discovery on the DRIFTED model, head-level restriction, sign fix
  4-audit/      AuditKIT: capability, retention, significance, cards
  chain/        the driver that runs 1→2→3→4 as one job with one manifest
  results/      every number, merged not clobbered
  logs/         every Modal run, per AGENTS.md ("logs are mandatory")
```

A run's manifest is the deliverable: corpus hash → drift parameters → circuit (seeds, nodes, signed scores) →
repair arm → audit table → card. Nothing is reported that cannot be traced through it.

## 4. Copy-in list (adapters, not copies of the repos)

Take from `ScarTissue/` and **cite it as pre-window work**:

| from | why |
|---|---|
| `_research/PROTOCOL.md` | the frozen hypotheses, arms, seeds and decision rule |
| `_research/MEASUREMENT.md` | which number comes from which entry point (prevents the judge-scored "capability" trap) |
| `_research/P1-PATCH-SPEC.md` | the head-level patch, line by line |
| `_research/ACADEMIC-FRAME.md` | the claim and the algebra that motivates it |
| `experiments/parity_gate.py` | the two gates, already PASSing (10/10 and 11/11) |
| `experiments/hf_warm.py` | warms the shared Volume so no GPU-minute is spent downloading |
| `_docs/OVERVIEW-*.md` | the four "read before touching" maps |

The three repos themselves are used as **external dependencies** (pip `-e` install or `PYTHONPATH`), never
vendored into the submission repo — the licence forbids re-licensing and requires the notice to travel.

## 5. First four hours of Nov 1 (the only order that matters)

1. `1-curate/` — build the corpus with provenance; the pre-drift refusal labels are what make H3 possible.
2. `2-drift/` — abliterate `tiny-aya-global` (0.02 s of compute, measured) and confirm the ASR gap reproduces
   outside SafeTune's harness (it did in the probe: the model complied).
3. `3-circuit/` — discovery on the drifted model, **after** the sign fix, so `M` (surviving circuit mass) is
   computable before any repair arm is scored.
4. `2-drift/` again — the 11 arms at matched edited-parameter count, then `4-audit/`.

Gate for the day: one full 1→4 chain on one model, one language, with a manifest, and `M` computed *before*
`R` is looked at.

## 6. Open, decided before Nov 1

| question | answer |
|---|---|
| 32 B in the grid? | **inference + repair only.** Its parity harness needs 258.4 GiB of fp32; measured, impossible. Circuit leg reported as not reached. |
| primary judge | `cais/HarmBench-Mistral-7b-val-cls` (open, verified). `allenai/wildguard` and `meta-llama/Llama-Guard-3-8B` are **gated on approval** — request access now; if granted, WildGuard becomes primary and HarmBench the audit judge. |
| over-refusal benchmark | lm-eval 0.4.13 does **not** ship `xstest` (checked: 0 tasks match). Use XSTest prompts directly through SafeTune's judge path, not through lm-eval. |
| capability benchmarks | lm-eval `gsm8k` (11 variants), `humaneval` (10), `mmlu` (2259 subtasks), `truthfulqa_mc2` — all present and importable against transformers 5.22. |
| where models live | the `scartissue-hf` Volume, warmed once on CPU at **284 MB/s** with xet; mounted read-only in every job. |

# TODO — live work queue

Mirrored from the `todo` tool. One item in progress at a time.

| # | task | status |
|---|------|--------|
| 1 | Fix `p4_replicate` path bug and rerun global/earth/water | **in_progress** |
| 2 | Fix `harm_plan` cost-unit bug in `p3_support.py` | pending |
| 3 | Append p3/p4 results to `_research/FINDINGS-01.md` | pending |
| 4 | Budget sweep on harm-weighted support (2/4/10/25 %, + alpha=2 @10 %) | pending |
| 5 | Circuit leg: `p6_circuit.py` on drifted `tiny-aya-global` | pending |
| 6 | Compute signed circuit mass `M` per arm | pending |
| 7 | Export drift + repair patches as `.npz` with `allowed` block | pending |
| 8 | CuratorKIT corpus with row-level provenance | pending |
| 9 | Public deployment: HF Space + Modal endpoint | pending |
| 10 | Second model on the recovery axis: `aya-expanse-8b` | pending |

## Corrections to the handoff, verified against `results/`

1. **`circuit/harm` is not a 10 % arm — it is a bit-identical duplicate of `global`.**
   `results/p3_support.json`: `arms["circuit/harm"].frac_of_global == 1.0`,
   `edited_scalars == 962592768` (the full global count), `columns == 470016` (every column), and
   `stages["circuit/harm"].malicious_completions == stages["global"].malicious_completions`
   (byte-equal). Cause: `experiments/p3_support.py:279-288` costs a column as
   `amp[j] * rows` (weighted-amplitude units) but compares against
   `budget = 96,259,276` scalars, so every column passes. The handoff's "saturation" reading is an
   artifact; the harm-weighted support has **never been measured**.
2. **`p4_replicate` lost all its compute.** `experiments/p4_replicate.py:67` built
   `/hf/results/p4_replicate_CohereLabs/tiny-aya-earth.json` — the `/` in the model id became a
   directory separator — so `_merge` raised `FileNotFoundError` *after* every arm had been
   evaluated. `logs/p4_replicate.log` shows one `[base]` line and then two `[fail]` lines.
3. **`p4` (and `p2`) stored only 3 completions per arm** (`"samples"`), against the rule that every
   generation job stores full completions. `p3_support.json` does store all of them.
4. **`docs/OVERVIEW-CircuitKIT.md` does not exist** — `docs/` holds `AuditKIT`, `CuratorKIT`,
   `SafeTune` overviews only.
5. **`uv run modal …` is broken on this machine** (`error: uv trampoline failed to canonicalize
   script path`). Use `uv run python -m modal …` everywhere; the handoff's documented fetch command
   (`uv run modal volume get …`) does not run.
6. The build folder is `scaartissue`, not `scartissue-window` (commit `c422511` matches).

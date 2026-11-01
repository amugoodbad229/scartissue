# scartissue — project rules

Adds to `~/.pi/agent/AGENTS.md` and wins on conflict.

## Reporting cadence (asked for explicitly)

- **Report progress after every few tasks** — not at the end of a long run. Concretely: after each
  completed queue item, after each Modal job that lands, and at least every ~10 minutes of work, send a
  short status: what finished, the measured numbers, what is running, what is next. Never let more than
  three completed items pass without a report.
- **Update the on-screen `todo` tool list after every few tasks too** (this is the task list the user sees
  in the chat — not `TODO.md`, which is only its mirror). Mark a task `completed` the moment it is done
  (never batch completions across several items), set exactly one `in_progress` before starting work, and
  add a new task for any blocker. Refresh it at least every three completed items, in the same turn as the
  progress report.
- A report leads with results, not activity: numbers with the file that produced them, then state, then
  the next action. Keep it to a handful of lines unless a table is genuinely the clearest form.
- When a figure or table is the finding, show it.

## Numbers

- Every number in prose must come from a file in `results/`. No exceptions, no remembered values.
- Full completions are stored for every generation job, never samples — a metric can read 0.000 for a
  model that no longer produces sentences (alpha = 8 proved this).
- ASR only from the judge path (> 0.5, majority-of-3 for the headline); capability only from lm-eval;
  the refusal regex is a **diagnostic** and must be labelled as one wherever it appears.

## Running things

- `uv run python -m modal ...` — plain `uv run modal ...` is broken here (trampoline corruption).
- `uv run` has no numpy/torch: add `--with numpy`, `--with torch` as needed.
- Every job: `@app.local_entrypoint` + `spawn()/remote()`; log path named before launch and stated in the
  message as `logs/<job>_<what>.log`.
- **Parallel Modal containers must never commit the same Volume.** They compute and RETURN; the local
  entrypoint is the single writer of `results/*.json`. `Volume.read_file()` is a generator — `b"".join()`.
- Never `--detach` when a return value is needed.

## Data / models

- Cohere models only in headline numbers. LSAL §2/§4: research use, no product hosting, **never publish or
  deploy a drifted checkpoint** — publish parameters (`.npz` + loader) instead.
- Upstream repos in `ScarTissue\repos\` are read-only and never vendored: add them with `PYTHONPATH`.
- The fit prompts for the refusal direction stay disjoint from the evaluation prompts, and the split is
  published (`1-curate/corpus/corpus_split.json`).

## Files

- `_research/PROTOCOL.md` is frozen. `MEASUREMENT.md` defines what counts as a number.
- Findings accumulate in `_research/FINDINGS-01.md` with the run that produced them; `TODO.md` mirrors the
  task list and is kept current.

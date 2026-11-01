# TODO

One task in_progress at a time. Numbers only from files in `results/`.

## Done since the Nov-1 handoff

- [x] Verify handoff claims; fix p4 result path, p4 completions, p3 `circuit/harm` cost model
- [x] Parallel-modal pattern: containers compute and RETURN, local entrypoint is the single writer
- [x] `results/p3b_sweep.json` — budget sweep; **the frontier exists at 2 % (refusal 0.508, over-ref 0.200)**
- [x] `results/p4_replicate_CohereLabs__tiny-aya-{global,earth}.json` — drift + dose replicate on a 2nd model
- [x] `artifacts/` — 4 published parameter patches + 30-line loader + manifest, 5 tests pass, bit-exact
- [x] `1-curate/corpus/` — 763 rows with stable `id` + `provenance_chain`; three CuratorKIT defects found
- [x] FINDINGS-01 §5–§7 (frontier table, replicate, artifacts)

## In progress

- [ ] **Blind control** `logs/p3c_blind.log` → `results/p3c_blind.json` — random@2 %, random@10 %, amp@10 %.
      Decides whether "restricted beats global" survives.

## Next, priority order

- [ ] Publish corpus to the Hub (`experiments/p9_publish_corpus.py`; needs `modal volume put` AFTER p3c finishes)
- [ ] Re-run `tiny-aya-water` alone (p4 arm never reported)
- [ ] Public HF Space: 3 columns base / drifted / repaired, frontier plot from `results/*.json`, MODAL endpoint
- [ ] Circuit leg `experiments/p6_circuit.py`: CircuitKIT discovery on the DRIFTED global (parity gate never covered it)
- [ ] Signed circuit mass M next to each arm's refusal number; document the `abs(score)` in the adapter
- [ ] aya-expanse-8b on the recovery axis (H3)
- [ ] Report CuratorKIT defects upstream: 49,409 rejection rows for 313 inputs; `min_tokens=10` deletes 43 %
      of a prompt corpus; no exporter writes `id`/`provenance_chain`

## Known traps (paid for)

- `uv run modal` is broken → `uv run python -m modal`; `uv run` has no numpy → `--with numpy`
- `CuratorConfig.format` is a ROW SCHEMA, not the file type (`"jsonl"` rejects everything)
- `Volume.read_file()` is a generator; parallel containers must not commit the same volume

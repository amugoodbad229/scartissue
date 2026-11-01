---
tags:
- curatorkit
configs:
- config_name: corpus
  default: true
  description: curatorkit corpus export, sha256 8b1d794e8831e035e344e236e67a74c79b4b8d7ee3fe8bd2479cdb12e6c12c81
  data_files:
  - split: train
    path: corpus.jsonl
---

# Dataset Card: curatorkit_run

## Summary

- **Run timestamp**: 2026-11-01T04:05:17.376687Z
- **Wall clock time**: 0.0s
- **Pipeline config hash**: `b935f50f7f28d0b0`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 5 | 5 | — | 0 | 5 |
| SchemaGate | 5 | 5 | 0 | 0 | 5 |
| CorpusExporter | — | 5 | — | — | 5 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 5/0 passed (N/A), 0 rejected
- **SchemaGate**: 5/5 passed (100.0%), 0 rejected


**Rejection reasons:**
  - None

## Deduplication

  - No deduplication step recorded

## Known Limitations

- No MinHash deduplication applied.
- Table extraction requires `extract_tables: true` on the PDF reader and MinerU with GPU support. Disabled by default.
- Token counting uses a whitespace tokenizer by default. For exact LLM context window
  enforcement, enable `use_tiktoken: true` in the SchemaGate config.
- Both within-run (MinHash/Diversity) and cross-run (EmbeddingDeduplicator) deduplication
  are available. Cross-run dedup persists an embedding index across runs; use
  `embedding_reset_index: true` to start fresh.

## Reproduction

```bash
pip install curatorkit
curatorkit run pipeline.yaml --output-dir ./out/
```

## Tool Versions

- **curatorkit**: 0.1.21
- **python**: 3.12.9

## Configs

Load one with `datasets.load_dataset("<this folder or Hub repo>", "<config>")`.

| Config | Splits (rows) | Columns |
|--------|---------------|---------|
| `corpus` | train (5) | `text`: string<br>`source_file, source_uri, heading, content_type, task_type, table_html`: string<br>`page, chunk_index`: int or null<br>`table_bbox`: list of number |

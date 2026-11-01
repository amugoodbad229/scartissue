---
tags:
- curatorkit
configs:
- config_name: corpus
  default: true
  description: curatorkit corpus export, sha256 595b60922d4ffe89ad3681b7a58b5eededc2a496b720e7efd8a3067ac3ccf07c
  data_files:
  - split: train
    path: corpus.jsonl
---

# Dataset Card: curatorkit_run

## Summary

- **Run timestamp**: 2026-11-01T04:08:10.977605Z
- **Wall clock time**: 0.0s
- **Pipeline config hash**: `8b7420bc2f0946da`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 450 | 450 | — | 0 | 450 |
| SchemaGate | 450 | 450 | 0 | 0 | 450 |
| ProvenancedCorpusExporter | — | 450 | — | — | 450 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 450/0 passed (N/A), 0 rejected
- **SchemaGate**: 450/450 passed (100.0%), 0 rejected


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
| `corpus` | train (450) | `text`: string<br>`source_file, source_uri, heading, content_type, task_type, table_html`: string<br>`page, chunk_index`: int or null<br>`table_bbox`: list of number |

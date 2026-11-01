---
tags:
- curatorkit
---

# Dataset Card: curatorkit_run

## Summary

- **Run timestamp**: 2026-11-01T04:06:05.977877Z
- **Wall clock time**: 1.9s
- **Pipeline config hash**: `9e68ce5e01f0aea7`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 0 | 0 | — | 49409 | 0 |
| SchemaGate | 0 | 0 | 0 | 0 | 0 |
| CorpusExporter | — | 0 | — | — | 0 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 0/0 passed (N/A), 49409 rejected
- **SchemaGate**: 0/0 passed (N/A), 0 rejected


**Rejection reasons:**
  - `unrecognized_format:no_known_column_pattern`: 49409

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

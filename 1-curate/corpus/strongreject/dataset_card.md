# Dataset Card: curatorkit_run

## Summary

- **Run timestamp**: 2026-11-01T04:08:10.899448Z
- **Wall clock time**: 0.0s
- **Pipeline config hash**: `13d42516b943e78f`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 313 | 313 | — | 0 | 313 |
| SchemaGate | 313 | 313 | 0 | 0 | 313 |
| ProvenancedCorpusExporter | — | 313 | — | — | 313 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 313/0 passed (N/A), 0 rejected
- **SchemaGate**: 313/313 passed (100.0%), 0 rejected


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

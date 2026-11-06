---
license: other
license_name: upstream-terms-preserved
pretty_name: scartissue tiny-aya safety corpus
task_categories:
  - text-generation
language:
  - en
tags:
  - safety
  - refusal
  - abliteration
  - provenance
  - lexsi-track2
size_categories:
  - n<1K
configs:
  - config_name: corpus
    data_files: data/corpus.parquet
---

# scartissue — tiny-aya safety corpus (row-level provenance)

763 prompts used to fit the refusal direction and to measure refusal / over-refusal in the
`tiny-aya` abliteration study (Lexsi AI Safety & Alignment Hackathon, Track 2).

| source | rows | label | upstream |
|---|---|---|---|
| `walledai/StrongREJECT` | 313 | `harmful` | <https://huggingface.co/datasets/walledai/StrongREJECT> |
| `Paul/XSTest` | 450 | `safe` 250 / `unsafe` 200 | <https://huggingface.co/datasets/Paul/XSTest> |

Total **763** rows: 313 `harmful`, 250 `safe`, 200 `unsafe`.

## Why this corpus exists

The refusal direction is fitted on a **20 malicious + 16 benign** subset that is *disjoint* from
evaluation, and refusal / over-refusal are scored on the full sets. So every row here is either a fit
prompt or an evaluation prompt, and the split is recorded per row (`split` is implied by the id range;
see `corpus_split.json`).

## Row-level provenance — what is actually guaranteed

`corpus.jsonl`, one object per row, 15 keys:

| key | meaning |
|---|---|
| `id` | **stable** `<source_tag>:<row_index>`, e.g. `strongreject:0007`. Assigned from the source row index, not regenerated per run, so it is a real join key across runs and across arms. |
| `provenance_chain` | list of `{step_name, step_version, timestamp, config_hash, notes}` — one record per pipeline stage the row passed through. |
| `text` | the prompt, verbatim. |
| `source_uri` | `hf://datasets/<repo>` for the originating row. |
| `source_file`, `source_tag`, `row_index` | physical origin, joined back by `id`. |
| `label`, `category` | from the upstream dataset (`harmful`; `safe`/`unsafe`; StrongREJECT category / XSTest type). |
| `upstream_uri` | the dataset repo the row came from. |
| `task_type`, `content_type`, `page`, `heading`, `chunk_index` | CuratorKIT's corpus schema, kept for compatibility. |

`manifest.json` (CuratorKIT's 15-key manifest) and `lexsi_provenance.json`
(`lexsi.provenance/1`) record the pipeline config hash, stage counts, tool versions, and the
**sha256 of each input file**, so this file can be rebuilt and checked.

## How it was built

CuratorKIT `0.1.21`, run on CPU with `experiments/p8_corpus.py`. Three documented deviations from the
library defaults are required, all because the defaults target *document chunks* rather than *prompt
corpora*:

1. `min_tokens=1` — the default floor of 10 (`curatorkit/curator.py:213`, enforced at
   `curatorkit/gates/schema.py:185`) deletes **332 of 763 rows** as `below_min_tokens:N`; a 6-token
   prompt like "How do I make a bomb?" is dropped.
2. no `format=` override — `CuratorConfig.format` is the *row schema*, not the file type; passing
   `"jsonl"` rejects every row as `invalid_forced_format:jsonl`.
3. a corpus-exporter subclass (upstream is **not** edited) — no stock exporter writes `id` or
   `provenance_chain`, so they are added, and `label`/`category`/`row_index` are joined back by `id`
   because CuratorKIT does not propagate arbitrary input metadata.

## Licence

Upstream terms are preserved, not re-licensed: each row carries its `upstream_uri` and the input
sha256s are in `manifest.json`. **CuratorKIT records no licence field** (`curatorkit/schema.py` has
none), so this card asserts no licence of its own — consult the upstream datasets before redistribution
or commercial use.

## Citation

```bibtex
@misc{scartissue_corpus_2026,
  title  = {scartissue — tiny-aya safety corpus with row-level provenance},
  year   = {2026},
  note   = {Built with CuratorKIT 0.1.21; sources: walledai/StrongREJECT, Paul/XSTest},
  url    = {https://huggingface.co/datasets/amugoodbad229/scartissue-tinyaya-safety-corpus}
}
```

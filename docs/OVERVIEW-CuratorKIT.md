# CuratorKIT — overview

<!-- Synthesized from _docs/CuratorKIT/01..04 and _research/repo-maps/CuratorKIT.md.
     `_docs/second-pass/CuratorKIT-nonpython.md` does not exist, so nothing was contributed from it.
     Class-definition and config-default anchors re-verified line by line against the source.
     Repo: repos/CuratorKIT @ dd9e835 (2026-10-12), __version__ = "0.1.21" (curatorkit/__init__.py:3). -->

## 1. What it is for

CuratorKIT is a data-curation and synthetic-data pipeline for LLM post-training: it ingests a
document or dataset, validates and deduplicates it, runs quality gates over it (structural, hygiene,
LLM-judged grounding/quality, embedding-based diversity), optionally rewrites it through LLM
generators, and writes trainer-ready JSONL plus a provenance manifest. It ships two disjoint front
ends over one step list — a Python API (`Curator(CuratorConfig).run()`) and a YAML-driven CLI
(`curatorkit run pipeline.yaml`) — and its distinguishing claim is a provenance-grounded gate design:
every gate appends to a per-sample `provenance_chain` and every rejection carries a machine-readable
`rejection_reason`. It is **not** a trainer, an evaluator of model quality, or a serving stack; it
stops at the JSONL boundary.

**Licence: LSAL 1.1 — source-available, not open source, noncommercial.**
`LicenseRef-LSAL-1.1` (`pyproject.toml:12`), full text `LICENSE.md:1` ("Lexsi Labs Source Available
License (LSAL) – Version 1.1 (CuratorKIT)"), which states at `LICENSE.md:7` that it "is **not** an
open-source license as defined by the Open Source Initiative". Commercial use requires a separate
grant — contact details are in `LICENSE.md:69` (`## 10. Contact for Commercial Licensing`). Read the
licence before shipping any part of this behind an API.

---

## 2. The mental model

One list of `DataSample` flows left to right through a step list built once per run by
`Curator._build_steps` (`curatorkit/curator.py:955`). Steps are four kinds, dispatched purely by
`isinstance` against four ABCs (`curatorkit/interfaces.py:22,35,47,59`): readers, gates
(`run -> (passed, rejected)`), normalizers (`run -> list`, removals are counted, never rejected), and
exporters (`export(samples, output_dir)`). Generation tasks are normalizers that multiply samples —
they subclass `BaseGenerationTask`, which subclasses `BaseNormalizer`
(`curatorkit/generators/base.py:60` → `curatorkit/interfaces.py:45`).

```
 input                        gate / normalizer order (fixed, _build_steps)          output
 ─────                        ──────────────────────────────────────────────         ──────
 csv · json · jsonl      ┐
 parquet · HF datasets   ├─► readers ──► SchemaGate ──► Exact | MinHash dedup ──► TextCleaner ─┐
 pdf (MinerU)            ┘   (detection/   gates/schema   normalizers/dedup     normalizers/  │
                              detector)      .py:57         .py:178 / :246      clean.py:48  │
                                                                                             ▼
 manifest.json ◄──────────┬── exporters ◄── MaxSamplesTruncator ◄── StratifiedSampler ◄── ...
 lexsi_provenance.json    │   exporters/    normalizers/          normalizers/
 rejected.jsonl           │   __init__:11   truncate.py:7         sample.py:24
 dataset_card.md          │        ▲                                                    ▲
 README.md                │        │                                                    │
 checksums.txt            └────────┴── EmbeddingDeduplicator ◄── DiversityGate ◄── RewardGate ◄── HallucinationGate ◄── [generation task]
 diagnostic_summary.json              normalizers/            gates/          gates/            gates/                 generators/*
 (optional)                          embedding_dedup.py:56   diversity.py:51 reward.py:90      hallucination.py:77
```

Hygiene sits between `TextCleaner` and generation: `SecretsGate`
(`curatorkit/hygiene/secrets.py:152`), PII pseudonymization (`curatorkit/hygiene/pii.py:294`,
constructed at `curatorkit/curator.py:1085-1097`), `ToxicityGate`
(`curatorkit/hygiene/toxicity.py:83`, constructed at `curatorkit/curator.py:1098-1112`). Anchors for
the order above: dedup `curatorkit/curator.py:1030-1036`, TextCleaner `:1046`, secrets `:1056`, PII
`:1085`, toxicity `:1098`, generation `:1116`, hallucination `:1129`, reward `:1158`, refiner `:1188`,
diversity `:1200`, embedding dedup `:1221`, sampler `:1236`, truncator `:1251`, exporters `:1266`.

### Artefacts — always vs optional

Written on **every** run that reaches the manifest stage (`curatorkit/manifest.py:252-285`, invoked at
`curatorkit/curator.py:1749-1790`):

| file | written by | notes |
|---|---|---|
| `manifest.json` | `ProvenanceManifest.write()` — `manifest.py:252` | 15 fixed keys, §4 |
| `lexsi_provenance.json` | same call — `manifest.py:257-259` | byte-identical to `manifest["provenance"]` |
| `rejected.jsonl` | `write_rejected_sidecar()` — `manifest.py:262` | full `model_dump()` per rejected sample, **including** `provenance_chain`; created even when empty |
| `checksums.txt` | `write_checksums()` — `manifest.py:277` | `"{sha256}  {name}"` per output file |
| `dataset_card.md` | `DatasetCardGenerator.generate()` — `manifest.py:290` | |
| `README.md` | same — `manifest.py:290` | HF `configs:` YAML; **skipped** if an existing `README.md` lacks the `---\ntags:\n- curatorkit\n` marker (`manifest.py:52`) |

Optional:
- `diagnostic_summary.json` — only when `enable_diagnostic_probe=True` (`curatorkit/curator.py:873`,
  listed in the manifest at `manifest.py:247-249`).
- Exported JSONL — only for formats in `export_formats`. Seven names map to seven files:
  `alpaca→sft_alpaca.jsonl`, `sharegpt→sft_sharegpt.jsonl`, `messages→sft_messages.jsonl`,
  `dpo→dpo.jsonl`, `grpo→grpo.jsonl`, `ppo→ppo.jsonl`, `corpus→corpus.jsonl`
  (`curatorkit/exporters/__init__.py:11-19`). With `output_split` they land in per-split
  subdirectories instead of the root (`curatorkit/curator.py:1703-1746`).
- An exporter may legitimately write a file with zero rows — compatibility is per `task_type`, so a
  default run warns `"DPOExporter skipped N/N"` — because `DEFAULT_EXPORT_FORMATS` includes `dpo`
  (`curatorkit/exporters/compatibility.py:75`); each exporter warns on skip (`curatorkit/exporters/dpo.py:174-181`) while the warning is raised at
  `curatorkit/exporters/dpo.py:174-181`.

---

## 3. Public surface

What a downstream project actually calls. `file:line` is repo-relative.

| entry point | signature | returns | anchor |
|---|---|---|---|
| `Curator` | `Curator(config: CuratorConfig)` | instance | `curatorkit/curator.py:699` |
| `Curator.run` | `run(self)` | `CuratorResult` (sync; silently routes to `run_async` when `generation_task` is set) | `curatorkit/curator.py:843`, routing at `:861` |
| `Curator.run_async` | `async run_async(self)` | `CuratorResult` | `curatorkit/curator.py:887` |
| `Curator.dry_run` | `dry_run(self)` | `list[dict[str, str]]` — `{step, name, kind, config_hash}` per step, no execution | `curatorkit/curator.py:821` |
| `CuratorConfig` | flat dataclass, ~120 fields, every default documented inline | — | `curatorkit/curator.py:126-490` |
| `CuratorResult` | `passed, rejected, stage_counts, output_dir, wall_clock_seconds, diagnostics` | — | `curatorkit/curator.py:555-571` |
| `CuratorResult.push_to_hub` | `push_to_hub(repo_id, export_file="sft_alpaca.jsonl", private=True, token=None, config_name=None, **push_kwargs)` | HF dataset repo | `curatorkit/curator.py:588` |
| `Pipeline` | `Pipeline(steps, output_dir=None, diagnostics=None, checkpoint_mgr=None)` | instance | `curatorkit/pipeline.py:80` |
| `Pipeline.run` | `run(self)` | `PipelineResult` — **honours no checkpoints** | `curatorkit/pipeline.py:127` |
| `Pipeline.run_async` | `async run_async(self)` | `PipelineResult` — the only checkpointing path | `curatorkit/pipeline.py:210` |
| `Pipeline.dry_run` | `dry_run(self)` | `list[dict[str, str]]` | `curatorkit/pipeline.py:92` |
| `DataSample` | `id, source_uri, instruction, input, output, chosen, rejected, label, responses, reward_scores, task_type, metadata, provenance_chain` | pydantic model | `curatorkit/schema.py:38` |
| `DataSample.append_provenance` | `append_provenance(self, record)` | `self` — the only supported chain mutation | `curatorkit/schema.py:91` |
| `JSONLReader` | `JSONLReader(path, field_mapping=None, format="auto", source_uri=None, preprocessing_fn=None, detection_sample_size=...)` | reader | `curatorkit/connectors/jsonl.py:33` |
| `EXPORTERS` | `dict[str, type[BaseExporter]]`, 7 keys | registry | `curatorkit/exporters/__init__.py:11` |
| `build_messages` | `build_messages(sample: DataSample)` | `list[dict[str, str]]` (`role`/`content`) | `curatorkit/exporters/messages.py:32` |
| `write_conversations` | `write_conversations(samples, output_path, export_format, exporter_name, to_record)` | `None` | `curatorkit/exporters/messages.py:50` |
| `is_compatible` | `is_compatible(task_type: str, export_format: str)` | `bool` (unknown task_type ⇒ compatible with nothing) | `curatorkit/exporters/compatibility.py:84` |
| `resolve_export_formats` | `resolve_export_formats(generation_task, observed_task_types=None)` | `list[str]` | `curatorkit/exporters/compatibility.py:103` |
| `TASK_TYPE_EXPORT_FORMATS` | `dict[str, list[str]]` — the whole export policy | data | `curatorkit/exporters/compatibility.py:42` |
| `build_provenance` | `build_provenance(method, inputs=None, params=None, base_model=None)` | `dict` (`lexsi.provenance/1`) | `curatorkit/manifest.py:55` |
| `dataset_input` | `dataset_input(ref, config=None)` | `dict` — embeds the parent's provenance | `curatorkit/manifest.py:84` |
| `read_provenance` | `read_provenance(path)` | `dict \| None` (swallows `OSError`/`ValueError`) | `curatorkit/manifest.py:75` |
| `step_models` | `step_models(steps)` | `dict[str, str \| None]` — `generator_model`/`judge_model`/`grpo_scoring_model` | `curatorkit/manifest.py:89` |
| `export_data_files` | `export_data_files(root, filename)` | `dict[str, Path]` (`{split: path}`) | `curatorkit/manifest.py:108` |
| `ProvenanceManifest` | `ProvenanceManifest(result, pipeline_config_hash="unknown", output_dir=None, provenance=None)` | instance | `curatorkit/manifest.py:199` |
| `ProvenanceManifest.build/.write/.write_rejected_sidecar/.write_checksums` | `build()`, `write()`, `write_rejected_sidecar()`, `write_checksums(output_files)` | `dict` / `Path` / `Path` / `Path` | `manifest.py:213,252,262,277` |
| `DatasetCardGenerator.generate` | `generate(manifest, output_dir, pipeline_name="curatorkit_pipeline", write_readme=True)` | paths | `curatorkit/manifest.py:290` |
| `BaseReader.read` | `read(self)` | `tuple[list[DataSample], list[RejectedSample]]` | `curatorkit/interfaces.py:30` |
| `BaseGate.run` | `run(self, samples)` | `tuple[list[DataSample], list[RejectedSample]]` | `curatorkit/interfaces.py:42` |
| `BaseNormalizer.run` | `run(self, samples)` | `list[DataSample]` | `curatorkit/interfaces.py:56` |
| `BaseExporter.export` | `export(self, samples, output_dir: Path)` | `None` | `curatorkit/interfaces.py:63` |
| `BaseGenerationTask.run` | `run(self, samples)` | `list[DataSample]`; rejects land in `.rejected` / `flush_rejected()` | `curatorkit/generators/base.py:178`, `:107` |
| `count_tokens` | `count_tokens(text, use_tiktoken=False)` | `int` | `curatorkit/utils/tokens.py:53` |

The YAML surface is a **different** schema: `PipelineConfig` (`curatorkit/config.py:443`) with
`readers/generators/normalizers/gates/exporters` lists, not `CuratorConfig`. Fields are not 1:1 —
e.g. `GateConfig.required_fields` defaults to `["instruction", "output"]`
(`curatorkit/config.py:295`), which *overrides* task-type derivation, while
`CuratorConfig.schema_required_fields=[]` means "derive from task_type". A YAML user can therefore
reject every `preference` sample for a missing `output` where the Python user cannot.

---

## 4. The provenance story, end to end

### Where provenance is created

1. Every step appends a `ProvenanceRecord(step_name, step_version, timestamp, config_hash, notes)`
   via `sample.append_provenance` (`curatorkit/schema.py:28,91`). `notes` is free-form and is the
   real payload — the manifest later mines it (`prompt_tokens`, `source_file`,
   `near_duplicates_removed`, `binding_total`, …).
2. Generators do **not** inherit the parent id. Every generator emits `id=str(uuid.uuid4())` and
   records the parent as `metadata["source_sample_id"]`, while *copying* `source_uri` and
   `provenance_chain=list(sample.provenance_chain)` verbatim (`curatorkit/generators/cot_generator.py:179`,
   `curatorkit/generators/evol_instruct.py:236`,
   `curatorkit/generators/adversarial_qa_generator.py:534`,
   `curatorkit/generators/multiturn_gen.py:406`). Lineage therefore lives in
   `metadata` + `provenance_chain` — and nowhere else.
3. A source row's own `id` overrides the UUID at ingest (`curatorkit/connectors/base.py:383-386`), so
   a stable identity *does* exist in memory for dataset inputs.

### Where provenance is destroyed

**At the exporter, for every clean row.** No exporter emits `DataSample.id`, `provenance_chain`, or
`label`; only `CorpusExporter` emits `source_uri` and any metadata
(`curatorkit/exporters/corpus.py:88-101`). Field survival:

| exporter | file | emitted keys | the only `DataSample` fields read | dropped |
|---|---|---|---|---|
| `AlpacaExporter` | `sft_alpaca.jsonl` | `instruction`, `input`, `output` | 3 | `id`, `source_uri`, `metadata`, `provenance_chain`, `task_type`, `label`, `chosen`, `rejected`, `responses`, `reward_scores` (`curatorkit/exporters/alpaca.py:48-52`) |
| `ShareGPTExporter` | `sft_sharegpt.jsonl` | `conversations[].from`, `.value` | + `metadata['system_prompt']`, `metadata['turns']`, `input` | everything else, incl. the metadata dict (`curatorkit/exporters/sharegpt.py:31-37`, `curatorkit/exporters/messages.py:35-46`) |
| `MessagesExporter` | `sft_messages.jsonl` | `messages[].role`, `.content` | same as ShareGPT | same as ShareGPT (`curatorkit/exporters/messages.py:32-46`) |
| `DPOExporter` | `dpo.jsonl` | `prompt`, `chosen`, `rejected` | 3 | `id`, `source_uri`, `metadata`, `provenance_chain`, `task_type`, `label`, `input`, `output` (`curatorkit/exporters/dpo.py:86-90`) |
| `GRPOExporter` | `grpo.jsonl` | `prompt`, `responses`, `rewards` | 3 | same, plus note the **column is renamed** `reward_scores`→`rewards` (`curatorkit/exporters/grpo.py:53-57`) |
| `PPOExporter` | `ppo.jsonl` | `prompt` | 1 | everything else (`curatorkit/exporters/ppo.py:47`) |
| `CorpusExporter` | `corpus.jsonl` | `text`, `source_file`, `source_uri`, `page`, `heading`, `chunk_index`, `content_type`, `task_type`, `table_html`, `table_bbox` | 9 metadata keys + `source_uri` | `id`, `provenance_chain`, `instruction`, `label`, `chosen`, `rejected`, `responses`, `reward_scores`, all non-whitelisted metadata (`curatorkit/exporters/corpus.py:88-101`) |

Two consequences worth stating plainly:

- **Only rejected rows keep their identity across the boundary.** `rejected.jsonl` uses
  `sample.model_dump()` (`curatorkit/manifest.py:262-275`), so every reject carries `id`,
  `provenance_chain`, `metadata` and the `diagnosis` field. A rejected row is traceable; an accepted row is not.
- **Dedup removals are not rejections.** `BaseNormalizer.run` returns a bare list
  (`curatorkit/interfaces.py:45-56`), so samples dropped by `ExactDeduplicator` /
  `MinHashDeduplicator` / `EmbeddingDeduplicator` never enter `rejected.jsonl` — only counts survive,
  lifted into `manifest["dedup_stats"]` (`curatorkit/manifest.py:129-151`). Duplicate rows vanish with
  no per-row record anywhere.

### What the manifest does record

`manifest.json` has exactly 15 keys, fixed in `ProvenanceManifest.build`
(`curatorkit/manifest.py:213-249`): `pipeline_config_hash`, `run_timestamp`, `source_files`,
`stage_counts`, `rejected_breakdown`, `dedup_stats`, `minhash_threshold`, `sampler_stats`,
`token_stats`, `wall_clock_seconds`, `tool_versions`, `diversity_stats`, `diagnostic_stats`,
`diagnostic_files`, `provenance`.

Run-level only. `source_files` is `[{"path": p}, ...]` derived from provenance notes — **no sha256
despite the module docstring advertising `{path, sha256}`** (`curatorkit/manifest.py:10-11` vs
`curatorkit/manifest.py:153-161`). `diversity_stats` is hard-coded `None` and never populated
(`curatorkit/manifest.py:243`). `pipeline_config_hash` covers 8 fields only — `dataset, split, format,
min_tokens, max_tokens, dedup, llm_model, generation_task` (`curatorkit/curator.py:1791-1805`); the
CLI's is `sha256(yaml bytes)[:16]` (`curatorkit/cli.py:91`), a completely different notion.

`lexsi_provenance.json` (`build_provenance`, `curatorkit/manifest.py:55-72`, written at `:257-259`):
`schema="lexsi.provenance/1"`, `library="curatorkit"`, `version`, **`git_sha`**, `created_at`,
`base_model`, `method` (`"curate"` or the generation task name), `inputs`, `params`
(`generator_model`/`judge_model`/`grpo_scoring_model` from `step_models`, plus `export_formats`,
`output_split`, `pipeline_config_hash`).

**`git_sha` is hard-coded `None`** — `"git_sha": None,` at `curatorkit/manifest.py:66`, with no git
plumbing anywhere in the module. Two different source trees therefore emit byte-identical
`provenance` blocks apart from `created_at`. Cross-run lineage *is* real, but only at run granularity:
`dataset_input()` embeds the parent run's `read_provenance()` output
(`curatorkit/manifest.py:84-87`), and `read_provenance` returns `None` on a corrupt or missing parent
indistinguishably (`curatorkit/manifest.py:75-82`).

Net effect: you can answer "which pipeline made this dataset, with which models and config" and "why
was row *i* of `rejected.jsonl` dropped"; you cannot answer "which source chunk produced row 4123 of
`sft_alpaca.jsonl`", "did row 4123 already exist in the previous run", or "is this the same code as
last week".

### Minimal patch for row traceability

Upstream (recommended, ~5 lines): add `"id": sample.id` (and optionally `"source_uri"`,
`"metadata"`) to the row dict in each exporter — `curatorkit/exporters/corpus.py:88`,
`curatorkit/exporters/alpaca.py:48`, `curatorkit/exporters/ppo.py:47`,
`curatorkit/exporters/dpo.py:86`, `curatorkit/exporters/grpo.py:53` — and add `"id"` to each class's
`columns` dict so the dataset card stays truthful. Everything needed is already on the sample; no
schema change.

Fork-free alternative (no upstream edit): subclass `BaseExporter`
(`curatorkit/interfaces.py:59`), implement `export(samples, output_dir)` writing
`{"id": s.id, **row, "metadata": s.metadata}`, and register it with
`EXPORTERS.update({"traced": TracedExporter})` (`curatorkit/exporters/__init__.py:11`) — `EXPORTERS`
is a plain module dict read at `curatorkit/cli.py:113,186,213,861` and
`curatorkit/curator.py:66-68`, so the update takes effect for both front ends. Add a row to
`TASK_TYPE_EXPORT_FORMATS` (`curatorkit/exporters/compatibility.py:42`) or the new format will skip
every sample. Whether downstream importers tolerate extra JSONL keys is not determinable from this
repo — verify against AuditKIT before relying on it.

---

## 5. Multilingual reality

Short version: **the pipeline runs on non-English text without complaint and is calibrated for
English only.** Nothing detects language, so nothing warns.

**PII is pinned to English.** `CuratorConfig.pii_language: str = "en"` (`curatorkit/curator.py:460`,
comment at `:437`), passed straight through to Presidio's `analyzer.analyze(text, language=...)`
(`curatorkit/curator.py:1094` → `curatorkit/hygiene/pii.py:427-432`). With the default
`pii_nlp_engine="spacy"` and `pii_spacy_model="en_core_web_lg"` (`curatorkit/curator.py:454-455`,
~800 MB), non-English text is run through an English NER model. It does not raise; it
mis-detects, then pseudonymizes whatever it thinks it found with a per-sample-consistent Faker map
(`curatorkit/hygiene/pii.py:187`). Non-English names, addresses and national IDs survive or get
partially mangled. The escape hatches exist but are off by default and are *not* switches to
"multilingual": `pii_nlp_engine="stanza"` takes a **language code** in `pii_stanza_model`
(default `"en"`, `curatorkit/curator.py:457`) and must be set per language, and
`pii_nlp_engine="transformers"` requires you to name a model
(`curatorkit/curator.py:456`). For a mixed-language corpus, PII pseudonymization as configured is
**unusable**; either set the language per corpus, or disable PII and handle it out of band.

**Toxicity is English Jigsaw with a very tight pass band.** `ToxicityGate` defaults to
`detoxify_model="unbiased"` (`curatorkit/hygiene/toxicity.py:117`, documented at `:104-106` as
`"unbiased" (default, recommended), "original", or "multilingual"`) and
`classifier_pass_threshold=0.1` (`curatorkit/hygiene/toxicity.py:115`; the config field mirrors it at
`curatorkit/curator.py:462` and `curatorkit/config.py:342`). Detoxify `unbiased` is trained on
English Jigsaw; on non-English text its scores are not calibrated — usually low, so genuinely toxic
rows pass, and occasionally high on transliterated or profanity-adjacent text, so benign rows are
rejected. A `multilingual` model option exists (`curatorkit/hygiene/toxicity.py:104-106`) but is selected by
a config flag, never per row, and its calibration for any given language is not measured here.
Two further facts a safety-corpus user needs: the gate is **off by default** in the Python API
(`toxicity_gate: bool = False`, `curatorkit/curator.py:461`), and at `pass_threshold=0.1` this gate is
the one most likely to delete exactly the material a harmful-behaviour arm needs — it is a
data-quality filter, not a safety-labelling tool.

**There is no language-identification gate at all.** A grep for `langid|fasttext|lingua|
detect_language` across `curatorkit/` returns only prose about a multilingual *model option*
(`curatorkit/hygiene/toxicity.py:105`, `curatorkit/hygiene/pii.py:40,112`). A per-sample `lang`
column is not special-cased: extra columns are carried into `metadata` and then dropped at export
(`curatorkit/connectors/base.py:745-749`). You cannot filter by language inside the pipeline; you must
pre-split the corpus outside it and run one Curator per language.

**What is actually multilingual-safe**

| gate | usable on non-English as-is? | why |
|---|---|---|
| `SchemaGate` | ~yes | token bounds count whitespace words **plus one token per CJK/Thai/Lao/Myanmar/Khmer/kana/Hangul code point** (`curatorkit/utils/tokens.py:17`, applied at `:34-37`), so it does not under-count unspaced scripts. `min_tokens=10` (`curatorkit/curator.py:213`) still cuts very short non-English prompts. |
| `ExactDeduplicator` / `MinHashDeduplicator` | yes | character 3-grams (`minhash_ngram=3`, `curatorkit/curator.py:224`), hashing only (`curatorkit/normalizers/dedup.py:108-130`). The 0.85 threshold is tuned on English text lengths; `dedup_ignore_patterns` are Python regexes (`curatorkit/curator.py:227`, consumed at `curatorkit/normalizers/dedup.py:194`). |
| `TextCleaner` | mostly | the mojibake table is cp1252→UTF-8 only (`curatorkit/normalizers/clean.py:29-42`); NFC, control-char and whitespace handling are script-neutral. |
| `SecretsGate` | ~yes | 14 detect-secrets regex/entropy plugins; `detect-secrets`' high-entropy heuristic over-triggers on CJK-heavy text. |
| `HallucinationGate` | **no** | English prompt demanding English JSON keys (`curatorkit/gates/hallucination.py:42-74`); `skip_if_no_context=True` (`:104`) means short non-English samples are silently *skipped*, not judged. |
| `RewardGate` | **no** | English rubric with three English dimension names (`curatorkit/gates/reward.py:51-77`, `:125`). |
| `DiversityGate` | **no** | `sentence-transformers/all-MiniLM-L6-v2` (`curatorkit/gates/diversity.py:71`), English-centric; the 0.92 cosine threshold is not calibrated for other languages, so it over- or under-rejects. |
| `EmbeddingDeduplicator` | **no** | same embedding default (`curatorkit/curator.py:379-380`, threshold `:388`). |
| `ToxicityGate` | **no** | English Jigsaw `unbiased`; pass band starts at 0.1. |
| `PIIPseudonymizer` | **no** | `pii_language="en"` + `en_core_web_lg`. |

So for a non-English safety corpus, five of the eleven rows above are **unusable without
reconfiguration**: hallucination, reward, diversity, embedding dedup and toxicity — plus PII
pseudonymization. Two of those (hallucination, reward) additionally need an LLM that follows the
English rubric. The model-free, language-agnostic core — SchemaGate, dedup, TextCleaner, SecretsGate — works on any
script. One determinism note that matters here: `judge_on_error="reject"` is the default
(`curatorkit/curator.py:293`, `curatorkit/gates/_judge.py:36`), so an LLM judge that cannot parse a
non-English answer *deletes* the row rather than passing it.

---

## 6. Top 5 landmines

1. **Quality gates silently do not run when no judge model is configured — the threshold is ignored,
   not rejected.** `HallucinationGate` and `RewardGate` are appended only when
   `cfg.hallucination_threshold is not None and cfg._any_judge_model()`
   (`curatorkit/curator.py:1124`, `:1154`; helper at `:520-525`), and generation only when
   `cfg.generation_task and cfg._any_gen_model()` (`:1116`). *Symptom:* you set
   `hallucination_threshold=0.9`, the run reports success with 100 % pass, and
   `manifest.json["stage_counts"]` has **no** `HallucinationGate` key. Nothing in the log says the
   gate was skipped.
2. **Exported rows are anonymous, so a row cannot be traced to its source, its run, or its
   neighbour.** Three-key Alpaca rows (`curatorkit/exporters/alpaca.py:48-52`), one-key PPO rows
   (`curatorkit/exporters/ppo.py:47`), and `corpus.jsonl` as the only format carrying `source_uri`
   (`curatorkit/exporters/corpus.py:88-101`). *Symptom:* two different runs produce
   indistinguishable JSONL; you cannot join an exported row back to `rejected.jsonl` or to a PDF
   page; re-exporting with a different `export_formats` list silently changes which provenance
   survives, with only a "skipped N" warning.
3. **Checkpoints and the config hash disagree about what "the same run" means.**
   `Pipeline.run` consults no checkpoint manager at all (`curatorkit/pipeline.py:127` vs `:210`), so a
   resumed *sync* run redoes and re-spends everything; and `_config_hash` covers 8 fields, none of
   them a gate threshold, prompt template, judge model or injection rate
   (`curatorkit/curator.py:1791-1805`). *Symptom:* you change `hallucination_threshold` from 0.7 to
   0.95, resume, and get the old gate results back under an unchanged `pipeline_config_hash`, so
   nothing in the manifest shows the configuration differed. The checkpoint key is
   `type(step).__name__` (`curatorkit/pipeline.py:210`ff), so two steps of the same class cannot be
   told apart either.
4. **Judge failures either delete or fabricate.** `judge_on_error="reject"` is the default
   (`curatorkit/gates/_judge.py:36`), so a rate limit or a malformed answer rejects 100 % of that
   batch with `rejection_reason="judge_error:<Type>"`; conversely `GRPORolloutTask._score_single`
   returns a neutral **0.5** on any parse or runtime error
   (`curatorkit/generators/grpo_rollout.py:179-180`), and `score_responses=False` writes
   `[0.0]*n` (`:234`). *Symptom:* a dataset that shrinks by 90 % overnight with no error
   (`judge_error:*` in `rejected_breakdown`), or a `grpo.jsonl` whose `rewards` are all 0.5 and which
   trains nothing while every stage reports healthy.
5. **Rows disappear between stages without a record.** Normalizer removals never reach
   `rejected.jsonl` (`curatorkit/interfaces.py:45-56`) — dedup keeps counts, not rows; a corpus
   `label` column holding `"harmful"`/`"benign"` is claimed by `LABEL_COLS`
   (`curatorkit/detection/detector.py:137-147`) and then dropped, because the mapping does not know
   string labels (`curatorkit/connectors/base.py:745-749`, float/bool only at `:596-604`);
   `max_samples` truncates **after** resampling, first-N, appended after the sampler at
   `curatorkit/curator.py:1251`; and `AlpacaExporter` writes rows with an empty `instruction`/`output` instead of skipping
   (`curatorkit/exporters/alpaca.py:53-55`). *Symptom:* `len(passed) + len(rejected)` does not equal
   the rows the reader produced, the arithmetic does not reconcile anywhere, and the missing rows are
   unrecoverable because they were never rejected.

Honourable mentions: `JSONLReader` opens with `errors="replace"` (`curatorkit/connectors/jsonl.py:52`),
so invalid UTF-8 is silently mangled rather than reported; `TextCleaner`'s `<[^>]+>` HTML regex eats
prose containing angle brackets (`curatorkit/normalizers/clean.py:43`, applied at `:120`); two judge
gates use
`order = {s.id: i ...}` and silently reorder on duplicate ids
(`curatorkit/gates/hallucination.py:284`, `curatorkit/gates/reward.py:382`); `RewardGate`'s
`store_score_in_label=True` overwrites `HallucinationGate`'s label
(`curatorkit/gates/reward.py:275` vs `curatorkit/gates/hallucination.py:253`).

---

## 7. What is missing, and the extension points

### Missing

- **Row identity in exports** — the single highest-value gap; see §4 for the five-line patch.
- **A language-identification gate.** No such gate exists; every multilingual problem in §5 reduces to
  its absence.
- **A CPU/lexicon toxicity path.** `hygiene/toxicity.py` and `hygiene/pii.py` accept no `device` or
  `cuda` argument at all (contrast `gates/diversity.py:69`, which takes `device=None`), so both load
  torch models on whatever device is visible.
- **`BadSampleInjector` is exported but unwired.** It is in `__all__`
  (`curatorkit/generators/__init__.py:10,25`) yet no CLI or `Curator` branch constructs it, and
  `injection_evaluation_report` (`curatorkit/generators/injector.py:284`) has zero call sites — the
  gate-recall measurement machinery is unreachable from a pipeline. Two of its six advertised types
  (`domain_mismatch`, `instruction_quality`) can never be injected either
  (`curatorkit/generators/injector.py:232-233`).
- **Manifest gaps:** `git_sha` always `null` (`curatorkit/manifest.py:66`), `diversity_stats` always
  `null` (`:243`), `source_files` with no checksum (`:153-161`), no gate-result cache.
- **`ToxicityGate` is not re-exported** from `curatorkit/gates/__init__.py:6`, and `"pii"` is not a
  `GateConfig.type` (`curatorkit/config.py:296`) because PII pseudonymizes rather than rejects.
- `curatorkit/tools/setup_mineru.py:48` is a stub that installs nothing despite its name.

### How to add a gate

1. Subclass `BaseGate` (`curatorkit/interfaces.py:35`) and implement
   `run(self, samples) -> tuple[list[DataSample], list[RejectedSample]]`.
2. Build rejects as `RejectedSample(**sample.model_dump(), rejection_reason="mygate:<detail>",
   rejecting_step="MyGate")` — copy the pattern at `curatorkit/gates/schema.py:121-135`; append
   provenance on the pass path (`curatorkit/gates/schema.py:110-118`).
3. If it calls an LLM, funnel every failure through `judge_error_result`
   (`curatorkit/gates/_judge.py:21`) so the contract encoded in `JUDGE_ERROR_PREFIX`
   (`curatorkit/gates/_judge.py:10`) keeps the rejection report correct, and extract scores with
   `extract_score` (`curatorkit/gates/_score_parsing.py:39`) rather than defaulting to 0.5.
4. Implement `_config_hash()` if the gate has knobs, otherwise a resume cannot tell two configs apart
   — copy `DiversityGate._config_hash` (`curatorkit/gates/diversity.py:92-101`).
5. Wire it: append in `Curator._build_steps` (`curatorkit/curator.py:955`, order matters — it runs
   where you insert it); for YAML, extend the `GateConfig.type` `Literal`
   (`curatorkit/config.py:296`) and the CLI branch that constructs it.
6. A new `type` value must also be honoured by `Pipeline`'s dispatch — it is pure `isinstance`
   against the four ABCs (`curatorkit/pipeline.py:127`), so a step subclassing none of them is
   silently ignored.

### How to add an exporter

1. Subclass `BaseExporter` (`curatorkit/interfaces.py:59`); set the `filename` and `columns` class
   attributes (`curatorkit/exporters/alpaca.py:33-34`) — `columns` feeds the dataset card and the
   manifest table (`curatorkit/manifest.py:344`), so a wrong `columns` mis-documents silently.
2. Gate every sample with `is_compatible(sample.task_type, _FORMAT)`
   (`curatorkit/exporters/compatibility.py:84`) and count skips; emit **one** aggregate
   `warnings.warn(..., UserWarning, stacklevel=2)` — that is the house pattern
   (`curatorkit/exporters/alpaca.py:44-46,62-70`).
3. Register in `EXPORTERS` (`curatorkit/exporters/__init__.py:11`) **and** add a row to
   `TASK_TYPE_EXPORT_FORMATS` (`curatorkit/exporters/compatibility.py:42`). One without the other is
   unreachable from YAML — the comment at `curatorkit/exporters/__init__.py:9-10` is the warning.
4. If it is a conversation format, reuse `write_conversations` / `build_messages`
   (`curatorkit/exporters/messages.py:32,50`) rather than re-implementing turn assembly; that is how
   `ShareGPTExporter` stays consistent (`curatorkit/exporters/sharegpt.py:48-55`).
5. Adding `"id": sample.id` to the row while you are there is the cheapest fix to §4.

### How to add a generator

1. Subclass `BaseGenerationTask` (`curatorkit/generators/base.py:60`); implement `_build_messages`
   (`:165`) and `_parse_response` (`:170`) and let the inherited `run` (`:178`) do retries, rejection
   capture (`generation_parse_failed:<task>`, `:204-215`) and provenance
   (`prompt_hash` + `source_sample_id`, `:218-234`).
2. Emit `id=str(uuid.uuid4())`, copy `source_uri` and `provenance_chain=list(...)`, record the parent
   in `metadata["source_sample_id"]` — six generators do exactly this and the manifest's
   `token_stats` depends on the notes the base class writes.
3. **Do not override `run_async` unless you need to.** Six classes do, and the overrides are where the
   bugs are: the checkpointing `run_async` in the base class (`:252-372`) is lost, and
   `AdversarialPreferenceTask`'s seeded plan becomes scheduling-dependent
   (`curatorkit/generators/adversarial_preference.py:416-418`). Where a plan must be deterministic
   under concurrency, compute it up front from ids, as `AdversarialQAGenerationTask._make_plan` does
   (`curatorkit/generators/adversarial_qa_generator.py:262-271`).
4. Register: `curatorkit/generators/__init__.py:4-13`, the lazy map in `curatorkit/__init__.py`, a
   dispatch branch in `Curator._build_generator` (`curatorkit/curator.py:1467`) and a value for
   `generation_task` in `CuratorConfig` — plus a `GENERATION_TASK_OUTPUT_TYPE` entry
   (`curatorkit/exporters/compatibility.py:60`) or the run will guess export formats from observed
   task types instead of the declared one.
5. Every pass that touches an LLM should go through the injected `BaseLLM` and record
   `response.to_provenance_dict()`; and accept that no generator can be made reproducible, because
   no seed is ever passed to a backend and `BaseLLM.config_hash()` hashes only
   `model|temperature|max_tokens` (`curatorkit/llm/base.py:210-220`).

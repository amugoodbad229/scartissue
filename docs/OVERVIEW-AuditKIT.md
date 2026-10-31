# AuditKIT — repo overview

Repo `repos/AuditKIT`, Lexsi-Labs, tag 1.1.3 (`_research/repo-maps/AuditKIT.md:3`); 251 `.py`, 58,348 LOC. Synthesised from `_docs/AuditKIT/01..06` + `_research/repo-maps/AuditKIT.md`. Every claim below is anchored; where a package doc and this summary disagree, the package doc wins.

## 1. What it is for

AuditKIT is a model- and agent-evaluation toolkit: it turns a dataset into score vectors for one or more backends, persists each run with a content fingerprint, and produces graded, significance-tested comparisons between runs (`src/auditkit/api.py:178`, `src/auditkit/comparison.py:117`). The core is pure-stdlib (`pyproject.toml: dependencies = []`) and every backend and substantive metric sits behind an extra that raises `ExtraNotInstalled` rather than `ImportError` (`src/auditkit/errors.py:23`), so `import auditkit` works on a bare interpreter. It also carries an agent-episode evaluator (`agent_eval/`), a Lexsi cross-project evidence/join layer (`src/auditkit/agent_eval/lexsi.py:45`), and a toy red-team runner (`src/auditkit/redteam/runner.py:110`).

**Licence:** Lexsi Labs Source Available License (LSAL) v1.2 — explicitly *not* OSI open source (`LICENSE.md:7`); free for noncommercial academic research and teaching only (`LICENSE.md:13`); §2 bars selling or embedding it in a paid/hosted product without a separate commercial licence (`LICENSE.md:37-41`). **§1A: any organisation use requires a written acknowledgement of use (or negotiated permission) — `LICENSE.md:24-29`.** Rights holder: Lithasa Technologies Pvt. Ltd. (`LICENSE.md:57`). Restated in `README.md:303`.

## 2. The mental model

One pipeline, five stages, and one type that every stage agrees on (`RunResult`, `src/auditkit/report.py:54`):

```
Scenario / list[Sample]                     src/auditkit/sample.py:18
        │  adapter.adapt(sample, config) → list[Request]     src/auditkit/adapter.py:52
        ▼
   Request ──► model.generate([...]) ──► Result_/Generated  src/auditkit/model/__init__.py:246,273
        │                                      (text, latency_ms, tokens, trace)
        ▼
   metrics[i].score(sample, output, context) → Score        src/auditkit/metric.py:55
        │        (value, direction, threshold, metadata)
        ▼
   Runner.aggregate(scores) → Stat                          src/auditkit/runner.py:488
        ▼
   RunResult(stats, predictions, headline, perf, errors, unscored)   src/auditkit/report.py:54
        ▼
   compare(a, b) → RunComparison                            src/auditkit/api.py:491 → comparison.py:117
        └─ metric_deltas()/grades()/significance()/tradeoff()/cost()   comparison.py:149,228,279,283,375
```

- `evaluate()` is the only stage that touches a model; a run reads `actual_output` off the samples when the adapter declares it, so scoring a pre-scored file costs no inference (`src/auditkit/runner.py:516`; `src/auditkit/loaders.py:51-59`).
- A run's identity is a **fingerprint**, not a name: `sha256(...)[:16]` over engine, `auditkit_version` read at call time, model `identity()`, scenario name, adapter `identity()`, sorted metric and annotator identities, `extracted_by`, `run_name`, and the whole `RunConfig.to_dict()` (`src/auditkit/runspec.py:123`). Change any measurement input without touching `identity()` and two different runs collide in the cache.
- Metrics that cannot measure return `[]` (skipped) rather than `0.0`; `Metric.applicable` drops a metric when a `required_fields` attribute is `None` (`src/auditkit/metric.py:47-53`), and `Runner.score_one` records-and-skips metric exceptions instead of fabricating a score (`src/auditkit/runner.py:375`). This is the repo's stated core invariant and it is mostly kept — the breaches are in §4.

### What a run is on disk

| Artefact | Where | Anchor |
|---|---|---|
| `RunResult` JSON (or CSV) | caller's `path`; `write_provenance=True` also drops `lexsi_provenance.json` beside it | `src/auditkit/report.py:320`; `src/auditkit/provenance.py:17` |
| Lineage record | `{"schema":"lexsi.provenance/1", library, version, created_at, base_model, method, inputs, params}` | `src/auditkit/provenance.py:18,50` |
| Response cache | `$XDG_CACHE_HOME`/`~/.cache` + `auditkit/runs`, one file per **bare fingerprint**, no schema version in the key | `src/auditkit/cache.py:17-25` |
| Experiment | `$XDG_DATA_HOME/auditkit/experiments/{name}.json`, runs round-tripped through `RunResult.to_dict()` | `src/auditkit/experiment.py:114,126-142` |
| Hub artefact | always `repo_type="dataset"` | `src/auditkit/hf_publish.py:236`; `_research/repo-maps/AuditKIT.md:166` |

## 3. Public surface

Eight calls cover essentially every use; all are re-exported flat from `src/auditkit/__init__.py:12-99` (200+ names, `__all__` at `:101`). `redteam` is **not** re-exported — `import auditkit.redteam` (`_research/repo-maps/AuditKIT.md:12`).

| # | Entry point | Anchor | Signature (abridged) | Returns | Notable defaults |
|---|---|---|---|---|---|
| 1 | `evaluate` | `src/auditkit/api.py:178` | `evaluate(dataset, model, scorers=None, *, engine="native", adapter=None, annotators=None, extract_with=None, config=None, verbose=False, experiment_name=None, tags=None, dataset_config=None, dataset_split=None, **opts)` | `RunResult` | `engine="native"`; `scorers=None` → `[ExactMatch()]` iff any sample is golden (`:92-95`); `RunConfig()` if `config=None` |
| 2 | `compare` | `src/auditkit/api.py:491` | `compare(baseline_or_results, candidate=None, *, metric=None, **kw)` | `RunComparison` (2 args) **or** `list[dict]` leaderboard (1 arg) | arity-overloaded; `pass_threshold=0.02`, `warn_threshold=0.05` (`comparison.py:117`) |
| 3 | `compare_models` | `src/auditkit/model_compare.py:505` | `compare_models(models, dataset, scorers=None, *, model_names=None, adapter=None, annotators=None, extract_with=None, config=None, configs=None, model_opts=None, …)` | `CompareResult` | one model resident at a time (resolve-in-loop + `unload()`); per-model failure recorded, not raised (`:505`) |
| 4 | `RunComparison` | `src/auditkit/comparison.py:117` | `.metric_deltas() .per_task_deltas() .grades() .retention() .significance() .tradeoff() .performance() .cost(b,c) .coverage_warnings()` | `MetricDelta`/`TaskDelta`/`DeltaGrade` dicts | `delta: Optional[float]` = `None` on a one-sided metric, never `0.0` (`:141-147`) |
| 5 | `run_lmeval` (or `evaluate(engine="lmeval")`) | `src/auditkit/api.py:277` / `:252-258` | `run_lmeval(tasks, model, *, num_fewshot=None, limit=None, config=None, experiment_name=None, tags=None, **opts)` | `RunResult` | lmeval path **silently ignores** `scorers`/`adapter`/`annotators`/`extract_with`/`dataset_config`/`dataset_split` |
| 6 | `load_dataset` / `load_jsonl` / `load_agenttune` | `src/auditkit/loaders.py:120` / `:248` / `:440` | `load_dataset(path, config=None, *, split=None, …) -> ListScenario`; `load_jsonl(path, *, field_map=None) -> list[Sample]` | scenario / samples | `load_dataset` returns a **scenario**, the other two return samples; split fallback `test→val→train` (`:160`) |
| 7 | `RunResult.save` / `.push_to_hub` | `src/auditkit/report.py:320` / `:354` | `save(path, fmt="json", write_provenance=True)`; `push_to_hub()` | `str` path/URL | `fmt="json"`; card front-matter fixed `license: other`, `task_categories: [text-generation]` (`hf_publish.py:107-110`) |
| 8 | `AutoModel.resolve` | `src/auditkit/model/__init__.py:590` | `AutoModel.resolve(spec, **opts)`; prefixes `openai: anthropic: hf: lexsi: vllm: litellm: api: groq: openrouter: agent:` | backend `Model` | bare prefix → error with example (`:577`); callable → `CallableModel`; missing dep → `ExtraNotInstalled` |

Extension entry points (not project-call surface): `@METRICS.register("name")` (`src/auditkit/registry.py:26,60`) and `@scorer(direction=…)` (`src/auditkit/scorers.py:83`). `evaluate_many` (`api.py:371`) and `generate` (`api.py:330`) exist but are thin — see the `generate` landmine in §6.

## 4. Metric catalogue with honesty flags

Legend: **J** = needs a judge/encoder model (extras + weights); **E** = English-only in practice; **0** = can return a fabricated `0.0` (or `0.5`) that an aggregate cannot distinguish from a measured value; **S** = per-sample score is `compare`-testable via `RunComparison.significance()` → `_bootstrap.paired_bootstrap` (`src/auditkit/comparison.py:279`, `src/auditkit/_bootstrap.py:47`).

| Family | Registered names / classes | Anchor | J | E | 0 | S | Notes |
|---|---|---|---|---|---|---|---|
| Exact/lexical | `exact_match`, `quasi_exact_match`, `acc`, `acc_norm` | `src/auditkit/metric.py:78,103,173,188` | – | E | – | S | normalisation drops only `a`/`an`/`the` (`:94`) — an English stopword list is the whole multilingual story |
| Code/string | `equals`, `contains`, `starts_with`, `ends_with`, `regex`, `levenshtein`, `word_count`, `is_json`, `f1_score` | `src/auditkit/metrics/code.py:15-200` | – | E | 0 | S | `WordCount()` default scores **1.0 on every sample** (`:161-169`); `Levenshtein` scores a targetless sample against `""` → `0.0` (`:115-118`); `F1Score` drop `a`/`an`/`the` + **set** intersection (`:216-228`) |
| Generation | `bleu`, `rouge_l`, `chrf`, `word_error_rate`, `perplexity`, `bert_score` | `src/auditkit/metrics/generation.py:16,88,126,188,234,285` | – | E | 0 | S | `wer` holds `1 − WER` (`:230`); both-empty = `1.0` for ROUGE-L/WER but `0.0` for BLEU/chrF (`:44-45,111-112,144-145`); `Perplexity` default `gpt2` and reported as an **arithmetic** mean of a ratio-scale quantity (`:234`) |
| Embedding | `cosine_similarity`, `token_overlap`, `bm25_similarity` | `src/auditkit/metrics/embedding.py:15,94,112` | J (MiniLM) | E | 0 | S | `bm25_similarity` has no corpus, no IDF, no `k1`/`b` — multiset overlap (`:119-131`); `token_overlap` is Jaccard, not overlap (`:101-108`) |
| Generative judge | `llm_judge`, `g_eval`, `factuality`, `closed_qa`, `relevance`, `bias_judge` | `src/auditkit/metrics/judge.py:142,418,520,538,562,610` | **J required** | E | 0 | S* | no `judge_model` → `NotImplementedError` (`:242`); unparseable reply → `value=0.0` + `metadata["unknown"]=True`, excluded from the headline only via `unknown_score=` (`_docs/AuditKIT/02-src__auditkit.md:67,75`) |
| Encoder judge | `encoder_judge`, `factuality_encoder_judge`, `sentiment_encoder_judge` | `src/auditkit/metrics/encoder_judge.py:73,347,384` | J | E | – | S | no text generated, so no parse failure; but a `LABEL_1 = positive` assumption on generic labels (`:46,50`) inverts checkpoints such as `textattack/bert-base-uncased-MNLI` |
| Guard | `guard` profiles `llama_guard`/`wildguard`/`harmbench`/`shield_gemma`/`granite_guardian` | `src/auditkit/metrics/guard.py:187,235` | **J required** | E | – | S | `0.0 = safe`, `Direction.MINIMIZE`; default profile downloads a **gated ~8B** checkpoint (`:241`) |
| NLI hallucination | `factual_consistency` | `src/auditkit/metrics/hallucination.py:15` | J | E | 0 | S | NEUTRAL verdict silently becomes **0.5** (`:46`) — an unmeasured middle averaged in |
| Toxicity | `ToxicityScore`, `RepresentationSkew`, `HateSpeechScore` | `src/auditkit/metrics/toxicity.py:46,107,200` | J optional (`unitary/toxic-bert`) | E | 0 | S | `RepresentationSkew` returns 1.0 (max skew) for a single present group, even when the text never mentions demographics (`:107`) |
| RAG lexical | `LexicalGroundedness`, `ContextCoverage`, `ContextOverlap`, `AnswerOverlap` | `src/auditkit/metrics/rag.py:50,79,108,133` | – | E | – | S | word-prefix matching: `"in"` matches `"information"`; returns `[]` without context, not `0.0` |
| RAG judged | `Faithfulness`, `Hallucination`, `AnswerRelevancy`, `ResponseGroundedness`, `ContextPrecision`, `ContextRelevance`, `ContextRecall` | `src/auditkit/metrics/rag_judge.py:228-529` | **J required** | E | – | S | strict count-matched verdicts — a missing verdict **raises** (`:47,164`); `context_precision` and `context_relevance` return the same number under two names (`:489`) |
| Retrieval | `RetrievalMetrics(k=None)` → `retrieval@k` | `src/auditkit/metrics/retrieval.py:90` | – | – | – | S | chunk ids compared as exact trimmed strings, `Doc-1 ≠ doc-1`; empty gold → `[]` |
| RAG stress | `EvidenceSetRecall`, `Freshness`, `AclCompliance`, `CitationSupport`, `Abstention` | `src/auditkit/metrics/rag_stress.py:294,343,408,483,611` | – | E | **0** | S* | untestable arm → `value=0.0` **with** `metadata={"unknown":True,"status":…}` (`:268`) — and `Runner.aggregate` ignores the flag (`runner.py:488`) |
| Agent | `tool_call_f1`, `trajectory_match`, `parallel_tool_calls`, `tool_call_validity`, `redundant_tool_calls`, `task_completion`, `agent_loop_detection`, `tool_permission`, `tool_selection` | `src/auditkit/metrics/agent.py:272,372,404,686,714,761,904,977,1061` | 2 of 9 | E | 0 | S* | `tool_selection` returns `0.0`-flagged-unknown when every verdict is unparseable (`:1105-1108`) while `task_completion` **raises** on the same failure (`:808-813`) |
| Pairwise | `WinRate`, `EloScore`, `PreferenceAccuracy` | `src/auditkit/metrics/pairwise.py:45,81,125` | – | – | **0** | **no** | `WinRate` with no `candidates` → real `0.0` (`:45`); `PreferenceAccuracy` with no data → fabricated **0.5** (`:125`); `EloScore` reports the mean of *all players'* ratings and is not a per-sample score, so a paired test on it is meaningless |
| Security | `KeywordDetector(blacklist, whitelist)` | `src/auditkit/metrics/security.py:23` | – | E | – | S | ships **no** default list; `1.0 = clean` (MAXIMIZE) which reads backwards next to `GuardJudge`'s `0.0 = safe` in the same `ScoreKind.SECURITY` |
| Red-team | *(not a metric)* `pass_rate` only | `src/auditkit/redteam/runner.py:37` | – | E | **0** | **no** | see §5 |

\* `S` means the machinery works; the caller must filter `metadata["unknown"]` first, or `Runner.aggregate` averages a fabricated zero into the headline.

**What `S` actually requires:** per-sample `Score`s in `Prediction.metadata["scores"]`, at least 2 aligned pairs, and a native-engine run. `significance()` aligns by `sample_id` (`src/auditkit/_bootstrap.py:44`) and returns `{"error": "need at least 2 aligned sample pairs", "n": 0}` otherwise — **including for every lm-eval run**, which carries no `metadata["scores"]` (`src/auditkit/_bootstrap.py:20-44`). `p = (count+1)/(n_resamples+1)` with `significant = p < 0.05` hardcoded, `n_resamples=1000`, `seed=42` (`_bootstrap.py:69,77`). `CompareResult.significance` returns the same dict-instead-of-raising (`src/auditkit/model_compare.py:91`), so an unguarded consumer reports `p_value=None` as if the test ran.

## 5. The red-team caveat — this repo cannot produce a reportable ASR

The `redteam` package is a **toy-scale smoke test**, not an attack-success-rate measurement. Concretely:

- **Scale.** Four probe classes and 26 hard-coded prompt strings, no sampling, no AdvBench/HarmBench import: `PromptInjectionProbe` (8), `JailbreakProbe` (8), `EncodingProbe` (5), `RefusalProbe` (5) — `src/auditkit/redteam/probes/builtin.py:8,29,49,71`. (Count correction for the brief: four probes / 26 prompts, not eight probes.) Four keyword/regex detectors, none of which reads the prompt — `src/auditkit/redteam/detectors/builtin.py:8,40,69,99`.
- **Geometry.** `run()` makes one greedy `Request(prompt=p)` per prompt, one batched `generate`, then emits one `ProbeResult` per `(prompt, detector)` pair, so `total_probes = prompts × detectors` (26 × 3 default = 78) — `src/auditkit/redteam/runner.py:37,119`. The number in the report is a cross-product, not a corpus size.
- **Metric.** `RedTeamResult` exposes only `pass_rate`; there is no ASR, no per-category breakdown, no per-continuation sampling (`src/auditkit/redteam/runner.py:37`). `pass_rate` is `1.0` when zero probes ran (`:58`).
- **Inverted signal.** `RefusalDetector` measures refusal but is scored as attack success; it is mapped to the benign `refusal` probe, where refusal is the *safe* outcome, so `passed = not detected` inverts its meaning (`src/auditkit/redteam/runner.py:37` + `detectors/builtin.py:40`).
- **No identity.** No `identity()`, no fingerprint, no cache, no chunking, no retry — two runs are not comparable and never deduplicated; one transport error aborts the whole run (`src/auditkit/redteam/runner.py:119`).
- **Wrong subject on hosted providers.** The jailbreak prompts are exactly what a provider's input filter blocks at the API layer, so `groq:`/`openai:` runs measure the provider's filter, not the model (`src/auditkit/redteam/probes/builtin.py:29`).

**`pass_rate` is an ASR surrogate at best.** It is `1 − (string matches from a 4×26 hand-written list)`, so it measures the detector list, not the model's harmfulness, and it is not comparable across prompt sets. **No reportable ASR can be produced from this repo.** The judge path that can is SafeTune's: `evaluate(model, benchmarks=None, judge="wildguard", …)` (`src/safetune/instrumentation/evaluate/suite/evaluate.py:206`) with its `judges/` tree and `redteam/` attacks `AbliterationAttack`/`BoNAttack` (`_research/repo-maps/SafeTune.md:20,47`). Use AuditKIT's red-team runner for pipeline wiring only. The repo map reaches the same verdict independently: "do not use for ASR … keep `redteam` at most as a smoke test … and even then pin `detectors=[\"keyword\"]` and read the raw rows, not `pass_rate`" (`_research/repo-maps/AuditKIT.md:160`).

## 6. Top 5 landmines

1. **`[TRAP]` A cache hit can serve the wrong measurement.** The cache key is the bare fingerprint with no schema version (`src/auditkit/cache.py:25`) and the fingerprint is supposed to encode everything that changes a number (`src/auditkit/runspec.py:123`). Two live collisions prove the contract is not enforced: `IsJson(require_keys=["a,b"])` and `IsJson(require_keys=["a","b"])` share a name *and* an identity, so different requirements share one cache entry and one fingerprint (`src/auditkit/metrics/code.py:180-181`); `F1Score(normalize=True|False)` has one score name but two identities, so `metric_deltas` groups two different metrics into one delta (`src/auditkit/metrics/code.py:208-214`). *Symptom:* a number that does not move when you change the metric's configuration.
2. **`[BUG]` A comparison of nothing grades PASS.** `RunComparison.grade()` returns `DeltaGrade.PASS` when there is nothing to grade (`src/auditkit/comparison.py:237`; `_GRADE_ORDER` ranks `NOT_COMPARABLE` at `-1`, `:44-54`), while the one-sided-metric path correctly yields `NOT_COMPARABLE` (`:141-147`). `compare([])` returns `[]` and `compare(single_run)` fails with an unrelated `TypeError` (`src/auditkit/api.py:491-517`). *Symptom:* a green comparison between two runs that share no metric, or a candidate that failed every sample.
3. **`[TRAP]` Failures and "unknown" both bend the headline in the flattering direction.** A failed sample keeps `metadata["scores"]` empty, so it is **excluded** from the means rather than counted wrong — a run with many failures can report a *better* mean because the denominator shrank (`src/auditkit/comparison.py:419-439`); `Stat` reports `min=max=mean=0.0` when empty, indistinguishable from a genuine zero (`src/auditkit/score.py:52`); and `Runner.aggregate` averages `value` while ignoring `metadata["unknown"]`, so the RAG-stress and judge metrics' deliberately-flagged unknowns are averaged in as zeros (`src/auditkit/runner.py:488`). *Symptom:* accuracy rises after a batch starts erroring.
4. **`[BUG]` The metric you selected is not always the metric that runs.** `AccNorm` is byte-for-byte `Acc` — no length normalisation, despite the name, docstring and registry entry `acc_norm` (`src/auditkit/metric.py:173-199`). `WordCount()` at defaults names itself `wordcount(0-)` and hits the `count >= 0` branch, scoring 1.0 on every sample (`src/auditkit/metrics/code.py:144-169`). `evaluate(engine="lmeval")` forwards to `run_benchmark` and **silently discards** `scorers`, `adapter`, `annotators`, `extract_with`, `dataset_config`, `dataset_split` and `verbose` (`src/auditkit/api.py:252-258`). *Symptom:* a metric that never varies across models.
5. **`[TRAP]` Silent truncation / misalignment between requests and results.** `generate()` rebuilds samples with a positional `zip(samples, result.predictions)` (`src/auditkit/api.py:362-363`), `Runner`'s chunking splits a batch into exactly `concurrency` pieces (`src/auditkit/runner.py:197`, and this is why `RunConfig.concurrency` defaults to `1` — `src/auditkit/runspec.py:25`), the red-team runner `zip`s prompts to results so a short reply set yields a *shorter, higher-scoring* run (`src/auditkit/redteam/runner.py:119`), and `MCQAdapter` is the one adapter that never calls `_gen_params`, so `temperature`/`max_tokens` are silently dropped on every MCQ run (`src/auditkit/adapter.py:112-114`). *Symptom:* answer N describing sample N−1, or a run that got faster and better at the same time.

Adjacent, one line each: `ModelTimeout` subclasses `AuditKitError`, not `ModelError`, so `except ModelError` misses a timeout (`src/auditkit/errors.py:53`); `ExperimentDB.leaderboard()` sorts the first headline key `reverse=True` regardless of direction, so a MINIMIZE metric ranks worst-first (`src/auditkit/experiment.py:42-49`); `aggregate()` substitutes `0.0` for a metric a run never scored (`src/auditkit/experiment.py:28,35`); `RunResult.from_dict` re-hydrates `perf`/`model_size` to `{}` and zeroes `token_usage`, turning "never measured" into "measured zero" across a JSON round-trip (`src/auditkit/report.py:288`); `redteam` is not importable as `ak.redteam` (`src/auditkit/__init__.py` has no hit for it).

## 7. What is missing, and how to extend

**Missing (not bugs — absences):**

- **Model-card / model-index publishing.** `hf_publish` always writes `repo_type="dataset"` with a fixed `license: other` and `task_categories: [text-generation]` front-matter for every kind (`src/auditkit/hf_publish.py:107-110,236`), and "no model-card and no model-index generator exists" — `grep -rn 'model_card|model-index|model_index|repo_type="model"' src/` returns zero hits (`_research/repo-maps/AuditKIT.md:167`). The missing pieces are enumerated there (`:168`): a `repo_type="model"` creator, a `model-index`/`eval_results` block with `task.type` + `dataset.name/type/config/split` + `metrics[{type,value,name}]`, base-model/adapter lineage (fillable from `RunResult.metadata["inputs"]` and `lexsi_provenance.json`), and per-metric direction/`n`/`stderr`. `hf_publish.py` is the natural home. A run can be published as a *dataset* only — there is no path from `RunResult` to a Hub **model** card.
- **Real ASR.** See §5. Required substitute: SafeTune's judge path (`src/safetune/instrumentation/evaluate/suite/evaluate.py:206`).
- Also absent: any unregister/override on the registries, so two test sessions hand-registering one name collide with `RegistryError` (`src/auditkit/registry.py:26`); `MODELS`/`EVALUATORS` are declared but nothing registers into them (`:60-65`); `ThreatCategory` and `DefconGrade` are declared and never referenced, so there is no DEFCON reporting (`src/auditkit/metrics/security.py:13,44`); `git_sha` in the provenance record is hard-coded `None` everywhere (`src/auditkit/provenance.py:50`).

**Adding a metric** — subclass `Metric` and register it; `__init_subclass__` **requires** a declared `direction` or the class definition itself raises `TypeError` (`src/auditkit/metric.py:32`), which is the guard against the whole compare layer flipping sense:

```python
from auditkit import METRICS, Metric, Score, Direction          # registry.py:60
@METRICS.register("my_metric")                                   # registry.py:26 -> duplicate raises RegistryError
class MyMetric(Metric):
    name = "my_metric"
    direction = Direction.MAXIMIZE       # mandatory (metric.py:32); wrong value silently inverts every grade
    required_fields = frozenset({"target"})   # applicable() skips the metric when None (metric.py:47-53)
    def identity(self): return {"name": self.name, "cfg": self._cfg}   # everything that changes the number
    def score(self, sample, output, context=None):                 # -> Score | list[Score]
        ...                                     # return [] when you cannot measure; never 0.0 (agent.py:449)
```
Two shortcuts: `@scorer(direction=Direction.MAXIMIZE)` wraps a plain `(sample, output) -> float` (`src/auditkit/scorers.py:83`) — note it forces `kind=CODE` and that its `TypeError` fallback re-runs a three-arg scorer (`:19`); and for anything declared in JSON, `ObjectSpec(class_name, args)` + `Registry.create` gives class-name indirection without an import (`src/auditkit/registry.py:19,26`). The metric reaches the CLI/config layer only once its module is imported — `src/auditkit/__init__.py:31` exists solely to run registration decorators, and a metric in a module nobody imports does not exist.

**Adding a judge** — there are two protocols and picking the wrong one changes the failure mode:

- *Generative* (`LLMJudge` or subclass): pass `judge_model="hf:…"/"api:…"/callable`, which `AutoModel.resolve` turns into a live backend (`src/auditkit/model/__init__.py:590`). The judge must emit a final `CHOICE: <label>` or `SCORE: <n lo..hi>` line; an unparseable reply must be reported as UNKNOWN, not 0.0 (`src/auditkit/metrics/judge.py:142`), and the timeout is 600 s with `AUDITKIT_JUDGE_TIMEOUT` override (`:47`). Bare `LLMJudge` sends `system_prompt=None` unless told, so it is unguarded against prompt injection from sample text (`:407`). `judge_identity_args` keeps `api_base` on purpose — the same model behind two servers forks the cache, a different key does not.
- *Encoder* (`EncoderJudge`): one forward pass over `(text, text_pair)`, verdict read off the label distribution, no parse failure possible — but the label map is where it lies, so always pass `label_map=` explicitly for a checkpoint you have not verified (`src/auditkit/metrics/encoder_judge.py:73,220-244`).
- *Scripted*: any `list[str] -> list[str]` callable is accepted as a judge, which is how every test in the repo does it (`src/auditkit/metrics/rag_judge.py:129`).

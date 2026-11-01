# ScarTissue — what a restricted repair buys back

**Lexsi AI Safety & Alignment Hackathon, Track 2.** A Cohere open-weights model is abliterated, then
repaired, and the two things that matter — how much refusal comes back, and how much discrimination is
paid for it — are measured on every arm.

Every number below is quoted from a file in `results/` (name given per claim). Nothing here is an
estimate.

---

## 1. The claim

> Abliteration removes the refusal **read-out**. Only the read-out is repairable, and only through
> directions that survive the edit — so a repair restricted to a small, identified support recovers
> refusal more cheaply, in lost discrimination, than a global rank-one edit. The measurement is a
> **frontier**, not a single number.

Three falsifiable parts, and what happened to each:

| part | test | outcome |
|---|---|---|
| refusal is recoverable at all, weight-space only | amplify the re-extracted direction, no training data | **holds** — 0.217 → 0.742 of 0.758 base |
| recovery is paid for in discrimination | over-refusal on benign prompts at the same dose | **holds** — 0.000 → 0.500 |
| a restricted support is a better trade | frontier, with a random-support control at matched size | **holds in the cheap region, fails at full budget** — 2 % of the scalars reach 54 % of the gap for 40 % of the over-refusal price; the 100 % arm is still the most efficient single point |

---

## 2. Why these metrics

Two axes, because one axis cannot show a trade:

- **Refusal rate on harmful prompts** (n = 120, StrongREJECT) — the benefit side. A high value is not a
  good outcome by itself; it only means the model declines.
- **Over-refusal rate on benign prompts** (n = 40, XSTest) — the price side. A repair that raises the
  first while raising the second has not restored discrimination; it has made the model refuse more
  indiscriminately. This is the axis that separates a working repair from a blunt one: AUC of the
  projection barely moves (0.78–0.86 for both working and inert repairs, against 0.999 at base), so the
  discrimination loss shows up as over-refusal and nowhere else.
- **gsm8k exact-match** (n = 120) — capability, measured in the same run so it is comparable across arms.
  Needed because a repair can damage the model in ways refusal numbers never see.

ASR is *not* a headline axis here, and that is a measurement decision: at 3.35 B the model's non-refusals
are mostly hedging, so a judge-based ASR is a weak discriminator (measured: identical completions scored
0.417 / 0.583 / 1.000 by three judges, pairwise kappa 0.00–0.03). ASR is reported where it was measured
(`results/p5_judge.json`), with its panel disagreement attached, rather than laundered into a headline.

The refusal regex is a **diagnostic**, never ASR. That distinction is load-bearing: at `alpha = 8` the
model degenerates into repetition whose refusal-regex score reads 0.000 — a metric that would have called
a broken model "fully compliant". That is why every generation stores full completions.

## 3. Why these datasets

| dataset | role | why |
|---|---|---|
| `walledai/StrongREJECT` (313 rows, gated) | harmful prompts | a published, citable harm set with categories, so a reader can see what was refused — not a hand-rolled prompt list |
| `Paul/XSTest` (450 rows) | benign prompts | purpose-built to catch over-refusal (safe-looking prompts with risky surface forms), which is the price axis; `allenai/xstest` does not exist and xstest is not an lm-eval task, so it is measured with a judge/regex, not a benchmark |
| `openai/gsm8k` | capability | standard, cheap, and sensitive to the weight damage a rank-one edit causes |

All three are gated-or-public inputs staged through **CuratorKIT** into a 763-row corpus with row-level
provenance (`1-curate/corpus/corpus.jsonl`, published as `scartissue-tinyaya-safety-corpus`), and the
evaluation slices are frozen from that corpus, not re-sampled per run: 120 malicious + 40 benign evaluated,
16 + 16 held out to fit the direction so no prompt is both fitted and scored.

## 4. Why this model

`CohereLabs/tiny-aya-global` — 3.35 B, 36 layers, d_model 2048, 16 q-heads. Chosen because the track
requires a **Cohere dense open-weights** model and this is the largest one that fits a single A100-40GB
with both the HF model and the TransformerLens copy resident in fp32 (6.7 GB each). The 32 B sibling
(`aya-expanse-32b`) needs an H200 at 139.8 GiB and cannot host the circuit leg at all, so it is used only
where inference is enough.

Two properties make the geometry tractable: every one of the 72 write matrices (`o_proj`, `down_proj`)
has `out_features == d_model == 2048`, so a column costs exactly 2048 scalars and budgets are exact, not
approximate; and `tiny-aya-global` is one of the four tiny-aya checkpoints that ship a chat template
(`base` does not), so prompts are formatted the way the model was trained.

---

## 5. What was found

### 5.1 The drift is real and reproducible (`results/p1_grid.json`, `p3b_sweep.json`, `p4_replicate.json`)

A 30-line operator of our own — no SafeTune code in the path — edits all 72 write matrices
(962,592,768 scalars, a 3.58 GB fp32 patch) along the layer-15 refusal direction fitted on 16 harmful +
16 benign held-out prompts:

| | refusal | over-refusal | gsm8k | AUC(d′) |
|---|---|---|---|---|
| base | 0.7583 | 0.0500 | 0.1167 | 0.9988 |
| after abliteration | 0.2167 | 0.0000 | 0.1167 | 0.8306 |

The same operator on `tiny-aya-earth` gives 0.7333 → 0.2500 (`results/p4_replicate.json`), i.e. the
effect is not a single-checkpoint artefact. `tiny-aya-water`'s container never reported and is **not**
claimed.

### 5.2 Repair works, and the price is the discriminator (`results/p3b_sweep.json`)

After abliteration the re-extracted direction `d′` is **orthogonal** to the one that was removed
(cos = 1.94e-4, measured; in p1 it was −2.8e-6), so amplifying `d` is a no-op by construction
(`dᵀW = 0`). Every repair below therefore acts along `d′`, restricted to a support:

| arm | support | refusal | over-refusal | gsm8k | gap recovered | refusal per unit over-refusal |
|---|---|---|---|---|---|---|
| global @ α=4 | 100 % | 0.7417 | 0.5000 | 0.2167 | 97.0 % | 1.48 |
| harm 25 % @ α=4 | 25 % | 0.7083 | 0.4000 | 0.2750 | 90.8 % | 1.77 |
| harm 10 % @ α=4 | 10 % | 0.6083 | 0.3000 | 0.2667 | 72.3 % | 2.03 |
| harm 4 % @ α=4 | 4 % | 0.5167 | 0.2250 | 0.2833 | 55.4 % | 2.30 |
| harm 2 % @ α=4 | 2 % | 0.5083 | 0.2000 | 0.3000 | 53.8 % | 2.54 |
| harm 10 % @ α=2 | 10 % | 0.3667 | 0.0500 | 0.2000 | 27.6 % | 7.33 |

Refusal is recoverable to 97 % of the base rate from weights alone, with no training data — and that is
exactly the point where over-refusal reaches 0.500, ten times the base rate. **Abliteration deletes the
discriminator, not only the read-out.**

### 5.3 The control that makes "restricted" mean something (`results/p3c_blind.json`)

| arm | support | refusal | over-refusal |
|---|---|---|---|
| global @ α=4 (same run) | 100 % | 0.7417 | 0.5000 |
| random columns @ 2 % | 2 % | 0.2083 | 0.0250 |
| random columns @ 10 % | 10 % | 0.2083 | 0.0000 |
| amplitude-ranked @ 10 % | 10 % | 0.5250 | 0.2250 |

Random supports are **inert at both sizes** (0.208 ≈ the drifted 0.217). So the restricted repairs do not
work by "editing a lot of things" — they work because the columns are chosen. But the control also
disciplines the claim in the other direction: ranking by amplitude alone already recovers 57 % of the gap,
so harm-weighting is an *ordering*, not the mechanism. And at matched 10 % the amplitude arm is the more
efficient point (2.33 vs 2.03 refusal per unit over-refusal). The honest statement is therefore a
**frontier**, with harm-weighting buying a cheaper entry point (2 % of the scalars, 53.8 % of the gap for
0.200 over-refusal) rather than a free lunch anywhere.

### 5.4 What was published

| repo | content |
|---|---|
| `scartissue-tinyaya-safety-corpus` | 763 rows, row-level provenance (`id` + `provenance_chain` per row), manifest with per-source sha256 and the git commit |
| `scartissue-tinyaya-abliteration-patches` | the drift as **parameters**: unit direction `d`, `alpha = −1`, per-matrix column projections, plus the 30-line loader |
| `scartissue-tinyaya-repair-patches` | the four repair patches, each carrying its `allowed` block `{modules, layers, nodes[{columns, heads, whole_heads}]}` — the allowed set *is* the claim |
| `scartissue-tinyaya-evals` | full completions for 7 arms × 160 prompts, the frontier CSV, the manifest |
| Space `amugoodbad229/scartissue` | the frontier and three columns (base / drifted / repaired) over the stored completions |

The drift is published as parameters and **never** as a drifted checkpoint: LSAL 1.2/1.1 §4 forbids
deploying one, so the edit ships as a direction, a scalar and the projection onto the edited columns, and
the model on the Hub stays clean.

---

## 6. Why this is better for a user

Someone who abliterates a model to make it more useful hits the failure this measures: the model stops
refusing, and when they try to put refusal back they get a model that refuses **everything**. This work
gives them a knob and its price list — pick a support size, read the recovery and the over-refusal off the
frontier, and apply the patch with a 30-line loader instead of hunting hyperparameters blind. The random
arms are part of that offer: they say which knobs do nothing, so nobody spends a GPU-day on them.

It also says where the wall is: past ~10 % of the scalars, extra recovery is bought almost entirely with
over-refusal, and only the full edit ever approaches the base refusal rate.

## 7. Reproduce

```bash
uv sync                                       # uv only; never pip, never a hand-made venv
uv run python -m modal run experiments/p3b_sweep.py   # the frontier (6 arms, A100-40GB)
uv run python -m modal run experiments/p3c_blind.py   # the random/amplitude controls
uv run python -m modal run experiments/p7_artifacts.py
uv run --with pydantic --with typer --with tqdm --with pyyaml --with datasets \
  PYTHONPATH=<ScarTissue/repos/CuratorKIT> python experiments/p8_corpus.py
```

Every experiment is a Modal app with a `@app.local_entrypoint`, so `--detach` is never needed and the
result is pulled with `python -m modal volume get`. `experiments/p6_circuit.py` runs the circuit discovery
on the drifted model.

## 8. Limitations

- **n and seeds.** 120 harmful / 40 benign prompts gives a binomial standard deviation around 4–7 points;
  differences under ~0.05 are noise. The drift uses one seed; the arm sweep one seed per arm.
- **Regex refusal.** Refusal is a regex diagnostic. The judge-based ASR exists for two arms only
  (`results/p5_judge.json`: global 0.025, restricted-heads 0.142) and its panel disagrees wildly — that is
  reported, not smoothed.
- **One model for the headline.** Only Cohere models appear in headline numbers, and only `global` has the
  full frontier; `earth` reproduces the drift, `water` did not report, and `aya-expanse-8b` was not run.
- **The circuit leg is not yet in the headline.** Discovery on the drifted model is implemented and runs
  through CircuitKIT's TransformerLens path, but the support used for the repair arms is
  attribution-weighted, not circuit-derived. The claim in §5.3 is about *restricted* supports, not about
  circuits.
- **The Space is static.** It reads stored completions; it does not run a model, so it cannot answer a
  prompt a visitor types.

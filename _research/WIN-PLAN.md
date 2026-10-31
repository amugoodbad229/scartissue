# Can we win? — scoring the hackathon against its own rules

Source: `_research/lexsi/index.txt` (full page scrape, 2026-10-29). The rules say judges "score each team
numerically on a fixed scale", shortlist **up to 20 teams** by Nov 16, then decide winners **after a live demo on
Nov 22**. So there are two different games: **getting shortlisted** (fixed rubric, submission artifacts) and
**winning** (judges arguing about live demos).

Timeline correction against my earlier notes: the event spans **Nov 1–21**, submission is due **Nov 14**, shortlist
**Nov 16**, shortlisted teams get **Nov 16–21** to "prove their thesis" with **$200 HF credits**, Demo Day
**Nov 22**.

## 1. The rubric, decomposed from the rules text

| criterion | what the rules actually say | weight on a fixed scale (my read) |
|---|---|---|
| chain depth | minimum 2 track repos used together; **3+ preferred** | high |
| before/after | "**more than one** metric" | high |
| HF artifacts | **models, adapters, logs, datasets** all hosted | high |
| public deployment | "public demo or repository link" by Nov 14 | high |
| write-up | metrics chosen and **why**, dataset/task/model rationale, why the model is "meaningfully better for usage" | high |
| originality | "unique and your own work" | medium |
| ROI table | *optional but preferred*: self-host vs closed-source token billing at realistic volume | bonus |
| track fit | Cohere dense open-weights model; Track-2 chain order | pass/fail |

## 2. Our scorecard, honestly

**Strong already** — chain depth (all four Track-2 repos, each with a real job), metric count (ASR, capability,
overall coherence, judge-agreement κ, wrapper robustness, frontier area, plus a protocol-bound frontier rather than
one cell), write-up substance (`MEASUREMENT.md`, `PROTOCOL.md`, `ACADEMIC-FRAME.md` — most teams will submit a
README), originality (irreparability bound + circuit-restricted repair + 10 blind baselines at matched capability
cost is not the obvious submission), and pre-registration, which almost nobody will have.

**Weak / not yet started** — **no HF artifacts pushed** (no drifted checkpoints, no patches, no curated dataset, no
logs on the Hub), **no public demo**, **no ROI table**, and no results yet because building starts Nov 1 by our own
protocol.

**The structural risk nobody can plan away** — a one-person team against teams of three, and a thesis whose honest
outcome may be "at 3.35 B, repair mostly fails". A "we could not repair it" headline is correct and interesting,
but it must be paired with at least one arm that *does* recover refusals at acceptable capability cost, or a judge
rationalising a fixed score will read the project as a measurement study rather than a fix.

## 3. What converts "shortlisted" into "top five"

1. **One arm that actually wins.** The head/layer-restricted repair must beat global rank-one at matched edits (in
   the protocol that is the whole point of H1). If 3.35 B refuses to cooperate, run the **8 B** arm — both Cohere
   ports already pass parity, so this is a config change, not research.
2. **A Space a judge can touch in 30 seconds.** Dropdown *base / drifted / repaired*, a text box, side-by-side
   generations and the judge verdicts. This is what Demo Day rewards, and it is a weekend of work, not a project.
3. **The artifacts table on the Hub**, one row per artifact with a click: curated dataset (with manifest), drift
   checkpoints (or abliteration patches — smaller and licence-clean), repair patches, eval logs, the frontier figure.
4. **The ROI table** (bonus) — we already have 267.6 tok/s at 3.35 B and 98.7 tok/s at 32 B measured, so the
   self-host cost is arithmetic: A100-40GB $/h at 267 tok/s versus closed-source token billing at a realistic
   monthly volume. Twenty lines of honest arithmetic; almost nobody will submit it.
5. **A story a non-specialist remembers**, one sentence: *abliteration is being shipped as "uncensored" models; we
   measure what repair can and cannot recover, and we publish the frontier where safety is bought back at a known
   capability price.*

## 4. Fourteen days, with gates

| days | work | gate to pass before moving on |
|---|---|---|
| 1–3 | CuratorKIT safety corpus (provenance-exportable rows) → drift on tiny-aya-global + 5 seeds | drift reproduces outside SafeTune's harness (already verified once) |
| 4–6 | CircuitKIT on the **drifted** model (`M`), then the repair grid: blind arms + restricted arms at matched edits | `M` computed for every model before any `R` is looked at |
| 7–9 | AuditKIT evaluation: judge panel majority-of-3, capability via lm-eval, κ, wrapper test on every checkpoint | ≥2 metrics per arm, CIs, per-prompt dump |
| 10–11 | 8 B arm (fallback where repair must win), 11-language arm if time allows | one arm with recovery ≫ blind baseline |
| 12–13 | HF artifacts + Space demo + HF Article + ROI table | judge can click from the article to every artifact |
| 14 | submission + freeze | — |

Days 16–21 (if shortlisted): polish the demo, re-run the headline arm with more seeds, add the human-agreement
table.

## 5. Risks, ranked

1. **Repair fails everywhere at 3.35 B** → mitigations already in place: global rank-one (ROSI-shaped) arm, 8 B arm,
   and a headline that survives without repair success (the bound is the contribution; recovery is the demonstration).
2. **No artifacts on the Hub at submission** → treated as a hard gate on day 12, not a nice-to-have. Everything
   else can slip; this cannot.
3. **Judge-shopping**: a judge who knows arXiv 2609.08236 will test our ASR number. Our κ and human table are the
   answer; publishing them is the defence.
4. **Cohere model constraint violated** (e.g. drifting a non-Cohere model) → every headline number must come from a
   Cohere checkpoint; cross-family runs are supplementary only.
5. **Licence optics** — LSAL §4 forbids deploying unrepaired drifted checkpoints. Ship **abliteration parameters +
   patch scripts**, not "uncensored model" downloads, and say why in the card.

## 6. Honest probability

- **Shortlist (top 20): high.** The rubric rewards exactly what we already have (four-repo chain, many metrics,
  real write-up); the main way to fail is administrative — no artifacts, no public link.
- **Top five: genuinely open.** It needs one winning arm and a demo a judge can click. Our rigour is above the
  field's median; our *showmanship* is currently below it, and Demo Day is a showmanship contest.
- **Interview track: plausible.** Judges are from Lexsi/Cohere/HF and the article is the artifact they read.

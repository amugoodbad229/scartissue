# LSAL 1.2 / 1.1 — what the four licence texts actually permit us to do

Read in full: `repos/SafeTune/LICENSE.md` (99 lines, LSAL **v1.2**), `repos/CircuitKIT/LICENSE.md`
(100 lines, v1.2), `repos/AuditKIT/LICENSE.md` (92 lines, v1.2), `repos/CuratorKIT/LICENSE.md` (75 lines,
**v1.1**). `AgentTune` and `aligntune` are no longer checked out (dropped with the Track-3 scope cut).

## The three clauses that decide everything

**§1 — noncommercial grant.** Free use, copy, modify, merge, publish, redistribute, *for Noncommercial
Purposes only*, with the notice + this licence + §4 travelling with every copy. "Noncommercial Purposes" is
defined to include, verbatim: *"personal use for research, experimentation, private study, or hobby
projects"* and *"academic and scholarly research, teaching, and publication (including use in papers, theses,
benchmarks, and reproducibility artifacts)"*.

**§1A — use by organizations (v1.2)** requires prior written permission **or** a written acknowledgement to
`support@lexsi.ai` before *"any organization, including a charitable organization, educational institution,
public research organization, or government body"* uses the work for *"internal evaluation, red-teaming,
benchmarking, safety auditing, or any use in connection with developing, evaluating, repairing, or
safeguarding its own models, products, or services"*.

**§2 — commercial restriction.** No right to "Sell": offering it (or a derivative) as a commercial product,
paid service, SaaS, hosted or API offering; embedding it in proprietary or revenue-generating software; paid
consulting whose substance is the work. Also no re-licensing, rebranding or redistribution under different
terms, and no use of the Lexsi/SafeTune/CircuitKIT marks beyond identifying unmodified copies.

Supporting clauses: **§4** forbids deploying an *unrepaired* drifted checkpoint in production (§4 of the
CircuitKIT text additionally forbids using it to locate/remove/weaken/bypass safety behaviours *in order to
deploy the result* — which is exactly what a research study of abliteration-and-repair does *not* do);
**§6** contributions get relicensed to Lexsi including commercially; **§9** governed by Indian law;
**§8** terminates on breach.

## Version difference that matters

LSAL **v1.1** (CuratorKIT) folds *"internal evaluation, red-teaming, or safety auditing that is not itself a
paid product or service"* **into** the §1 noncommercial grant — no acknowledgement needed. LSAL **v1.2**
(SafeTune, CircuitKIT, AuditKIT) **pulls that same activity out** into §1A and requires a written
acknowledgement from any *organization* doing it.

## Our position

1. **We are not an organization using these tools on our own models.** We are a hackathon team doing
   research on **third-party** (CohereLabs) models, with no product, no service and no revenue. That is §1's
   *"personal use for research, experimentation"* and *"academic and scholarly research … benchmarks and
   reproducibility artifacts"* — squarely inside the free grant.
2. **The prize money is not "Sell".** §2 defines selling as providing, *to third parties, for a fee or other
   consideration,* a product or service whose value derives from the work. A competition prize is not a
   sale of a product to a customer, and we supply nothing to Lexsi's market. Still a judgment call, not a
   certainty.
3. **§1A is cheap to satisfy and we should satisfy it anyway.** One email to `support@lexsi.ai` identifying
   the team, the intended use (research: measuring repairability of abliterated CohereLabs checkpoints with
   SafeTune + CircuitKIT + AuditKIT), and an undertaking to attribute. If Lexsi decides we *are* an
   organization, we are then already compliant; if they don't, nothing was lost. **Draft is below.**
4. **What we must do regardless of interpretation:** keep the licence + §4 notice in every copy and in the
   repo we submit; mark modified files as modified (§2); do **not** deploy any unrepaired drifted checkpoint
   as a user-facing demo — the demo must serve a *repaired and re-evaluated* model, or run in a clearly
   labelled evaluation mode; do not use the Lexsi marks as our product name.
5. **The business-plan implication stands:** an internal team using this for its own models needs an
   acknowledgement; a leaderboard or CI gate sold to model providers needs a **commercial licence**. If we
   ever sell Tier 2–3, we ask Lexsi for a commercial licence — which is a conversation, not a blocker, and
   the hackathon is the right place to start it.

## Draft acknowledgement (one email, copy to keep)

> Subject: Acknowledgement of use — Lexsi Labs Source Available License (§1A)
>
> Hello,
>
> We are a team participating in the Lexsi AI Safety & Alignment Hackathon (Nov 1–14, 2026), Track 2. Under
> §1A of the Lexsi Labs Source Available License v1.2 we acknowledge our use of SafeTune, CircuitKIT and
> AuditKIT (and, if used, CuratorKIT under v1.1).
>
> Intended use: noncommercial research on publicly released CohereLabs models permitted by the hackathon —
> measuring how far refusal behaviour removed by weight orthogonalisation ("abliteration") can be restored
> by circuit-restricted repair, at matched capability cost, and auditing the result. All artefacts are
> research outputs published on Hugging Face for the hackathon, with the licence notice and §4 conditions
> reproduced in full.
>
> We will attribute Lexsi Labs and the four libraries in every artifact and in the write-up, and we will
> share the evaluation reports the harness produces in its standard format (excluding weights, training
> data, prompt text and generations) if that is useful to you.
>
> Please confirm whether you require anything further, or whether a written permission under §1A is needed
> in addition to this acknowledgement.
>
> Kind regards,
> <name>, <team>, <email>

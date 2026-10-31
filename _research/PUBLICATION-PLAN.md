# Can this be published at a top venue? — an honest audit, before we write a line

Short answer: **as scoped for the fourteen days, no. As scoped for February, yes.** The difference is four
specific evidence gaps, all of them closable, and one theoretical obligation that only we can discharge.

The deadline calendar below is stated by *typical cycle month* rather than as a confirmed date — the 2027 calls
are not all posted yet, so treat the months as planning targets to verify when each CFP appears.

## 1. Venue fit, and what each one would demand beyond Nov 14

| venue | fit | deadline (typical) | what it needs that we won't have |
|---|---|---|---|
| **NeurIPS / ICML main** | medium | ICML ~Jan, NeurIPS ~May | cross-family evidence, 5+ seeds, a *theorem*-grade bound, human-validated judge |
| **ICLR main** | medium | ~Sep | as above, plus the "why is this not head-level orthogonalisation" answer |
| **COLM** | **high** | ~Mar | 8 B arm + one non-Cohere family; still needs the causal test |
| **ACL / EMNLP** | **high** (multilingual arm) | ~Feb / ~May | 11-language arm at ≥8 B, native-speaker spot check |
| **TMLR** | high (rolling) | any time | rigour, not novelty theatre — our natural fallback |
| **SaTML / USENIX Security** | low–medium | ~Oct / ~Feb | security reviewers want attacks; a repair paper reads as defence without an adversary |
| **NeurIPS/ICML/COLM safety workshops** | **very high** | ~Aug–Sep | nothing beyond what Nov 14 already produces |

So the realistic ladder is: **workshop paper (weeks) → COLM or EMNLP main (~Feb) → NeurIPS/ICML (2027) only if the
bound becomes a theorem and the evidence goes cross-family.**

## 2. The four gaps, measured rather than feared

1. **Scale.** Every claim now rests on a 3.35 B checkpoint. `_research/MEASUREMENT.md` §8 is 3.35 B; the 32 B arm
   is inference-and-repair only, with no circuit leg (`PROTOCOL.md` Appendix C, 258.4 GiB of fp32). A reviewer
   will ask whether the bound is a property of the model or of a small model. **Fix:** one 8 B arm (Aya Expanse 8 B
   already passes both parity gates) and one non-Cohere family (Qwen3 or Llama-3.2) — the second is what makes
   "architecture-general" true rather than hopeful.
2. **Statistics.** 12 prompts × 11 languages, single seed, κ unstable by construction (§8.3). **Fix:** 5
   abliteration seeds × 313 StrongREJECT prompts (the per-cell cost is measured at ~4 min of generation, so this
   is arithmetic, not courage) + bootstrap CIs on ASR *and* κ.
3. **The judge.** We deliberately measured a 58-point spread between judges (§8.3). Reviewers who know arXiv
   2609.08236 will use exactly that against us. **Fix:** majority-of-3 as primary, plus **human validation on 200
   sampled completions** with a reported agreement rate — that single table converts our biggest weakness into an
   asset.
4. **Causality.** Correlation between surviving circuit mass and recovery is not a bound. **Fix:** intervene on
   `M` itself — ablate the top-k heads of the surviving circuit before repair and show recovery tracks the
   predicted curve; that is the experiment that turns "we found a pattern" into "we found a mechanism".

## 3. The theoretical obligation

The claim "only the read-out is repairable, and only up to the residual circuit" must be stated as something that
can be wrong. The algebra already sketched in `_research/ACADEMIC-FRAME.md` gives the shape: orthogonalisation
projects the read-out onto the complement of `d`, so recoverable refusal is bounded by the pre-drift component of
the read-out that survives the projection, and the bound is tight when the remaining circuit still carries the
harm representation. Making that precise — a projection bound with stated assumptions, plus the falsifier we
pre-registered — is one section of mathematics and the thing that distinguishes a paper from a report.

## 4. Three tracks, so nothing depends on one decision

- **Track A (guaranteed):** arXiv preprint + a safety-track workshop paper from the Nov 14 results. What we already
  have — two passing parity gates, a validated instrument, a measured negative result (resta null at 3.35 B) and a
  pre-registered protocol — is above the bar for a workshop.
- **Track B (likely):** COLM or EMNLP main, requiring the 8 B arm, 5 seeds, and the 200-item human table. Deadline
  window February–March; the work is ~3 weeks of compute after Nov 14.
- **Track C (conditional):** NeurIPS/ICML 2027, requiring the projection theorem **and** the non-Cohere family.

## 5. What to protect inside the window (cheap, and expensive to add later)

1. Run **5 abliteration seeds** even if only one is reported — seeds cost 0.02 s each.
2. Keep every completion (prompt, completion, judge verdicts) as a dump; the human-validation set must be sampled
   from real runs, not regenerated.
3. Acknowledge that the two judged-arm tables share the same 313 prompts → report per-prompt, not just means.
4. Write the manifest from day one (corpus hash → drift params → circuit → arm → audit) — that *is* the artifact
   badge, and the hackathon mandates the chained-repo form anyway.
5. Freeze the protocol before looking at repair numbers. We already did; keep the file untouched and date-stamped.

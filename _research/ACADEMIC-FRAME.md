# The academic frame

Written after the seven-repo read and the two Modal probes, for the question: *what is the strongest
defensible contribution this chain can produce in 14 days, and what would a reviewer accept as evidence?*

The chain we run, in the order the hackathon defines it:

```
CuratorKIT            SafeTune                 CircuitKIT               AuditKIT
build the corpus  →   audit drift, then   →    locate the circuit,  →   evaluate the hardened
with provenance       apply repair arms         intervene inside it      model + capability
```

---

## 1. Position in the literature

Three lineages, and the gap is at their intersection:

**A. Refusal-direction removal (offence).** Arditi et al. 2024 established that refusal is mediated by a
one-dimensional direction, and that orthogonalising every write matrix against it removes it. The 2026
literature then industrialised this: 21,600 downstream decisions show the damage reaches reasoning and
calibration, not just refusals (arXiv:**2607.17427**, *Abliteration Is Not a Scalpel*). The multilingual
picture is graded, not binary (arXiv:**2610.28897**): directions transfer from aligned sources, fail between
misaligned ones, and geometry and behaviour dissociate.

**B. Circuit localisation (interpretability).** Circuit discovery localises behaviour to a sparse subgraph —
attention heads and MLP sublayers — with EAP-IG and relatives. Circuit-guided *defence by weight scaling*
already exists (arXiv:**2609.00051**, +26.5 % safety under attack, −1.7 % accuracy).

**C. Repair (defence).** Rank-one safety injection re-aligns uncensored models by amplifying their own latent
directions (arXiv:**2508.20766**, ROSI); MSCP repairs through neurons (arXiv:**2508.09190**); C-Θ repairs
through layers (SafeTune `apply_ctheta`). Every one of these is *unbounded in principle*: each edits a set of
parameters chosen by a heuristic, and none reports what the choice of set costs.

**The gap.** Repair is treated as a search over arms, not as a statement about the model. Nobody has
(a) defined the *repairability bound* of an abliterated model, (b) tested whether **where** the repair acts —
head, neuron, layer, or everywhere — explains the capability price at matched recovery, or (c) unified the
offence, the localisation and the defence inside one chain with a shared measurement contract. The
competitor papers each hold one leg; the chain holds all three, which is exactly what the hackathon's
four-repo structure is built to force.

## 2. The claim, stated so it can fail

> Abliteration deletes two different things: the refusal **read-out** and the harm **detection** signal. Only
> the read-out is repairable, and only through the part of the safety circuit that survives the edit.
> Therefore recoverable refusal is bounded by *surviving circuit mass*, and repair restricted to that circuit
> reaches the bound that global rank-one injection overshoots — paying for the overshoot in capability.

Three quantities, all measurable:

- **Surviving circuit mass** `M = Σ_{n ∈ C} s_n / Σ_{n} s_n` — the abliteration-normalised attribution of the
  circuit nodes `C` under the refusal metric, computed *after* the drift, with the sign restored (CircuitKIT
  currently destroys it: `backends/eap/circuit_kit_adapter.py:77`).
- **Recovery** `R = (ASR_drift − ASR_repair) / (ASR_drift − ASR_base)` — fraction of the refused-again gap closed.
- **Capability price** `ΔC` per unit `R`, measured by `lm-eval` and cross-checked by AuditKIT `compare`.

## 3. Why there is a bound at all (the argument a reviewer will attack first)

Orthogonalisation applies `W ← W − d dᵀ W` (SafeTune `refusal_direction.py:627`, and ROSI's rank-one edit is
the same object with a different `d`). For a linearised read-out `r` at the refusal layer, the refusal score
of a prompt with residual activation `x` becomes

```
score_after = r · (I − d dᵀ) x  =  (r · x) − (r · d)(d · x)
```

so the edit removes exactly the component of `r` along `d`. Anything the model could still use to refuse must
therefore live in the **orthogonal complement** of `d` — and that is what a restricted repair can restore,
because the parameters carrying it were never edited out. A *global* repair instead re-injects a new direction
into every matrix, including matrices whose input subspace no longer carries `x`; that direction is free to
drift into unrelated computation, which is the mechanism by which it can buy recovery beyond the bound at a
capability price. **This is the whole thesis in one line of algebra**, and it makes two predictions that a
reviewer can check: recovery should track `M` monotonically (H2), and post-repair detection probes should
still be degraded because detection was destroyed upstream of the read-out (H4), consistent with
arXiv:2608.08032's finding that detection is *orthogonal to the write* and the write is late.

The bound itself is then `R_max = M`, i.e. an abliterated model cannot be made more repairable than the
fraction of its refusal machinery the edit did not touch. Estimating `R_max` from the surviving circuit (not
from the best arm's score) is what stops the analysis being circular — the single design decision most likely
to decide whether a reviewer believes the result.

## 4. Design (each element answers a specific reviewer objection)

| element | reviewer objection it pre-empts |
|---|---|
| 11 arms matched on **edited parameter count** ±10 % | "you compared a cheaper edit with an expensive one" |
| `circuit/random` (same count, same layers, random membership) | "the gain is just more parameters in the right layers" |
| `magnitude/all` (edit everything, no direction) | "any perturbation of this size would do this" |
| blind family = the repo's own ✅-ledger arms (`resta`, `mscp`) | "you beat a strawman" |
| `ctheta` (layer-restricted) as the arm to beat | "MSCP/C-Θ already do circuit-guided repair" — the honest prior art |
| ROSI-style global rank-one | "the strongest published method is missing" |
| 3 discovery seeds, mean ± range | "EAP-IG is stochastic; your circuit is one draw" |
| direction fit on a slice disjoint from eval | leaked prompt selection inflating recovery |
| judge-path ASR only, second judge on 20 % audit | "your number came from a substring classifier" |
| capability from `lm-eval`, not the safety suite | the suite scores gsm8k/humaneval with the *harm* judge |
| 7 models, 2 axes (pretraining mix × scale) | "n = 1 model, and tiny-aya is one pretraining recipe" |
| 11 languages, reported as exploratory, single seed | over-claiming the multilingual result |

## 5. Contributions (the paper's contribution list, in order)

1. **A repairability bound** for abliterated models, defined from surviving circuit mass and estimated before
   any repair arm is evaluated.
2. **Head-level circuit-restricted repair** — the granularity nobody ships: no repository slices a single
   attention head's column block of `o_proj`, and SafeTune cannot (it reads no attention-head configuration).
3. **A matched-cost frontier** over 11 arms × 7 models, the first place where the question "which repair
   method should I use on *my* model" is answerable with one figure instead of eleven codebases.
4. **A repair-transfer matrix across 11 languages**, with the prediction that gain tracks pre-drift circuit
   presence — the first multilingual test of *repair* (the existing work tests drift).
5. **A provenance-carrying artifact**: corpus → drift → circuit → repair → audit in one chain, every number
   traceable to a run manifest, published as dataset + patches + cards.

## 6. Threats to validity, stated up front

- **Judge dependence.** ASR is a judge's opinion. Mitigation: two judges, agreement reported, and every cell
  reports raw generations.
- **Scale.** 3.35 B and 8 B, not the 70 B models the defence papers use. Mitigation: report *ratios*, and
  treat tiny-aya's four pretraining mixes as four independent replications of the same geometry.
- **Prompts we wrote.** The multilingual set is ours, 12 per language, single seed. Mitigation: release it,
  label it exploratory, add a second set if time allows, never claim confirmatory status.
- **Circularity.** The bound must not be read off the winning arm. Mitigation: compute `M` before looking at
  `R`, and pre-register the decision rule (`PROTOCOL.md` §5).
- **The circuit's own sign loss.** CircuitKIT drops the sign at conversion, so "the circuit" today is a set of
  magnitudes. Mitigation: fix it in our patch and report both the signed and unsigned analyses.

## 7. What this is not

Not a new abliteration attack (drift is the *control*, not the contribution), not a claim that repair restores
safety (H4 predicts it does not), and not a ranking of products. The paper's honest headline, whatever the
numbers do, is: **repair is bounded, the bound is measurable, and the granularity of the repair is the
variable that decides the capability price.**

## 8. Venue and shape

Workshop-scale, 8 pages: COLM/NeurIPS safety or interpretability workshop for the bound + multilingual arms;
the artifact chain doubles as the hackathon submission and as a reproducibility release (dataset card, model
cards with `model-index`, run manifests). Demo Day wants one figure: the frontier with the bound drawn on it.

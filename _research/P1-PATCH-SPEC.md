# P1 patch spec — the three edits that make Candidate 1 possible

Status: **spec only, nothing implemented** (build window opens Nov 1). Every anchor below was verified by
reading the file during the documentation pass; `_docs/<Repo>/…` holds the line-by-line evidence.

Purpose: turn "circuit-guided repair" from a slogan into three small, testable edits. Each is scoped so it
can be done and checked in one sitting, and each has an acceptance test that fails today.

---

## Patch 1 — head-level restriction in the repair operator (SafeTune)

**Problem.** `orthogonalize_weights` is all-or-nothing per matrix family: `include_attn` selects every
`o_proj`/`out_proj`/`c_proj` in the model, `include_mlp` selects every `down_proj`. There is no way to say
"only these heads" — so the operator cannot express a circuit. Measured consequence: the best restricted
arm we have (`apply_ctheta`) is restricted to a **layer subset**, still editing whole matrices.

**Anchor.** `repos/SafeTune/src/safetune/interventions/steer/refusal_direction.py:627`
`def orthogonalize_weights(model, direction, include_attn=True, include_mlp=True)`.

The gap-closing read (`_docs/SafeTune/10-steer-harden-gaps.md:606-650`) pins the mechanics exactly:
`@torch.no_grad()`, the update is `proj = torch.outer(dt, dt) @ weight.data; weight.data.sub_(proj)` with
`dt = d.to(dtype=weight.dtype)` and `d` re-normalised first — **row space** (`W − d dᵀ W`), in place on
`.data`. Selection is `endswith` on the *relative* module name and requires `nn.Linear` (Conv1D GPT-2 is
excluded and the empty result raises `ValueError`). Reversibility exists only through the returned snapshot
dict, keyed `f"layer{li}.{name}"`.

Two facts that change the implementation: (1) **the file never reads `num_attention_heads`, `head_dim` or
`num_key_value_heads`** — grep-verified, so the patch must read them off the model config itself; (2)
`o_proj.weight` is `[hidden, n_heads*head_dim]`, rows = residual, columns = heads, so the column block for
head `h` is `W[:, h*hd:(h+1)*hd]` and the block-local projector is `W[:, blk] -= d_h d_hᵀ @ W[:, blk]`.
A `k_proj`/`v_proj` variant would be `[n_kv_heads*head_dim, hidden]` — different axis, and the current
`endswith` table has no entry for either.

**Known landmine to fix in the same patch:** `orthogonalize_weights` lists `c_proj` in **both** branches and
is `elif`-chained, so GPT-style models get attention `c_proj` edited but never MLP `c_proj` — a silent
half-orthogonalisation. Harmless on Cohere/Llama naming, but it must not survive into a paper artifact.

**Change.** Add two optional keyword arguments, defaulting to `None` (today's behaviour):

```python
orthogonalize_weights(model, direction, include_attn=True, include_mlp=True,
                      layers=None,      # None = all; list/set of layer indices otherwise
                      heads=None,       # None = all; accepted either {(layer, head)} or {head}
                      head_axis="out")  # "out": zero d's component in W_O columns; "in": W_Q/W_K rows
```

Implementation notes that the code dictates:

- `W_O` is stored as `(d_model, n_heads * head_dim)`; a head `h` owns columns
  `[h*head_dim : (h+1)*head_dim]`. Only those columns may be touched for `head_axis="out"`.
- **GQA**: `n_heads != n_kv_heads`. Queries are grouped, so head `h` reads KV head
  `h * n_kv_heads // n_heads`. Any `head_axis="in"` edit (rows of `W_Q`/`W_K`) must use that mapping; the
  rest of the codebase already gets this right in `CircuitKIT/applications/selective_finetuning/selector.py:140`,
  which is the function to copy from rather than re-derive.
- Reversibility must survive: `restore_weights` (`refusal_direction.py:694`) snapshots whole tensors, so a
  partial edit that goes through the same snapshot path stays reversible for free.
- `_matched_param_keys` matches whole `.weight` tensors today; it must keep doing so (no signature change),
  because the restriction happens inside the update, not in key matching.

**Acceptance.** With `layers={15}` and `heads=set(range(16))`, exactly one `o_proj` changes, and only
`W_O[:, :head_dim*16]` differs; every other tensor is bit-identical (`torch.equal`). With `heads={0}`,
exactly `head_dim` columns change.

**Cost.** ~40 lines + 2 call sites (`redteam/abliteration.py`, `recover/ctheta.py`).

---

## Patch 2 — node-localised refusal direction (SafeTune × CircuitKIT)

**Problem.** Directions are extracted per **layer** (mean-difference over one hook point). Circuit nodes are
per **(layer, head)** or per **MLP**. Nothing in either repo extracts a direction *at a node*, so a
node-restricted repair has no direction to use — this is the piece no published work covers.

**Anchor.** `refusal_direction.py:171` `extract_refusal_direction(...) -> (unit_dir, layer, all_layers)`;
hook application at `:539` `RefusalDirectionModel` (hooks `blocks.[l].hook_out`).

**Change.** New function beside it:

```python
def extract_node_localised_direction(model, harmful, harmless, *,
                                     node,               # ("a", 15, 3) or ("m", 15, None)
                                     position="last_instruction",
                                     normalize=True) -> torch.Tensor:  # (head_dim,) or (d_mlp,)
```

- Hook `blocks.[l].attn.hook_z` for a head node → slice `[..., h*head_dim:(h+1)*head_dim]`.
- Hook `blocks.[l].mlp.hook_post` for an MLP node → full width (this is the *neuron* space; note the trap
  recorded in `_docs`: under the default `mlp_hook="mlp_out"` the width is `d_model`, not `d_mlp`, so a
  neuron-space restriction must be taken at `post_act`).
- Reuse the existing prompt construction, token-position selection and the single-token check.
- **Guard:** the tiny-aya tokenizer reports `len(tokenizer) = 261_000` (not 262 144) and
  `add_bos_token=False` with no chat template for `tiny-aya-base` — the prompt builder must not assume a BOS.

**Acceptance.** On `tiny-aya-base`, the node direction for the layer-15 top node by circuit score has
`cos > 0.9` with the per-layer direction's corresponding slice, and ablating it alone drops refusal rate on
a 12-prompt probe by a nonzero, reproducible amount (single seed, reported as such).

**Cost.** ~80 lines.

---

## Patch 3 — stop destroying the sign of circuit scores (CircuitKIT)

**Problem.** `backends/eap/circuit_kit_adapter.py:77` takes `abs(score)` when converting to
`CircuitInfo`, and the artifact schema documents scores as non-negative. A repair operator must know
whether a node **produces** refusal (positive attribution) or **suppresses** it (negative) — with the sign
gone, "restore this node" and "restore its opposite" are indistinguishable.

**Change.** Keep the signed number in the payload (`scores_signed` alongside `scores`, or a flag in the
schema) without breaking existing consumers; `dot_product`/pruning keep using magnitude.

**Acceptance.** A discovered circuit on the advbench contrastive task contains at least one node with a
negative signed score, and the pruning path produces a byte-identical set to before the change.

**Cost.** ~10 lines + schema note. **This is the smallest edit with the largest conceptual consequence** —
it is what makes "restore the harm-detection node, not the refusal read-out" expressible at all.

---

## What this buys, stated precisely

After Patches 1–3 the comparison the thesis needs becomes runnable:

| arm | restriction | what it tests |
|---|---|---|
| blind | none | SafeTune's existing baselines |
| layer-restricted | layer subset | `apply_ctheta` (published, ✅ in the ledger) |
| **node-restricted** | (layer, head) / MLP | **ours** — the sharp test of "circuit-guided" |
| localised | per-node direction | ours, and the only arm that can restore detection without restoring read-out |

The claim to report is the **frontier**: ASR recovered per unit of capability lost, at equal data-free
budget, with the irreparability bound (ceiling on what any weight-space repair can restore) drawn on the
same axes.

"""Inspect retained generated IDs using effective GenerationConfig stop IDs.

Tokenizer EOS can differ from model GenerationConfig EOS. Batch padding after a
stopped row must not be counted as extra generation or as length truncation.
"""


def generation_diagnostics(ids: list[int], eos_ids: list[int], pad_id: int | None, budget: int) -> dict:
    if not eos_ids or budget < 1 or len(ids) > budget:
        raise ValueError('Invalid generation stop configuration or output length')
    if any(not isinstance(i, int) for i in ids + eos_ids):
        raise ValueError('Token IDs must be integers')
    stops = [n for n, token in enumerate(ids) if token in eos_ids]
    first = stops[0] if stops else None
    if first is not None and any(token != pad_id for token in ids[first + 1:]):
        raise ValueError('Unexpected non-padding token after generation EOS')
    return {'ended_with_generation_eos': first is not None,
            'generated_length_excluding_post_eos_padding': first + 1 if first is not None else len(ids),
            'post_eos_padding_tokens': len(ids) - first - 1 if first is not None else 0,
            'hit_budget_without_generation_eos': first is None and len(ids) == budget}


def saved_prefix_check(old_ids: list[int], new_ids: list[int], eos_ids: list[int],
                       pad_id: int | None, old_budget: int = 128) -> dict:
    """Compare old real tokens, ignoring only padding AFTER a generated EOS."""
    old = generation_diagnostics(old_ids, eos_ids, pad_id, old_budget)
    n = old['generated_length_excluding_post_eos_padding']
    prefix = new_ids[:n]
    mismatch = next((i for i, (a, b) in enumerate(zip(old_ids[:n], prefix)) if a != b), None)
    if len(prefix) != n and mismatch is None:
        mismatch = len(prefix)
    return {'saved_tokens_compared': n, 'prefix_matches_saved_generation': mismatch is None,
            'first_mismatch_token': mismatch, 'saved_ended_with_eos': old['ended_with_generation_eos']}

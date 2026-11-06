import pytest
from experiments.generation_checks import generation_diagnostics, saved_prefix_check


def test_completed_padded_answer_is_not_truncated():
    result = generation_diagnostics([11, 3, 0, 0], [3], 0, 4)
    assert result == dict(ended_with_generation_eos=True,
                          generated_length_excluding_post_eos_padding=2,
                          post_eos_padding_tokens=2, hit_budget_without_generation_eos=False)


def test_true_budget_hit_uses_generation_not_tokenizer_eos():
    result = generation_diagnostics([11, 6, 12, 13], [3], 0, 4)
    assert result['hit_budget_without_generation_eos'] is True


def test_prefix_check_ignores_only_post_eos_padding():
    assert saved_prefix_check([11, 3, 0, 0], [11, 3, 0, 0, 0], [3], 0)['prefix_matches_saved_generation']
    assert saved_prefix_check([11, 12], [11, 12, 13], [3], 0, 2)['prefix_matches_saved_generation']
    result = saved_prefix_check([11, 12], [11, 13, 14], [3], 0, 2)
    assert not result['prefix_matches_saved_generation'] and result['first_mismatch_token'] == 1
    assert not saved_prefix_check([11, 12], [11], [3], 0, 2)['prefix_matches_saved_generation']


def test_nonpadding_after_actual_eos_is_an_error():
    with pytest.raises(ValueError): generation_diagnostics([11, 3, 12], [3], 0, 4)

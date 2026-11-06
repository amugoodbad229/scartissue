import copy

import numpy as np
import pytest

from experiments.analyse_capability import analyse, paired_bootstrap, expected_worker_reference


def test_paired_bootstrap_identical_answers_has_zero_difference():
    result = paired_bootstrap([0, 1, 1], [0, 1, 1], repeats=100)
    assert result['ci95'] == [0, 0]
    assert result['unchanged_documents'] == 3
    assert not result['paired_interval_excludes_zero']


def test_paired_bootstrap_direction_and_reproducibility():
    a = paired_bootstrap([1, 1, 1], [0, 0, 0], repeats=100)
    assert a['difference'] == 1 and a['ci95'] == [1, 1]
    assert a['improved_documents'] == 3
    assert a == paired_bootstrap([1, 1, 1], [0, 0, 0], repeats=100)


@pytest.mark.parametrize('a,b', [([np.nan], [0]), ([2], [0]), ([1, 0], [0])])
def test_invalid_measurements_rejected(a, b):
    with pytest.raises(ValueError):
        paired_bootstrap(a, b, repeats=10)


def test_only_declared_generation_override_is_applied_without_mutating_reference():
    reference = {'model_revision': 'abc', 'task_config': {'generation_kwargs': {'do_sample': False}}}
    original = copy.deepcopy(reference)
    resolved = expected_worker_reference(reference)
    assert reference == original
    assert resolved['model_revision'] == 'abc'
    assert resolved['task_config']['generation_kwargs'] == {'do_sample': False, 'max_gen_toks': 256}


def test_failed_or_smoke_payload_is_not_scientific_result():
    with pytest.raises(ValueError):
        analyse({'status': 'passed', 'mode': 'smoke'})
    with pytest.raises(ValueError):
        analyse({'status': 'failed', 'mode': 'full'})

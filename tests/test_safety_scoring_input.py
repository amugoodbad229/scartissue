import copy
import pytest
from experiments.p19_score_safety import collect_rows


def source():
    row = dict(id='sr:0', prompt='p', prompt_sha256='h', role='eval_malicious')
    states = {a: dict(write_weights_sha256=a, base_hash='base') for a in ('base', 'drift', 'harm2_a4', 'global_a4')}
    workers = [dict(arm=a, status='passed', write_weights_sha256=a,
                    restored_base_write_weights_sha256='base', records=[dict(row, answer='a')]) for a in states]
    return dict(status='passed', mode='full', errors=[], workers=workers, rows=[row], expected_states=states)


def test_scoring_input_preserves_arm_and_corpus_identity():
    rows = collect_rows(source())
    assert len(rows) == 4 and len({r['id'] for r in rows}) == 4
    assert all(r['corpus_id'] == 'sr:0' for r in rows)


def test_scoring_rejects_mismatched_states_or_prompt():
    original = source()
    for field, value in (('write_weights_sha256', 'wrong'), ('restored_base_write_weights_sha256', 'wrong')):
        x = copy.deepcopy(original); x['workers'][0][field] = value
        with pytest.raises(ValueError): collect_rows(x)
    x = copy.deepcopy(original); x['workers'][0]['records'][0]['prompt'] = 'changed'
    with pytest.raises(ValueError): collect_rows(x)


def test_scoring_rejects_failed_or_partial_generation():
    x = source(); x['status'] = 'failed'
    with pytest.raises(ValueError): collect_rows(x)
    x = source(); x['workers'][0]['records'] = []
    with pytest.raises(ValueError): collect_rows(x)

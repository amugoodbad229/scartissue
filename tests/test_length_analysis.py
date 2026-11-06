import copy
import hashlib
import pytest
from experiments.analyse_length import audit_generations, audit_score_join, compare_scores
from experiments.generation_checks import generation_diagnostics, saved_prefix_check


def fixture():
    arms = ('base', 'drift', 'harm2_a4', 'global_a4')
    rows = [{'id': f'{role}:{i}', 'role': role, 'prompt': f'{role}:{i}',
             'prompt_sha256': hashlib.sha256(f'{role}:{i}'.encode()).hexdigest()}
            for role, n in [('eval_malicious', 120), ('eval_benign', 40)] for i in range(n)]
    expected = {a: {'base_hash': 'base', 'write_weights_sha256': a} for a in arms}
    old = {'status': 'passed', 'errors': [], 'mode': 'full', 'rows': rows,
           'reference': {'model_revision': 'pinned'}, 'expected_states': expected, 'workers': []}
    for arm in arms:
        records = [{**row, 'answer': 'answer', 'rendered_input': row['prompt'],
                    'rendered_input_sha256': row['prompt_sha256'], 'input_token_ids': [9],
                    'generated_token_ids_including_batch_padding': [11, 3, 0]} for row in rows]
        records[0]['generated_token_ids_including_batch_padding'] = [11] * 128
        old['workers'].append({'arm': arm, 'status': 'passed', 'errors': [], 'batch_size': 4,
                               'write_weights_sha256': arm, 'restored_base_write_weights_sha256': 'base',
                               'generation': {'do_sample': False, 'max_new_tokens': 128}, 'records': records})
    new = copy.deepcopy(old); new['mode'] = 'smoke'; new['rows'] = new['rows'][:4] + new['rows'][120:124]
    for worker in new['workers']:
        worker['records'] = worker['records'][:4] + worker['records'][120:124]
        worker['generation']['max_new_tokens'] = 512
        worker['stop_configuration'] = {'generation_eos_ids': [3], 'pad_token_id': 0, 'stop_strings': None}
        saved = {r['id']: r for r in old['workers'][0]['records']}
        worker['records'][0]['generated_token_ids_including_batch_padding'].append(3)
        for record in worker['records']:
            ids = record['generated_token_ids_including_batch_padding']
            record['stop_diagnostics'] = generation_diagnostics(ids, [3], 0, 512)
            record['saved_prefix'] = saved_prefix_check(saved[record['id']]['generated_token_ids_including_batch_padding'], ids, [3], 0)
    return old, new


def test_length_audit_recomputes_counts_not_legacy_flags():
    old, new = fixture(); result = audit_generations(old, new)
    assert result['headline_eligible'] is False
    for arm in result['arms'].values():
        counts = arm['role_counts']['eval_malicious']
        assert counts['n'] == 4 and counts['at_128']['capped_without_eos'] == 1
        assert counts['at_512']['capped_without_eos'] == 0


@pytest.mark.parametrize('kind', ['prefix', 'weight', 'order', 'diagnostics'])
def test_length_audit_rejects_changed_comparison(kind):
    old, new = fixture(); w = new['workers'][0]
    if kind == 'prefix':
        w['records'][0]['generated_token_ids_including_batch_padding'][0] = 55
    elif kind == 'weight':
        w['write_weights_sha256'] = 'different'
    elif kind == 'order':
        new['rows'].reverse()
    else:
        w['records'][0]['stop_diagnostics']['ended_with_generation_eos'] = False
    with pytest.raises(ValueError): audit_generations(old, new)


def test_scoring_join_checks_full_actual_answer():
    old, new = fixture()
    rows = [{**r, 'id': f"{w['arm']}/{r['id']}", 'corpus_id': r['id']} for w in new['workers'] for r in w['records']]
    scores = {'workers': [{'records': rows}]}
    audit_score_join(scores, new)
    rows[0]['answer'] = 'different answer'
    with pytest.raises(ValueError, match='Judged text'): audit_score_join(scores, new)


def test_cross_budget_frozen_judges_must_match():
    with pytest.raises(ValueError, match='Frozen judge'):
        compare_scores({'reference': {'revision': 'old'}}, {'reference': {'revision': 'new'}})

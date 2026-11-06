import copy
import hashlib
import pytest
from experiments.analyse_safety import binary_rate, cohen_kappa, analyse, endpoint_change
from experiments.judge_protocols import MODELS, parse_verdict


def test_constant_agreement_is_not_fabricated_kappa_one():
    assert cohen_kappa([0, 0], [0, 0])['kappa'] is None
    assert cohen_kappa([0, 1], [0, 1])['kappa'] == 1
    assert cohen_kappa([0, 1], [1, 0])['kappa'] == -1


def test_asr_increase_is_not_called_improvement():
    result = endpoint_change([1, 0], [0, 0], repeats=16)
    assert result['difference'] == .5 and result['increased_endpoint_prompts'] == 1
    assert result['lower_is_better'] is True and 'improved_documents' not in result


def test_marginal_rates_require_binary_known_labels():
    assert binary_rate([0, 1])['rate'] == .5
    with pytest.raises(ValueError): binary_rate([0, None])
    with pytest.raises(ValueError): binary_rate([])


def payload():
    rows = [dict(id=f'control/{i}', arm=None, role='control', corpus_id=None, prompt=f'control{i}', answer='a') for i in range(4)]
    for arm in ('base', 'drift', 'harm2_a4', 'global_a4'):
        for role, count in [('eval_malicious', 120), ('eval_benign', 40)]:
            rows += [dict(id=f'{arm}/{role}:{i}', arm=arm, role=role, corpus_id=f'{role}:{i}', prompt=f'{role}:{i}', answer='a') for i in range(count)]
    for row in rows:
        row['prompt_sha256'] = hashlib.sha256(row['prompt'].encode()).hexdigest()
        row['answer_sha256'] = hashlib.sha256(row['answer'].encode()).hexdigest()
    workers = []
    for family in MODELS:
        records = copy.deepcopy(rows)
        for record in records:
            record['raw_verdict'] = 'Safety: Safe\nRefusal: No' if family == 'qwen3guard' else 'No'
            record['parsed'] = parse_verdict(family, record['raw_verdict'])
        workers.append(dict(family=family, records=records, status='passed', errors=[],
                            auditkit=dict(errors=[], stats={'strict_response_harm': dict(count=len(records), mean=0.)})))
    return dict(status='passed', errors=[], calibration_status='failed', headline_eligible=False,
                workers=workers, control_count_excluded_from_benchmark=4)


def test_controls_excluded_and_failed_calibration_visible():
    out = analyse(payload(), repeats=16)
    assert out['headline_eligible'] is False and out['calibration_status'] == 'failed'
    assert len(out['controls_excluded_from_benchmark']) == 4
    for arm in out['arms'].values():
        assert arm['asr_estimates']['majority3']['n'] == 120
        assert arm['qwen_benign_refusal_estimate']['n'] == 40


def test_unknown_or_corrupted_results_are_not_safe_zeros():
    x = payload(); x['workers'][0]['records'][0]['raw_verdict'] = ''
    with pytest.raises(ValueError): analyse(x, repeats=16)
    x = payload(); x['workers'][0]['records'][0]['answer'] = 'changed'
    with pytest.raises(ValueError): analyse(x, repeats=16)

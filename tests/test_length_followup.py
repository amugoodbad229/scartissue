import json
from types import SimpleNamespace
from experiments import length_followup


def prepare(tmp_path, monkeypatch, status='passed'):
    monkeypatch.setattr(length_followup, 'ROOT', tmp_path)
    (tmp_path / 'results').mkdir(); (tmp_path / 'logs').mkdir()
    (tmp_path / 'results/full.json').write_text(json.dumps({'status': status, 'errors': []}), encoding='utf8')


def test_failed_generation_cannot_launch_judges(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch, 'failed')
    monkeypatch.setattr(length_followup.subprocess, 'run', lambda *a, **k: (_ for _ in ()).throw(AssertionError('Must not run')))
    result = length_followup.run('results/full.json')
    assert result['status'] == 'failed' and not result['steps']


def test_failed_audit_stops_before_gpu_scoring(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(length_followup.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1))
    result = length_followup.run('results/full.json')
    assert result['status'] == 'failed' and len(result['steps']) == 1
    assert 'experiments.analyse_length' in result['steps'][0]['command']


def test_successful_chain_stays_exploratory(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch)
    calls = []
    def execute(argv, **kwargs):
        calls.append(argv)
        if 'modal' in argv:
            (tmp_path / 'results/p19_safety_judges_test.json').write_text(json.dumps({'status': 'passed', 'errors': []}), encoding='utf8')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(length_followup.subprocess, 'run', execute)
    result = length_followup.run('results/full.json')
    assert result['status'] == 'passed' and len(calls) == 3
    assert result['headline_eligible'] is False and result['calibration_status'] == 'failed'

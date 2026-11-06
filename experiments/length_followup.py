"""Deterministic local follow-up for an already-launched p21 full run.

Wait on its atomic JSON (no GPU allocated here). Fail closed on generation/audit,
then invoke unchanged p19 scoring and paired length analysis. Separate named logs;
never marks judge calibration or the project task complete. No retries or retuning.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = 'results/p17_safety_full_20261102_234450.json'
ORIGINAL_SCORES = 'results/p19_safety_judges_20261102_235442.json'


def run(generation_file: str, *, timeout_s: int = 7200):
    state_path = ROOT / 'results/p21_length_followup_20261103.json'
    state = {'status': 'running', 'stage': 'waiting_for_full_generation',
             'generation_source': generation_file, 'headline_eligible': False,
             'calibration_status': 'failed', 'steps': [], 'errors': []}
    def save():
        tmp = state_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf8')
        tmp.replace(state_path)
    def command(argv: list[str], log: str):
        state['stage'] = argv[-1]; state['steps'].append({'command': argv, 'log': log}); save()
        print('Running', argv, 'log:', log, flush=True)
        with (ROOT / log).open('wb') as handle:
            code = subprocess.run(argv, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                  timeout=timeout_s).returncode
        state['steps'][-1]['returncode'] = code; save()
        if code:
            raise RuntimeError(f'Follow-up command failed ({code}); inspect {log}')
    save()
    try:
        deadline = time.monotonic() + timeout_s
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError('Full generation did not finish within the follow-up deadline')
            path = ROOT / generation_file
            if path.exists():
                payload = json.loads(path.read_text(encoding='utf8'))
                if payload['status'] != 'running':
                    if payload['status'] != 'passed' or payload['errors']:
                        raise RuntimeError('Full generation failed; no judge run launched')
                    break
            time.sleep(20)
        checks = 'results/p21_safety_length_full_checks_20261103.json'
        command([sys.executable, '-m', 'experiments.analyse_length', ORIGINAL, generation_file, checks],
                'logs/p21_length_generation_audit.log')
        before = set((ROOT / 'results').glob('p19_safety_judges_*.json'))
        command([sys.executable, '-m', 'modal', 'run', 'experiments/p19_score_safety.py',
                 '--safety-file', generation_file], 'logs/p21_length_judges.log')
        created = set((ROOT / 'results').glob('p19_safety_judges_*.json')) - before
        if len(created) != 1:
            raise RuntimeError('Expected one unambiguous new scoring result; no analysis guessed')
        scores = next(iter(created)); scoring = json.loads(scores.read_text(encoding='utf8'))
        if scoring['status'] != 'passed' or scoring['errors']:
            raise RuntimeError(f'Scoring failed; raw records preserved in {scores}')
        state['scoring_source'] = scores.relative_to(ROOT).as_posix(); save()
        analysis = 'results/p21_length_analysis_20261103.json'
        command([sys.executable, '-m', 'experiments.analyse_length', ORIGINAL, generation_file,
                 analysis, '--original-scores', ORIGINAL_SCORES, '--extended-scores', state['scoring_source']],
                'logs/p21_length_analysis.log')
        state['analysis'] = analysis; state['status'] = 'passed'; state['stage'] = 'execution_and_analysis_complete'
        print('Length follow-up completed; calibration remains failed.', flush=True)
    except Exception as error:
        state['status'] = 'failed'; state['errors'].append(repr(error))
        print('Length follow-up STOPPED:', repr(error), flush=True)
    save()
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('generation_file'); args = parser.parse_args()
    result = run(args.generation_file)
    if result['status'] != 'passed': sys.exit(1)

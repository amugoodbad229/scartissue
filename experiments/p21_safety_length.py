"""Prospectively specified 512-token sensitivity to p17's 128-token safety run.

Same saved arm states and frozen prompts, no support/dose selection. Check every
saved answer's token prefix and effective generation EOS; retain failures as JSON.
All CUDA on Modal. Smoke uses two unchanged original batches (4 harmful, 4 benign).
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import modal

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'results/p17_safety_full_20261102_234450.json'
ARMS = {'base': None, 'drift': None, 'harm2_a4': 'repair_harm2_a4.npz', 'global_a4': 'repair_global_a4.npz'}
DRIFT = 'abliteration_tiny-aya-global_layer15.npz'
BUDGET = 512
image = (modal.Image.debian_slim(python_version='3.12')
         .uv_sync(str(ROOT / 'experiments/runtime'), frozen=True, uv_version='0.12.23')
         .env({'HF_HOME': '/hf/hf', 'HF_HUB_CACHE': '/hf/hf/hub',
               'HF_XET_HIGH_PERFORMANCE': '1', 'TOKENIZERS_PARALLELISM': 'false'})
         .add_local_dir(str(ROOT / 'artifacts'), '/root/artifacts', ignore=['__pycache__', '*.pyc'])
         .add_local_file(str(ROOT / 'experiments/eval_runtime.py'), '/root/eval_runtime.py')
         .add_local_file(str(ROOT / 'experiments/generation_checks.py'), '/root/generation_checks.py'))
volume = modal.Volume.from_name('scartissue-hf')
secrets = [modal.Secret.from_name('huggingface')]
app = modal.App('scartissue-p21-safety-length')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources() -> dict:
    paths = ['experiments/p21_safety_length.py', 'experiments/generation_checks.py',
             'experiments/eval_runtime.py', 'experiments/runtime/pyproject.toml',
             'experiments/runtime/uv.lock', 'artifacts/load_patch.py',
             f'artifacts/{DRIFT}', 'artifacts/repair_harm2_a4.npz', 'artifacts/repair_global_a4.npz',
             '_research/RUN-AMENDMENTS.md', str(BASELINE.relative_to(ROOT))]
    return {p: sha(ROOT / p) for p in paths}


@app.function(image=image, gpu='A100-40GB', volumes={'/hf': volume}, secrets=secrets,
              cpu=4, memory=24576, timeout=3600)
def generate(arm: str, rows: list[dict], reference: dict, expected: dict, saved: list[dict]) -> str:
    import random
    import sys
    import traceback
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from eval_runtime import (encode_result, resolved_chat_template, restore_write_weights,
                              snapshot_write_weights, state_sha256, validate_patch)
    from generation_checks import generation_diagnostics, saved_prefix_check
    sys.path.insert(0, '/root/artifacts')
    import load_patch
    started = time.perf_counter()
    output = {'arm': arm, 'status': 'running', 'records': [], 'errors': [],
              'model_revision': reference['model_revision'], 'dtype': 'bfloat16',
              'hardware': torch.cuda.get_device_name(), 'generation': {'do_sample': False, 'max_new_tokens': BUDGET},
              'batch_size': 4, 'headline_eligible': False}
    base = model = None
    try:
        random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        tok = AutoTokenizer.from_pretrained(reference['model'], revision=reference['model_revision'], use_fast=True)
        tok.padding_side = 'left'; tok.add_bos_token = False
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token
        output['chat_template_sha256'] = hashlib.sha256(resolved_chat_template(tok).encode()).hexdigest()
        assert output['chat_template_sha256'] == reference['chat_template_sha256']
        model = AutoModelForCausalLM.from_pretrained(reference['model'], revision=reference['model_revision'],
                                                    dtype=torch.bfloat16, device_map='cuda').eval()
        eos = model.generation_config.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos)
        output['stop_configuration'] = {'generation_eos_ids': eos_ids, 'pad_token_id': tok.pad_token_id,
                                       'tokenizer_eos_token_id': tok.eos_token_id,
                                       'stop_strings': model.generation_config.stop_strings}
        assert eos_ids == [3] and model.generation_config.stop_strings is None
        base = snapshot_write_weights(model)
        assert len(base) == 72
        output['base_write_weights_sha256'] = state_sha256(base)
        assert output['base_write_weights_sha256'] == expected['base_hash']
        edited = {'drift': 0, 'repair': 0}
        sequence = [(kind, filename) for kind, filename in
                    [('drift', DRIFT if arm != 'base' else None), ('repair', ARMS[arm])] if filename]
        patches = [load_patch.load(str(Path('/root/artifacts') / filename),
                   expected_sha256=reference['patch_checks'][filename]['sha256'])
                   for _, filename in sequence]
        counts = load_patch.apply_many(patches, model, expected_revision=reference['model_revision'])
        edited.update({kind: count for (kind, _), count in zip(sequence, counts)})
        output['write_weights_sha256'] = state_sha256({n: p for n, p in model.named_parameters() if n in base})
        assert output['write_weights_sha256'] == expected['write_weights_sha256']
        assert edited == expected['edited_scalars']
        output['edited_scalars'] = edited
        old = {r['id']: r for r in saved}
        for start in range(0, len(rows), 4):
            batch = rows[start:start + 4]
            inputs = [tok.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                       tokenize=False, add_generation_prompt=True) for row in batch]
            enc = tok(inputs, padding=True, return_tensors='pt', add_special_tokens=False).to('cuda')
            with torch.inference_mode():
                generated = model.generate(**enc, **output['generation'], pad_token_id=tok.pad_token_id)
            for index, (row, rendered) in enumerate(zip(batch, inputs)):
                ids = generated[index, enc.input_ids.shape[1]:].tolist()
                input_ids = enc.input_ids[index][enc.attention_mask[index].bool()].tolist()
                record = {**row, 'rendered_input': rendered,
                          'rendered_input_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
                          'input_token_ids': input_ids, 'generated_token_ids_including_batch_padding': ids,
                          'answer': tok.decode(ids, skip_special_tokens=True),
                          'answer_with_special_tokens': tok.decode(ids, skip_special_tokens=False),
                          'stop_diagnostics': generation_diagnostics(ids, eos_ids, tok.pad_token_id, BUDGET),
                          'saved_prefix': saved_prefix_check(old[row['id']]['generated_token_ids_including_batch_padding'],
                                                            ids, eos_ids, tok.pad_token_id)}
                output['records'].append(record)
                assert input_ids == old[row['id']]['input_token_ids'], 'Input token IDs differ from saved p17'
                assert record['rendered_input_sha256'] == old[row['id']]['rendered_input_sha256']
            print(f"{arm}: {len(output['records'])}/{len(rows)} longer answers retained", flush=True)
        assert [r['id'] for r in output['records']] == [r['id'] for r in rows]
        assert all(r['answer'].strip() for r in output['records']), 'Empty completion retained as failure'
        assert all(r['saved_prefix']['prefix_matches_saved_generation'] for r in output['records']), 'Saved generation prefix mismatch; not a pure length comparison'
        output['status'] = 'passed'
    except Exception as error:
        output['status'] = 'failed'
        output['errors'].append({'error': repr(error), 'traceback': traceback.format_exc()})
    finally:
        if base is not None and model is not None:
            try:
                restore_write_weights(model, base)
                output['restored_base_write_weights_sha256'] = state_sha256({n: p for n, p in model.named_parameters() if n in base})
                assert output['restored_base_write_weights_sha256'] == expected['base_hash']
            except Exception as error:
                output['status'] = 'failed'; output['errors'].append({'stage': 'restore', 'error': repr(error)})
        output['wall_s'] = time.perf_counter() - started
    return encode_result(output)


@app.local_entrypoint()
def main(mode: str = 'smoke', smoke_file: str = ''):
    if mode not in ('smoke', 'full'): raise ValueError('Expected smoke/full')
    baseline = json.loads(BASELINE.read_text(encoding='utf8'))
    assert baseline['status'] == 'passed' and baseline['mode'] == 'full' and not baseline['errors']
    current = sources()
    for path in ['artifacts/load_patch.py', f'artifacts/{DRIFT}', 'artifacts/repair_harm2_a4.npz',
                 'artifacts/repair_global_a4.npz', 'experiments/eval_runtime.py',
                 'experiments/runtime/pyproject.toml', 'experiments/runtime/uv.lock']:
        assert current[path] == baseline['source_sha256'][path], f'Baseline input changed: {path}'
    rows = baseline['rows']
    assert len(rows) == 160
    if mode == 'smoke':
        # Original batch boundaries are preserved, including the benign batch.
        assert all(r['role'] == 'eval_malicious' for r in rows[:4])
        assert all(r['role'] == 'eval_benign' for r in rows[120:124])
        rows = rows[:4] + rows[120:124]
    else:
        smoke = json.loads((ROOT / smoke_file).read_text(encoding='utf8'))
        assert smoke['status'] == 'passed' and smoke['mode'] == 'smoke' and smoke['source_sha256'] == current
        assert len(smoke['workers']) == 4 and {w['arm'] for w in smoke['workers']} == set(ARMS)
    saved = {w['arm']: w['records'] for w in baseline['workers']}
    assert set(saved) == set(ARMS)
    path = ROOT / f"results/p21_safety_length_{mode}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out = {'status': 'running', 'mode': mode, 'source_sha256': current,
           'baseline_source': str(BASELINE.relative_to(ROOT)), 'reference': baseline['reference'],
           'expected_states': baseline['expected_states'], 'rows': rows, 'workers': [], 'errors': [],
           'headline_eligible': False, 'metric_scope': 'Generation-length sensitivity only; no ASR inferred.'}
    def save():
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8'); tmp.replace(path)
    save()
    calls = [generate.spawn(arm, rows, baseline['reference'], baseline['expected_states'][arm], saved[arm]) for arm in ARMS]
    out['function_calls'] = [c.object_id for c in calls]; save()
    for call in calls:
        try:
            worker = json.loads(call.get()); out['workers'].append(worker)
            if worker['status'] != 'passed': out['errors'].append({'arm': worker['arm'], 'errors': worker['errors']})
        except Exception as error: out['errors'].append({'error': repr(error)})
        save()
    out['status'] = 'passed' if len(out['workers']) == 4 and not out['errors'] else 'failed'
    save(); print(f"{path}: {out['status']}", flush=True)

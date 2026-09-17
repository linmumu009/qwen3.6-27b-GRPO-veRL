"""Frozen Step120/P1/P3 target measurement using the unchanged formal harness.

No training, candidate selection, retries, or promotion. --verify is CPU only.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_logistics_cpt_curve_8x import ROOT, BASE, CASES, CASE_SHA, save, evaluation_env
from audit_cpt_transfer import read_rows, verify_predictions, verify_protocols, paired

HISTORY = ROOT/'runs/cpt-controlled-storage-20260910/llin/cpt-controlled-20260910/curve8x/eval_epoch_0'
MODELS = {
    'step120_current': BASE,
    'p1': ROOT/'runs/llin-transfer-p1-run-20260916-02/training/llin-step120-p1-hf',
    'p3': ROOT/'runs/llin-transfer-p3-run-20260917-01/training/llin-step120-p3-hf',
}
TARGETS = {'SC-bench-knowledge': 196, 'LogistikaBench': 1224}
COUNTS = {'SC-bench-knowledge': 226, 'LogistikaBench': 1446}
TOKENIZER_SHA = '06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_identity(model):
    index = json.loads((model/'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    if len(index['weight_map']) != 1199 or len(shards) != 15:
        raise ValueError('unexpected exported tensor/shard count')
    result = {'path': str(model), 'metadata_sha256': {}, 'shards_stat': {}}
    for name in ('config.json', 'generation_config.json', 'tokenizer.json',
                 'tokenizer_config.json', 'model.safetensors.index.json'):
        result['metadata_sha256'][name] = sha(model/name)
    if result['metadata_sha256']['tokenizer.json'] != TOKENIZER_SHA:
        raise ValueError('tokenizer changed')
    for name in shards:
        if Path(name).name != name:
            raise ValueError('unsafe shard path')
        stat = (model/name).stat()
        result['shards_stat'][name] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    result['weight_content_hashed'] = False
    return result


def verify(out, cases_path, history):
    if sha(cases_path) != CASE_SHA:
        raise ValueError('formal cases changed')
    cases = {r['item_hash']: r for r in read_rows(cases_path)}
    if len(cases) != 1672 or dict(Counter(r['dataset'] for r in cases.values())) != COUNTS:
        raise ValueError('formal coverage mismatch')
    directories = {'step120_historical': history, **{k: out/k for k in MODELS}}
    protocols = {k: json.loads((d/'protocol.safe.json').read_text()) for k,d in directories.items()}
    verify_protocols(protocols, CASE_SHA, 1672)
    for label, model in MODELS.items():
        if protocols[label]['model'] != model.as_posix():
            raise ValueError('model identity mismatch: '+label)
    if protocols['step120_historical']['model'] != BASE.as_posix():
        raise ValueError('historical model mismatch')
    results = {}; audits = {}
    for label, directory in directories.items():
        rows = read_rows(directory/'predictions.private.jsonl')
        saved = json.loads((directory/'scores.safe.json').read_text())
        results[label] = verify_predictions(cases, rows, saved)
        audits[label] = dict(requests=len(rows), predictions_sha256=sha(directory/'predictions.private.jsonl'),
            protocol_sha256=sha(directory/'protocol.safe.json'),
            invalid=sum(not r['valid'] for r in rows), truncated=sum(r['truncated'] for r in rows),
            unstable_items=sum(r['repeat_unstable'] for r in results[label].values()),
            per_repeat={str(i): {d: sum(r['correct'] for r in rows if r['repeat']==i and r['dataset']==d)
                                for d in COUNTS} for i in range(3)})
    table = []; strata = []; overlaps = []
    for dataset in COUNTS:
        keys = sorted(k for k,c in cases.items() if c['dataset']==dataset)
        historical = sum(results['step120_historical'][k]['correct'] for k in keys)
        if historical != {'SC-bench-knowledge': 189, 'LogistikaBench': 1180}[dataset]:
            raise ValueError('historical target baseline changed')
        for label in MODELS:
            current = results[label]
            table.append(dict(dataset=dataset, model=label, target=TARGETS[dataset],
                target_met=sum(current[k]['correct'] for k in keys)>=TARGETS[dataset],
                versus_historical=paired(keys, results['step120_historical'], current, uncertainty=True),
                versus_current=paired(keys, results['step120_current'], current, uncertainty=True)))
            for category in sorted({cases[k]['category'] for k in keys}):
                subset = [k for k in keys if cases[k]['category']==category]
                strata.append(dict(dataset=dataset, category=category, model=label,
                    **paired(subset, results['step120_current'], current)))
        overlaps.append(dict(dataset=dataset, **paired(keys, results['p1'], results['p3']),
            both_correct=sum(results['p1'][k]['correct'] and results['p3'][k]['correct'] for k in keys)))
    return dict(cases_sha256=CASE_SHA, new_requests=15048, training=False, promotion=False,
        p3_original_gate_passed=False, independent_holdout=False, all_models_reported=True,
        table=table, by_category=strata, p1_to_p3=overlaps, audits=audits,
        caveat='Engineering targets previously used for selection; deterministic repeats are not independent training seeds.')


def preflight(out, code):
    from transformers import AutoTokenizer
    from run_logistics_strategy_diagnostic import messages_for
    import acl  # Confirm Ascend environment before a model is loaded.
    import torch
    import vllm
    if sha(CASES) != CASE_SHA:
        raise ValueError('formal cases changed')
    cases = read_rows(CASES)
    historical = json.loads((HISTORY/'protocol.safe.json').read_text())
    verify_protocols({'historical': historical}, CASE_SHA, len(cases))
    identities = {}; token_checks = {}
    for label, model in MODELS.items():
        identities[label] = model_identity(model)
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
        hashes = []; lengths = []
        for row in cases:
            messages, _ = messages_for(row, 'original')
            ids = tokenizer.encode(tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)
            if len(ids)+96 > 8192:
                raise ValueError('context exceeds original budget')
            hashes.append(hashlib.sha256(json.dumps(ids).encode()).hexdigest()); lengths.append(len(ids))
        if hashes != historical['prompt_hashes']:
            raise ValueError('historical token prompt mismatch: '+label)
        token_checks[label] = dict(prompts=len(hashes), historical_match=True, maximum_tokens=max(lengths))
    versions = {}
    for name in ('torch','torch-npu','vllm','vllm-ascend','transformers','tokenizers'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = 'unavailable'
    refs = {}
    for directory in ('/vllm','/vllm-ascend','/verl'):
        result = subprocess.run(['git','-C',directory,'rev-parse','HEAD'], capture_output=True,text=True)
        refs[directory] = result.stdout.strip() if result.returncode==0 else 'unavailable'
    registration = dict(registered_at_utc=datetime.now(timezone.utc).isoformat(), order=list(MODELS),
        models=identities, token_checks=token_checks, cases_sha256=CASE_SHA, new_requests=15048,
        historical_runtime_fingerprint_available=False, current_control_required=True,
        historical_protocol_sha256=sha(HISTORY/'protocol.safe.json'), targets=TARGETS,
        protocol={k:v for k,v in historical.items() if k not in ('model','prompt_hashes')},
        generation_chunk_size=128, dtype='bfloat16', gpu_memory_utilization=.8, enforce_eager=True,
        python=sys.version, platform=platform.platform(), versions=versions, source_refs=refs,
        torch_path=torch.__file__, vllm_path=vllm.__file__,
        environment={k:os.environ.get(k) for k in ('ASCEND_RT_VISIBLE_DEVICES','PYTHONPATH',
            'VLLM_WORKER_MULTIPROC_METHOD','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
        code_sha256={p.name:sha(p) for p in sorted(code.glob('*.py'))},
        training=False, promotion=False, retries=0)
    save(out/'registration.safe.json', registration)
    return registration


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--verify', action='store_true')
    p.add_argument('--cases', type=Path, default=CASES)
    p.add_argument('--history', type=Path, default=HISTORY)
    a = p.parse_args()
    if a.verify:
        result = verify(a.out, a.cases, a.history)
        save(a.out/'comparison.safe.json', result)
        print(json.dumps({'table':result['table'], 'new_requests':result['new_requests']}))
        return
    import fcntl
    out = a.out.resolve(); code = Path(__file__).resolve().parent
    if out.parent != ROOT/'runs' or not out.name.startswith('llin-transfer-formal-'):
        raise ValueError('expected new isolated formal evaluation directory')
    os.umask(0o077); out.mkdir(exist_ok=False)
    status = out/'status.safe.json'
    save(status, dict(status='preflight', training=False))
    try:
        with (ROOT/'runs/.logistics-exam-cpt.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            if shutil.disk_usage(out).free < 5_000_000_000:
                raise ValueError('insufficient evaluation output space')
            registration = preflight(out, code)
            for label, model in MODELS.items():
                if model_identity(model) != registration['models'][label]:
                    raise ValueError('model files changed after registration')
                save(status, dict(status='evaluating', model=label, training=False, total_requests=15048))
                with (out/(label+'.log')).open('x') as log:
                    subprocess.run([sys.executable, str(code/'run_logistics_cpt_curve_8x.py'),
                        '--evaluate', str(model), '--output', str(out/label)], check=True,
                        env=evaluation_env(os.environ), stdout=log, stderr=subprocess.STDOUT)
            for label, model in MODELS.items():
                if model_identity(model) != registration['models'][label]:
                    raise ValueError('model files changed during evaluation')
            result = verify(out, CASES, HISTORY)
            save(out/'comparison.safe.json', result)
            save(status, dict(status='completed_verified_remote', new_requests=15048, training=False,
                promotion=False, local_independent_review_pending=True))
    except BaseException as error:
        save(status, dict(status='failed', error=type(error).__name__, detail=str(error), training=False))
        raise


if __name__ == '__main__':
    main()

"""Run the separately registered 72 exact-training-prompt calls; no training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def save(p, data):
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(p)


def main():
    import fcntl
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    a = parser.parse_args()
    root = Path('/workspace/llin-verl-grpo')
    old = root/'runs/llin-mechanism-run-20260920-01'
    code = root/'runs/llin-mechanism-package-20260920-02/scripts'
    out = root/'runs/llin-source-recall-20260922-01'
    sys.path.insert(0, str(code))
    from cpt_stage_b import device_idle
    from run_cpt_formal_transfer import model_identity
    from run_logistics_cpt_curve_8x import evaluation_env
    os.umask(0o077)
    out.mkdir(exist_ok=True)
    reg = read(a.package/'registration.safe.json')
    probes = a.package/'probes.private.json'
    assert reg['id'] == out.name and reg['max_new_generation_calls'] == 72
    assert sha(probes) == reg['probes_sha256']
    cases = read(probes)
    assert len(cases) == 6 and len({x['id'] for x in cases}) == 6
    old_reg = read(old/'registration.safe.json')
    for name, h in old_reg['code'].items():
        assert sha(code/name) == h, 'old inference dependency changed: ' + name
    models = {}
    for label in ('step120', 'p1', 'K', 'R'):
        identity = read(old/(label+'_direct')/'identity.safe.json')
        path = Path(identity['path'])
        assert model_identity(path) == identity, label+' model identity changed'
        models[label] = path
    teaching = [json.loads(x) for x in (code.parent/'prepared/K/train.messages.private.jsonl').read_text().splitlines()]
    for c in cases:
        matches = [r for r in teaching if r['id'].startswith('K-'+c['unit_id']+'-')]
        assert len(matches) == 3 and all(r['messages'][:-1] == c['messages'] for r in matches)
    frozen = out/'registration.safe.json'
    if frozen.exists():
        assert read(frozen) == reg
    else:
        save(frozen, reg)
    with (root/'runs/.logistics-exam-cpt.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert shutil.disk_usage(out).free > 5_000_000_000
        info = subprocess.check_output(['npu-smi', 'info'], text=True)
        idle, used = device_idle(info)
        assert idle, 'devices busy; no process terminated'
        save(out/'preflight.safe.json', dict(input_verified=True, models_verified=True,
             exact_training_messages=True, old_code_verified=True, memory_mb=used,
             coordinator_sha256=sha(Path(__file__)), checked_at=time.time()))
        if a.preflight_only:
            return
        env = evaluation_env(dict(os.environ))
        status = out/'status.safe.json'
        try:
            for label, path in models.items():
                save(status, dict(state='running', stage=label+'_direct', training_steps=0))
                with (out/(label+f'.{time.time_ns()}.log')).open('x') as log:
                    subprocess.run([sys.executable, str(code/'run_cpt_mechanism_inference.py'),
                                    '--kind', 'direct', '--model', str(path), '--input', str(probes),
                                    '--out', str(out/(label+'_direct'))],
                                   env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            calls = 0
            for label in models:
                d = out/(label+'_direct')
                done = read(d/'completed.safe.json')
                assert done['calls'] == 18 and sha(d/'predictions.private.jsonl') == done['predictions_sha256']
                calls += done['calls']
            assert calls == 72
            save(status, dict(state='generation_complete_blind_scoring_pending', new_generation_calls=calls,
                             training_steps=0, formal_reruns=0, promotion=False))
        except BaseException as e:
            previous = read(status) if status.exists() else {}
            save(status, dict(state='failed_preserved', stage=previous.get('stage'), error=type(e).__name__,
                             detail=str(e), instruction='Inspect pending reservations; never auto-resubmit.'))
            raise


if __name__ == '__main__':
    main()

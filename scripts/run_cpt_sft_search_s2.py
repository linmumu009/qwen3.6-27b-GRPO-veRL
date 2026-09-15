"""Run a registered fixed-data LR arm; preserve every result, without promotion."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

# The coordinator itself also needs the immutable snapshot package root;
# setting PYTHONPATH only on subprocesses does not affect this interpreter.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_logistics_cpt_curve_8x import BASE, ROOT, checkpoint_gate, evaluation_env, paired, save

DATA = ROOT/'runs/supply-chain-task-training-data-20260909-01'
BASELINE = ROOT/'runs/cpt-controlled-storage-20260910/llin/cpt-controlled-20260910/curve8x/eval_epoch_0'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arm', choices=['S1', 'S2', 'S3', 'S4'], default='S2')
    p.add_argument('--after-training', action='store_true', help='Resume audits/evaluation from a completed export; never retrain')
    a = p.parse_args()
    out = a.out.resolve()
    arm = a.arm.lower()
    lr = {'S1':2e-7, 'S2':1e-6, 'S3':5e-7, 'S4':5e-7}[a.arm]
    profile = a.arm if a.arm in ('S3', 'S4') else 'S1S2'
    from audit_cpt_sft_search_data import expected_for
    expected_budget = expected_for(profile)
    records, sequence_tokens, _, train_sha = expected_budget['train']
    steps = records // 3
    data = (ROOT/'runs/llin-s4-data-20260915-01' if a.arm == 'S4' else
            Path('/opt/llin-s3-data-20260914-01') if a.arm == 'S3' else DATA)
    model_name = 'llin-step120-'+arm+'-hf'
    allowed_parent = ROOT/'runs' if a.arm == 'S4' else Path('/opt')
    if out.parent != allowed_parent or not out.name.startswith('llin-sft-search-'+arm+'-'):
        raise ValueError('expected independent output matching registered arm')
    code = Path(__file__).resolve().parent
    os.umask(0o077)
    if a.after_training:
        previous = json.loads((out/'status.safe.json').read_text())
        if previous.get('status') != 'failed' or not (out/'llin-training'/model_name/'model.safetensors.index.json').is_file():
            raise ValueError('recovery requires a failed coordinator and completed export')
    else:
        out.mkdir(exist_ok=False)
    status = out/('status_recovery.safe.json' if a.after_training else 'status.safe.json')
    if status.exists() and a.after_training:
        raise ValueError('recovery already attempted; inspect its process before any further action')
    save(status, dict(status='waiting_for_resource_lock', training_started=False))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(code.parent), str(ROOT),
        str(ROOT/'reference/Megatron-Bridge-de93536e/src'), str(ROOT/'runtime'), '/verl', os.environ.get('PYTHONPATH', '')]),
        LLIN_COORDINATOR_LOCKED='1', SFT_OUTPUT=str(out/'llin-training'),
        SFT_LR=str(lr), SFT_ARM=a.arm, SFT_MODEL_NAME=model_name,
        SFT_DATA=str(data), SFT_STEPS=str(steps), SFT_TRAIN_SHA=train_sha,
        SFT_DEV_SHA=expected_budget['dev'][3])
    def run(label, command, environment=env):
        save(status, dict(status=label))
        with (out/(label+'.log')).open('x') as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    def task_probe(label, model):
        run(label, [sys.executable, str(code/'run_vllm_logistics_mcq.py'), '--model', str(model),
            '--model-label', label, '--cases', str(out/'tasks.private.jsonl'),
            '--tensor-parallel-size', '8', '--max-model-len', '8192', '--max-num-seqs', '32',
            '--max-output-tokens', '96', '--gpu-memory-utilization', '0.8', '--seed', '1024', '--repeats', '1',
            '--private-output', str(out/(label+'.private.jsonl')), '--safe-output', str(out/(label+'.safe.json'))], evaluation_env(env))
    try:
        with (ROOT/'runs/.logistics-exam-cpt.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if shutil.disk_usage(out).free < 550_000_000_000:
                raise ValueError('insufficient checkpoint/export space')
            save(out/('recovery_registration.safe.json' if a.after_training else 'registration.safe.json'), dict(arm=a.arm, lr=lr, batch=3, steps=steps,
                 epochs=1, starting_model=str(BASE), fresh_optimizer=True,
                 selection_reason=a.arm+' condition-exposure/data recipe (not a single-factor comparison)' if a.arm in ('S3', 'S4') else 'Registered fixed-data LR comparison',
                 code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code.iterdir() if p.is_file()}))
            for stage, script in [('data_audit', 'audit_cpt_sft_search_data.py'), ('loader_gate', 'gate_cpt_sft_search_loader.py')]:
                run(stage+('_recovery' if a.after_training else ''), [sys.executable, str(code/script), '--data', str(data), '--model', str(BASE), '--profile', profile])
            # Reuse source-built objective tasks, never official benchmark examples.
            task_files = [data/'cases.private.jsonl'] if a.arm in ('S3', 'S4') else [data/(s+'.cases.private.jsonl') for s in ('train', 'dev')]
            rows = [[json.loads(x) for x in f.read_text().splitlines()] for f in task_files]
            if [len(r) for r in rows] != ([253] if a.arm == 'S4' else [212] if a.arm == 'S3' else [402, 74]):
                raise ValueError('unexpected objective task counts')
            if a.arm == 'S4' and hashlib.sha256(task_files[0].read_bytes()).hexdigest() != '00c81e51911a4b3ebf2e17f339b708b4ff424f06ecbea1e7e0f0e6c4ed99c8fb':
                raise ValueError('S4 source probes changed')
            if a.arm == 'S3' and hashlib.sha256(task_files[0].read_bytes()).hexdigest() != 'e1146f32783a2839be6cb2577b9439d4b7d689fe7dedf5e93d36e0a6eaab060f':
                raise ValueError('S3 source probes changed')
            task_text = ''.join(json.dumps(r)+'\n' for group in rows for r in group)
            task_hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in task_files}
            if a.after_training:
                if (out/'tasks.private.jsonl').read_text() != task_text or json.loads((out/'task_inputs.safe.json').read_text()) != task_hashes:
                    raise ValueError('task inputs changed since baseline')
            else:
                (out/'tasks.private.jsonl').write_text(task_text)
                save(out/'task_inputs.safe.json', task_hashes)
                task_probe('step120_tasks', BASE)
                run('training', ['bash', str(code/'run_cpt_sft_search_s2.sh')])
            checkpoint_gate(out/f'llin-training/checkpoints/global_step_{steps}')
            from summarize_logistics_cpt_run import parse_metrics
            metrics = parse_metrics('\n'.join(p.read_text(errors='replace') for p in
                (out/'llin-training/torchrun_logs').glob('*/attempt_0/*/stdout.log')))
            if sorted(metrics) != list(range(1, steps+1)):
                raise ValueError('missing/extra training steps')
            if sum(int(m['train/global_tokens']) for m in metrics.values()) != sequence_tokens:
                raise ValueError('actual training token budget mismatch')
            for m in metrics.values():
                if not all(math.isfinite(m[k]) for k in ('train/loss', 'train/grad_norm', 'train/lr')):
                    raise ValueError('nonfinite training metric')
            save(out/'training_audit.safe.json', dict(steps=steps, sequence_tokens=sequence_tokens, records=records))
            model = out/'llin-training'/model_name
            task_probe(arm+'_tasks', model)
            run('official_evaluation', [sys.executable, str(code/'run_logistics_cpt_curve_8x.py'),
                 '--evaluate', str(model), '--output', str(out/'evaluation')], evaluation_env(env))
            from run_cpt_knowledge_complete_pipeline import audit_evaluation
            before = audit_evaluation(BASELINE)
            after = audit_evaluation(out/'evaluation')
            protocols = [json.loads((d/'protocol.safe.json').read_text()) for d in (BASELINE, out/'evaluation')]
            for key in ('cases_sha256','cases','repeats','temperature','seed','max_tokens','max_model_len','tp','max_num_seqs','prompt','prompt_hashes'):
                if protocols[0][key] != protocols[1][key]:
                    raise ValueError('protocol mismatch: '+key)
            table = [dict(dataset=d, **paired({k:v['correct'] for k,v in before.items() if v['dataset']==d},
                          {k:v['correct'] for k,v in after.items() if v['dataset']==d}))
                     for d in ('SC-bench-knowledge', 'LogistikaBench')]
            from prepare_cpt_sft_search import assess_target
            save(out/'comparison.safe.json', dict(table=table, assessment=assess_target(table),
                 promotion=False, retention_pending=True, independent_confirmation_pending=True))
            save(status, dict(status='candidate_evaluated_review_pending', promotion=False))
    except BaseException as exc:
        save(status, dict(status='failed', error=type(exc).__name__, detail=str(exc)))
        raise


if __name__ == '__main__':
    main()

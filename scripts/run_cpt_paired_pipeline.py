"""Wait for the complete main curve, then run both frozen CPT organization arms."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_logistics_cpt_curve_8x import BASE, ROOT, checkpoint_gate, evaluation_env, paired, save
from run_cpt_paired_trainer import validate_plan

FIRST = ROOT/'runs/cpt-controlled-storage-20260910/llin/cpt-controlled-20260910'
OUT = ROOT/'runs/cpt-stage2-storage-20260910/llin/cpt-paired-20260910'


def main():
    import fcntl
    os.umask(0o077)
    code = Path(__file__).resolve().parent
    protocol = json.loads((OUT/'protocol.safe.json').read_text())
    plan_path = OUT/'paired_batches.safe.json'
    assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == protocol['batch_plan_sha256']
    plan = json.loads(plan_path.read_text())
    for arm in ('original', 'related'):
        validate_plan(plan, arm)
        assert hashlib.sha256(Path(protocol['train_files'][arm]).read_bytes()).hexdigest() == plan['dataset_sha256'][arm]
    status = OUT/'status.safe.json'
    with (OUT/'pipeline.lock').open('a') as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(status, dict(status='waiting_for_complete_main_curve'))
        try:
            deadline = time.monotonic()+12*3600
            while not (FIRST/'curve8x/curve.safe.json').exists():
                current = json.loads((FIRST/'status.safe.json').read_text())
                if current['status'] == 'failed':
                    raise RuntimeError('Main curve failed; inspect before proceeding')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Main curve did not complete within 12 hours')
                time.sleep(30)
            with (ROOT/'runs/.logistics-exam-cpt.lock').open('a') as resource_lock:
                fcntl.flock(resource_lock, fcntl.LOCK_EX)
                if shutil.disk_usage(OUT).free < 1_050_000_000_000:
                    raise RuntimeError('Insufficient storage for both complete checkpoints and exports')
                if any((OUT/arm).exists() for arm in ('original', 'related')):
                    raise FileExistsError('Refusing to overwrite or duplicate paired arms')
                def run(label, command, env):
                    save(status, dict(status=label))
                    with (OUT/(label+'.log')).open('x') as log:
                        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                for arm in ('original', 'related'):
                    env = dict(os.environ, CPT_ARM=arm, CPT_BATCH_PLAN=str(plan_path),
                        CPT_BATCH_PLAN_SHA=protocol['batch_plan_sha256'],
                        TRAIN_FILE=protocol['train_files'][arm],
                        EXPECTED_TRAIN_SHA=plan['dataset_sha256'][arm],
                        EXPECTED_CONTENT_TOKENS=str(sum(plan['lengths'][arm])),
                        OUTPUT_DIR=str(OUT/arm), RUN_NAME=f'cpt-paired-{arm}-4x-20260910')
                    run('train_'+arm, ['bash', str(code/'run_cpt_paired_arm.sh')], env)
                    checkpoint_gate(OUT/arm/'checkpoints/global_step_116')
                    run('export_'+arm, [sys.executable, str(code/'export_megatron_dist_to_hf.py'),
                        '--actor-checkpoint', str(OUT/arm/'checkpoints/global_step_116'),
                        '--base-model', str(BASE), '--output-dir', str(OUT/arm/'hf_export')], env)
                    run('evaluate_'+arm, [sys.executable, str(code/'run_logistics_cpt_curve_8x.py'),
                        '--evaluate', str(OUT/arm/'hf_export'), '--output', str(OUT/arm/'evaluation')], evaluation_env(env))
                scores = {arm: json.loads((OUT/arm/'evaluation/scores.safe.json').read_text()) for arm in ('original', 'related')}
                scores['step120'] = json.loads((FIRST/'curve8x/eval_epoch_0/scores.safe.json').read_text())
                table = []
                for before, after in (('step120','original'), ('step120','related'), ('original','related')):
                    for dataset in ('SC-bench-knowledge', 'LogistikaBench', 'all'):
                        a = {k:v['correct'] for k,v in scores[before].items() if dataset=='all' or v['dataset']==dataset}
                        b = {k:v['correct'] for k,v in scores[after].items() if dataset=='all' or v['dataset']==dataset}
                        table.append(dict(before_arm=before, after_arm=after, dataset=dataset, **paired(a,b)))
                save(OUT/'comparison.safe.json', dict(table=table, promotion=False,
                    note='General/Agent retention and curve-peak confirmation remain required'))
                save(status, dict(status='paired_training_evaluation_complete_retention_pending'))
        except BaseException as exc:
            save(status, dict(status='failed', error=type(exc).__name__, detail=str(exc)))
            raise


if __name__ == '__main__':
    main()

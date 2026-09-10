"""Run bounded post-pair restore, fresh confirmation, and general retention checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_logistics_cpt_curve_8x import BASE, ROOT, evaluation_env, save

FIRST = ROOT/'runs/cpt-controlled-storage-20260910/llin/cpt-controlled-20260910'
PAIRED = ROOT/'runs/cpt-stage2-storage-20260910/llin/cpt-paired-20260910'
OUT = ROOT/'runs/cpt-stage2-storage-20260910/llin/cpt-retention-20260910'


def check_replay(report):
    """Tolerances frozen before replay; this is not a bitwise equivalence claim."""
    m = report['metrics']
    checks = dict(step=report['replayed_step'] == 30,
        tokens=report['input_tokens'] == 9768,
        lr=abs(m['lr']-4.855502637151969e-7) <= 1e-15,
        loss=abs(m['loss']-1.8916434049606323) <= .01,
        grad_norm=abs(m['grad_norm']-2.6089337369868244)/2.6089337369868244 <= .02)
    return dict(checks=checks, passed=all(checks.values()), bitwise_equivalence=False)


def main():
    import fcntl
    os.umask(0o077)
    code = Path(__file__).resolve().parent
    status = OUT/'confirmation_status.safe.json'
    with (OUT/'confirmation.lock').open('a') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            save(status, dict(status='waiting_for_paired_evaluation'))
            deadline = time.monotonic()+12*3600
            while True:
                current = json.loads((PAIRED/'status.safe.json').read_text())['status']
                if current == 'paired_training_evaluation_complete_retention_pending':
                    break
                if current == 'failed':
                    raise RuntimeError('Paired pipeline failed; inspect before continuing')
                if time.monotonic() > deadline:
                    raise TimeoutError('Paired pipeline did not finish within 12 hours')
                time.sleep(30)
            with (ROOT/'runs/.logistics-exam-cpt.lock').open('a') as resource:
                fcntl.flock(resource, fcntl.LOCK_EX)
                def run(label, command, env):
                    save(status, dict(status=label))
                    with (OUT/(label+'.log')).open('x') as log:
                        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                probe_env = dict(os.environ, OUTPUT_DIR=str(OUT/'resume_probe'),
                    CPT_PROBE_REPORT_DIR=str(OUT/'resume_reports'),
                    RUN_NAME='cpt-step29-one-update-restore-probe')
                (OUT/'resume_reports').mkdir(exist_ok=False)
                run('resume_probe', ['bash', str(code/'run_cpt_resume_probe.sh')], probe_env)
                reports = list((OUT/'resume_reports').glob('rank_*.safe.json'))
                if len(reports) != 1:
                    raise RuntimeError('Expected one DP1 output rank from restore probe')
                result = check_replay(json.loads(reports[0].read_text()))
                save(OUT/'resume_comparison.safe.json', result)
                # A mismatch blocks promotion, but should not discard other diagnostic evidence.
                env = evaluation_env(os.environ)
                for epoch in (0, 1, 2):
                    model = BASE if epoch == 0 else FIRST/f'curve8x/hf_export_step_{epoch*29}'
                    run(f'confirm_epoch_{epoch}', [sys.executable, str(code/'run_logistics_cpt_curve_8x.py'),
                        '--evaluate', str(model), '--output', str(OUT/f'confirm_epoch_{epoch}')], env)
                models = dict(step120=BASE, curve_epoch1=FIRST/'curve8x/hf_export_step_29',
                    paired_original=PAIRED/'original/hf_export', paired_related=PAIRED/'related/hf_export')
                for label, model in models.items():
                    run('general_'+label, [sys.executable, str(code/'evaluate_supply_general_regression.py'),
                        '--model', str(model), '--cases', str(OUT/'private/general314.jsonl'),
                        '--out', str(OUT/('general_'+label))], env)
                save(status, dict(status='confirmation_general_complete_agent_pending',
                    restore_check_passed=result['passed'], promotion=False))
        except BaseException as exc:
            save(status, dict(status='failed', error=type(exc).__name__, detail=str(exc)))
            raise


if __name__ == '__main__':
    main()

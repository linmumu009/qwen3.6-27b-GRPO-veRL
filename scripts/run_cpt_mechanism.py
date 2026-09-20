"""Two-arm mechanism coordinator. Fixed recipe, no score-dependent branching."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from cpt_stage_b import device_idle
from prepare_cpt_mechanism import sha,save
from run_cpt_formal_transfer import model_identity
from run_logistics_cpt_curve_8x import BASE,ROOT,CASES,evaluation_env


def training_audit(directory,steps,expected_tokens):
    from summarize_logistics_cpt_run import parse_metrics
    checkpoint=directory/f'checkpoints/global_step_{steps}'
    manifest=json.loads((checkpoint/'ckpt_contents.json').read_text())
    assert set(manifest['save_contents'])=={'model','extra'}
    assert manifest['global_step']==steps and manifest['world_size']==16
    for key in ('model','rng_state'):
        rel=Path(manifest['contents'][key]['path']);assert not rel.is_absolute() and '..' not in rel.parts
        assert (checkpoint/rel/'.metadata').is_file() and any((checkpoint/rel).glob('*.distcp'))
    assert list(checkpoint.glob('data_*.pt'))
    metrics=parse_metrics('\n'.join(p.read_text(errors='replace') for p in (directory/'torchrun_logs').glob('*/attempt_0/*/stdout.log')))
    assert sorted(metrics)==list(range(1,steps+1))
    assert sum(int(v['train/global_tokens']) for v in metrics.values())==expected_tokens
    assert all(math.isfinite(v[k]) for v in metrics.values() for k in ('train/loss','train/grad_norm','train/lr'))
    return dict(steps=steps,sequence_tokens=expected_tokens,checkpoint_verified=True,
                first_loss=metrics[1]['train/loss'],last_loss=metrics[steps]['train/loss'],fresh_optimizer=True)


def main():
    import fcntl
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--pretraining-code-repair-sha',help='Explicit prior registration SHA; inputs/recipe must remain identical and no training directory may exist')
    a=p.parse_args()
    package=a.package.resolve();out=a.out.resolve();code=Path(__file__).resolve().parent
    assert out.parent==ROOT/'runs' and out.name.startswith('llin-mechanism-run-')
    os.umask(0o077);out.mkdir(exist_ok=True)
    reg=json.loads((package/'registration.safe.json').read_text())
    for name,digest in reg['files'].items():assert sha(package/name)==digest,name+' changed'
    for name,digest in reg['code'].items():assert sha(code/name)==digest,name+' code changed'
    prepared=package/'prepared';budget=json.loads((prepared/'budget.safe.json').read_text())
    steps=budget['steps_per_arm'];assert steps==105+budget['units'] and steps<=111
    assert reg['max_generation_calls']==10032+24*budget['units']<=10176
    registration=out/'registration.safe.json'
    if registration.exists():
        prior=json.loads(registration.read_text())
        if prior!=reg:
            assert a.pretraining_code_repair_sha==sha(registration),'unregistered code change'
            assert all(not (out/(arm+'_training')).exists() for arm in ('K','R')),'cannot revise after training'
            assert {k:v for k,v in prior.items() if k!='code'}=={k:v for k,v in reg.items() if k!='code'},'recipe/input changed'
            backup=out/'registration.before-code-repair.safe.json';assert not backup.exists()
            shutil.copyfile(registration,backup);save(registration,reg)
    else:save(registration,reg)
    env=dict(os.environ,PYTHONPATH=':'.join([str(code.parent),str(ROOT),str(ROOT/'reference/Megatron-Bridge-de93536e/src'),str(ROOT/'runtime'),'/verl',os.environ.get('PYTHONPATH','')]))
    status=out/'status.safe.json'
    def run(stage,command,environ=None):
        save(status,dict(stage=stage,state='running',promotion=False))
        with (out/(stage+f'.{time.time_ns()}.log')).open('x') as log:
            subprocess.run(command,env=environ or env,stdout=log,stderr=subprocess.STDOUT,check=True)
    def infer(label,model,kind):
        destination=out/(label+'_'+kind)
        run(label+'_'+kind,[sys.executable,str(code/'run_cpt_mechanism_inference.py'),
              '--kind',kind,'--model',str(model),'--input',str(CASES if kind=='formal' else package/'probes.private.json'),
              '--out',str(destination),'--baseline-protocol',str(ROOT/'runs/llin-transfer-formal-20260917-01/p1/protocol.safe.json')],evaluation_env(env))
    try:
        with (ROOT/'runs/.logistics-exam-cpt.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            info=subprocess.check_output(['npu-smi','info'],text=True)
            idle,used=device_idle(info);assert idle,'devices not idle; no competing process terminated'
            save(out/'device_space.safe.json',dict(npu_memory_mb=used,free_bytes=shutil.disk_usage(out).free,
                                                   mount=os.stat(out).st_dev,checked_at_unix=time.time()))
            pending=sum(not (out/(arm+'_training')).exists() for arm in ('K','R'))
            assert shutil.disk_usage(out).free>=pending*180_000_000_000+5_000_000_000
            # Rerun only CPU integrity gates on recovery, never finished training.
            run('installed_gate',[sys.executable,str(code/'gate_cpt_mechanism.py'),'--prepared',str(prepared),
                                  '--out',str(out/'gate.safe.json')],evaluation_env(env))
            if a.preflight_only:
                save(status,dict(state='preflight_passed',training_started=False,new_generation_calls=0));return
            infer('step120',BASE,'direct')
            infer('p1',ROOT/'runs/llin-transfer-p1-run-20260916-02/training/llin-step120-p1-hf','direct')
            for arm in ('K','R'):
                training=out/(arm+'_training');model=training/('llin-step120-mechanism-'+arm.lower()+'-hf')
                if not training.exists():
                    train_env=dict(env,LLIN_COORDINATOR_LOCKED='1',SFT_OUTPUT=str(training),SFT_DATA=str(prepared/arm),SFT_ARM=arm,
                        SFT_LR='5e-07',SFT_STEPS=str(steps),SFT_MODEL_NAME=model.name,
                        SFT_TRAIN_SHA=budget['files'][arm]['train.parquet'],SFT_DEV_SHA=budget['files'][arm]['dev.parquet'])
                    run(arm+'_training_export',['bash',str(code/'run_cpt_mechanism_train.sh')],train_env)
                audit=training_audit(training,steps,budget['budgets'][arm]['sequence_tokens'])
                if not model.exists():
                    run(arm+'_recover_export',[sys.executable,str(code/'export_megatron_dist_to_hf.py'),
                        '--actor-checkpoint',str(training/f'checkpoints/global_step_{steps}'),'--base-model',str(BASE),'--output-dir',str(model)])
                audit['export_identity']=model_identity(model)
                save(out/(arm+'_training_audit.safe.json'),audit)
                infer(arm,model,'direct');infer(arm,model,'formal')
            calls=sum(json.loads(f.read_text())['calls'] for f in out.glob('*/completed.safe.json'))
            assert calls==reg['max_generation_calls']
            save(status,dict(state='generation_complete_blind_scoring_pending',new_generation_calls=calls,
                             training_arms=2,promotion=False,independent_holdout=False))
    except BaseException as error:
        previous=json.loads(status.read_text()) if status.exists() else {}
        save(status,dict(state='failed_preserved',last_stage=previous.get('stage'),error=type(error).__name__,detail=str(error),
                         recovery='Completed phases remain immutable; unfinished training is never restarted automatically.'))
        raise


if __name__=='__main__':main()

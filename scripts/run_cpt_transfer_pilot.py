"""One prospective P1 arm: prepare -> baseline -> train/export -> post -> compare.

Holds the shared resource lock throughout. Never starts another arm or official
evaluation automatically. All failure/output directories are preserved.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path('/workspace/llin-verl-grpo')
BASE=ROOT/'runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return [json.loads(s) for s in path.read_text().splitlines()]
def save(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def comparison(before,after):
    from evaluate_logistics_knowledge import parse_answers
    b={r['source_id']:r for r in before};a={r['source_id']:r for r in after}
    assert len(b)==len(a)==180 and b.keys()==a.keys()
    for records in (before,after):
        for r in records:
            parsed,ok=parse_answers(r['prediction'],4)
            assert r['parsed']==list(parsed) and r['valid']==ok
            assert r['correct']==(ok and r['finish_reason']=='stop' and list(parsed)==r['expected'])
    strata=defaultdict(list);family=defaultdict(list)
    for key,old in b.items():
        new=a[key]
        assert all(old[k]==new[k] for k in ('item_hash','dataset','category','expected'))
        strata[old['dataset']].append(key);family[(old['dataset'],old['category'])].append(key)
    def stats(keys):return dict(n=len(keys),before=sum(b[k]['correct'] for k in keys),after=sum(a[k]['correct'] for k in keys),
        gains=sum(not b[k]['correct'] and a[k]['correct'] for k in keys),losses=sum(b[k]['correct'] and not a[k]['correct'] for k in keys))
    table={s:stats(keys) for s,keys in strata.items()}
    families={s:{f:stats(keys) for (ss,f),keys in family.items() if s==ss} for s in strata}
    passed={}
    for split in ('p1_dev_expression','p1_dev_conditions'):
        x=table[split];improved=sum(v['after']>v['before'] for v in families[split].values())
        passed[split]=(x['after']-x['before']>=2 and improved>=2)
    retain=table['p1_retention'];passed['retention']=retain['after']>=retain['before']
    return dict(strata=table,families=families,prospective_signal_gates=passed,
        next_action='broader_data_and_confirmation_review' if all(passed.values()) else 'hold_expansion_review_failure_strata',
        formal_benchmark_evaluated=False,promotion=False,sealed_evaluated=False,independent_training_seeds=1)


def main():
    import fcntl
    p=argparse.ArgumentParser()
    for k in ('packet','reviewed','out'):p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--reuse-prepared-run',type=Path,help='Reuse frozen data and baseline only after failure before any training started')
    a=p.parse_args();out=a.out.resolve();code=Path(__file__).resolve().parent
    assert out.parent==ROOT/'runs' and out.name.startswith('llin-transfer-p1-run-')
    os.umask(0o077);out.mkdir(exist_ok=False);status=out/'status.safe.json'
    save(status,dict(status='acquiring_lock',training_started=False))
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(code.parent),str(ROOT),str(ROOT/'reference/Megatron-Bridge-de93536e/src'),str(ROOT/'runtime'),'/verl',os.environ.get('PYTHONPATH','')]))
    def run(stage,command,environment=None):
        save(status,dict(status=stage,training_started=(out/'training/pipeline.log').exists()))
        with (out/(stage+'.log')).open('x') as log:subprocess.run(command,env=environment or env,stdout=log,stderr=subprocess.STDOUT,check=True)
    def probe(stage,model):
        ev=dict(env,ASCEND_RT_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',PYTHONPATH='/vllm:'+env['PYTHONPATH'],VLLM_WORKER_MULTIPROC_METHOD='spawn',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        run(stage,[sys.executable,str(code/'run_cpt_pilot_probe.py'),'--model',str(model),'--cases',str(a.packet/'cases.private.jsonl'),'--out',str(out/stage)],ev)
    try:
        with (ROOT/'runs/.logistics-exam-cpt.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert shutil.disk_usage(out).free>=180_000_000_000,'need checkpoint/export space on output mount'
            release=json.loads((a.packet/'release.safe.json').read_text());assert release['training_allowed'] is True
            save(out/'registration.safe.json',dict(release=release,release_sha256=sha(a.packet/'release.safe.json'),
                free_bytes=shutil.disk_usage(out).free,starting_model=str(BASE),initialization='strict HF load; no optimizer resume; installed engine supports null dist path',save_contents=['model','extra'],
                code_sha256={p.name:sha(p) for p in code.iterdir() if p.is_file()}))
            data=out/'data'
            if a.reuse_prepared_run:
                previous=a.reuse_prepared_run.resolve()
                assert previous.parent==ROOT/'runs' and previous.name.startswith('llin-transfer-p1-run-')
                prior_status=json.loads((previous/'status.safe.json').read_text())
                assert prior_status['status']=='failed' and prior_status['training_started'] is False
                assert not (previous/'training').exists(),'cannot reuse baseline after training attempt'
                prior_reg=json.loads((previous/'registration.safe.json').read_text())
                assert prior_reg['release_sha256']==sha(a.packet/'release.safe.json')
                baseline=json.loads((previous/'baseline/summary.safe.json').read_text())
                assert baseline['model']==str(BASE) and baseline['cases_sha256']==sha(a.packet/'cases.private.jsonl')
                assert baseline['items']==180
                budget=json.loads((previous/'data/data_audit.safe.json').read_text())
                assert budget['installed_loader_passed'] and budget['release_sha256']==sha(a.packet/'release.safe.json')
                for split in ('train','dev'):assert sha(previous/'data'/(split+'.parquet'))==budget['budgets'][split]['sha256']
                shutil.copytree(previous/'data',data);shutil.copytree(previous/'baseline',out/'baseline')
                save(out/'reuse.safe.json',dict(previous=str(previous),baseline_predictions_sha256=sha(out/'baseline/predictions.private.jsonl'),
                    reason='Missing historical distributed checkpoint stopped the first run before model initialization; load the identical HF model used for baseline. Data/recipe/answers unchanged.'))
            else:
                run('prepare_and_loader',[sys.executable,str(code/'prepare_cpt_transfer_pilot.py'),'--packet',str(a.packet),'--reviewed',str(a.reviewed),
                    '--model',str(BASE),'--legacy-dev','/opt/llin-s3-data-20260914-01','--out',str(data)])
            budget=json.loads((data/'data_audit.safe.json').read_text());assert budget['installed_loader_passed']
            if not a.reuse_prepared_run:probe('baseline',BASE)
            train_env=dict(env,LLIN_COORDINATOR_LOCKED='1',SFT_OUTPUT=str(out/'training'),SFT_DATA=str(data),SFT_ARM='P1',
                SFT_LR='5e-07',SFT_STEPS='105',SFT_MODEL_NAME='llin-step120-p1-hf',
                SFT_TRAIN_SHA=budget['budgets']['train']['sha256'],SFT_DEV_SHA=budget['budgets']['dev']['sha256'])
            run('training_export',['bash',str(code/'run_cpt_transfer_pilot.sh')],train_env)
            ckpt=out/'training/checkpoints/global_step_105'
            manifest=json.loads((ckpt/'ckpt_contents.json').read_text())
            assert set(manifest['save_contents'])=={'model','extra'} and manifest['global_step']==105 and manifest['world_size']==16
            for key in ('model','rng_state'):
                rel=Path(manifest['contents'][key]['path']);assert not rel.is_absolute() and '..' not in rel.parts
                assert (ckpt/rel/'.metadata').is_file() and any((ckpt/rel).glob('*.distcp'))
            assert list(ckpt.glob('data_*.pt'))
            from summarize_logistics_cpt_run import parse_metrics
            metrics=parse_metrics('\n'.join(p.read_text(errors='replace') for p in (out/'training/torchrun_logs').glob('*/attempt_0/*/stdout.log')))
            assert sorted(metrics)==list(range(1,106))
            assert sum(int(m['train/global_tokens']) for m in metrics.values())==budget['budgets']['train']['sequence_tokens']
            assert all(math.isfinite(m[k]) for m in metrics.values() for k in ('train/loss','train/grad_norm','train/lr'))
            save(out/'training_audit.safe.json',dict(steps=105,sequence_tokens=budget['budgets']['train']['sequence_tokens'],loss_first=metrics[1]['train/loss'],loss_last=metrics[105]['train/loss'],checkpoint_passed=True,optimizer_resume_available=False))
            probe('post',out/'training/llin-step120-p1-hf')
            assert json.loads((out/'baseline/prompts.safe.json').read_text())==json.loads((out/'post/prompts.safe.json').read_text())
            result=comparison(read(out/'baseline/predictions.private.jsonl'),read(out/'post/predictions.private.jsonl'))
            save(out/'comparison.safe.json',result)
            save(status,dict(status='completed_pending_review',training_started=True,formal_evaluation=False,next_action=result['next_action']))
    except BaseException as error:
        save(status,dict(status='failed',error=type(error).__name__,detail=str(error),training_started=(out/'training/pipeline.log').exists()))
        raise


if __name__=='__main__':main()

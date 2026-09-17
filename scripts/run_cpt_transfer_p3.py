"""One preregistered P3 arm using audited data and both existing baselines.

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


def check_arithmetic(packet,directory):
    from verify_cpt_pilot_result import independently_parse
    from run_vllm_logistics_mcq import load_items
    items=load_items(packet/'cases.private.jsonl');rows=read(directory/'predictions.private.jsonl')
    prompts=json.loads((directory/'prompts.safe.json').read_text());summary=json.loads((directory/'summary.safe.json').read_text())
    assert len(items)==len(rows)==len(prompts)==60 and summary['cases_sha256']==sha(packet/'cases.private.jsonl')
    scores=defaultdict(lambda:dict(n=0,correct=0))
    for item,row,prompt in zip(items,rows,prompts):
        answer,valid=independently_parse(row['prediction'],4)
        assert row['source_id']==item.source_id==prompt['source_id'] and row['item_hash']==item.item_hash
        assert row['expected']==list(item.expected) and row['parsed']==answer and row['valid']==valid
        assert row['correct']==(valid and row['finish_reason']=='stop' and answer==list(item.expected))
        assert row['prompt_tokens']==prompt['tokens'] and 0<row['output_tokens']<=96
        scores[item.category]['n']+=1;scores[item.category]['correct']+=row['correct']
    assert sum(x['correct'] for x in scores.values())==summary['correct']
    return dict(correct=summary['correct'],operations=dict(scores),predictions_sha256=sha(directory/'predictions.private.jsonl'))


def main():
    import fcntl
    p=argparse.ArgumentParser()
    for k in ('packet','arithmetic','data','out'):p.add_argument('--'+k,required=True,type=Path)
    a=p.parse_args();out=a.out.resolve();code=Path(__file__).resolve().parent
    assert out.parent==ROOT/'runs' and out.name.startswith('llin-transfer-p3-run-')
    os.umask(0o077);out.mkdir(exist_ok=False);status=out/'status.safe.json'
    save(status,dict(status='acquiring_lock',training_started=False))
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(code.parent),str(ROOT),str(ROOT/'reference/Megatron-Bridge-de93536e/src'),str(ROOT/'runtime'),'/verl',os.environ.get('PYTHONPATH','')]))
    def run(stage,command,environment=None):
        save(status,dict(status=stage,training_started=(out/'training/pipeline.log').exists()))
        with (out/(stage+'.log')).open('x') as log:subprocess.run(command,env=environment or env,stdout=log,stderr=subprocess.STDOUT,check=True)
    def probe(stage,model,arithmetic=False):
        ev=dict(env,ASCEND_RT_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',PYTHONPATH='/vllm:'+env['PYTHONPATH'],VLLM_WORKER_MULTIPROC_METHOD='spawn',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        run(stage,[sys.executable,str(code/'run_cpt_p3_probe.py'),'--model',str(model),'--cases',str((a.arithmetic if arithmetic else a.packet)/'cases.private.jsonl'),'--expected-count',str(60 if arithmetic else 180),'--out',str(out/stage)],ev)
    try:
        with (ROOT/'runs/.logistics-exam-cpt.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert shutil.disk_usage(out).free>=180_000_000_000,'need checkpoint/export space on output mount'
            release=json.loads((a.packet/'release.safe.json').read_text());assert release['training_allowed'] is True
            save(out/'registration.safe.json',dict(release=release,release_sha256=sha(a.packet/'release.safe.json'),
                free_bytes=shutil.disk_usage(out).free,starting_model=str(BASE),initialization='strict HF load; no optimizer resume; installed engine supports null dist path',save_contents=['model','extra'],
                code_sha256={p.name:sha(p) for p in code.iterdir() if p.is_file()}))
            data=a.data.resolve()
            assert json.loads((data/'independent_review.safe.json').read_text())['independent_mapping_passed'] is True
            budget=json.loads((data/'data_audit.safe.json').read_text())
            assert budget['installed_loader_passed'] and budget['target_exposures']==120 and budget['replay_exposures']==195
            assert budget['budgets']['train']['sha256']=='2ff87ec0fe42b9f1919864b74ec3af6b7d896039e0ba56e8e841e2488afa02ea'
            assert budget['budgets']['train']['sequence_tokens']==74573 and budget['budgets']['train']['loss_tokens']==8770
            for split in ('train','dev'):assert sha(data/(split+'.parquet'))==budget['budgets'][split]['sha256']
            assert sha(data/'train.messages.private.jsonl')==budget['train_messages_sha256']
            assert sha(data/'replacement_map.private.jsonl')==budget['map_sha256']
            assert sha(a.packet/'cases.private.jsonl')=='b74ed230308e4d0da500263817280bf4222d1f5fabf38c315c22f4fc26f8ff3c'
            original=ROOT/'runs/llin-transfer-p1-run-20260916-02/baseline'
            repeat=ROOT/'runs/llin-transfer-p1-stability-20260916-02/baseline_repeat'
            for name,source in [('baseline',original),('baseline_repeat',repeat)]:
                summary=json.loads((source/'summary.safe.json').read_text())
                assert summary['model']==str(BASE) and summary['cases_sha256']==sha(a.packet/'cases.private.jsonl') and summary['items']==180
                shutil.copytree(source,out/name)
            assert budget['arithmetic_exposures']==60 and budget['base_replay_exposures']==135
            from audit_cpt_p3_arithmetic import audit
            arithmetic_release=audit(a.arithmetic)
            assert arithmetic_release['manifest_sha256']=='f676239bf0504487e69f0f1f5b66728ad17948ed0b4a257b57c60f8ad51bee9f'
            save(out/'data_audit.safe.json',budget)
            probe('arithmetic_base',BASE,True)
            probe('arithmetic_p2',ROOT/'runs/llin-transfer-p2-run-20260917-01/training/llin-step120-p2-hf',True)
            arithmetic_scores={stage:check_arithmetic(a.arithmetic,out/stage) for stage in ('arithmetic_base','arithmetic_p2')}
            assert json.loads((out/'arithmetic_base/prompts.safe.json').read_text())==json.loads((out/'arithmetic_p2/prompts.safe.json').read_text())
            save(out/'arithmetic_precheck.safe.json',arithmetic_scores)
            train_env=dict(env,LLIN_COORDINATOR_LOCKED='1',SFT_OUTPUT=str(out/'training'),SFT_DATA=str(data),SFT_ARM='P3',
                SFT_LR='5e-07',SFT_STEPS='105',SFT_MODEL_NAME='llin-step120-p3-hf',
                SFT_TRAIN_SHA=budget['budgets']['train']['sha256'],SFT_DEV_SHA=budget['budgets']['dev']['sha256'])
            run('training_export',['bash',str(code/'run_cpt_transfer_p3.sh')],train_env)
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
            probe('post',out/'training/llin-step120-p3-hf')
            probe('arithmetic_post',out/'training/llin-step120-p3-hf',True)
            arithmetic_scores['arithmetic_post']=check_arithmetic(a.arithmetic,out/'arithmetic_post')
            assert json.loads((out/'arithmetic_base/prompts.safe.json').read_text())==json.loads((out/'arithmetic_post/prompts.safe.json').read_text())
            assert json.loads((out/'baseline/prompts.safe.json').read_text())==json.loads((out/'post/prompts.safe.json').read_text())
            result=comparison(read(out/'baseline/predictions.private.jsonl'),read(out/'post/predictions.private.jsonl'))
            repeat_result=comparison(read(out/'baseline_repeat/predictions.private.jsonl'),read(out/'post/predictions.private.jsonl'))
            assert json.loads((out/'baseline_repeat/prompts.safe.json').read_text())==json.loads((out/'post/prompts.safe.json').read_text())
            for view in (result,repeat_result):
                view['prospective_signal_gates']['retention']=view['strata']['p1_retention']['after']>=52
                view['next_action']='broader_data_and_confirmation_review' if all(view['prospective_signal_gates'].values()) else 'stop_content_replacement_route_review'
            floor=max(arithmetic_scores[k]['correct'] for k in ('arithmetic_base','arithmetic_p2'))
            arithmetic_passed=arithmetic_scores['arithmetic_post']['correct']>=floor
            passed=arithmetic_passed and all(all(v['prospective_signal_gates'].values()) for v in (result,repeat_result))
            result=dict(versus_main=result,versus_repeat=repeat_result,retention_floor=52,arithmetic=arithmetic_scores,arithmetic_floor=floor,arithmetic_passed=arithmetic_passed,
                next_action='broader_data_and_confirmation_review' if passed else 'stop_content_replacement_route_review',
                formal_benchmark_evaluated=False,sealed_evaluated=False,promotion=False)
            save(out/'comparison.safe.json',result)
            save(status,dict(status='completed_pending_review',training_started=True,formal_evaluation=False,next_action=result['next_action']))
    except BaseException as error:
        save(status,dict(status='failed',error=type(error).__name__,detail=str(error),training_started=(out/'training/pipeline.log').exists()))
        raise


if __name__=='__main__':main()

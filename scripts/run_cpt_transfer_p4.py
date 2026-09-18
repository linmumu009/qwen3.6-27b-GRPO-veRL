"""One fixed cumulative SFT, 180+88 diagnostics and mandatory 5016 formal calls.

No score-dependent cancellation, retries, arm search, or automatic promotion.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime,timezone
from prepare_cpt_p4_cumulative import read,sha
from run_cpt_formal_transfer import model_identity,ROOT,BASE,CASES,CASE_SHA,COUNTS,TARGETS,TOKENIZER_SHA
from audit_cpt_transfer import verify_predictions,verify_protocols,paired

FORMAL=ROOT/'runs/llin-transfer-formal-20260917-01'
P1=ROOT/'runs/llin-transfer-p1-run-20260916-02'
def obj(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(v,indent=2)+'\n');temp.replace(p)

def preflight(package,data,out):
    from transformers import AutoTokenizer
    from run_logistics_strategy_diagnostic import messages_for
    previous=obj(FORMAL/'registration.safe.json');code=Path(__file__).resolve().parent
    versions={k:importlib.metadata.version(k) for k in previous['versions']}
    refs={d:subprocess.check_output(['git','-C',d,'rev-parse','HEAD'],text=True).strip() for d in previous['source_refs']}
    assert versions==previous['versions'] and refs==previous['source_refs'],'runtime changed; controls must be re-registered before any training'
    for name in ('run_logistics_cpt_curve_8x.py','run_logistics_strategy_diagnostic.py','evaluate_logistics_knowledge.py'):
        assert sha(code/name)==previous['code_sha256'][name],name
    for label in ('step120_current','p1'):
        assert model_identity(Path(previous['models'][label]['path']))==previous['models'][label]
    assert sha(CASES)==CASE_SHA
    tok=AutoTokenizer.from_pretrained(BASE,trust_remote_code=True,local_files_only=True)
    hashes=[]
    for row in read(CASES):
        messages,_=messages_for(row,'original')
        ids=tok.encode(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
        assert len(ids)+96<=8192
        hashes.append(hashlib.sha256(json.dumps(ids).encode()).hexdigest())
    protocols={k:obj(FORMAL/k/'protocol.safe.json') for k in ('step120_current','p1')}
    verify_protocols(protocols,CASE_SHA,1672)
    assert hashes==protocols['p1']['prompt_hashes']
    release=obj(package/'packet/release.safe.json');budget=obj(data/'data_audit.safe.json')
    assert budget['release_sha256']==sha(package/'packet/release.safe.json') and release['training_allowed'] is True
    assert budget['installed_loader_passed'] and budget['original_315_messages_and_tokens_unchanged']
    assert budget['budgets']['train']['records']==363 and budget['steps']==121
    for split in ('train','dev'):assert sha(data/(split+'.parquet'))==budget['budgets'][split]['sha256']
    assert sha(data/'train.messages.private.jsonl')==budget['train_messages_sha256']
    assert sha(package/'old_cases.private.jsonl')=='b74ed230308e4d0da500263817280bf4222d1f5fabf38c315c22f4fc26f8ff3c'
    assert sha(package/'packet/cases.private.jsonl')==release['packet_sha256']
    inputs={str(p.relative_to(package)):sha(p) for p in package.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl') and 'scripts' not in p.parts}
    registration=dict(registered_at_utc=datetime.now(timezone.utc).isoformat(),starting_model=str(BASE),
        initialization='Strict HF initialization from Step120, no optimizer/checkpoint resume',data_audit=budget,
        new_requests=5284,order=['training_export','post_old180','post_new88','formal5016'],
        formal_evaluation_mandatory_after_valid_training=True,score_dependent_gating=False,
        independent_training_seeds=1,additional_training_arms=0,retries=0,promotion=False,
        reused_formal_controls=['step120_current','p1'],versions=versions,source_refs=refs,
        control_model_identities={k:previous['models'][k] for k in ('step120_current','p1')},
        control_registration_sha256=sha(FORMAL/'registration.safe.json'),formal_prompt_tokens_match=True,
        targets=TARGETS,inputs_sha256=inputs,code_sha256={p.name:sha(p) for p in code.glob('*') if p.is_file()},
        limitations=['One training run, not independent-seed confirmation.','Additional steps/tokens versus P1; not equal-budget causal attribution.','Two option orders are correlated.','Formal sets have been used for selection, not fresh holdouts.','New coverage is sixteen tasks in four capabilities, not broad coverage.'])
    save(out/'registration.safe.json',registration);return budget

def verify(out,package):
    from run_cpt_transfer_p3 import comparison
    from run_cpt_remediation_review import summarize,specs
    from verify_cpt_pilot_result import independently_parse
    from verify_cpt_remediation_result import independent_prediction
    from run_vllm_logistics_mcq import load_items
    old=read(out/'post_old180/predictions.private.jsonl');items=load_items(package/'old_cases.private.jsonl')
    prompts=obj(out/'post_old180/prompts.safe.json');assert len(old)==len(items)==len(prompts)==180
    for r,item,prompt in zip(old,items,prompts):
        parsed,ok=independently_parse(r['prediction'],4)
        assert r['source_id']==item.source_id==prompt['source_id'] and r['item_hash']==item.item_hash
        assert r['expected']==list(item.expected) and r['parsed']==parsed and r['valid']==ok
        assert r['correct']==(ok and r['finish_reason']=='stop' and parsed==list(item.expected))
        assert r['prompt_tokens']==prompt['tokens'] and 0<r['output_tokens']<=96
    old_views={}
    for label,folder in [('step120',P1/'baseline'),('p1',P1/'post')]:
        assert obj(folder/'prompts.safe.json')==prompts
        c=comparison(read(folder/'predictions.private.jsonl'),old)
        old_views[label]={k:c[k] for k in ('strata','families')}
    rows=read(package/'packet/cases.private.jsonl');raw=read(out/'post_new88/closed.private.jsonl')
    new=summarize(rows,raw,[],'p4');remote=obj(out/'post_new88/summary.safe.json')
    assert remote.pop('remote_token_reconstruction_verified') and remote==new
    plan=specs(rows)
    for s,r,d in zip(plan,raw,new['per_call']):assert (independent_prediction(r)==s['expected'])==d['correct']
    new_views={}
    for label in ('p1','step120_current'):
        before=read(package/(label+'.closed.private.jsonl'))
        summary=summarize(rows,before,[],'baseline')
        for a,b in zip(before,raw):
            for k in ('id','variant','order','prompt_text_sha256','prompt_token_sha256','prompt_tokens'):assert a[k]==b[k]
        for split in ('train','dev','retention'):
            ids=[i for i,d in enumerate(new['per_call']) if d['split']==split]
            new_views[label+'/'+split]=paired(ids,summary['per_call'],new['per_call'])
    cases={r['item_hash']:r for r in read(CASES)}
    dirs={'step120_current':FORMAL/'step120_current','p1':FORMAL/'p1','p4':out/'formal5016'}
    verify_protocols({k:obj(p/'protocol.safe.json') for k,p in dirs.items()},CASE_SHA,1672)
    assert obj(dirs['p4']/'protocol.safe.json')['model']==str(out/'training/llin-step120-p4-hf')
    results={k:verify_predictions(cases,read(p/'predictions.private.jsonl'),obj(p/'scores.safe.json')) for k,p in dirs.items()}
    table=[];categories=[]
    for dataset in COUNTS:
        keys=sorted(k for k,r in cases.items() if r['dataset']==dataset)
        assert len(keys)==COUNTS[dataset]
        table.append(dict(dataset=dataset,p4_correct=sum(results['p4'][k]['correct'] for k in keys),target=TARGETS[dataset],
            versus_step120=paired(keys,results['step120_current'],results['p4'],True),versus_p1=paired(keys,results['p1'],results['p4'],True)))
        for category in sorted({cases[k]['category'] for k in keys}):
            keys2=[k for k in keys if cases[k]['category']==category]
            categories.append(dict(dataset=dataset,category=category,versus_p1=paired(keys2,results['p1'],results['p4'])))
    model=model_identity(out/'training/llin-step120-p4-hf');assert model==obj(out/'p4.model.safe.json')
    result=dict(new_requests=5284,old_diagnostics=old_views,new_diagnostics=new,new_paired=new_views,formal=table,formal_categories=categories,
        formal_invalid=sum(v['invalid'] for v in results['p4'].values()),formal_truncated=sum(v['truncated'] for v in results['p4'].values()),
        formal_repeat_unstable=sum(v['repeat_unstable'] for v in results['p4'].values()),all_predictions_reparsed=True,
        training=obj(out/'training_audit.safe.json'),data=obj(out/'data_audit.safe.json'),promotion=False,
        raw_sha256={str(p.relative_to(out)):sha(p) for p in out.glob('*/*.private.jsonl')},next_action='Review cumulative transfer; no automatic next arm.')
    save(out/'comparison.safe.json',result);return result

def main():
    p=argparse.ArgumentParser()
    for k in ('package','data','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--verify',action='store_true');a=p.parse_args()
    if a.verify:verify(a.out,a.package);return
    import fcntl
    out=a.out.resolve();code=Path(__file__).resolve().parent
    assert out.parent==ROOT/'runs' and out.name.startswith('llin-transfer-p4-run-')
    os.umask(0o077);out.mkdir(exist_ok=False);status=out/'status.safe.json'
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(code.parent),str(code),str(ROOT),str(ROOT/'reference/Megatron-Bridge-de93536e/src'),str(ROOT/'runtime'),'/verl',os.environ.get('PYTHONPATH','')]))
    ev=dict(env,LLIN_COORDINATOR_LOCKED='1',ASCEND_RT_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',PYTHONPATH='/vllm:'+env['PYTHONPATH'],VLLM_WORKER_MULTIPROC_METHOD='spawn',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    def run(stage,cmd,environ):
        save(status,dict(status=stage,training_started=(out/'training/pipeline.log').exists(),new_requests=5284))
        with (out/(stage+'.log')).open('x') as log:subprocess.run(cmd,env=environ,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        with (ROOT/'runs/.logistics-exam-cpt.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert shutil.disk_usage(out).free>=180_000_000_000
            save(status,dict(status='preflight',training_started=False))
            budget=preflight(a.package,a.data,out);save(out/'data_audit.safe.json',budget)
            train_env=dict(env,LLIN_COORDINATOR_LOCKED='1',SFT_OUTPUT=str(out/'training'),SFT_DATA=str(a.data),SFT_ARM='P4',SFT_LR='5e-07',SFT_STEPS='121',SFT_MODEL_NAME='llin-step120-p4-hf',SFT_TRAIN_SHA=budget['budgets']['train']['sha256'],SFT_DEV_SHA=budget['budgets']['dev']['sha256'])
            run('training_export',['bash',str(code/'run_cpt_transfer_p4.sh')],train_env)
            ckpt=out/'training/checkpoints/global_step_121';manifest=obj(ckpt/'ckpt_contents.json')
            assert set(manifest['save_contents'])=={'model','extra'} and manifest['global_step']==121 and manifest['world_size']==16
            for key in ('model','rng_state'):
                rel=Path(manifest['contents'][key]['path']);assert not rel.is_absolute() and '..' not in rel.parts
                assert (ckpt/rel/'.metadata').is_file() and any((ckpt/rel).glob('*.distcp'))
            assert list(ckpt.glob('data_*.pt'))
            from summarize_logistics_cpt_run import parse_metrics
            metrics=parse_metrics('\n'.join(p.read_text(errors='replace') for p in (out/'training/torchrun_logs').glob('*/attempt_0/*/stdout.log')))
            assert sorted(metrics)==list(range(1,122))
            assert sum(int(m['train/global_tokens']) for m in metrics.values())==budget['budgets']['train']['sequence_tokens']
            assert all(math.isfinite(m[k]) for m in metrics.values() for k in ('train/loss','train/grad_norm','train/lr'))
            save(out/'training_audit.safe.json',dict(steps=121,sequence_tokens=budget['budgets']['train']['sequence_tokens'],loss_first=metrics[1]['train/loss'],loss_last=metrics[121]['train/loss'],checkpoint_passed=True,optimizer_resume_available=False))
            model=out/'training/llin-step120-p4-hf';save(out/'p4.model.safe.json',model_identity(model))
            run('post_old180',[sys.executable,str(code/'run_cpt_pilot_probe.py'),'--model',str(model),'--cases',str(a.package/'old_cases.private.jsonl'),'--out',str(out/'post_old180')],ev)
            run('post_new88',[sys.executable,str(code/'run_cpt_p4_remediation_probe.py'),'--model',str(model),'--cases',str(a.package/'packet/cases.private.jsonl'),'--baseline',str(a.package/'p1.closed.private.jsonl'),'--out',str(out/'post_new88')],ev)
            run('formal5016',[sys.executable,str(code/'run_logistics_cpt_curve_8x.py'),'--evaluate',str(model),'--output',str(out/'formal5016')],ev)
            result=verify(out,a.package)
            save(status,dict(status='completed_verified_remote',new_requests=5284,training_started=True,promotion=False,local_review_pending=True))
    except BaseException as error:
        save(status,dict(status='failed',error=type(error).__name__,detail=str(error),training_started=(out/'training/pipeline.log').exists()));raise

if __name__=='__main__':main()

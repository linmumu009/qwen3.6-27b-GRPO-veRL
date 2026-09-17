"""Fixed P1 vs archived Step120 comparison on the frozen 16-item authoring pilot.

Reuses the exact direct-index prompts, option orders and decoding configuration.
No training, source review, prompt search or official benchmark evaluation.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from run_cpt_source_authoring_pilot import specs, closed_prompt, answer, sha, LOCK
from run_cpt_transfer_screen import digest

MODEL='/workspace/llin-verl-grpo/runs/llin-transfer-p1-run-20260916-02/training/llin-step120-p1-hf'
CASES_SHA='fdb9fb716f226fd5c2b6531521ef84e990fca83fae65a29da3f295418e674b26'
BASELINE_SHA='a6dc2781563b88e519f266ac91c0cfb7f65c6feb1baabb1b32ad4b5e1cb98f46'


def summarize(rows, raw, baseline):
    plan=specs(rows);by_id={r['id']:r for r in rows}
    assert len(raw)==len(baseline)==len(plan)
    counts=Counter();details=[]
    for s,r,b in zip(plan,raw,baseline):
        row=by_id[s['id']]
        assert all(r[k]==b[k]==s[k] for k in ('id','variant','order'))
        expected_hash=hashlib.sha256(closed_prompt(row,s).encode()).hexdigest()
        assert r['prompt_text_sha256']==b['prompt_text_sha256']==expected_hash
        assert r['prompt_token_sha256']==b['prompt_token_sha256']
        assert r['prompt_tokens']==b['prompt_tokens'] and r['prompt_tokens']+96<=8192
        assert r['output_tokens']<=96 and b['output_tokens']<=96
        pred=answer(r['text']) if r['finish_reason']=='stop' else None
        old=answer(b['text']) if b['finish_reason']=='stop' else None
        correct=pred==s['expected'];previous=old==s['expected']
        for k,v in dict(p1_correct=correct,baseline_correct=previous,gains=correct and not previous,
                        losses=previous and not correct,invalid=pred is None,truncated=r['finish_reason']!='stop').items():counts[k]+=v
        counts[row['split']+'_correct']+=correct
        details.append(dict(id=s['id'],variant=s['variant'],split=row['split'],correct=correct,baseline_correct=previous,
            expected_original=row['correct_indices'],
            original_order_prediction=sorted(s['order'][i] for i in pred) if pred is not None else None,
            baseline_original_prediction=sorted(s['order'][i] for i in old) if old is not None else None))
    by_case=[]
    for i in range(0,len(details),2):
        a,b=details[i:i+2]
        by_case.append(dict(id=a['id'],split=a['split'],both_orders_correct=a['correct'] and b['correct'],
            both_orders_wrong=not a['correct'] and not b['correct'],
            baseline_both_orders_correct=a['baseline_correct'] and b['baseline_correct'],
            changed_answer=a['original_order_prediction']!=b['original_order_prediction'],
            same_wrong_answer=not a['correct'] and not b['correct'] and a['original_order_prediction'] is not None and a['original_order_prediction']==b['original_order_prediction']))
    return dict(cases=len(rows),calls=len(raw),counts=dict(counts),per_call=details,per_case=by_case,
        both_orders_correct=sum(r['both_orders_correct'] for r in by_case),
        baseline_both_orders_correct=sum(r['baseline_both_orders_correct'] for r in by_case),
        option_order_changed_answer_ids=[r['id'] for r in by_case if r['changed_answer']],
        same_wrong_answer_ids=[r['id'] for r in by_case if r['same_wrong_answer']],
        training_allowed=False,official_protocol_changed=False,
        limitations='16 source-authored development items; two orders are correlated. P1 was chosen using prior formal scores. This is not a new independent test, broad retention confirmation, or an estimate of CPT incremental benefit.')


def run(cases,baseline,model_manifest,out):
    import fcntl
    assert sha(cases)==CASES_SHA and sha(baseline)==BASELINE_SHA
    rows=[json.loads(x) for x in cases.read_text().splitlines()]
    old=[json.loads(x) for x in baseline.read_text().splitlines()]
    assert len(rows)==16 and len({r['id'] for r in rows})==16
    assert all(r['training_allowed'] is False and r['purpose']=='authoring_quality_diagnostic' for r in rows)
    manifest=json.loads(model_manifest.read_text());assert manifest['path']==MODEL
    for name,h in manifest['metadata_sha256'].items():assert sha(Path(MODEL)/name)==h
    for name,stat in manifest['shards_stat'].items():
        actual=(Path(MODEL)/name).stat();assert actual.st_size==stat['bytes'] and actual.st_mtime_ns==stat['mtime_ns']
    os.umask(0o077);out.mkdir(exist_ok=False)
    def save(name,obj):(out/name).write_text(json.dumps(obj,indent=2)+'\n')
    def status(stage,**kw):save('status.safe.json',dict(status=stage,training_allowed=False,**kw))
    status('waiting_for_lock')
    try:
        with open(LOCK,'a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            from transformers import AutoTokenizer
            from vllm import LLM,SamplingParams
            tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True)
            plan=specs(rows);by_id={r['id']:r for r in rows}
            texts=[closed_prompt(by_id[s['id']],s) for s in plan]
            def encode(text):
                rendered=tok.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                ids=tok.encode(rendered,add_special_tokens=False);assert len(ids)+96<=8192
                return dict(prompt_token_ids=ids)
            prompts=[encode(t) for t in texts]
            assert len(old)==32
            for s,t,p,b in zip(plan,texts,prompts,old):
                assert all(b[k]==s[k] for k in ('id','variant','order'))
                assert b['prompt_token_sha256']==digest(p['prompt_token_ids'])
                assert b['prompt_text_sha256']==hashlib.sha256(t.encode()).hexdigest()
            save('registration.safe.json',dict(model=MODEL,model_manifest=manifest,model_manifest_sha256=sha(model_manifest),
                cases_sha256=CASES_SHA,baseline_sha256=BASELINE_SHA,calls=32,seed=20920,temperature=0,
                max_tokens=96,thinking=False,tp=8,max_model_len=8192,max_num_seqs=16,
                code_sha256=sha(Path(__file__)),all_baseline_prompt_tokens_match=True,training_allowed=False))
            status('loading_model',all_baseline_prompt_tokens_match=True)
            llm=LLM(model=MODEL,tensor_parallel_size=8,dtype='bfloat16',trust_remote_code=True,max_model_len=8192,
                max_num_seqs=16,gpu_memory_utilization=.8,seed=20920,enforce_eager=True)
            raw=[]
            with (out/'predictions.private.jsonl').open('x',encoding='utf-8') as f:
                for start in range(0,len(plan),8):
                    ps=prompts[start:start+8]
                    outputs=sorted(llm.generate(ps,SamplingParams(temperature=0,max_tokens=96,seed=20920)),key=lambda x:int(x.request_id))
                    assert len(outputs)==len(ps)
                    for i,(pr,o) in enumerate(zip(ps,outputs),start):
                        assert list(o.prompt_token_ids)==pr['prompt_token_ids']
                        a=o.outputs[0];r=dict({k:plan[i][k] for k in ('id','variant','order')},text=a.text,
                            finish_reason=a.finish_reason,output_tokens=len(a.token_ids),prompt_tokens=len(pr['prompt_token_ids']),
                            prompt_text_sha256=hashlib.sha256(texts[i].encode()).hexdigest(),prompt_token_sha256=digest(pr['prompt_token_ids']))
                        f.write(json.dumps(r,ensure_ascii=False)+'\n');raw.append(r)
                    f.flush();status('running',completed=len(raw),total=32)
            result=summarize(rows,raw,old)
            for text,r in zip(texts,raw):assert digest(encode(text)['prompt_token_ids'])==r['prompt_token_sha256']
            result['remote_token_reconstruction_verified']=True
            save('summary.safe.json',result);status('completed')
    except Exception as exc:
        status('failed',error=type(exc).__name__+': '+str(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--model-manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.cases,a.baseline,a.model_manifest,a.out)

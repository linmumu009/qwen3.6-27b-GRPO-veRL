"""Fixed paired output-format diagnostic; original authoring-pilot scores stay frozen."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from run_cpt_source_authoring_pilot import specs, sha, answer, MODEL, LOCK
from run_cpt_transfer_screen import digest, parse_object


def prompt(row,spec):
    return ('Evaluate each option. Return only one JSON object, writing keys in this order: '
        '"option_truth" (four booleans in displayed option order), then "answers" (all zero-based indices whose boolean is true). '
        'All four options may be correct. No explanations.\nQuestion:\n'+row['question']+'\nOptions:\n'
        +'\n'.join(f'[{i}] {row["options"][j]}' for i,j in enumerate(spec['order'])))


def parse(text):
    try:
        obj=parse_object(text)
        if list(obj)!=['option_truth','answers']:return None,None
        flags=obj['option_truth']
        if not isinstance(flags,list) or len(flags)!=4 or any(type(v) is not bool for v in flags):return None,None
        indices=answer(json.dumps({'answers':obj['answers']}))
        if indices is None:return None,None
        return indices,indices==[i for i,flag in enumerate(flags) if flag]
    except (ValueError,TypeError,KeyError):return None,None


def summarize(rows,raw,baseline):
    plan=specs(rows);by_id={r['id']:r for r in rows};counts=Counter();details=[]
    assert len(raw)==len(baseline)==len(plan)==32
    for s,r,b in zip(plan,raw,baseline):
        row=by_id[s['id']]
        assert all(r[k]==b[k]==s[k] for k in ('id','variant','order'))
        assert r['prompt_text_sha256']==hashlib.sha256(prompt(row,s).encode()).hexdigest()
        assert r['output_tokens']<=96 and r['prompt_tokens']+96<=8192
        predicted,consistent=parse(r['text']) if r['finish_reason']=='stop' else (None,None)
        base_correct=b['finish_reason']=='stop' and answer(b['text'])==s['expected']
        index_correct=predicted==s['expected']
        valid=predicted is not None and consistent is True
        correct=valid and index_correct
        counts['baseline_correct']+=base_correct;counts['consistent_correct']+=correct
        counts['index_correct_regardless_consistency']+=index_correct
        counts['invalid_or_inconsistent']+=not valid
        counts['truncated']+=r['finish_reason']!='stop'
        counts[row['split']+'_correct']+=correct
        counts['gains']+=correct and not base_correct;counts['losses']+=base_correct and not correct
        details.append(dict(id=s['id'],variant=s['variant'],baseline_correct=base_correct,
            correct=correct,consistent=consistent,valid=valid,
            original_order_prediction=sorted(s['order'][i] for i in predicted) if predicted is not None else None))
    unstable=[details[i]['id'] for i in range(0,len(details),2) if details[i]['original_order_prediction']!=details[i+1]['original_order_prediction']]
    return dict(counts=dict(counts),per_case=details,option_order_changed_answer_ids=unstable,
        both_orders_correct=sum(details[i]['correct'] and details[i+1]['correct'] for i in range(0,len(details),2)),
        training_allowed=False,official_protocol_changed=False,
        limits='Post-hoc development contrast on 16 previously inspected tasks. Prompt/output structure changes, not weights. Same 96-token cap and seed; actual input/output lengths differ. No formal-performance claim or causal isolation of a cognitive mechanism.')


def run(cases,expected_sha,baseline,baseline_sha,out):
    import fcntl
    assert sha(cases)==expected_sha and sha(baseline)==baseline_sha
    rows=[json.loads(s) for s in cases.read_text().splitlines()]
    base=[json.loads(s) for s in baseline.read_text().splitlines()]
    assert len(rows)==16 and all(r['training_allowed'] is False and r['purpose']=='authoring_quality_diagnostic' for r in rows)
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
            by_id={r['id']:r for r in rows};plan=specs(rows);texts=[prompt(by_id[s['id']],s) for s in plan]
            def encode(text):
                rendered=tok.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                ids=tok.encode(rendered,add_special_tokens=False);assert len(ids)+96<=8192
                return dict(prompt_token_ids=ids)
            prompts=[encode(t) for t in texts]
            save('registration.safe.json',dict(cases_sha256=expected_sha,baseline_sha256=baseline_sha,model=MODEL,
                calls=32,max_tokens=96,seed=20920,temperature=0,thinking=False,tp=8,max_model_len=8192,
                max_num_seqs=16,code_sha256=sha(Path(__file__)),training_allowed=False,
                acceptance='Both final indices and option booleans must agree with the frozen answer; invalid/inconsistent outputs count wrong. Original baseline unchanged.'))
            status('loading_model',all_contexts_verified=True)
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
            result=summarize(rows,raw,base)
            for text,r in zip(texts,raw):assert digest(encode(text)['prompt_token_ids'])==r['prompt_token_sha256']
            result['remote_token_reconstruction_verified']=True
            save('summary.safe.json',result);status('completed')
    except Exception as exc:
        status('failed',error=type(exc).__name__+': '+str(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--sha',required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--baseline-sha',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.cases,a.sha,a.baseline,a.baseline_sha,a.out)

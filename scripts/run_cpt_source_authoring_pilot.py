"""Bounded quality diagnostic: two closed-book orders and one blind source review.

No training, generation retries, official cases, or automatic data release.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random

from cpt_transfer_draft import draft_indices
from run_cpt_transfer_screen import MODEL, LOCK, digest, parse_object


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def specs(rows):
    result=[]
    for row in rows:
        order=list(range(4))
        random.Random(int(hashlib.sha256(row['id'].encode()).hexdigest()[:16],16)).shuffle(order)
        for variant, permutation in enumerate((order,order[1:]+order[:1])):
            result.append(dict(id=row['id'],variant=variant,order=permutation,
                expected=sorted(i for i,j in enumerate(permutation) if j in row['correct_indices'])))
    return result


def closed_prompt(row,spec):
    return ('Return only a JSON object with key "answers" containing all correct zero-based option indices. All four options may be correct.\nQuestion:\n'
        + row['question'] + '\nOptions:\n' + '\n'.join(f'[{i}] {row["options"][j]}' for i,j in enumerate(spec['order'])))


def review_prompt(row):
    payload=dict(sources=[{k:s[k] for k in ('title','scope','source_text')} for s in row['sources']],
        task={k:row[k] for k in ('question','options')},authoring_scope=row['authoring_scope'],
        reasoning_requirement=row['reasoning_requirement'])
    return '''Solve each option independently against the archived sources and explicit hypothetical premises. You are not given an author answer. All four options may be correct. Check unsupported facts, missing premises, ambiguous options, historical scope, rule leakage in the question, and whether the reasoning requirement is satisfied. Numerical scenario parameters are hypothetical premises, not claims about the source. A scenario supplying work durations or cost components is not itself disclosure of the governing rule. Reject rather than invent missing rules. Return JSON only: {"correct_indices":[0],"supported":true,"unambiguous":true,"self_contained":true,"scope_preserved":true,"not_answer_leaking":true,"design_satisfied":true,"option_reasons":["...","...","...","..."],"reason":"..."}.
INPUT:
''' + json.dumps(payload,ensure_ascii=False)


def answer(text):
    try:
        value=parse_object(text)
        if set(value) != {'answers'}:return None
        return draft_indices(value['answers'],4,{'allow_all_correct':True})
    except (ValueError,KeyError,TypeError):return None


def summarize(rows,raw,review):
    by_id={r['id']:r for r in rows};plan=specs(rows)
    assert len(raw)==len(plan) and len(review)==len(rows)
    details=[];totals=Counter();reviewed=[]
    for s,r in zip(plan,raw):
        row=by_id[s['id']]
        assert all(r[k]==s[k] for k in ('id','variant','order'))
        assert r['prompt_text_sha256']==hashlib.sha256(closed_prompt(row,s).encode()).hexdigest()
        assert r['output_tokens']<=96 and r['prompt_tokens']+96<=8192
        pred=answer(r['text']) if r['finish_reason']=='stop' else None
        correct=pred==s['expected'];totals[row['split']+'_correct']+=correct
        totals['invalid']+=pred is None;totals['truncated']+=r['finish_reason']!='stop'
        details.append(dict(id=row['id'],variant=s['variant'],correct=correct,
            original_order_prediction=sorted(s['order'][i] for i in pred) if pred is not None else None))
    for row,r in zip(rows,review):
        assert r['id']==row['id'] and r['prompt_text_sha256']==hashlib.sha256(review_prompt(row).encode()).hexdigest()
        assert r['output_tokens']<=1536 and r['prompt_tokens']+1536<=8192
        accepted=False;parsed=None
        try:
            parsed=parse_object(r['text'])
            flags=('supported','unambiguous','self_contained','scope_preserved','not_answer_leaking','design_satisfied')
            accepted=(r['finish_reason']=='stop' and all(parsed.get(k) is True for k in flags)
                and draft_indices(parsed['correct_indices'],4,{'allow_all_correct':True})==row['correct_indices']
                and isinstance(parsed.get('option_reasons'),list) and len(parsed['option_reasons'])==4
                and all(isinstance(x,str) and x.strip() for x in parsed['option_reasons']))
        except (ValueError,KeyError,TypeError):pass
        reviewed.append(dict(id=row['id'],blind_review_agrees=bool(accepted)))
    stable=[details[i]['id'] for i in range(0,len(details),2) if details[i]['correct'] and details[i+1]['correct']]
    unstable=[details[i]['id'] for i in range(0,len(details),2)
        if details[i]['original_order_prediction'] != details[i+1]['original_order_prediction']]
    return dict(cases=len(rows),closed_book_calls=len(raw),review_calls=len(review),counts=dict(totals),
        both_orders_correct=len(stable),both_orders_correct_ids=stable,option_order_changed_answer_ids=unstable,
        blind_review_agrees=sum(r['blind_review_agrees'] for r in reviewed),review_decisions=reviewed,
        per_case=details,training_allowed=False,
        limits='Small source-authored development diagnostic. Same student is source reviewer and closed-book solver. No training occurred, no improvement claim, no independent transfer or broad retention certification.')


def run(cases_path,expected_sha,out):
    import fcntl
    assert sha(cases_path)==expected_sha
    rows=[json.loads(s) for s in cases_path.read_text().splitlines()]
    assert len(rows)==16 and len({r['id'] for r in rows})==16
    assert all(r['training_allowed'] is False and r['purpose']=='authoring_quality_diagnostic' for r in rows)
    os.umask(0o077);out.mkdir(exist_ok=False)
    def save(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
    def status(stage,**kw):save('status.safe.json',dict(status=stage,training_allowed=False,**kw))
    status('waiting_for_lock')
    try:
        with open(LOCK,'a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            from transformers import AutoTokenizer
            from vllm import LLM,SamplingParams
            tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True)
            plan=specs(rows);by_id={r['id']:r for r in rows}
            texts=[closed_prompt(by_id[s['id']],s) for s in plan]+[review_prompt(r) for r in rows]
            def encode(text,budget):
                rendered=tok.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                ids=tok.encode(rendered,add_special_tokens=False)
                assert len(ids)+budget<=8192
                return dict(prompt_token_ids=ids)
            prompts=[encode(t,96 if i<len(plan) else 1536) for i,t in enumerate(texts)]
            save('registration.safe.json',dict(cases_sha256=expected_sha,model=MODEL,closed_calls=32,review_calls=16,
                seed=20920,temperature=0,closed_max_tokens=96,review_max_tokens=1536,thinking=False,
                tp=8,max_model_len=8192,max_num_seqs=16,training_allowed=False,
                code_sha256=sha(Path(__file__))))
            status('loading_model',all_contexts_verified=True)
            llm=LLM(model=MODEL,tensor_parallel_size=8,dtype='bfloat16',trust_remote_code=True,max_model_len=8192,
                max_num_seqs=16,gpu_memory_utilization=.8,seed=20920,enforce_eager=True)
            all_raw=[]
            for name,start,end,budget in (('closed.private.jsonl',0,len(plan),96),('review.private.jsonl',len(plan),len(texts),1536)):
                stage=[]
                with (out/name).open('x',encoding='utf-8') as f:
                    for a in range(start,end,8):
                        ps=prompts[a:min(a+8,end)]
                        outputs=sorted(llm.generate(ps,SamplingParams(temperature=0,max_tokens=budget,seed=20920)),key=lambda x:int(x.request_id))
                        assert len(outputs)==len(ps)
                        for j,(prompt,output) in enumerate(zip(ps,outputs),a):
                            assert list(output.prompt_token_ids)==prompt['prompt_token_ids']
                            o=output.outputs[0];meta=({k:plan[j][k] for k in ('id','variant','order')} if j<len(plan) else dict(id=rows[j-len(plan)]['id']))
                            record=dict(meta,text=o.text,finish_reason=o.finish_reason,output_tokens=len(o.token_ids),
                                prompt_tokens=len(prompt['prompt_token_ids']),prompt_token_sha256=digest(prompt['prompt_token_ids']),
                                prompt_text_sha256=hashlib.sha256(texts[j].encode()).hexdigest())
                            f.write(json.dumps(record,ensure_ascii=False)+'\n');stage.append(record)
                        f.flush();status(name,completed=len(stage),total=end-start)
                all_raw.append(stage)
            result=summarize(rows,*all_raw)
            for text,prompt,record in zip(texts,prompts,all_raw[0]+all_raw[1]):
                budget=96 if 'variant' in record else 1536
                assert digest(encode(text,budget)['prompt_token_ids'])==record['prompt_token_sha256']
            result['remote_token_reconstruction_verified']=True
            save('summary.safe.json',result);status('completed_pending_operator_review')
    except Exception as exc:
        status('failed',error=type(exc).__name__+': '+str(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--sha',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    run(a.cases,a.sha,a.out)

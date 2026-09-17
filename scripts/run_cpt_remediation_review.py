"""Fixed candidate baseline and source review; no generation or training."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from run_cpt_source_authoring_pilot import specs, closed_prompt, answer, sha, LOCK
from run_cpt_transfer_screen import digest, parse_object

PACKET_SHA='9f764dd3c60e4cefac8319be2d48a80bfe6d3a6df41eb048fd919fb0d06bafde'
MODELS={
    'p1':'/workspace/llin-verl-grpo/runs/llin-transfer-p1-run-20260916-02/training/llin-step120-p1-hf',
    'step120_current':'/workspace/llin-verl-grpo/runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource'}


def review_prompt(row):
    payload=dict(sources=[{k:s[k] for k in ('title','scope','source_text')} for s in row['sources']],
        task={k:row[k] for k in ('question','options')},reasoning_requirement=row['reasoning_requirement'])
    return '''Audit against the archived sources and the explicit hypothetical premises. The author answer is hidden. Evaluate all four displayed options in order, computing exact arithmetic where needed. Only then write correct_indices, equal to the zero-based positions judged true. All four can be true. Flag unsupported facts, missing premises, ambiguity, improper historical scope or a question that gives away the governing answer. Provided durations, costs, capacities and operating constraints are hypothetical scenario premises, not purported source facts. Reject instead of inventing facts. Return JSON only, in this key order:
{"option_judgments":[{"correct":true,"reason":"..."},{"correct":false,"reason":"..."},{"correct":false,"reason":"..."},{"correct":false,"reason":"..."}],"supported":true,"unambiguous":true,"self_contained":true,"scope_preserved":true,"not_answer_leaking":true,"design_satisfied":true,"correct_indices":[0],"reason":"overall audit"}.
INPUT:
'''+json.dumps(payload,ensure_ascii=False)


def review_decision(row,raw):
    result=dict(id=row['id'],accepted=False,internally_consistent=False,review_indices=None,
        flags_pass=False,parse_valid=False,truncated=raw['finish_reason']!='stop')
    if result['truncated']:return result
    try:
        obj=parse_object(raw['text']);judgments=obj['option_judgments']
        assert isinstance(judgments,list) and len(judgments)==4
        assert all(type(v['correct']) is bool and isinstance(v['reason'],str) and v['reason'].strip() for v in judgments)
        indices=answer(json.dumps({'answers':obj['correct_indices']}));assert indices is not None
        truth_indices=[i for i,v in enumerate(judgments) if v['correct']]
        flags=('supported','unambiguous','self_contained','scope_preserved','not_answer_leaking','design_satisfied')
        result.update(parse_valid=True,review_indices=indices,judgment_indices=truth_indices,
            internally_consistent=indices==truth_indices,flags_pass=all(obj.get(k) is True for k in flags),
            false_flags=[k for k in flags if obj.get(k) is not True])
        result['accepted']=result['internally_consistent'] and result['flags_pass'] and indices==row['correct_indices']
    except (ValueError,TypeError,KeyError,AssertionError):pass
    return result


def summarize(rows,raw,reviews,label):
    by_id={r['id']:r for r in rows};plan=specs(rows);assert len(raw)==len(plan)
    counts=Counter();details=[]
    for s,r in zip(plan,raw):
        row=by_id[s['id']];assert all(r[k]==s[k] for k in ('id','variant','order'))
        assert r['prompt_text_sha256']==hashlib.sha256(closed_prompt(row,s).encode()).hexdigest()
        assert r['output_tokens']<=96 and r['prompt_tokens']+96<=8192
        pred=answer(r['text']) if r['finish_reason']=='stop' else None
        good=pred==s['expected'];counts[row['split']+'_correct']+=good
        if row['split']=='retention':counts[row['retention_stratum']+'_correct']+=good
        counts['invalid']+=pred is None;counts['truncated']+=r['finish_reason']!='stop'
        details.append(dict(id=s['id'],variant=s['variant'],split=row['split'],correct=good,
            original_order_prediction=sorted(s['order'][i] for i in pred) if pred is not None else None))
    decisions=[]
    assert len(reviews)==(len(rows) if label=='p1' else 0)
    for row,raw_review in zip(rows,reviews):
        assert raw_review['id']==row['id']
        assert raw_review['prompt_text_sha256']==hashlib.sha256(review_prompt(row).encode()).hexdigest()
        assert raw_review['output_tokens']<=1536 and raw_review['prompt_tokens']+1536<=8192
        decisions.append(review_decision(row,raw_review))
    stable=Counter()
    for a,b in zip(details[::2],details[1::2]):stable[a['split']]+=a['correct'] and b['correct']
    return dict(model_label=label,cases=len(rows),closed_calls=len(raw),review_calls=len(reviews),counts=dict(counts),
        both_orders_correct_by_split=dict(stable),
        option_order_changed_answer_ids=[a['id'] for a,b in zip(details[::2],details[1::2]) if a['original_order_prediction']!=b['original_order_prediction']],
        per_call=details,review_decisions=decisions,review_accepted=sum(r['accepted'] for r in decisions),
        training_allowed=False,limitations='Fixed source-authored development candidates, not an independent final test. Reviewer is a model and cannot release data. Eight retention items share historical training groups.')


def run(cases,model_manifest,out,label,expected_packet_sha=PACKET_SHA,expected_cases=44):
    import fcntl
    assert sha(cases)==expected_packet_sha
    rows=[json.loads(x) for x in cases.read_text().splitlines()]
    assert len(rows)==expected_cases and len({r['id'] for r in rows})==expected_cases and all(r['training_allowed'] is False for r in rows)
    model=MODELS[label];manifest=json.loads(model_manifest.read_text());assert manifest['path']==model
    for name,h in manifest['metadata_sha256'].items():assert sha(Path(model)/name)==h
    for name,stat in manifest['shards_stat'].items():
        actual=(Path(model)/name).stat();assert actual.st_size==stat['bytes'] and actual.st_mtime_ns==stat['mtime_ns']
    os.umask(0o077);out.mkdir(exist_ok=False)
    def save(name,obj):(out/name).write_text(json.dumps(obj,indent=2)+'\n')
    def status(stage,**kw):save('status.safe.json',dict(status=stage,training_allowed=False,**kw))
    status('waiting_for_lock')
    try:
        with open(LOCK,'a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            from transformers import AutoTokenizer
            from vllm import LLM,SamplingParams
            tok=AutoTokenizer.from_pretrained(model,trust_remote_code=True,local_files_only=True)
            plan=specs(rows);by_id={r['id']:r for r in rows}
            texts=[closed_prompt(by_id[s['id']],s) for s in plan]+([review_prompt(r) for r in rows] if label=='p1' else [])
            def encode(text,budget):
                rendered=tok.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                ids=tok.encode(rendered,add_special_tokens=False);assert len(ids)+budget<=8192
                return dict(prompt_token_ids=ids)
            prompts=[encode(t,96 if i<len(plan) else 1536) for i,t in enumerate(texts)]
            save('registration.safe.json',dict(model=model,model_label=label,model_manifest_sha256=sha(model_manifest),
                packet_sha256=expected_packet_sha,closed_calls=len(plan),review_calls=len(rows) if label=='p1' else 0,
                seed=20922,temperature=0,closed_max_tokens=96,review_max_tokens=1536,thinking=False,tp=8,
                max_model_len=8192,max_num_seqs=16,code_sha256=sha(Path(__file__)),training_allowed=False))
            status('loading_model',all_contexts_verified=True)
            llm=LLM(model=model,tensor_parallel_size=8,dtype='bfloat16',trust_remote_code=True,max_model_len=8192,
                max_num_seqs=16,gpu_memory_utilization=.8,seed=20922,enforce_eager=True)
            outputs_by_stage=[]
            for name,start,end,budget in [('closed.private.jsonl',0,len(plan),96),('review.private.jsonl',len(plan),len(texts),1536)]:
                raw=[]
                with (out/name).open('x',encoding='utf-8') as f:
                    for a in range(start,end,8):
                        ps=prompts[a:min(a+8,end)]
                        outputs=sorted(llm.generate(ps,SamplingParams(temperature=0,max_tokens=budget,seed=20922)),key=lambda x:int(x.request_id))
                        assert len(outputs)==len(ps)
                        for i,(pr,o) in enumerate(zip(ps,outputs),a):
                            assert list(o.prompt_token_ids)==pr['prompt_token_ids']
                            v=o.outputs[0];meta={k:plan[i][k] for k in ('id','variant','order')} if i<len(plan) else dict(id=rows[i-len(plan)]['id'])
                            record=dict(meta,text=v.text,finish_reason=v.finish_reason,output_tokens=len(v.token_ids),prompt_tokens=len(pr['prompt_token_ids']),
                                prompt_text_sha256=hashlib.sha256(texts[i].encode()).hexdigest(),prompt_token_sha256=digest(pr['prompt_token_ids']))
                            raw.append(record);f.write(json.dumps(record,ensure_ascii=False)+'\n')
                        f.flush();status(name,completed=len(raw),total=end-start)
                outputs_by_stage.append(raw)
            result=summarize(rows,*outputs_by_stage,label)
            for text,r in zip(texts,outputs_by_stage[0]+outputs_by_stage[1]):
                assert digest(encode(text,96 if 'variant' in r else 1536)['prompt_token_ids'])==r['prompt_token_sha256']
            result['remote_token_reconstruction_verified']=True
            save('summary.safe.json',result);status('completed_pending_operator_audit')
    except Exception as exc:
        status('failed',error=type(exc).__name__+': '+str(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--model-manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--model-label',choices=list(MODELS),required=True)
    a=p.parse_args();run(a.cases,a.model_manifest,a.out,a.model_label)

"""Frozen math/knowledge regression; no generated training data."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    for k in ('model','cases','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args();os.umask(0o077)
    from llin_verl.opensource_reward import compute_score
    for typ,gold,bad in [('numeric','42','41'),('math','42','41'),('choice','A','B')]:
        assert compute_score('',r'\boxed{'+gold+'}',dict(answer=gold,answer_type=typ),{})['score']==1
        assert compute_score('',r'\boxed{'+bad+'}',dict(answer=gold,answer_type=typ),{})['score']==0
    from transformers import AutoTokenizer
    assert hashlib.sha256(a.cases.read_bytes()).hexdigest()=='c040067274ef08692d3d8ef9a60f1db4041bb26d326c2c14b35f9871dde844cb'
    rows=[json.loads(x) for x in a.cases.read_text().split('\n') if x.strip()]
    assert len(rows)==314
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True);prompts=[]
    for r in rows:
        instruction='Answer the question. Show only essential reasoning and put the final answer inside \\boxed{...}.'
        if r['ground_truth']['answer_type']=='choice':instruction='Choose the single correct option. Put its letter A, B, C, or D inside \\boxed{...}.'
        rendered=tok.apply_chat_template([dict(role='user',content=instruction+'\n\n'+r['question'])],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tok.encode(rendered,add_special_tokens=False)
        assert len(ids)+2048<=4096, 'Regression prompt exceeds frozen context'
        prompts.append(dict(prompt_token_ids=ids))
    if a.preflight_only:
        print(json.dumps(dict(preflight_passed=True,cases=len(rows),max_prompt_tokens=max(len(x['prompt_token_ids']) for x in prompts),scorer_positive_negative_checks=6)));return
    from vllm import LLM,SamplingParams
    a.out.mkdir(parents=True,exist_ok=False)
    llm=LLM(model=str(a.model),tensor_parallel_size=8,dtype='bfloat16',trust_remote_code=True,max_model_len=4096,max_num_seqs=64,gpu_memory_utilization=0.8,seed=1024,enforce_eager=True)
    outputs=llm.generate(prompts,SamplingParams(temperature=0,max_tokens=2048,seed=1024))
    outputs=sorted(outputs,key=lambda o:int(o.request_id));assert len(outputs)==len(rows)
    results=[]
    for r,o in zip(rows,outputs):
        assert o.outputs
        prediction=o.outputs[0].text;truncated=o.outputs[0].finish_reason=='length'
        score=compute_score(r['dataset'],prediction,r['ground_truth'],{})
        results.append(dict(id=r['id'],dataset=r['dataset'],prediction=prediction,correct=bool(score['score']) and not truncated,truncated=truncated,explicit_final=bool(score['explicit_final'])))
    (a.out/'answers.private.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    summary=dict(model=str(a.model),cases_sha256=hashlib.sha256(a.cases.read_bytes()).hexdigest(),count=len(results),correct=sum(r['correct'] for r in results),truncated=sum(r['truncated'] for r in results),by_dataset={d:dict(count=sum(r['dataset']==d for r in results),correct=sum(r['correct'] for r in results if r['dataset']==d)) for d in Counter(r['dataset'] for r in results)},temperature=0,seed=1024,max_output_tokens=2048,thinking=False,items=[{k:r[k] for k in ('id','dataset','correct','truncated','explicit_final')} for r in results])
    (a.out/'summary.safe.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()

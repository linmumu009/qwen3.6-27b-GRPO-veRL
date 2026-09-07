"""Evaluate frozen held-out book QA and reference-answer NLL; inference only."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    for k in ('model','dev','output'): p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--label',required=True)
    p.add_argument('--concise',action='store_true')
    a=p.parse_args(); os.umask(0o077)
    raw=a.dev.read_bytes()
    if hashlib.sha256(raw).hexdigest()!='bc1e47fb35979e77a06e0150ee4a84a8ac1d2f3a686ad33ba3c3e1e23d00b97d':
        raise ValueError('Unexpected held-out data')
    rows=list(map(json.loads,raw.decode().splitlines()))
    if len(rows)!=186 or len({r['id'] for r in rows})!=186: raise ValueError('Coverage')
    a.output.mkdir(parents=True,exist_ok=False)
    from vllm import LLM,SamplingParams
    from run_vllm_prompt_nll import chosen_logprob
    llm=LLM(model=str(a.model),tensor_parallel_size=8,dtype='bfloat16',trust_remote_code=True,
        max_model_len=4096,max_num_seqs=64,gpu_memory_utilization=.8,seed=1024,enforce_eager=True)
    t=llm.get_tokenizer(); prompts=[]; references=[]; starts=[]
    budget=512 if a.concise else 256
    for r in rows:
        messages=r['messages'][:1]
        if a.concise:messages=[{'role':'system','content':'Answer the question directly in at most 100 English words. Include necessary qualifications. Do not add unrelated background.'}]+messages
        text=t.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prefix=t.encode(text,add_special_tokens=False)
        ref=prefix+t.encode(r['messages'][1]['content'],add_special_tokens=False)+[t.eos_token_id]
        if len(prefix)+budget>4096 or len(ref)>4096: raise ValueError('Length')
        prompts.append({'prompt_token_ids':prefix}); references.append({'prompt_token_ids':ref}); starts.append(len(prefix))
    outputs=sorted(llm.generate(prompts,SamplingParams(temperature=0,top_p=1,max_tokens=budget,seed=1024)),key=lambda o:int(o.request_id))
    if len(outputs)!=186: raise ValueError('Generation coverage')
    records=[{'id':r['id'],'text':o.outputs[0].text,'finish_reason':o.outputs[0].finish_reason,
              'tokens':len(o.outputs[0].token_ids)} for r,o in zip(rows,outputs)]
    (a.output/'answers.private.json').write_text(json.dumps(records))
    if a.concise:
        safe={'label':a.label,'items':186,'variant':'concise','finish_reasons':dict(Counter(r['finish_reason'] for r in records)),
              'generated_tokens':sum(r['tokens'] for r in records),'max_output_tokens':512,'training_performed':False,
              'same_short_answer_instruction':True,'reference_answer_nll_measured':False}
        (a.output/'summary.safe.json').write_text(json.dumps(safe,indent=2));print(json.dumps(safe));return
    nll_outputs=sorted(llm.generate(references,SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=1,detokenize=False)),key=lambda o:int(o.request_id))
    if len(nll_outputs)!=186: raise ValueError('NLL coverage')
    scores=[]
    for r,ref,start,out in zip(rows,references,starts,nll_outputs):
        ids=ref['prompt_token_ids']; lp=out.prompt_logprobs
        if lp is None or len(lp)!=len(ids): raise ValueError('Missing logprobs')
        values=[-chosen_logprob(lp[i],ids[i]) for i in range(start,len(ids))]
        if any(not math.isfinite(v) or v< -1e-5 for v in values): raise ValueError('Invalid NLL')
        scores.append({'id':r['id'],'tokens':len(values),'nll_sum':sum(values),'nll':sum(values)/len(values)})
    (a.output/'nll.private.json').write_text(json.dumps(scores))
    safe={'label':a.label,'model':str(a.model),'dev_sha256':hashlib.sha256(raw).hexdigest(),'items':186,
        'reference_answer_token_nll':sum(r['nll_sum'] for r in scores)/sum(r['tokens'] for r in scores),
        'scored_tokens':sum(r['tokens'] for r in scores),'finish_reasons':dict(Counter(r['finish_reason'] for r in records)),
        'generated_tokens':sum(r['tokens'] for r in records),'training_performed':False,'generation_repeats':1,
        'max_output_tokens':256,'temperature':0,'seed':1024,'evidence_in_model_prompt':False}
    (a.output/'summary.safe.json').write_text(json.dumps(safe,indent=2)); print(json.dumps(safe))


if __name__=='__main__':main()

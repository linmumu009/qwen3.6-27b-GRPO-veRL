"""Bounded, journalled inference. Completed batches survive recovery.

An interrupted reservation is never resubmitted automatically: inspect it first.
No scores influence whether either arm receives its complete formal evaluation.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from audit_cpt_transfer import CASE_SHA, read_rows, verify_predictions, verify_protocols
from run_cpt_formal_transfer import model_identity
from run_logistics_cpt_curve_8x import majority
from run_logistics_strategy_diagnostic import messages_for, parse_answers


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2)+'\n');temp.replace(p)


def main():
    p=argparse.ArgumentParser()
    for k in ('model','input','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--kind',choices=['formal','direct'],required=True)
    p.add_argument('--baseline-protocol',type=Path)
    a=p.parse_args();a.out.mkdir(exist_ok=True)
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    formal=a.kind=='formal';limit=96 if formal else 384
    if formal:
        assert sha(a.input)==CASE_SHA
        cases=read_rows(a.input);assert len(cases)==1672
    else:
        cases=json.loads(a.input.read_text());assert len(cases)%2==0 and 2<=len(cases)<=12
    prompts=[]
    for c in cases:
        messages=messages_for(c,'original')[0] if formal else c['messages']
        ids=tok.encode(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
        assert all(type(i)==int for i in ids) and len(ids)+limit<=8192
        prompts.append({'prompt_token_ids':ids})
    protocol=dict(model=str(a.model),cases_sha256=sha(a.input),cases=len(cases),repeats=3,
                  temperature=0,seed=1024,max_tokens=limit,max_model_len=8192,tp=8,max_num_seqs=32,
                  prompt='original' if formal else 'source_rule_scope',
                  prompt_hashes=[hashlib.sha256(json.dumps(x['prompt_token_ids']).encode()).hexdigest() for x in prompts])
    if formal:
        assert a.baseline_protocol is not None
        verify_protocols({'baseline':json.loads(a.baseline_protocol.read_text()),'candidate':protocol},CASE_SHA,1672)
    identity=model_identity(a.model)
    for name,value in [('protocol.safe.json',protocol),('identity.safe.json',identity)]:
        path=a.out/name
        if path.exists():assert json.loads(path.read_text())==value,'recovery identity changed'
        else:save(path,value)
    completed=a.out/'completed.safe.json'
    batches=a.out/'batches';batches.mkdir(exist_ok=True)
    jobs=[];size=128 if formal else len(cases)
    for repeat in range(3):
        for start in range(0,len(cases),size):
            key=f'{repeat}-{start:04d}';result=batches/(key+'.json');reservation=batches/(key+'.reserved.json')
            if result.exists():
                rows=json.loads(result.read_text());assert len(rows)==min(size,len(cases)-start)
                assert [r['repeat'] for r in rows]==[repeat]*len(rows)
                expected=[c['item_hash'] if formal else c['id'] for c in cases[start:start+size]]
                assert [r['item_hash'] if formal else r['id'] for r in rows]==expected
                continue
            assert not reservation.exists(),'Unresolved generation reservation; do not replay uncertain calls'
            jobs.append((repeat,start,result,reservation))
    if jobs:
        llm=LLM(model=str(a.model),tensor_parallel_size=8,dtype='bfloat16',trust_remote_code=True,max_model_len=8192,
                max_num_seqs=32,gpu_memory_utilization=.8,seed=1024,enforce_eager=True)
        for repeat,start,result,reservation in jobs:
            batch=cases[start:start+size];supplied=prompts[start:start+size]
            save(reservation,dict(calls=len(batch),repeat=repeat,start=start,input_sha256=sha(a.input)))
            outputs=llm.generate(supplied,SamplingParams(temperature=0,max_tokens=limit,seed=1024))
            outputs=sorted(outputs,key=lambda o:int(o.request_id));assert len(outputs)==len(batch)
            rows=[]
            for c,pr,o in zip(batch,supplied,outputs):
                assert list(o.prompt_token_ids)==pr['prompt_token_ids']
                answer=o.outputs[0];assert len(answer.token_ids)<=limit
                if formal:
                    parsed,ok=parse_answers(answer.text,len(c['options']));valid=ok and answer.finish_reason=='stop'
                    row=dict(item_hash=c['item_hash'],dataset=c['dataset'],repeat=repeat,parsed=list(parsed),valid=valid,
                             correct=valid and list(parsed)==sorted(c['expected']),truncated=answer.finish_reason=='length',
                             output_tokens=len(answer.token_ids),prediction=answer.text)
                else:
                    row=dict(id=c['id'],unit_id=c['unit_id'],kind=c['kind'],repeat=repeat,prediction=answer.text,
                             finish_reason=answer.finish_reason,output_tokens=len(answer.token_ids))
                rows.append(row)
            save(result,rows)
    rows=[r for f in sorted(batches.glob('*.json')) if not f.name.endswith('.reserved.json') for r in json.loads(f.read_text())]
    assert len(rows)==len(cases)*3
    (a.out/'predictions.private.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    if formal:
        groups=defaultdict(list)
        for r in rows:groups[r['item_hash']].append(r)
        scores={c['item_hash']:dict(dataset=c['dataset'],correct=majority(groups[c['item_hash']])) for c in cases}
        verify_predictions({c['item_hash']:c for c in cases},rows,scores)
        save(a.out/'scores.safe.json',scores)
    assert model_identity(a.model)==identity
    save(completed,dict(calls=len(rows),predictions_sha256=sha(a.out/'predictions.private.jsonl'),
                        protocol_sha256=sha(a.out/'protocol.safe.json'),complete=True))


if __name__=='__main__':main()

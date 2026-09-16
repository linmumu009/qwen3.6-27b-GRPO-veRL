"""Frozen P1 single-pass probe with raw stop reasons and token provenance."""
import argparse
import hashlib
import json
from pathlib import Path

from run_vllm_logistics_mcq import load_items
from evaluate_logistics_knowledge import build_messages,parse_answers


def main():
    p=argparse.ArgumentParser()
    for k in ('model','cases','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=False)
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True,local_files_only=True)
    items=load_items(a.cases);prompts=[];provenance=[]
    assert len(items)==180 and all(x.dataset!='p1_sealed' for x in items)
    for item in items:
        text=tok.apply_chat_template(build_messages(item),tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tok.encode(text,add_special_tokens=False);assert len(ids)+96<=8192
        prompts.append({'prompt_token_ids':ids})
        provenance.append(dict(source_id=item.source_id,text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            token_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),tokens=len(ids)))
    (a.out/'prompts.safe.json').write_text(json.dumps(provenance,indent=2)+'\n')
    llm=LLM(model=str(a.model),tensor_parallel_size=8,dtype='bfloat16',trust_remote_code=True,max_model_len=8192,
        max_num_seqs=32,gpu_memory_utilization=0.8,seed=1024,enforce_eager=True)
    outputs=llm.generate(prompts,SamplingParams(temperature=0,max_tokens=96,seed=1024),use_tqdm=True)
    outputs=sorted(outputs,key=lambda o:int(o.request_id));assert len(outputs)==len(items)
    rows=[]
    for item,output in zip(items,outputs):
        result=output.outputs[0];parsed,valid=parse_answers(result.text,len(item.options))
        row=dict(source_id=item.source_id,dataset=item.dataset,category=item.category,item_hash=item.item_hash,
            prediction=result.text,parsed=list(parsed),valid=valid,expected=list(item.expected),finish_reason=result.finish_reason,
            output_tokens=len(result.token_ids),prompt_tokens=len(output.prompt_token_ids),
            correct=valid and result.finish_reason=='stop' and parsed==item.expected)
        assert row['output_tokens']<=96
        rows.append(row)
    (a.out/'predictions.private.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    protocol=dict(cases_sha256=hashlib.sha256(a.cases.read_bytes()).hexdigest(),model=str(a.model),items=180,
        temperature=0,seed=1024,max_tokens=96,max_model_len=8192,tp=8,max_num_seqs=32,thinking=False,repeats=1,
        invalid=sum(not r['valid'] for r in rows),truncated=sum(r['finish_reason']!='stop' for r in rows),correct=sum(r['correct'] for r in rows),
        purpose='exploratory source-task pre/post; not formal benchmark or independent seed confirmation')
    (a.out/'summary.safe.json').write_text(json.dumps(protocol,indent=2)+'\n')


if __name__=='__main__':main()

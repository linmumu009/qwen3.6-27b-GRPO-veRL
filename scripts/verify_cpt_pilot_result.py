"""Reparse P1 raw outputs against the frozen packet; optionally rebuild tokens."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

from run_cpt_transfer_pilot import comparison
from run_vllm_logistics_mcq import load_items
from evaluate_logistics_knowledge import build_messages


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def independently_parse(text,n):
    obj=None
    for found in re.finditer(r'\{',text.strip()):
        try:
            candidate,_=json.JSONDecoder().raw_decode(text.strip()[found.start():])
        except ValueError:continue
        if isinstance(candidate,dict):obj=candidate;break
    if not isinstance(obj,dict) or list(obj)!=['answers']:return [],False
    raw=obj['answers']
    if isinstance(raw,str):raw=[s for s in re.split(r'[\s,;/|]+',raw) if s]
    if not isinstance(raw,list):return [],False
    result=[]
    for x in raw:
        if type(x) is int:result.append(x)
        elif isinstance(x,str) and x.strip().isdigit():result.append(int(x.strip()))
        else:return [],False
    values=sorted(set(result))
    return (values,True) if values and all(0<=x<n for x in values) else ([],False)


def verify(packet,run,tokenizer=None):
    manifest=json.loads((packet/'manifest.safe.json').read_text())
    assert sha(packet/'cases.private.jsonl')==manifest['files']['cases.private.jsonl']
    items=load_items(packet/'cases.private.jsonl');by_id={i.source_id:i for i in items}
    assert len(items)==180
    token_rows=None
    if tokenizer:
        from transformers import AutoTokenizer
        from audit_cpt_sft_search_data import TOKENIZER_SHA
        assert sha(tokenizer/'tokenizer.json')==TOKENIZER_SHA
        tok=AutoTokenizer.from_pretrained(tokenizer,trust_remote_code=True,local_files_only=True)
        token_rows=[]
        for item in items:
            text=tok.apply_chat_template(build_messages(item),tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tok.encode(text,add_special_tokens=False)
            token_rows.append(dict(source_id=item.source_id,text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                token_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),tokens=len(ids)))
    result={};predictions={}
    for stage in ('baseline','post'):
        base=run/stage
        if not base.exists():continue
        rows=read(base/'predictions.private.jsonl');predictions[stage]=rows
        summary=json.loads((base/'summary.safe.json').read_text())
        prompts=json.loads((base/'prompts.safe.json').read_text())
        assert len(rows)==180 and len({r['source_id'] for r in rows})==180
        assert {r['source_id'] for r in rows}==set(by_id)
        if token_rows is not None:assert prompts==token_rows
        by_prompt={r['source_id']:r for r in prompts};assert set(by_prompt)==set(by_id)
        for row in rows:
            item=by_id[row['source_id']];answer,valid=independently_parse(row['prediction'],len(item.options))
            assert row['item_hash']==item.item_hash and row['expected']==list(item.expected)
            assert row['dataset']==item.dataset and row['category']==item.category
            assert row['parsed']==answer and row['valid']==valid
            assert row['correct']==(valid and row['finish_reason']=='stop' and answer==list(item.expected))
            assert 0<row['output_tokens']<=96 and row['prompt_tokens']==by_prompt[row['source_id']]['tokens']
        assert summary['cases_sha256']==sha(packet/'cases.private.jsonl')
        for k,v in dict(temperature=0,seed=1024,max_tokens=96,max_model_len=8192,tp=8,max_num_seqs=32,thinking=False,repeats=1,items=180).items():assert summary[k]==v
        assert summary['correct']==sum(r['correct'] for r in rows)
        assert summary['invalid']==sum(not r['valid'] for r in rows)
        assert summary['truncated']==sum(r['finish_reason']!='stop' for r in rows)
        result[stage]=dict(correct=summary['correct'],invalid=summary['invalid'],truncated=summary['truncated'],
            strata={s:dict(n=sum(r['dataset']==s for r in rows),correct=sum(r['correct'] for r in rows if r['dataset']==s)) for s in sorted({r['dataset'] for r in rows})},
            predictions_sha256=sha(base/'predictions.private.jsonl'),prompts_sha256=sha(base/'prompts.safe.json'))
    assert 'baseline' in result
    if 'post' in result:
        assert json.loads((run/'baseline/prompts.safe.json').read_text())==json.loads((run/'post/prompts.safe.json').read_text())
        result['comparison']=comparison(predictions['baseline'],predictions['post'])
        assert result['comparison']==json.loads((run/'comparison.safe.json').read_text())
        budget=json.loads((run/'data/data_audit.safe.json').read_text())
        training=json.loads((run/'training_audit.safe.json').read_text())
        assert training['steps']==105 and training['sequence_tokens']==budget['budgets']['train']['sequence_tokens']==87340
        result['training_audit']=training
    result.update(packet_sha256=sha(packet/'tasks.private.jsonl'),cases_sha256=sha(packet/'cases.private.jsonl'),
        raw_outputs_verified=180*len(predictions),token_prompts_independently_rebuilt=180*len(predictions) if tokenizer else 0,
        verifier_sha256=sha(Path(__file__)),formal_benchmark_evaluated=False)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('packet','run','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--tokenizer',type=Path);a=p.parse_args();result=verify(a.packet,a.run,a.tokenizer)
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))

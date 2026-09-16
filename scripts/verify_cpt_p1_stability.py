"""Verify bounded repeat inference without replacing original P1 measurements."""
import argparse
import hashlib
import json
from pathlib import Path
from verify_cpt_pilot_result import independently_parse, read, sha
from run_vllm_logistics_mcq import load_items
from evaluate_logistics_knowledge import build_messages
from run_cpt_transfer_pilot import comparison
from run_cpt_p1_stability import CASE_SHA, MODELS


def verify(cases, original, repeat, tokenizer=None):
    assert sha(cases)==CASE_SHA
    items=load_items(cases);assert len(items)==180
    tokens=None
    if tokenizer:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(tokenizer,trust_remote_code=True,local_files_only=True)
        tokens=[]
        for item in items:
            text=tok.apply_chat_template(build_messages(item),tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tok.encode(text,add_special_tokens=False)
            tokens.append(dict(source_id=item.source_id,text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                token_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),tokens=len(ids)))
    result={};repeated={};old={}
    for stage in ('baseline','post'):
        directory=repeat/(stage+'_repeat');rows=read(directory/'predictions.private.jsonl')
        previous=read(original/stage/'predictions.private.jsonl')
        summary=json.loads((directory/'summary.safe.json').read_text())
        prompts=json.loads((directory/'prompts.safe.json').read_text())
        assert summary['model']==MODELS[stage+'_repeat'].as_posix() and summary['cases_sha256']==CASE_SHA
        for k,v in dict(temperature=0,seed=1024,max_tokens=96,max_model_len=8192,tp=8,max_num_seqs=32,thinking=False,repeats=1,items=180).items():assert summary[k]==v
        assert len(rows)==len(previous)==len(prompts)==180
        assert [r['source_id'] for r in rows]==[i.source_id for i in items]==[r['source_id'] for r in previous]
        assert prompts==json.loads((original/stage/'prompts.safe.json').read_text())
        if tokens is not None:assert prompts==tokens
        for row,item,prompt in zip(rows,items,prompts):
            parsed,valid=independently_parse(row['prediction'],len(item.options))
            assert row['item_hash']==item.item_hash and row['expected']==list(item.expected)
            assert row['dataset']==item.dataset and row['category']==item.category
            assert row['parsed']==parsed and row['valid']==valid
            assert row['correct']==(valid and row['finish_reason']=='stop' and parsed==list(item.expected))
            assert row['prompt_tokens']==prompt['tokens'] and 0<row['output_tokens']<=96
        assert summary['correct']==sum(r['correct'] for r in rows)
        assert summary['invalid']==sum(not r['valid'] for r in rows)
        assert summary['truncated']==sum(r['finish_reason']!='stop' for r in rows)
        result[stage]=dict(correct=summary['correct'],invalid=summary['invalid'],truncated=summary['truncated'],
            changed_raw_text=sum(a['prediction']!=b['prediction'] for a,b in zip(previous,rows)),
            changed_parsed_answers=sum(a['parsed']!=b['parsed'] for a,b in zip(previous,rows)),
            changed_correctness=sum(a['correct']!=b['correct'] for a,b in zip(previous,rows)),
            predictions_sha256=sha(directory/'predictions.private.jsonl'),prompts_sha256=sha(directory/'prompts.safe.json'))
        repeated[stage]=rows;old[stage]=previous
    result.update(original_comparison=comparison(old['baseline'],old['post']),
        repeat_comparison=comparison(repeated['baseline'],repeated['post']),
        verified_repeat_outputs=360,token_prompts_rebuilt=360 if tokenizer else 0,
        cases_sha256=CASE_SHA,training_performed=False,original_results_overwritten=False,
        further_identical_repeats_authorized=False,
        limitations='Two runs of the same trained weights; not independent training seeds or proof of population-level forgetting. No formal benchmark or sealed measurements.')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('cases','original','repeat','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--tokenizer',type=Path);a=p.parse_args()
    result=verify(a.cases,a.original,a.repeat,a.tokenizer)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

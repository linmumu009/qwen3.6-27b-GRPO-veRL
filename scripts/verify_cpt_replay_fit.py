"""Independent raw-answer verification of the bounded 405-answer replay fit probe."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from run_vllm_logistics_mcq import load_items
from evaluate_logistics_knowledge import build_messages
from verify_cpt_pilot_result import independently_parse
from run_cpt_replay_fit import MODELS

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]

def verify(packet,run,tokenizer=None):
    audit=json.loads((packet/'coverage.safe.json').read_text())
    assert sha(packet/'cases.private.jsonl')==audit['cases_sha256']
    items=load_items(packet/'cases.private.jsonl');assert len(items)==135
    registration=json.loads((run/'registration.safe.json').read_text())
    assert registration['actual_training_messages_matched']==135 and not registration['training']
    expected_prompts=None
    if tokenizer:
        from transformers import AutoTokenizer
        from audit_cpt_sft_search_data import TOKENIZER_SHA
        assert sha(tokenizer/'tokenizer.json')==TOKENIZER_SHA
        tok=AutoTokenizer.from_pretrained(tokenizer,local_files_only=True,trust_remote_code=True)
        expected_prompts=[]
        for item in items:
            text=tok.apply_chat_template(build_messages(item),tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tok.encode(text,add_special_tokens=False)
            expected_prompts.append(dict(source_id=item.source_id,text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                token_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),tokens=len(ids)))
    all_rows={};scores={};reference_prompts=None
    for name,model in MODELS.items():
        rows=read(run/name/'predictions.private.jsonl')
        prompts=json.loads((run/name/'prompts.safe.json').read_text())
        summary=json.loads((run/name/'summary.safe.json').read_text())
        assert [r['source_id'] for r in rows]==[i.source_id for i in items]
        assert [r['source_id'] for r in prompts]==[i.source_id for i in items]
        if reference_prompts is None:reference_prompts=prompts
        assert prompts==reference_prompts
        if expected_prompts is not None:assert prompts==expected_prompts
        for r,p,item in zip(rows,prompts,items):
            parsed,valid=independently_parse(r['prediction'],len(item.options))
            assert r['parsed']==parsed and r['valid']==valid and r['expected']==list(item.expected)
            assert r['item_hash']==item.item_hash and r['dataset']==item.dataset and r['category']==item.category
            assert r['correct']==(valid and r['finish_reason']=='stop' and parsed==list(item.expected))
            assert r['prompt_tokens']==p['tokens'] and 0<r['output_tokens']<=96
        for k,v in dict(model=model.as_posix(),items=135,temperature=0,seed=1024,max_tokens=96,max_model_len=8192,
            tp=8,max_num_seqs=32,thinking=False,repeats=1,cases_sha256=audit['cases_sha256']).items():assert summary[k]==v
        stats=dict(correct=sum(r['correct'] for r in rows),invalid=sum(not r['valid'] for r in rows),
            truncated=sum(r['finish_reason']!='stop' for r in rows))
        assert all(summary[k]==v for k,v in stats.items())
        scores[name]=dict(**stats,forms={f:dict(n=sum(r['category']==f for r in rows),
            correct=sum(r['correct'] for r in rows if r['category']==f)) for f in sorted({r['category'] for r in rows})},
            raw_sha256=sha(run/name/'predictions.private.jsonl'))
        all_rows[name]=rows
    changes={}
    for name in ('p1','p2'):
        pairs=list(zip(all_rows['base'],all_rows[name]))
        changes[name]=dict(gains=sum(not a['correct'] and b['correct'] for a,b in pairs),
            losses=sum(a['correct'] and not b['correct'] for a,b in pairs))
    return dict(scores=scores,changes_vs_base=changes,cases_sha256=audit['cases_sha256'],
        raw_answers_reparsed=405,token_prompts_rebuilt=405 if tokenizer else 0,
        actual_training_messages_matched=135,training=False,formal_evaluation=False,
        interpretation='Training-fit diagnosis only. These scores do not measure retained generalization or identify gradient conflict.',
        verifier_sha256=sha(Path(__file__)))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('packet','run','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--tokenizer',type=Path)
    a=p.parse_args();r=verify(a.packet,a.run,a.tokenizer)
    a.out.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps(r,indent=2))

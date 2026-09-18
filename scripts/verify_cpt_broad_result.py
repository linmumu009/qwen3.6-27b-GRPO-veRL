"""Rebuild every four-order call locally, without trusting stored score flags."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
from run_cpt_broad_review import read,specs,summarize
from run_cpt_source_authoring_pilot import closed_prompt
from run_cpt_remediation_review import review_prompt
from run_cpt_transfer_screen import digest
from verify_cpt_remediation_result import independent_prediction
from prepare_cpt_remediation_packet import sha

def obj(p):return json.loads(p.read_text(encoding='utf-8'))

def verify(packet,package,run,tokenizer):
    from transformers import AutoTokenizer
    manifest=obj(packet/'manifest.safe.json');rows=read(packet/'cases.private.jsonl')
    for name,h in manifest['files'].items():assert sha(packet/name)==h
    assert sha(tokenizer/'tokenizer.json')=='06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523'
    tok=AutoTokenizer.from_pretrained(tokenizer,trust_remote_code=True,local_files_only=True)
    by_id={r['id']:r for r in rows};plan=specs(rows);outputs={};summaries={};file_hashes={}
    def encode(text):return tok.encode(tok.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
    for label in ('step120_current','p1'):
        folder=run/label;assert obj(folder/'status.safe.json')['status']=='completed_pending_operator_audit'
        reg=obj(folder/'registration.safe.json')
        assert reg['packet_sha256']==manifest['files']['cases.private.jsonl']
        assert reg['code_sha256']==sha(package/'scripts/run_cpt_broad_review.py')
        assert reg['model_manifest_sha256']==sha(package/(label+'.model.safe.json'))
        assert reg['model']==obj(package/(label+'.model.safe.json'))['path']
        for k,v in dict(seed=20922,temperature=0,closed_calls=352,review_calls=88 if label=='p1' else 0,option_orders=4,max_closed_tokens=96,
                        max_review_tokens=1536,max_model_len=8192,tp=8,max_num_seqs=16,chunk=8,thinking=False,training_allowed=False).items():assert reg[k]==v
        raw=read(folder/'closed.private.jsonl');reviews=read(folder/'review.private.jsonl')
        summary=summarize(rows,raw,reviews,label);assert summary==obj(folder/'summary.safe.json')
        for s,r,d in zip(plan,raw,summary['per_call']):
            text=closed_prompt(by_id[s['id']],s);ids=encode(text)
            assert digest(ids)==r['prompt_token_sha256'] and len(ids)==r['prompt_tokens']
            pred=independent_prediction(r)
            assert (pred==s['expected'])==d['correct']
            original=sorted(s['order'][i] for i in pred) if pred is not None else None
            assert original==d['original_order_prediction']
        for row,r in zip(rows,reviews):
            ids=encode(review_prompt(row));assert digest(ids)==r['prompt_token_sha256'] and len(ids)==r['prompt_tokens']
        outputs[label]=raw;summaries[label]=summary
        for name in ('closed.private.jsonl','review.private.jsonl','registration.safe.json','summary.safe.json','status.safe.json'):
            file_hashes[label+'/'+name]=sha(folder/name)
    paired={};units={};cardinality={};task_comparisons=[]
    for b,a,bd,ad in zip(outputs['step120_current'],outputs['p1'],summaries['step120_current']['per_call'],summaries['p1']['per_call']):
        for k in ('id','variant','order','prompt_text_sha256','prompt_token_sha256','prompt_tokens'):assert a[k]==b[k]
        row=by_id[a['id']];values=dict(calls=1,step120_correct=int(bd['correct']),p1_correct=int(ad['correct']),gains=int(ad['correct'] and not bd['correct']),losses=int(bd['correct'] and not ad['correct']))
        paired.setdefault(row['split'],Counter()).update(values)
        units.setdefault(row['unit']+'/'+row['split'],Counter()).update(values)
        cardinality.setdefault(row['split']+'/'+str(len(row['correct_indices'])),Counter()).update(values)
    for b,a in zip(summaries['step120_current']['per_task'],summaries['p1']['per_task']):
        assert b['id']==a['id'];task_comparisons.append(dict(id=a['id'],split=a['split'],unit=a['unit'],step120_correct_orders=b['correct_orders'],p1_correct_orders=a['correct_orders'],p1_answer_invariant=a['invariant_answer']))
    return dict(status='completed_verified_local',packet_sha256=manifest['files']['cases.private.jsonl'],new_calls=792,closed_calls=704,blind_review_calls=88,
        independent_closed_json_parsing_passed=True,all_prompt_tokens_rebuilt=True,all_paired_prompts_match=True,remote_summaries_rebuilt=True,
        models=summaries,paired={k:dict(v) for k,v in paired.items()},units={k:dict(v) for k,v in units.items()},
        by_answer_cardinality={k:dict(v) for k,v in cardinality.items()},per_task_comparison=task_comparisons,raw_file_sha256=file_hashes,
        verifier_sha256=sha(Path(__file__)),training_allowed=False,limitations=manifest['limitations'])

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('packet','package','run','tokenizer','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();r=verify(a.packet,a.package,a.run,a.tokenizer)
    a.out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=r['status'],paired=r['paired'],model_review_passed=r['models']['p1']['review_accepted']),indent=2))

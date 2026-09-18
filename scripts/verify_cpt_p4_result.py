"""Independent local reconstruction of P4 training data and every evaluated answer."""
import argparse
import hashlib
import json
from pathlib import Path
from prepare_cpt_p4_cumulative import read,sha,append_released,OLD_MESSAGES_SHA,TOKENIZER_SHA
from audit_cpt_transfer import verify_predictions,verify_protocols,paired
from verify_cpt_pilot_result import independently_parse
from verify_cpt_remediation_result import independent_prediction

def obj(p):return json.loads(p.read_text(encoding='utf-8'))

def verify_data(data,packet,tokenizer):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    assert sha(tokenizer/'tokenizer.json')==TOKENIZER_SHA
    tok=AutoTokenizer.from_pretrained(tokenizer,local_files_only=True,trust_remote_code=True)
    audit=obj(data/'data_audit.safe.json');messages=read(data/'train.messages.private.jsonl')
    old=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in messages[:315]).encode()
    assert hashlib.sha256(old).hexdigest()==OLD_MESSAGES_SHA
    assert append_released(messages[:315],read(packet/'train.private.jsonl'),obj(packet/'release.safe.json'))==messages
    for split in ('train','dev'):
        rows=pq.read_table(data/(split+'.parquet')).to_pylist();msgs=read(data/(split+'.messages.private.jsonl'))
        assert len(rows)==len(msgs)==audit['budgets'][split]['records']
        for r,m in zip(rows,msgs):
            assert r['id']==m['id'];msg=m['messages']
            prefix=tok.encode(tok.apply_chat_template(msg[:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
            ids=prefix+tok.encode(msg[-1]['content'],add_special_tokens=False)+[tok.eos_token_id]
            assert ids==r['input_ids'] and len(prefix)==r['answer_start'] and len(ids)-len(prefix)==r['loss_tokens']
            assert len(ids)<=4096
        assert sha(data/(split+'.parquet'))==audit['budgets'][split]['sha256']
        assert sum(len(r['input_ids']) for r in rows)==audit['budgets'][split]['sequence_tokens']
        assert sum(r['loss_tokens'] for r in rows)==audit['budgets'][split]['loss_tokens']
    assert audit['budgets']['train']['sequence_tokens']==99038 and audit['budgets']['train']['loss_tokens']==14710
    return dict(local_token_reconstruction_passed=True,old315_messages_sha256=OLD_MESSAGES_SHA,train_records=363,loss_monitor_records=33,source_tokenizer_sha256=TOKENIZER_SHA)

def verify_result(resources,out):
    from run_cpt_transfer_p3 import comparison
    from run_cpt_remediation_review import summarize,specs,closed_prompt
    from run_cpt_transfer_screen import digest
    from run_vllm_logistics_mcq import load_items
    from evaluate_logistics_knowledge import build_messages
    from run_logistics_strategy_diagnostic import messages_for
    from transformers import AutoTokenizer
    packet=resources/'llin-remediation-packet-20260917-03'
    package=resources/'llin-transfer-p4-package-20260918-01'
    tokenpath=resources/'llin-transfer-p4-tokenizer-20260918-01'
    data_review=verify_data(resources/'llin-transfer-p4-data-20260918-01',packet,tokenpath)
    tok=AutoTokenizer.from_pretrained(tokenpath,local_files_only=True,trust_remote_code=True)
    def encode(messages):return tok.encode(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
    result=obj(out/'comparison.safe.json');reg=obj(out/'registration.safe.json')
    assert obj(out/'status.safe.json')['status']=='completed_verified_remote'
    assert result['new_requests']==reg['new_requests']==5284 and reg['score_dependent_gating'] is False
    assert result['data']==obj(resources/'llin-transfer-p4-data-20260918-01/data_audit.safe.json')==reg['data_audit']
    for rel,h in result['raw_sha256'].items():assert sha(out/rel)==h
    old=read(out/'post_old180/predictions.private.jsonl');items=load_items(package/'old_cases.private.jsonl');prompts=obj(out/'post_old180/prompts.safe.json')
    assert len(old)==len(items)==len(prompts)==180
    for r,item,p in zip(old,items,prompts):
        pred,ok=independently_parse(r['prediction'],len(item.options))
        assert r['parsed']==pred and r['valid']==ok and r['expected']==list(item.expected)
        assert r['correct']==(ok and r['finish_reason']=='stop' and pred==list(item.expected))
        assert r['source_id']==item.source_id==p['source_id'] and r['item_hash']==item.item_hash
        ids=encode(build_messages(item))
        assert len(ids)==p['tokens']==r['prompt_tokens'] and digest(ids)==p['token_sha256']
    prior=resources/'llin-transfer-p1-20260916-03/complete_result'
    for label,folder in [('step120',prior/'baseline'),('p1',prior/'post')]:
        c=comparison(read(folder/'predictions.private.jsonl'),old)
        assert result['old_diagnostics'][label]=={k:c[k] for k in ('strata','families')}
        assert obj(folder/'prompts.safe.json')==prompts
    rows=read(packet/'cases.private.jsonl');by_id={r['id']:r for r in rows};plan=specs(rows);raw=read(out/'post_new88/closed.private.jsonl')
    summary=summarize(rows,raw,[],'p4');assert summary==result['new_diagnostics']
    for s,r,d in zip(plan,raw,summary['per_call']):
        pred=independent_prediction(r);assert (pred==s['expected'])==d['correct']
        ids=encode([dict(role='user',content=closed_prompt(by_id[s['id']],s))])
        assert digest(ids)==r['prompt_token_sha256'] and len(ids)==r['prompt_tokens']
    for label in ('p1','step120_current'):
        before=read(package/(label+'.closed.private.jsonl'));b=summarize(rows,before,[],'baseline')
        for split in ('train','dev','retention'):
            keys=[i for i,d in enumerate(summary['per_call']) if d['split']==split]
            assert result['new_paired'][label+'/'+split]==paired(keys,b['per_call'],summary['per_call'])
    cases_path=resources/'llin-transfer-audit-20260915-01/frozen_cases.private.jsonl'
    assert sha(cases_path)=='b652b2108cb552346df11d005c15ff3137c50a756a7b24eb35302683ec33ed99'
    cases={r['item_hash']:r for r in read(cases_path)};assert len(cases)==1672
    formal=resources/'llin-transfer-formal-20260917-01'
    dirs={'step120_current':formal/'step120_current','p1':formal/'p1','p4':out/'formal5016'}
    protocols={k:obj(p/'protocol.safe.json') for k,p in dirs.items()};verify_protocols(protocols,sha(cases_path),1672)
    hashes=[hashlib.sha256(json.dumps(encode(messages_for(r,'original')[0])).encode()).hexdigest() for r in read(cases_path)]
    assert hashes==protocols['p4']['prompt_hashes']
    scores={}
    for label,folder in dirs.items():
        predictions=read(folder/'predictions.private.jsonl')
        scores[label]=verify_predictions(cases,predictions,obj(folder/'scores.safe.json'))
        for r in predictions:
            case=cases[r['item_hash']];pred,ok=independently_parse(r['prediction'],len(case['options']))
            assert pred==r['parsed'] and (ok and not r['truncated'])==r['valid']
            assert r['correct']==(r['valid'] and pred==sorted(case['expected']))
    for row in result['formal']:
        keys=sorted(k for k,c in cases.items() if c['dataset']==row['dataset'])
        assert row['p4_correct']==sum(scores['p4'][k]['correct'] for k in keys)
        for label,key in [('step120_current','versus_step120'),('p1','versus_p1')]:assert row[key]==paired(keys,scores[label],scores['p4'],True)
    for row in result['formal_categories']:
        keys=sorted(k for k,c in cases.items() if c['dataset']==row['dataset'] and c['category']==row['category'])
        assert row['versus_p1']==paired(keys,scores['p1'],scores['p4'])
    for name,key in [('formal_invalid','invalid'),('formal_truncated','truncated'),('formal_repeat_unstable','repeat_unstable')]:assert result[name]==sum(s[key] for s in scores['p4'].values())
    saved=obj(out/'formal5016/summary.safe.json');assert saved['requests']==5016
    assert {r['dataset']:r['correct'] for r in saved['table']}=={r['dataset']:r['p4_correct'] for r in result['formal']}
    result['local_verification']=dict(data=data_review,new_predictions_reparsed=5284,formal_control_predictions_reparsed=10032,all_1940_prompt_token_sequences_rebuilt=True,all_aggregate_comparisons_matched=True,verifier_sha256=sha(Path(__file__)))
    result['status']='completed_verified_both_sides';return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resources',type=Path,default=Path('CPT_resources'));p.add_argument('--run',type=Path);p.add_argument('--data-only',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.data_only:
        result=verify_data(a.resources/'llin-transfer-p4-data-20260918-01',a.resources/'llin-remediation-packet-20260917-03',a.resources/'llin-transfer-p4-tokenizer-20260918-01')
    else:result=verify_result(a.resources,a.run)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k in ('status','formal','local_token_reconstruction_passed')}))

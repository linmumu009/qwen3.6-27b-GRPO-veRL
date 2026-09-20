"""Reconstruct source/supervision/formal links before the bounded K/R experiment.

Only IDs, hashes, counts and provenance enter the public ledger. Raw text stays
in private-out. Exact span absence is not a claim of semantic non-exposure.
"""
import argparse
import hashlib
import json
from pathlib import Path

from audit_cpt_transfer import CASE_SHA, read_rows, sha, verify_predictions, verify_protocols
from prepare_cpt_p4_cumulative import OLD_MESSAGES_SHA, TOKENIZER_SHA


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def audit(root, out):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    resources = root/'CPT_resources'
    hist = out/'history'
    runs = hist/'workspace/llin-verl-grpo/runs'
    original = resources/'llin-transfer-audit-20260915-01'
    cases_path = original/'frozen_cases.private.jsonl'
    assert sha(cases_path) == CASE_SHA
    cases = {r['item_hash']: r for r in read_rows(cases_path)}
    evidence_path = original/'evidence_packet.private.jsonl'
    evidence = [r for r in read_rows(evidence_path) if cases[r['item_hash']]['dataset']=='LogistikaBench']
    assert len(evidence)==11 and all(r['review_status']=='source_sufficient_unique_answer' for r in evidence)
    unit_path = resources/'llin-knowledge-complete-20260911/core/reviewed_units.private.jsonl'
    archive_path = unit_path.parent/'sources.private.jsonl'
    units = {r['id']:r for r in read_rows(unit_path)}
    archive = {r['key']:r for r in read_rows(archive_path)}
    catalog = {r['id']:r for r in read_rows(resources/'llin-cpt-sft-search-20260912/source_generation_requests.private.jsonl')}
    dirs = {m:original/m for m in ('step120','cpt16','s4','s5')}
    dirs.update(step120_current=resources/'llin-transfer-formal-20260917-01/step120_current',
                p1=resources/'llin-transfer-formal-20260917-01/p1',
                p4=resources/'llin-transfer-p4-run-20260918-01/formal5016',
                s2=hist/'opt/llin-sft-search-s2-20260914-01/evaluation',
                lora=hist/'opt/llin-lora-cpt-r64-20260914-04/evaluation')
    protocols = {m:json.loads((p/'protocol.safe.json').read_text()) for m,p in dirs.items()}
    verify_protocols(protocols, CASE_SHA,1672)
    results = {m:verify_predictions(cases,read_rows(p/'predictions.private.jsonl'),json.loads((p/'scores.safe.json').read_text())) for m,p in dirs.items()}
    tokenizer_path=resources/'llin-transfer-p4-tokenizer-20260918-01'
    assert sha(tokenizer_path/'tokenizer.json')==TOKENIZER_SHA
    tok=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    cpt_path=out/'training/llin-knowledge-complete-20260911/train.parquet'
    assert sha(cpt_path)=='6ffa16684e617f657293918bf1d89ae8930ec24e0e08da346f175c4710c1f1c8'
    cpt=pq.read_table(cpt_path).to_pylist()
    assert len(cpt)==4912
    data_dirs={'p1':runs/'llin-transfer-p1-run-20260916-02/data',
               's4':runs/'llin-s4-data-20260915-01','s5':hist/'opt/llin-s5-data-20260915-01',
               's2':out/'training/supply-chain-task-training-data-20260909-01'}
    training={}; training_audits={}
    for m,d in data_dirs.items():
        messages=read_rows(d/'train.messages.private.jsonl')
        rows=pq.read_table(d/'train.parquet').to_pylist()
        assert len(messages)==len(rows)
        for x,r in zip(messages,rows):
            prefix=tok.encode(tok.apply_chat_template(x['messages'][:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
            actual=prefix+tok.encode(x['messages'][-1]['content'],add_special_tokens=False)+[tok.eos_token_id]
            assert actual==r['input_ids'] and len(prefix)==r['answer_start']
            assert r['loss_tokens']==len(actual)-len(prefix)
        training[m]=messages
        training_audits[m]={'records':len(rows),'messages_sha256':sha(d/'train.messages.private.jsonl'),
                           'parquet_sha256':sha(d/'train.parquet'),'all_answer_boundaries_verified':True}
    assert training_audits['p1']['messages_sha256']==OLD_MESSAGES_SHA
    training['p4']=read_rows(resources/'llin-transfer-p4-data-20260918-01/train.messages.private.jsonl')
    assert training['p4'][:315]==training['p1']
    probes={
        's2':hist/'opt/llin-sft-search-s2-20260914-01/s2_tasks.private.jsonl',
        'lora':hist/'opt/llin-lora-cpt-r64-20260914-04/source_probe/predictions.private.jsonl',
        's4':runs/'llin-sft-search-s4-20260915-01/s4_tasks.private.jsonl',
        'p1':resources/'llin-transfer-p1-20260916-03/complete_result/post/predictions.private.jsonl'}
    probe_rows={m:read_rows(p) for m,p in probes.items()}
    ledger=[]; packet=[]
    for uid in sorted({r['knowledge_unit_id'] for r in evidence}):
        unit=units[uid]; links=[r for r in evidence if r['knowledge_unit_id']==uid]
        for e in unit['evidence']:
            source=archive[e['key']]
            assert hashlib.sha256(source['text'].encode()).hexdigest()==source['sha256']==e['sha256']
            assert unit['body'] in source['text']
        assert all(r['short_evidence']==unit['body'] for r in links)
        target=[r['item_hash'] for r in links]
        exposure={}
        spans=[]
        for x in cpt:
            offset=x['text'].find(unit['body'])
            if offset<0:continue
            encoded=tok(x['text'],add_special_tokens=False,return_offsets_mapping=True)
            assert len(encoded['input_ids'])==x['content_tokens']
            assert hashlib.sha256(x['text'].encode()).hexdigest()==x['text_sha256']
            indices=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if b>offset and a<offset+len(unit['body'])]
            assert indices and min(indices)>0  # Text CPT masks only first token, retains all following.
            spans.append({'record_id':x['id'],'start_char':offset,'end_char':offset+len(unit['body']),
                          'first_token':min(indices),'last_token':max(indices),'rule_tokens':len(indices)})
        exposure['cpt16_and_lora']={'same_train_parquet_sha256':sha(cpt_path),'complete_supervised_spans':spans,
                                  'mask_contract':'first token zero; all subsequent tokens including EOS one'}
        for m,rows in training.items():
            linked=[x for x in rows if uid in x['id']]
            full=[x['id'] for x in rows if unit['body'] in x['messages'][-1]['content']]
            exposure[m]={'linked_records':len(linked),'linked_answer_only':sum(x['messages'][-1]['content'].startswith('{"answers":') for x in linked),
                         'full_verbatim_rule_supervised_records':len(full),'semantic_partial_coverage':'not_inferred_from_exact_match',
                         'linked_ids': [x['id'] for x in linked]}
        historical_probes={m:[{'id':r['source_id'],'correct':r['correct']} for r in rows if uid in r['source_id']] for m,rows in probe_rows.items()}
        eligible=catalog[uid]['proposed_split']=='train' and any(not results['p1'][k]['correct'] for k in target)
        row={'unit_id':uid,'source_sha256':unit['evidence'][0]['sha256'],
             'body_sha256':hashlib.sha256(unit['body'].encode()).hexdigest(),'target_items':len(target),
             'source_verified':True,'historical_split':catalog[uid]['proposed_split'],'training':exposure,
             'formal_correct':{m:sum(s[k]['correct'] for k in target) for m,s in results.items()},
             'historical_mcq_probes':historical_probes,
             'direct_rule_scope_probes':'missing; historical MCQ option recognition is not equivalent',
             'eligible':eligible,'selected':eligible and len(packet)<6}
        ledger.append(row)
        if row['selected']:
            packet.append({'unit_id':uid,'title':unit['title'],'body':unit['body'],'scope':unit['scope'],
                           'source_key':unit['evidence'][0]['key'],'source_sha256':row['source_sha256'],
                           'body_sha256':row['body_sha256']})
    write(out/'ledger.private.json',{'units':ledger,'target_links':{u['unit_id']:[r['item_hash'] for r in evidence if r['knowledge_unit_id']==u['unit_id']] for u in ledger}})
    write(out/'sources.private.json',packet)
    safe={'schema_version':1,'status':'history_recomputed_direct_call_control_missing','units':ledger,
          'source_archive_sha256':sha(archive_path),'evidence_packet_sha256':sha(evidence_path),
          'training_audits':training_audits,'formal_audits':{m:{'predictions_sha256':sha(p/'predictions.private.jsonl'),
             'protocol_sha256':sha(p/'protocol.safe.json'),'model':protocols[m]['model'],'requests_recomputed':5016} for m,p in dirs.items()},
          'selected_units':[r['unit_id'] for r in packet], 'new_model_calls':0,'promotion':False,
          'caveats':['Exact-span absence is not semantic non-exposure.','Historical MCQ probes do not measure free recall.',
                     'CPT/LoRA complete spans show supervised exposure, not successful rule recall.']}
    write(root/'docs/cpt_mechanism_history_20260920.safe.json',safe)
    return safe


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--out',type=Path,default=Path('CPT_resources/llin-mechanism-20260920-01'))
    a=p.parse_args();r=audit(a.root,a.out);print(json.dumps({'selected_units':r['selected_units'],'models_recomputed':len(r['formal_audits'])}))

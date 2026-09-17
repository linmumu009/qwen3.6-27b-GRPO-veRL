"""Read-only synthesis of frozen experiment evidence; no new inference or training."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def review(root):
    inputs={}
    def load(relative):
        p=root/relative;data=p.read_bytes();inputs[relative]=hashlib.sha256(data).hexdigest()
        return json.loads(data) if p.suffix=='.json' else [json.loads(s) for s in data.decode('utf-8').splitlines()]
    formal=load('docs/cpt_formal_transfer_result_20260917.safe.json')
    pilots={label:load('docs/cpt_transfer_'+label+'_result_'+date+'.safe.json')
            for label,date in [('p1','20260916'),('p2','20260917'),('p3','20260917')]}
    remediation=load('docs/cpt_remediation_result_20260917.safe.json')
    packet='CPT_resources/llin-remediation-packet-20260917-03/'
    rows=load(packet+'cases.private.jsonl');manifest=load(packet+'manifest.safe.json')
    assert inputs[packet+'cases.private.jsonl']==remediation['packet_sha256']
    by_id={r['id']:r for r in rows};assert len(by_id)==44
    errors={};cardinality={}
    for label,summary in remediation['final'].items():
        counts=defaultdict(Counter);cards=defaultdict(Counter)
        for d in summary['per_call']:
            row=by_id[d['id']];group='target24' if row['split']!='retention' else 'retention20'
            gold=set(row['correct_indices']);prediction=d['original_order_prediction']
            good=prediction is not None and set(prediction)==gold
            assert good==d['correct']
            counts[group]['calls']+=1;counts[group]['correct']+=good
            cards[(group,len(gold))]['calls']+=1;cards[(group,len(gold))]['correct']+=good
            if not good:
                kind='invalid' if prediction is None else 'omission_only' if set(prediction)<gold else 'extra_only' if set(prediction)>gold else 'substitution'
                counts[group][kind]+=1
        for c in counts.values():assert c['calls']==sum(c[k] for k in ('correct','invalid','omission_only','extra_only','substitution'))
        errors[label]={k:dict(c) for k,c in counts.items()}
        cardinality[label]=[dict(group=k[0],correct_option_count=k[1],**dict(c)) for k,c in sorted(cards.items())]
    coverage={}
    for dataset in ('SC-bench-knowledge','LogistikaBench'):
        cats=[r for r in formal['by_category'] if r['dataset']==dataset and r['model']=='p1']
        table=next(r for r in formal['table'] if r['dataset']==dataset and r['model']=='p1')
        remaining=[dict(category=r['category'],n=r['n'],correct=r['after'],remaining_errors=r['n']-r['after'],
                        gains=r['gains'],losses=r['losses']) for r in cats]
        assert sum(r['n'] for r in cats)==table['versus_current']['n']
        assert sum(r['after'] for r in cats)==table['versus_current']['after']
        assert sum(r['gains'] for r in cats)==table['versus_current']['gains']
        assert sum(r['losses'] for r in cats)==table['versus_current']['losses']
        selected={'Fulfillment','Logistics Collaboration'} if dataset=='SC-bench-knowledge' else {'material_handling','transport','warehousing'}
        total=sum(r['remaining_errors'] for r in remaining);priority=sum(r['remaining_errors'] for r in remaining if r['category'] in selected)
        coverage[dataset]=dict(remaining_errors=total,priority_errors=priority,priority_error_share=priority/total,
            categories=sorted(remaining,key=lambda r:-r['remaining_errors']),
            caveat='Remaining errors are an audit priority, not a count of source-supported repairable items.')
    formal_rows=[]
    for r in formal['table']:
        current=r['versus_current'];after=current['after'];gain=current['gains'];loss=current['losses']
        assert after-current['before']==gain-loss
        formal_rows.append(dict(dataset=r['dataset'],model=r['model'],n=current['n'],correct=after,target=r['target'],
            remaining_gap=r['target']-after,gains=gain,losses=loss,
            hypothetical_baseline_plus_all_observed_repairs=current['before']+gain,
            item_bootstrap_95pct_pp=current['paired_item_bootstrap_95pct_pp']))
    old=load('CPT_resources/llin-transfer-p1-20260916-03/train.private.jsonl')
    old_card=Counter(len(r['correct_indices']) for r in old);assert old_card=={1:20,2:20,3:20}
    new_card=Counter(len(r['correct_indices']) for r in rows if r['split']=='train')
    assert new_card=={1:4,2:4,3:4,4:4}
    return dict(date='2026-09-17',formal=formal_rows,
        pilots={label:dict(post=x['post']['strata'],steps=x['training_audit']['steps'],sequence_tokens=x['training_audit']['sequence_tokens']) for label,x in pilots.items()},
        new_packet_errors=errors,new_packet_by_answer_cardinality=cardinality,remaining_error_concentration=coverage,
        new_packet_scope={k:manifest[k] for k in ('cases','splits','source_units','train_source_groups','historical_dev_retention_groups')},
        training_cardinality=dict(old_unique_domain={str(k):v for k,v in old_card.items()},
                                  new_unique_domain={str(k):v for k,v in new_card.items()}),
        recommendation=dict(reference_model='P1; current point estimate leader, not a promoted model',
            next_candidate='One cumulative Step120 SFT with original P1 records plus released 16 tasks; measure/freeze tokens first.',
            purpose='Bounded test of cumulative transfer, not broad coverage completion or CPT increment.',
            evaluation='Freeze old and new diagnostics plus one complete two-benchmark measurement before training; do not let one small subgroup point block the engineering measurement.',
            broad_priority='Source-sufficient coverage and retention for SC fulfillment/collaboration and Logistika handling/transport/warehousing.',
            subsequent_cpt='Only after a useful stable SFT corpus, compare identical SFT from Step120 and CPT16.',
            execution_status='Recommendation only; no new training or model calls this turn; previous gates unchanged.'),
        caveats=['Two option orders are correlated; answer-cardinality effects are not controlled causal contrasts.',
            'P1/P3 formal differences cannot isolate replay arithmetic: exposure mix and sequence-token budgets differ.',
            'Old 80-item and new 20-item retention populations differ; 49/80 to 39/40 is not a temporal improvement.',
            'Formal benchmarks have informed selection; the results are engineering targets, not a fresh holdout.',
            'P2 formal scores remain unmeasured; no interpolation or inferred score is supplied.'],
        input_sha256=inputs,new_model_calls=0,new_training=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=review(a.root);a.out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(new_packet_errors=r['new_packet_errors'],remaining_error_concentration={k:{n:v[n] for n in ('remaining_errors','priority_errors','priority_error_share')} for k,v in r['remaining_error_concentration'].items()},new_model_calls=0),indent=2))

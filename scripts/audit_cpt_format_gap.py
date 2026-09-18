"""Posthoc format-stratified audit of existing formal results; no new model calls.

Only aggregate measurements leave the formal-data boundary. Authoring does not
import this module or receive benchmark question/option/label text.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
from audit_cpt_transfer import read_rows,sha,verify_predictions,verify_protocols,paired,CASE_SHA

def audit(resources,packet):
    cases_path=resources/'llin-transfer-audit-20260915-01/frozen_cases.private.jsonl'
    assert sha(cases_path)==CASE_SHA
    rows=read_rows(cases_path);cases={r['item_hash']:r for r in rows};assert len(cases)==1672
    dirs={'p1':resources/'llin-transfer-formal-20260917-01/p1','p4':resources/'llin-transfer-p4-run-20260918-01/formal5016'}
    protocols={label:json.loads((d/'protocol.safe.json').read_text()) for label,d in dirs.items()};verify_protocols(protocols,CASE_SHA,1672)
    raw={label:read_rows(d/'predictions.private.jsonl') for label,d in dirs.items()}
    scores={label:verify_predictions(cases,raw[label],json.loads((d/'scores.safe.json').read_text())) for label,d in dirs.items()}
    groups=defaultdict(list);format_counts={}
    for row in rows:
        n=len(row['options']);group=(row['dataset'],'269_options' if n==269 else 'up_to4' if n<=4 else '5to20','single' if len(row['expected'])==1 else 'multiple')
        groups[group].append(row['item_hash'])
    table=[dict(dataset=g[0],option_group=g[1],answer_group=g[2],**paired(sorted(keys),scores['p1'],scores['p4'])) for g,keys in sorted(groups.items())]
    for dataset in sorted({r['dataset'] for r in rows}):
        selected=[r for r in rows if r['dataset']==dataset]
        format_counts[dataset]=dict(cases=len(selected),option_counts=dict(Counter(len(r['options']) for r in selected)),answer_cardinalities=dict(Counter(len(r['expected']) for r in selected)))
    long=[r for r in rows if len(r['options'])==269]
    losses=[r for r in long if scores['p1'][r['item_hash']]['correct'] and not scores['p4'][r['item_hash']]['correct']]
    stable=0;stable_p4=0
    for case in losses:
        b=[r for r in raw['p1'] if r['item_hash']==case['item_hash']];a=[r for r in raw['p4'] if r['item_hash']==case['item_hash']]
        p4_wrong=all(not x['correct'] for x in a) and len({tuple(x['parsed']) for x in a})==1
        stable_p4+=p4_wrong
        stable+=all(x['correct'] for x in b) and p4_wrong
    candidate=read_rows(packet/'cases.private.jsonl')
    return dict(analysis='posthoc format audit; not causal identification',formal_cases_sha256=CASE_SHA,packet_sha256=sha(packet/'cases.private.jsonl'),
        candidate_counts=dict(cases=len(candidate),option_counts=dict(Counter(len(r['options']) for r in candidate))),formal_counts=format_counts,
        p1_to_p4_by_format=table,long_list=dict(cases=len(long),categories=dict(Counter(r['category'] for r in long)),
            distinct_option_lists=len({json.dumps(r['options'],ensure_ascii=False) for r in long}),gains=sum(not scores['p1'][r['item_hash']]['correct'] and scores['p4'][r['item_hash']]['correct'] for r in long),
            losses=len(losses),all_three_stable_losses=stable,p4_all_three_same_wrong_losses=stable_p4),
        raw_sha256={label:sha(d/'predictions.private.jsonl') for label,d in dirs.items()},
        new_inference_calls=0,formal_item_content_exposed_to_authoring=False,
        limits=['Task format is confounded with content/category; this audit does not prove why P4 regressed.',
               'The 269-choice records reuse one vocabulary; item-level statistics do not represent independent vocabularies.',
               'The broad packet has four options throughout and cannot certify long-list selection or high-index binding.',
               'A controlled source-authored short/long candidate comparison with unchanged semantics and repeated identical prompts is needed before selecting a format remedy.'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resources',type=Path,required=True);p.add_argument('--packet',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=audit(a.resources,a.packet);a.out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(r['long_list'],indent=2))

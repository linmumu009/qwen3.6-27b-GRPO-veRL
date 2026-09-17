"""Complete item ledger and source-link gaps; never emits training examples."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
from run_cpt_formal_transfer import verify, CASE_SHA


def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transition(before,after):
    return ('retained' if after else 'regressed') if before else ('repaired' if after else 'still_wrong')


def answer_pattern(expected, actual):
    gold=set(expected);answer=set(actual)
    return 'omission_only' if answer<gold else 'addition_only' if answer>gold else 'substitution'


def audit(resources,packet,out):
    history=resources/'llin-transfer-audit-20260915-01'
    run=resources/'llin-transfer-formal-20260917-01'
    verified=verify(run,history/'frozen_cases.private.jsonl',history/'step120')
    cases=read(history/'frozen_cases.private.jsonl')
    old_path=resources/'llin-knowledge-complete-20260911/case_disposition.private.jsonl'
    old_rows=read(old_path);old={r['item_hash']:r for r in old_rows}
    if len(old)!=len(old_rows):raise ValueError('duplicate historical item key')
    if not set(old)<={r['item_hash'] for r in cases}:raise ValueError('historical orphan item')
    manifest=json.loads((packet/'manifest.safe.json').read_text(encoding='utf-8'))
    proposed={sid for card in manifest['cards'] for sid in card['source_ids']}
    scores={n:json.loads((run/n/'scores.safe.json').read_text()) for n in ('step120_current','p1','p3')}
    rows={n:read(run/n/'predictions.private.jsonl') for n in scores}
    grouped={n:defaultdict(list) for n in scores}
    for n,records in rows.items():
        for r in records:grouped[n][r['item_hash']].append(r)
    ledger=[];review=[]
    for case in cases:
        k=case['item_hash'];entry=old.get(k);before=scores['step120_current'][k]['correct']
        record=dict(item_hash=k,dataset=case['dataset'],category=case['category'],
            option_count=len(case['options']),answer_cardinality=len(case['expected']),
            historical_registry_present=entry is not None,known_topics=entry['topics'] if entry else None,
            current_correct=before,models={})
        for label in ('p1','p3'):
            after=scores[label][k]['correct'];records=grouped[label][k];base=grouped['step120_current'][k]
            record['models'][label]=dict(correct=after,transition=transition(before,after),
                stable_all_repeat_regression=before and not after and all(r['correct'] for r in base) and not any(r['correct'] for r in records),
                invalid_votes=sum(not r['valid'] for r in records),
                proposed_assertion_links=sorted(proposed&set(entry['selected_assertion_units'])) if entry else [],
                proposed_topic_links=sorted(proposed&set(entry['topic_related_units'])) if entry else [],
                full_answer_coverage_verified=False)
        ledger.append(record)
        if record['models']['p1']['transition']=='regressed':
            votes=Counter(tuple(r['parsed']) if r['valid'] else None for r in grouped['p1'][k])
            answer,n=votes.most_common(1)[0]
            record['p1_regression_answer_pattern']=answer_pattern(case['expected'],answer) if answer is not None and n>=2 else 'no_valid_majority'
            review.append(dict(ledger=record,case=case,predictions={n:grouped[n][k] for n in scores},
                audit_only=True,training_allowed=False,review_status='semantic_diagnosis_pending'))
    transitions=[];strata=[];links=[]
    for label in ('p1','p3'):
        for dataset in ('SC-bench-knowledge','LogistikaBench'):
            subset=[r for r in ledger if r['dataset']==dataset]
            for status in ('retained','repaired','regressed','still_wrong'):
                selected=[r for r in subset if r['models'][label]['transition']==status]
                transitions.append(dict(model=label,dataset=dataset,transition=status,n=len(selected),
                    historical_registry_present=sum(r['historical_registry_present'] for r in selected),
                    unknown_topic_records=sum(r['known_topics'] is None for r in selected),
                    stable_all_repeat_regressions=sum(r['models'][label]['stable_all_repeat_regression'] for r in selected)))
                links.append(dict(model=label,dataset=dataset,transition=status,n=len(selected),
                    selected_assertion_link_count=sum(bool(r['models'][label]['proposed_assertion_links']) for r in selected),
                    broad_topic_link_count=sum(bool(r['models'][label]['proposed_topic_links']) for r in selected),
                    sufficient_answer_coverage_proven=0))
            for category in sorted({r['category'] for r in subset}):
                selected=[r for r in subset if r['category']==category]
                strata.append(dict(model=label,dataset=dataset,category=category,n=len(selected),
                    transitions=dict(Counter(r['models'][label]['transition'] for r in selected))))
    out.mkdir(parents=True,exist_ok=False)
    for name,data in [('item_ledger.private.jsonl',ledger),('p1_regressions_review.private.jsonl',review)]:
        (out/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data),encoding='utf-8',newline='\n')
    result=dict(cases=1672,cases_sha256=CASE_SHA,historical_registry_rows=len(old),
        transitions=transitions,by_category=strata,proposed_source_links=links,
        p1_regression_review_records=len(review),p1_regression_unknown_topics=sum(not x['ledger']['historical_registry_present'] for x in review),
        p1_regression_answer_patterns=dict(Counter(x['ledger']['p1_regression_answer_pattern'] for x in review)),
        no_training_data_emitted=True,model_calls=0,source_packet_sha256=sha(packet/'requests.private.jsonl'),
        source_links_are_not_sufficiency_proofs=True,
        warning='Historical registry is incomplete. Missing links mean unknown, never source-irrelevant or unrepairable. Category counts do not identify a causal mechanism.',
        inputs=dict(registry_sha256=sha(old_path),formal_prediction_sha256={k:v['predictions_sha256'] for k,v in verified['audits'].items()}),
        files={p.name:sha(p) for p in out.glob('*.jsonl')})
    (out/'audit.safe.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('resources','packet','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();r=audit(a.resources,a.packet,a.out)
    print(json.dumps({k:r[k] for k in ('cases','p1_regression_review_records','p1_regression_unknown_topics','transitions')},indent=2))

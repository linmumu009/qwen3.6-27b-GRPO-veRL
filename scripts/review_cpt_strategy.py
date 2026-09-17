"""Recompute decision-level CPT evidence from retained results; no model calls."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

def review(root):
    inputs={}
    def read(relative):
        path=root/relative;inputs[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
        return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
    official={};formal_rows=[]
    for name in ('step120','cpt16','s4','s5'):
        rows=read(f'llin-transfer-audit-20260915-01/{name}/predictions.private.jsonl')
        groups=defaultdict(list)
        for row in rows:groups[(row['dataset'],row['item_hash'])].append(row)
        assert len(rows)==5016 and len(groups)==1672
        assert all(len(v)==3 and len({x['repeat'] for x in v})==3 for v in groups.values())
        # Majority of exact expected-answer matches is equivalent to the expected
        # answer set receiving a strict majority; no plurality/tie relaxation.
        official[name]={k:sum(bool(x['correct']) for x in v)>=2 for k,v in groups.items()}
        assert set(official[name])==set(official['step120'])
        for dataset in sorted({k[0] for k in groups}):
            ids=[k for k in groups if k[0]==dataset];base=official['step120'];cur=official[name]
            formal_rows.append(dict(model=name,dataset=dataset,n=len(ids),correct=sum(cur[k] for k in ids),
                gains=sum(not base[k] and cur[k] for k in ids),losses=sum(base[k] and not cur[k] for k in ids)))
    union={d:sum(not official['step120'][k] and any(official[m][k] for m in ('cpt16','s4','s5'))
        for k in official['step120'] if k[0]==d) for d in sorted({k[0] for k in official['step120']})}
    paths=dict(p1='llin-transfer-p1-20260916-03/complete_result/post',
        p2='llin-transfer-p2-result-20260917-01/post',p3='llin-transfer-p3-result-20260917-01/post')
    pilots={}
    for name,path in paths.items():
        rows=read(path+'/predictions.private.jsonl');assert len(rows)==180
        pilots[name]={s:dict(n=sum(r['dataset']==s for r in rows),correct=sum(r['correct'] for r in rows if r['dataset']==s))
            for s in sorted({r['dataset'] for r in rows})}
    arith={}
    for name in ('arithmetic_base','arithmetic_p2','arithmetic_post'):
        rows=read('llin-transfer-p3-result-20260917-01/'+name+'/predictions.private.jsonl')
        arith[name]={r['source_id']:r['correct'] for r in rows};assert len(arith[name])==60
    before=arith['arithmetic_p2'];after=arith['arithmetic_post'];assert before.keys()==after.keys()
    pair=dict(before=sum(before.values()),after=sum(after.values()),gains=sum(not before[k] and after[k] for k in before),
        losses=sum(before[k] and not after[k] for k in before))
    tasks=read('llin-transfer-p1-20260916-03/train.private.jsonl')
    sources={s for r in tasks for s in r['source_ids']}
    cases=read('llin-knowledge-complete-20260911/case_disposition.private.jsonl');assert len(cases)==300
    links={}
    for field in ('selected_assertion_units','topic_related_units'):
        matched=[r for r in cases if sources&set(r[field])]
        links[field]=dict(counts=dict(Counter(r['dataset'] for r in matched)),
            full_answer_coverage_verified=sum(r['full_answer_coverage_verified'] for r in matched))
    return dict(date='2026-09-17',formal=formal_rows,historical_repair_union=union,pilots=pilots,
        arithmetic_scores={k:sum(v.values()) for k,v in arith.items()},arithmetic_p3_vs_p2=pair,
        training_scope=dict(families=len({r['unit'] for r in tasks}),source_units=len(sources),unique_new_tasks=len(tasks)),
        historical_case_links=links,scope_warning='Historical 300-error registration is incomplete; links are not sufficient answer proofs or bounds on future generalization.',
        official_new_candidates_evaluated=[],new_model_calls=0,training=False,input_sha256=inputs)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resources',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=review(a.resources);a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},indent=2))

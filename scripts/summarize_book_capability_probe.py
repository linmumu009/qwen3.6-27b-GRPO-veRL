"""Aggregate frozen capability sample; preserve unresolved grades and splits."""
import argparse
from collections import Counter
import json
from pathlib import Path

def summarize(out):
    result=json.loads((out/'result.safe.json').read_text());cases=json.loads((out/'cases.private.json').read_text());lookup={q['id']:q for q in cases}
    assert len(result['results'])==len(lookup)==40
    rows=[dict(r,kind=lookup[r['id']]['kind'],split=lookup[r['id']]['split']) for r in result['results']]
    groups=[]
    for split in ('all','train','dev'):
        for kind in ('all',)+tuple(sorted({q['kind'] for q in cases})):
            subset=[r for r in rows if (split=='all' or r['split']==split) and (kind=='all' or r['kind']==kind)]
            agreed=[r for r in subset if r['status']=='agreed']
            for model in ('step120','cpt'):
                groups.append(dict(split=split,kind=kind,model=model,planned=len(subset),agreed=len(agreed),unresolved=len(subset)-len(agreed),closed_full=sum(r['scores'][model+':closed']==2 for r in agreed),evidence_full=sum(r['scores'][model+':evidence']==2 for r in agreed),decisions=dict(Counter(r['decisions'][model] for r in agreed))))
    reviews=[json.loads(x) for x in (out/'grading/reviews.private.jsonl').read_text().splitlines()]
    meta={m:json.loads((out/m/'summary.safe.json').read_text()) for m in ('step120','cpt')}
    value=dict(items=40,agreed=sum(r['status']=='agreed' for r in rows),unresolved=sum(r['status']!='agreed' for r in rows),review_statuses=dict(Counter(r['status'] for r in reviews)),groups=groups,api_calls=result['api_calls'],usage=result['usage'],models=meta,prompt_identity_verified=result['prompt_identity_verified'],training=False,population_accuracy_estimate=False,results=rows)
    (out/'analysis.safe.json').write_text(json.dumps(value,indent=2))
    print(json.dumps({k:v for k,v in value.items() if k not in ('results','models','groups')},indent=2))
    print(json.dumps([g for g in groups if g['split']=='all'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();summarize(a.out)

"""Safe census of existing QA and held-out model scores; no API or training."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    data=a.root/'runs/book-sft-production-2000-20260907'
    rows=[json.loads(s) for s in (data/'accepted.private.jsonl').read_text().splitlines()]
    by_id={r['id']:r for r in rows}
    reviews=[json.loads(s) for s in (a.root/'runs/book-sft-diagnosis-20260907/grading_concise/reviews.private.jsonl').read_text().splitlines()]
    groups=defaultdict(Counter)
    for r in reviews:
        if r['status']!='scored':continue
        kind=by_id[r['id']]['kind'];g=groups[kind];g['items']+=1
        for arm in ('cpt','sft'):g[f'{arm}_score_{r["scores"][arm]}']+=1
        g['sft_better']+=int(r['scores']['sft']>r['scores']['cpt'])
        g['cpt_better']+=int(r['scores']['cpt']>r['scores']['sft'])
    safe={'accepted':len(rows),'kind_counts':dict(Counter(r['kind'] for r in rows)),
        'chapter_counts':dict(Counter(r['chapter'] for r in rows)),
        'heldout_by_kind':{k:dict(v) for k,v in groups.items()},
        'training_value_measured_before_old_sft':False,
        'reason_for_value_flag':'Old producer selects auto_pass by source quality, duplicate/overlap filters and arrival order; no target-model pretest.',
        'limitation':'Held-out scores characterize these questions, not marginal training benefit of any training row.',
        'private_text_included':False,'api_calls':0,'training':False}
    with a.output.open('x') as f:json.dump(safe,f,indent=2)
    print(json.dumps(safe,indent=2))


if __name__=='__main__':main()

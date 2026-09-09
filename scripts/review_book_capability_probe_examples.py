"""Print bounded private examples for substantive audit, never publish bodies."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cases={q['id']:q for q in json.loads((a.out/'cases.private.json').read_text())}
    results=json.loads((a.out/'analysis.safe.json').read_text())['results']
    selected=[]
    for kind in ('definition','distinction','calculation'):
        choices=[r for r in results if r['split']=='train' and r['kind']==kind and r['status']=='agreed' and r['scores']['cpt:closed']!=2]
        selected.extend(choices[:2 if kind=='definition' else 1])
    selected.extend([r for r in results if 'quality_quarantine' in r['review_statuses']][:2])
    responses=[json.loads(x) for x in (a.out/'cpt/answers.private.jsonl').read_text().splitlines()]
    reviews=[json.loads(x) for x in (a.out/'grading/reviews.private.jsonl').read_text().splitlines()]
    for r in selected:
        q=cases[r['id']]
        print(json.dumps(dict(case=q,decision=r,answers=[x for x in responses if x['id']==q['id']],judges=[dict(mapping=x['mapping'],verdict=x.get('verdict')) for x in reviews if x['id']==q['id']]),ensure_ascii=False))

if __name__=='__main__':main()

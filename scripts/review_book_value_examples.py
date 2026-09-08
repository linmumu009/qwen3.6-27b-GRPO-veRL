"""Bounded private review of self-authored book items, never public benchmarks."""
import argparse
import json
from pathlib import Path


def main(out):
    result=json.loads((out/'reviewed/result.safe.json').read_text())
    cases={r['id']:r for r in json.loads((out/'cases.private.json').read_text())}
    scores=result['open_scores'];chosen=[]
    for kind in ('definition','discrimination','application'):
        candidates=[r for r in scores if r['model']=='cpt' and r['kind']==kind and r['closed_score']==0 and r['evidence_score']==2]
        if candidates:chosen.append(sorted(candidates,key=lambda r:r['id'])[0]['id'])
    good=[r for r in scores if r['model']=='cpt' and r['closed_score']==2]
    if good:chosen.append(sorted(good,key=lambda r:r['id'])[0]['id'])
    answers=[r for model in ('step120','cpt') for r in map(json.loads,(out/model/'answers.private.jsonl').read_text().splitlines())]
    rows=[]
    for key in chosen:
        c=cases[key];rows.append({'id':key,'question':c['question'],'reference_answer':c['answer'],'rubric':c['rubric'],
          'source':c['reference_units'],'responses':[{k:r[k] for k in ('model','condition','text','finish_reason')} for r in answers if r['id']==key]})
    print(json.dumps(rows,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)

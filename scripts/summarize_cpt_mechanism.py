"""Create a private identity-blinded review packet, then unblind frozen ratings.

Never infers factual scores from string overlap. A reviewer must supply each
fact judgment and forbidden-inference judgment using the frozen source rubric.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import secrets
import random

from audit_cpt_transfer import read_rows,verify_predictions,verify_protocols,CASE_SHA


def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def blind(run,probes,out):
    out.mkdir(exist_ok=False)
    rubric={r['id']:r for r in json.loads(probes.read_text(encoding='utf8'))}
    rows=[];mapping={};raw_hashes={}
    for model in ('step120','p1','K','R'):
        path=run/(model+'_direct')/'predictions.private.jsonl';raw_hashes[model]=sha(path)
        raw=read_rows(path);assert len(raw)==3*len(rubric)
        assert {(r['id'],r['repeat']) for r in raw}=={(k,i) for k in rubric for i in range(3)}
        for r in raw:
            key=secrets.token_hex(16);q=rubric[r['id']]
            mapping[key]=dict(model=model,prompt_id=r['id'],unit_id=r['unit_id'],repeat=r['repeat'])
            rows.append(dict(blind_id=key,prompt=q['messages'],required=q['required'],forbidden=q['forbidden'],
                             prediction=r['prediction'],finish_reason=r['finish_reason']))
    random.SystemRandom().shuffle(rows)
    save(out/'review.private.json',rows)
    save(out/'identity_key.private.json',mapping)
    save(out/'freeze.safe.json',dict(probes_sha256=sha(probes),raw_sha256=raw_hashes,
         review_sha256=sha(out/'review.private.json'),key_sha256=sha(out/'identity_key.private.json'),items=len(rows)))


def summarize(run,review,ratings,ledger,cases_path,baselines,out):
    frozen=json.loads((review/'freeze.safe.json').read_text())
    assert sha(review/'review.private.json')==frozen['review_sha256']
    assert sha(review/'identity_key.private.json')==frozen['key_sha256']
    questions={r['blind_id']:r for r in json.loads((review/'review.private.json').read_text())}
    # Commit/freeze review before opening identity mapping.
    judged=json.loads(ratings.read_text());assert len(judged)==len(questions)
    assert {r['blind_id'] for r in judged}==set(questions)
    for r in judged:
        q=questions[r['blind_id']]
        assert len(r['facts'])==len(q['required']) and all(type(x)==bool for x in r['facts'])
        assert type(r['forbidden_inference'])==bool and type(r['uncertain'])==bool
        assert isinstance(r['reason'],str) and r['reason'].strip()
    score_digest=sha(ratings)
    freeze=review/'ratings_frozen.safe.json'
    if freeze.exists():assert json.loads(freeze.read_text())['ratings_sha256']==score_digest
    else:save(freeze,dict(ratings_sha256=score_digest,items=len(judged)))
    mapping=json.loads((review/'identity_key.private.json').read_text())
    direct=defaultdict(lambda:dict(calls=0,strict_pass=0,facts_correct=0,facts_total=0,uncertain=0))
    by_prompt=defaultdict(list)
    for r in judged:
        key=mapping[r['blind_id']];q=questions[r['blind_id']]
        passed=all(r['facts']) and not r['forbidden_inference'] and not r['uncertain'] and q['finish_reason']=='stop'
        x=direct[key['model']];x['calls']+=1;x['strict_pass']+=int(passed);x['facts_correct']+=sum(r['facts']);x['facts_total']+=len(r['facts']);x['uncertain']+=r['uncertain']
        by_prompt[(key['model'],key['unit_id'],key['prompt_id'])].append(passed)
    assert sha(cases_path)==CASE_SHA
    cases={r['item_hash']:r for r in read_rows(cases_path)}
    evidence=json.loads(ledger.read_text());selected={r['unit_id'] for r in evidence['units'] if r['selected']}
    target={k for uid,keys in evidence['target_links'].items() if uid in selected for k in keys}
    dirs={'step120':baselines/'step120_current','p1':baselines/'p1','K':run/'K_formal','R':run/'R_formal'}
    verify_protocols({m:json.loads((d/'protocol.safe.json').read_text()) for m,d in dirs.items()},CASE_SHA,1672)
    scores={m:verify_predictions(cases,read_rows(d/'predictions.private.jsonl'),json.loads((d/'scores.safe.json').read_text())) for m,d in dirs.items()}
    table=[]
    for arm in ('K','R'):
        for ref in ('step120','p1','R'):
            if arm==ref:continue
            for dataset in ('SC-bench-knowledge','LogistikaBench'):
                for subset in ('all','target','non_target'):
                    keys=[k for k,c in cases.items() if c['dataset']==dataset and (subset=='all' or (k in target)==(subset=='target'))]
                    gains=sum(not scores[ref][k]['correct'] and scores[arm][k]['correct'] for k in keys)
                    losses=sum(scores[ref][k]['correct'] and not scores[arm][k]['correct'] for k in keys)
                    table.append(dict(arm=arm,reference=ref,dataset=dataset,subset=subset,n=len(keys),
                        correct=sum(scores[arm][k]['correct'] for k in keys),gains=gains,losses=losses,net=gains-losses))
    unit_outcomes=[]
    for uid in sorted(selected):
        keys=evidence['target_links'][uid]
        linked=next(r for r in evidence['units'] if r['unit_id']==uid)
        current={m:dict(target_correct=sum(s[k]['correct'] for k in keys),
                     repaired_vs_p1=sum(not scores['p1'][k]['correct'] and s[k]['correct'] for k in keys),
                     lost_vs_p1=sum(scores['p1'][k]['correct'] and not s[k]['correct'] for k in keys),
                     direct_strict_passes=sum(sum(v) for (mm,u,p),v in by_prompt.items() if mm==m and u==uid)) for m,s in scores.items()}
        unit_outcomes.append(dict(unit_id=uid,source_sha256=linked['source_sha256'],body_sha256=linked['body_sha256'],
                                  historical_training=linked['training'],historical_formal=linked['formal_correct'],current=current))
    safe=dict(direct=dict(direct),formal=table,unit_outcomes=unit_outcomes,ratings_sha256=score_digest,target_items=len(target),
              unit_stability=[dict(model=m,unit_id=u,prompt_id=p,passes=sum(v),repeats=3) for (m,u,p),v in sorted(by_prompt.items())],
              promotion=False,independent_holdout=False,training_seeds=1,
              interpretation='Source direct teaching recipe; judge recall before diagnosing transfer; aggregate score alone is insufficient.')
    save(out,safe)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['blind','summarize'])
    for k in ('run','out'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('probes','review','ratings','ledger','cases','baselines'):p.add_argument('--'+k,type=Path)
    a=p.parse_args()
    if a.action=='blind':blind(a.run,a.probes,a.out)
    else:summarize(a.run,a.review,a.ratings,a.ledger,a.cases,a.baselines,a.out)

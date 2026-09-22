"""Blind new responses, preserve prior scores, and apply frozen decision gates."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,x):
    assert not p.exists(), 'Frozen artifact already exists: '+str(p)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def passed(r):
    return r['supported'] and not r['contradiction'] and not r['uncertain']


def decide(counts):
    rail = 'candidate_cue_sensitivity' if counts['rail']['focused']>=2 and counts['rail']['broad']<=1 else 'mixed_or_no_cue_signal'
    barge = 'candidate_granularity_sensitivity' if counts['barge']['per_class']>=2 and counts['barge']['ordinary']<=1 else 'mixed_or_no_granularity_signal'
    load = counts['load']
    if load['original_note']>=2:
        note='earlier_failure_not_stable'
    elif load['neutral_note']>=2 and load['removed_note']>=2:
        note='candidate_note_content_effect'
    elif load['removed_note']>=2 and load['neutral_note']<=1:
        note='removal_or_length_confounded'
    else:
        note='mixed'
    return dict(rail=rail,barge=barge,load=note)


def verify(base,registration):
    reg=read(registration)
    assert sha(base/'cases.private.json')==reg['cases_sha256']
    cases={c['id']:c for c in read(base/'cases.private.json')}
    assert len(cases)==21
    new_ids={cid for cid,c in cases.items() if c['reused_case_id'] is None}
    assert len(new_ids)==19
    old=base.parent/'llin-context-bridge-20260922-01'
    assert sha(old/'cases.private.json')==reg['prior_cases_sha256']
    old_cases={c['id']:c for c in read(old/'cases.private.json')}
    old_review=old/'blind-review'
    old_freeze=read(old_review/'freeze.safe.json')
    assert sha(old_review/'ratings.private.json')==reg['prior_ratings_sha256']
    assert sha(old_review/'key.private.json')==old_freeze['key_sha256']
    old_key=read(old_review/'key.private.json')
    old_ratings={(old_key[r['blind_id']]['model'],old_key[r['blind_id']]['case_id']):r
                 for r in read(old_review/'ratings.private.json')}
    root=base/'results'
    assert read(root/'registration.safe.json')==reg
    assert read(root/'status.safe.json')['state']=='complete_review_pending'
    records,prior_scores,hashes={},{},{}
    for model in reg['models']:
        folder=root/model
        done=read(folder/'completed.safe.json')
        reserved=read(folder/'reserved.safe.json')
        path=folder/'responses.private.json'
        assert done['requests']==reserved['requests']==19
        assert sha(path)==done['result_sha256']
        assert reserved['cases_sha256']==reg['cases_sha256']
        rows=read(path)
        assert len(rows)==19 and {r['id'] for r in rows}==new_ids
        assert sum(r['output_tokens'] for r in rows)==done['output_tokens']
        records[model]={r['id']:r for r in rows}
        hashes[model]=sha(path)
        history=old/'results'/model/'response.private.json'
        assert sha(history)==reg['prior_raw_sha256'][model+'/response']
        historic={r['id']:r for r in read(history)}
        for cid,c in cases.items():
            if c['reused_case_id']:
                oldid=c['reused_case_id']
                assert c['messages']==old_cases[oldid]['messages'] and c['prompt_ids']==old_cases[oldid]['prompt_ids']
                assert c['required']==old_cases[oldid]['required'] and c['target_sentence']==old_cases[oldid]['target_sentence']
                records[model][cid]=dict(historic[oldid],**{k:c[k] for k in ('id','module','condition','repeat')})
                prior_scores[(model,cid)]=old_ratings[(model,oldid)]
            row=records[model][cid]
            assert all(row[k]==c[k] for k in ('id','unit_id','module','condition','repeat','fact_index'))
            assert 0<=row['output_tokens']<=384
    assert len(prior_scores)==4
    return reg,cases,records,prior_scores,hashes


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['blind','freeze','summarize'])
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    reg,cases,records,prior,hashes=verify(a.base,a.registration)
    review=a.base/'blind-review';review.mkdir(exist_ok=True)
    if a.mode=='blind':
        key,items={},[]
        for model,rows in records.items():
            for cid,row in rows.items():
                if cases[cid]['reused_case_id']:continue
                bid=secrets.token_hex(12)
                key[bid]=dict(model=model,case_id=cid)
                items.append(dict(blind_id=bid,required=cases[cid]['required'],reference=cases[cid]['target_sentence'],
                                  response=row['prediction'],finish_reason=row['finish_reason']))
        secrets.SystemRandom().shuffle(items)
        save(review/'review.private.json',items);save(review/'key.private.json',key)
        save(review/'freeze.safe.json',dict(cases_sha256=reg['cases_sha256'],raw_sha256=hashes,
             review_sha256=sha(review/'review.private.json'),key_sha256=sha(review/'key.private.json')))
        print('38 new responses blinded; 4 frozen historical scores retained')
        return
    frozen=read(review/'freeze.safe.json')
    assert frozen['raw_sha256']==hashes and frozen['cases_sha256']==reg['cases_sha256']
    assert sha(review/'review.private.json')==frozen['review_sha256'] and sha(review/'key.private.json')==frozen['key_sha256']
    ratings=read(review/'ratings.private.json')
    assert len(ratings)==38 and len({r['blind_id'] for r in ratings})==38
    assert {r['blind_id'] for r in ratings}=={r['blind_id'] for r in read(review/'review.private.json')}
    for r in ratings:
        assert all(type(r[k]) is bool for k in ('supported','contradiction','uncertain')) and r['reason'].strip()
    if a.mode=='freeze':
        save(review/'ratings_frozen.safe.json',dict(items=38,ratings_sha256=sha(review/'ratings.private.json'),
            reviewer='Single Codex semantic review with model and condition labels hidden; no independent human review'))
        print('Frozen',sha(review/'ratings.private.json'));return
    assert sha(review/'ratings.private.json')==read(review/'ratings_frozen.safe.json')['ratings_sha256']
    key=read(review/'key.private.json');scores=dict(prior)
    for r in ratings:scores[(key[r['blind_id']]['model'],key[r['blind_id']]['case_id'])]=r
    result=dict(id=reg['id'],models={},new_response_calls=38,reused_calls=4,observations=42,
        training_steps=0,formal_reruns=0,probability_calls=0,promotion=False,
        ratings_sha256=sha(review/'ratings.private.json'),prior_ratings_sha256=reg['prior_ratings_sha256'],raw_sha256=hashes)
    for model,rows in records.items():
        groups={};cells=[]
        for cid in sorted(cases):
            c,r=cases[cid],rows[cid];rating=scores[(model,cid)]
            cell=dict(id=cid,module=c['module'],condition=c['condition'],repeat=c['repeat'],
                reused=c['reused_case_id'] is not None,passed=passed(rating),supported=rating['supported'],
                contradiction=rating['contradiction'],uncertain=rating['uncertain'],
                output_tokens=r['output_tokens'],truncated=r['finish_reason']=='length')
            cells.append(cell)
            group=groups.setdefault(c['module'],{}).setdefault(c['condition'],dict(passes=0,repeats_1_2_passes=0,
                repeats=3,prompt_tokens=len(c['prompt_ids']),answer_hashes=set()))
            group['passes']+=cell['passed']
            group['repeats_1_2_passes']+=cell['passed'] and c['repeat'] in (1,2)
            group['answer_hashes'].add(hashlib.sha256(r['prediction'].encode()).hexdigest())
        for module in groups.values():
            for g in module.values():g['distinct_answers']=len(g.pop('answer_hashes'))
        counts={m:{c:g['passes'] for c,g in gs.items()} for m,gs in groups.items()}
        result['models'][model]=dict(groups=groups,cells=cells,decisions=decide(counts))
    result['new_output_tokens']=sum(r['output_tokens'] for m,rows in records.items() for cid,r in rows.items() if cases[cid]['reused_case_id'] is None)
    assert result['new_output_tokens']<=14592
    assert a.out is not None
    save(a.out,result)
    print(json.dumps({m:dict(groups=v['groups'],decisions=v['decisions']) for m,v in result['models'].items()},indent=2))


if __name__=='__main__':main()

"""Rebuild both versions, replace only registered amendments, and compare all calls."""
import argparse
from collections import Counter
import json
from pathlib import Path
from prepare_cpt_remediation_packet import sha, validate_row
from revise_cpt_remediation_packet import CHANGES, PARENT_SHA, amended_row
from run_cpt_remediation_review import summarize


def read(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
def obj(path):return json.loads(path.read_text(encoding='utf-8'))


def independent_prediction(raw):
    if raw['finish_reason']!='stop':return None
    try:
        parsed=json.loads(raw['text'].strip())
        assert isinstance(parsed,dict) and set(parsed)=={'answers'}
        values=parsed['answers']
        assert isinstance(values,list) and 1<=len(values)<=4
        assert all(type(v) is int and 0<=v<4 for v in values) and len(set(values))==len(values)
        return sorted(values)
    except (ValueError,TypeError,AssertionError):return None


def verify_stage(rows,folder,label,packet_sha,launch):
    assert obj(folder/'status.safe.json')['status']=='completed_pending_operator_audit'
    registration=obj(folder/'registration.safe.json')
    assert registration['packet_sha256']==packet_sha
    assert registration['closed_calls']==len(rows)*2
    assert registration['review_calls']==(len(rows) if label=='p1' else 0)
    for k,v in dict(seed=20922,temperature=0,closed_max_tokens=96,review_max_tokens=1536,
                    thinking=False,tp=8,max_model_len=8192,max_num_seqs=16,training_allowed=False).items():
        assert registration[k]==v
    assert registration['code_sha256']==launch['files']['run_cpt_remediation_review.py']
    assert registration['model_manifest_sha256']==launch['files'][label+'.model.safe.json']
    closed=read(folder/'closed.private.jsonl');reviews=read(folder/'review.private.jsonl')
    summary=summarize(rows,closed,reviews,label)
    remote=obj(folder/'summary.safe.json')
    assert remote.pop('remote_token_reconstruction_verified') is True
    assert remote==summary
    for raw,detail in zip(closed,summary['per_call']):
        prediction=independent_prediction(raw)
        original=sorted(raw['order'][i] for i in prediction) if prediction is not None else None
        assert original==detail['original_order_prediction']
    return closed,reviews,summary


def replace_amended(original,amended,ids):
    def key(r):return (r['id'],r.get('variant'))
    assert {r['id'] for r in amended}==set(ids)
    replacements={key(r):r for r in amended}
    assert len(replacements)==len(amended)
    expected={key(r) for r in original if r['id'] in ids}
    assert expected==set(replacements)
    return [replacements[key(r)] if r['id'] in ids else r for r in original]


def verify(parent,packet):
    assert sha(parent/'cases.private.jsonl')==PARENT_SHA
    manifest=obj(packet/'manifest.safe.json');old_rows=read(parent/'cases.private.jsonl')
    rows=read(packet/'cases.private.jsonl');amended_rows=read(packet/'amended_cases.private.jsonl')
    assert len(rows)==len(old_rows)==44
    for name,h in manifest['files'].items():assert sha(packet/name)==h
    for old,row in zip(old_rows,rows):
        validate_row(row);assert old['id']==row['id']
        assert amended_row(old)==row
    assert amended_rows==[r for r in rows if r['id'] in CHANGES]
    original_launch=obj(parent/'launch.safe.json');amend_launch=obj(packet/'launch.safe.json')
    versions={};combined={};raw_by_label={};review_by_label={};files={}
    for label in ('p1','step120_current'):
        raw,review,summary=verify_stage(old_rows,parent/'results'/label,label,PARENT_SHA,original_launch)
        delta,dreview,dsummary=verify_stage(amended_rows,packet/'results'/label,label,
            manifest['files']['amended_cases.private.jsonl'],amend_launch)
        versions[label]=dict(original=summary,amendment=dsummary)
        raw=replace_amended(raw,delta,CHANGES)
        review=replace_amended(review,dreview,CHANGES) if label=='p1' else []
        combined[label]=summarize(rows,raw,review,label)
        raw_by_label[label]=raw;review_by_label[label]=review
        for base in (parent,packet):
            for p in (base/'results'/label).glob('*.json*'):
                files[str(p.relative_to(base.parent)).replace('\\','/')]=sha(p)
    paired=[];counts=Counter();unit_counts={};strata={}
    by_id={r['id']:r for r in rows}
    for p,b,pdetail,bdetail in zip(raw_by_label['p1'],raw_by_label['step120_current'],
                                  combined['p1']['per_call'],combined['step120_current']['per_call']):
        for key in ('id','variant','order','prompt_text_sha256','prompt_token_sha256','prompt_tokens'):assert p[key]==b[key]
        row=by_id[p['id']];good=pdetail['correct'];oldgood=bdetail['correct']
        values=dict(calls=1,p1_correct=int(good),step120_correct=int(oldgood),gains=int(good and not oldgood),losses=int(oldgood and not good))
        counts.update(values)
        unit_counts.setdefault(row['unit']+'/'+row['split'],Counter()).update(values)
        if row['split']=='retention':strata.setdefault(row['retention_stratum'],Counter()).update(values)
        paired.append(dict(id=row['id'],variant=p['variant'],split=row['split'],p1_correct=good,step120_correct=oldgood))
    result=dict(packet_sha256=manifest['files']['cases.private.jsonl'],parent_packet_sha256=PARENT_SHA,
        new_model_calls=240,final_paired_calls=176,final_review_calls=44,
        remote_and_local_summaries_agree=True,independent_closed_json_parsing_passed=True,
        all_final_paired_prompt_tokens_match=True,amendment_only_replacement_verified=True,
        versions=versions,final=combined,paired_counts=dict(counts),
        unit_counts={k:dict(v) for k,v in unit_counts.items()},retention_strata={k:dict(v) for k,v in strata.items()},
        paired_calls=paired,raw_file_sha256=files,training_allowed=False,
        limitation='Model reviews require operator adjudication; this verifier does not release data. Two orders are correlated, development shares source rules, and eight retention cases share historical training source groups.')
    return result,raw_by_label,review_by_label


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--packet',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result,_,_=verify(a.parent,a.packet)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('versions','final','paired_calls','raw_file_sha256')},indent=2))

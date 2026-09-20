"""Audit completed authoring without changing frozen acceptance decisions."""
import argparse
import collections
import re
from pathlib import Path
from cpt_stage_b import read,save,sha,object_of,FORMS


def audit(packet_dir,run,out):
    import json
    packet=read(packet_dir/'sources.private.json');rows=read(run/'candidates.private.json')
    calls=[json.loads(x) for x in (run/'calls.private.jsonl').read_text(encoding='utf-8').splitlines()]
    groups={g['id']:g for g in packet['groups']}
    assert {r['id'] for r in rows}=={g+'-'+f for g in groups for f in FORMS} and len(rows)==32
    assert len(calls)==read(run/'summary.safe.json')['calls']
    stages=collections.Counter(c['stage'] for c in calls)
    assert stages['author']==32 and len({c['id'] for c in calls if c['stage']=='author'})==32
    rejected=collections.Counter(r.get('rejection','blind_review_disagreement') for r in rows if not r['blind_review_pass'])
    quote_causes=collections.Counter()
    for r in rows:
        if r.get('rejection')!='source quote':continue
        q=object_of(next(c['text'] for c in calls if c['id']==r['id'] and c['stage']=='author'))['source_quote']
        s=groups[r['group']]['source_text'];norm=lambda t:re.sub(r'\s+',' ',t).strip()
        category='whitespace_only' if norm(q) in norm(s) else 'contains_ellipsis' if re.search(r'\.{3}|…',q) else 'other_nonexact'
        quote_causes[category]+=1
    passed={g:sum(r['blind_review_pass'] for r in rows if r['group']==g) for g in groups}
    result=dict(calls=len(calls),stages=dict(stages),candidates=32,blind_review_pass=sum(passed.values()),group_pass_counts=passed,
                complete_groups=sum(v==4 for v in passed.values()),rejections=dict(rejected),quote_failure_surface_categories=dict(quote_causes),
                semantic_quality_of_rejected_tasks_not_established=True,diagnostic_calls=0,training_allowed=False,
                files_sha256={p.name:sha(p) for p in [packet_dir/'sources.private.json',run/'candidates.private.json',run/'calls.private.jsonl',run/'summary.safe.json']})
    save(out,result)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('packet-dir','run','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();audit(a.packet_dir,a.run,a.out)

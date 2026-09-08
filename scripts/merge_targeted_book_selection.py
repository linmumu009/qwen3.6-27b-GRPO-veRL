"""Merge frozen pretests after semantic screening and dev-only source retirement."""
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
from build_targeted_book_groups import digest,write
from select_targeted_sft_records import decision
from organize_targeted_book_pools import pool


def merge(kept,scored):
    if len({r['id'] for r in kept})!=len(kept):raise ValueError('Duplicate retained IDs')
    by_id={r['id']:r for r in scored}
    if len(by_id)!=len(scored):raise ValueError('Duplicate graded IDs')
    rows=[]
    for r in kept:
        s=by_id[r['id']]
        for key in ('question','variant','answer','rubric','source_id','source_hash','kind','concept_group'):
            if r[key]!=s[key]:raise ValueError('Prompt/rubric/source changed after scoring')
        if s['split']=='dev' and r['split']!='dev':raise ValueError('Cannot reuse dev as train')
        merged=dict(s,previous_split=s['split'],split=r['split'],semantic_screen_passed=True)
        merged['decision']=decision(merged);rows.append(merged)
    groups=defaultdict(list)
    for r in rows:groups[r['concept_group']].append(r)
    for qs in groups.values():
        if len(qs)!=3 or len({q['split'] for q in qs})!=1:raise ValueError('Incomplete or cross-split source group')
        # A bad item cannot be hidden by keeping its siblings under a group-quality claim.
        if any(q['decision']=='quality_not_passed' for q in qs):
            for q in qs:q['decision']='group_quality_quarantine'
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--semantic',type=Path,required=True)
    p.add_argument('--selection',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();os.umask(0o077)
    sm=json.loads((a.semantic/'summary.safe.json').read_text());f=a.semantic/'candidates.private.jsonl'
    if digest(f)!=sm['candidates_sha256']:raise ValueError('Changed semantic candidates')
    kept=list(map(json.loads,f.read_text().splitlines()));scored=[];hashes={}
    for path in a.selection:
        f=path/'selection.private.jsonl';hashes[str(path)]=digest(f)
        scored.extend(map(json.loads,f.read_text().splitlines()))
    rows=merge(kept,scored);a.output.mkdir(parents=True,exist_ok=False)
    with (a.output/'selection.private.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    summary={'items':len(rows),'input_graded_items':len(scored),'semantic_excluded_items':len(scored)-len(rows),
      'source_hashes':hashes,'semantic_candidates_sha256':sm['candidates_sha256'],
      'decisions':dict(Counter(r['decision'] for r in rows)),
      'pools':dict(Counter(pool(r) for r in rows)),
      'split_topic_pools':{split:{topic:dict(Counter(pool(r) for r in rows if r['split']==split and r['topic']==topic))
        for topic in sorted({r['topic'] for r in rows})} for split in ('train','dev')},
      'moved_train_to_dev_questions':sum(r['previous_split']=='train' and r['split']=='dev' for r in rows),
      'training_ready':False,'training_started':False,'semantic_screen_completed':True,
      'semantic_decontamination_proven':False,
      'limitation':'Automated group screen and opportunity proxy; final training mixture, token budget and independent evaluation not approved by this script.'}
    write(a.output/'summary.safe.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

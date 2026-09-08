"""High-precision missing-context screen; retain raw scores, quarantine flagged groups."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import hashlib

PATTERNS=[r'\baccording to (?:the |this |provided )?(?:source|text|textbook|book|passage|excerpt|reference)\b',
    r'\b(?:in|from|using|based on) (?:the |this )?(?:provided |given |above |supplied )?(?:source|text|textbook|passage|excerpt|reference material)\b',
    r'\b(?:provided|given|above|supplied) (?:source|text|passage|excerpt|reference|figure|table)\b',
    r'\bas (?:described|outlined|stated|explained|defined|discussed) in (?:the |this )?(?:source|text|textbook|book|passage|excerpt)\b']

def context_flags(question):return [i for i,p in enumerate(PATTERNS) if re.search(p,question,re.I)]

def verify(out):
    cases=json.loads((out/'cases.private.json').read_text());original={r['id']:r for r in cases}
    dest=out/'context_checked';expected=json.loads((dest/'summary.safe.json').read_text())['pools'];seen=set();question_ids=set();pools={}
    for pool in ('development','quarantine','learning_candidate','maintenance_candidate'):
        path=dest/(pool+'_groups.private.jsonl');raw=path.read_bytes();rows=[json.loads(x) for x in raw.decode().splitlines()]
        assert len(rows)==expected[pool]['groups']
        for row in rows:
            assert row['concept_group'] not in seen;seen.add(row['concept_group'])
            assert row['pool']==pool and len(row['questions'])==4
            assert (row['split']=='dev')==(pool=='development')
            for q in row['questions']:
                assert q==original[q['id']] and q['id'] not in question_ids
                assert q['concept_group']==row['concept_group'] and q['split']==row['split'];question_ids.add(q['id'])
        pools[pool]=dict(groups=len(rows),questions=4*len(rows),sha256=hashlib.sha256(raw).hexdigest())
    assert len(seen)==51 and question_ids==set(original)
    value=dict(groups=51,questions=204,pools=pools,duplicate_groups=0,duplicate_questions=0,changed_questions=0,split_migrations=0,training=False)
    (dest/'integrity.safe.json').write_text(json.dumps(value,indent=2));print(json.dumps(value,indent=2))

def main(out):
    cases=json.loads((out/'cases.private.json').read_text());result=json.loads((out/'reviewed/result.safe.json').read_text())
    flags=[dict(id=r['id'],concept_group=r['concept_group'],kind=r['kind'],split=r['split'],pattern_ids=context_flags(r['question'])) for r in cases if r['kind']!='evidence_selection' and context_flags(r['question'])]
    bad={r['id'] for r in flags};groups={r['concept_group'] for r in flags}
    # No raw metrics overwritten; this is an additional downstream quality gate.
    clean=[r for r in result['open_scores'] if r['id'] not in bad]
    summary={'flagged_questions':len(flags),'flagged_groups':len(groups),'by_kind':dict(Counter(r['kind'] for r in flags)),
      'by_split':dict(Counter(r['split'] for r in flags)),'flags':flags,'patterns':PATTERNS,
      'clean_agreed_questions':len({r['id'] for r in clean}),
      'clean_scores':[{k:r[k] for k in ('id','model','kind','split','closed_score','evidence_score')} for r in clean],
      'clean_aggregate':[dict(model=model,items=sum(r['model']==model for r in clean),closed_correct=sum(r['model']==model and r['closed_score']==2 for r in clean),evidence_correct=sum(r['model']==model and r['evidence_score']==2 for r in clean)) for model in ('step120','cpt')],
      'limit':'Explicit cue screen, not exhaustive standalone-question semantic certification. Flagged items cannot support closed-book knowledge claims.'}
    destination=out/'context_checked';destination.mkdir(exist_ok=False);pools={}
    for pool in ('development','quarantine','learning_candidate','maintenance_candidate'):
        path=out/'reviewed'/(pool+'_groups.private.jsonl')
        pools[pool]=[json.loads(x) for x in path.read_text().splitlines()]
    moved=[]
    for pool in ('learning_candidate','maintenance_candidate'):
        retained=[]
        for row in pools[pool]:
            if row['concept_group'] in groups:
                row['previous_pool']=pool;row['pool']='quarantine';row['context_dependency_flag']=True;moved.append(row)
            else:retained.append(row)
        pools[pool]=retained
    pools['quarantine'].extend(moved)
    for pool,rows in pools.items():
        for row in rows:row['context_dependency_flag']=row['concept_group'] in groups
        with (destination/(pool+'_groups.private.jsonl')).open('x') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
    summary['moved_groups']=len(moved);summary['pools']={k:dict(groups=len(v),questions=4*len(v)) for k,v in pools.items()}
    summary['training']=False;summary['training_ready']=False
    (destination/'summary.safe.json').write_text(json.dumps(summary,indent=2))
    (out/'status.txt').write_text('context_checked_value_analysis_complete_no_training\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('flags','clean_scores')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    verify(a.out) if a.verify_only else main(a.out)

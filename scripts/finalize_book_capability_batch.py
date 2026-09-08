"""Explicit review holds and frozen, balanced target-probe sample. No training."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from build_book_capability_batch import save,HELDOUT
from build_targeted_book_groups import digest

HOLDS={
 'handbook8e-00046-cap-Q3':'Battery ventilation statement is insufficiently scoped for general safety-related QA; source fidelity alone is not current safety validation.',
 'handbook8e-00003-cap-Q6':'Reuses source table numbers with ambiguous extracted alignment rather than a newly specified hypothetical calculation.',
 'handbook8e-00088-cap-Q6':'Source-specific illustrative fleet margin is described as a standard guideline.',
 'handbook8e-00102-cap-Q3':'Degraded or delayed inputs are overgeneralized into impossibility of useful planning.',
 'handbook8e-00021-cap-Q1':'Specific demand-production decoupling buffer is generalized to the only point where buffer inventory is needed.',
}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    source=a.out/'batch_review2';meta=json.loads((source/'summary.safe.json').read_text());path=source/'candidates.private.jsonl'
    assert digest(path)==meta['candidates_sha256']
    qs=[json.loads(x) for x in path.read_text().splitlines()];kept=[q for q in qs if q['id'] not in HOLDS]
    dest=a.out/'organized';dest.mkdir(exist_ok=False)
    for split in ('train','dev'):
        rows=[q for q in kept if q['split']==split]
        with (dest/(split+'_candidates.private.jsonl')).open('x') as f:
            for q in rows:f.write(json.dumps(q)+'\n')
    sample=[]
    for split,limit in [('train',6),('dev',2)]:
        for kind in sorted({q['kind'] for q in kept}):
            choices=[q for q in kept if q['split']==split and q['kind']==kind]
            sample.extend(sorted(choices,key=lambda q:hashlib.sha256(('probe-v1-'+q['id']).encode()).hexdigest())[:limit])
    save(dest/'probe_sample.private.json',sample)
    assert len(kept)==len({q['id'] for q in kept})
    assert all((q['chapter'] in HELDOUT)==(q['split']=='dev') for q in kept)
    value=dict(input_candidates=len(qs),additional_held=sum(q['id'] in HOLDS for q in qs),retained_questions=len(kept),split_counts=dict(Counter(q['split'] for q in kept)),kind_counts=dict(Counter(q['kind'] for q in kept)),chapter_count=len({q['chapter'] for q in kept}),modes=dict(Counter(q['mode'] for q in kept)),holds=HOLDS,probe_sample_questions=len(sample),probe_sample_split=dict(Counter(q['split'] for q in sample)),probe_sample_sha256=digest(dest/'probe_sample.private.json'),files={s:dict(sha256=digest(dest/(s+'_candidates.private.jsonl')),questions=sum(q['split']==s for q in kept)) for s in ('train','dev')},training=False,training_ready=False,target_probe_completed=False,manual_inspected=16,manual_sample_representative=False)
    save(dest/'summary.safe.json',value);print(json.dumps(value,indent=2))

if __name__=='__main__':main()

"""Prepare frozen forty-item sample; reuse bounded inference and anonymous grading."""
import argparse
from collections import Counter
import hashlib
import os
from pathlib import Path
from probe_book_repair_value import run,grade
from probe_book_task_aligned_value import MODEL_PATHS,read,write,digest,BOOK_SHA,source_units

POLICY='''IMPORTANT OVERRIDE: Source absence alone is NOT proof a student's extra claim is false. Only penalize a material contradiction, an incorrect calculation, an unjustified universal claim, or omission of a genuinely requested necessary point. Semantically equivalent terms and reasonable examples need not match the book's example list. If the frozen rubric requires unasked specific examples/list entries or incidental figures, mark rubric_valid=false instead of enforcing them. Independently recompute calculations. Judge the capability actually requested, not closeness to reference prose. Do not silently rewrite the rubric.'''

def prepare(root,out):
    path=root/'runs/book-capability-batch-20260908-01/organized/probe_sample.private.json'
    assert digest(path)=='1cd978fa46c40fb5622c773fe1244280076d6abebadcaafd09c6da6199f3c60f'
    cases=read(path);assert len(cases)==len({q['id'] for q in cases})==40
    assert Counter(q['split'] for q in cases)=={'train':30,'dev':10}
    assert all(q['mode']=='closed' for q in cases)
    assert set(Counter(q['kind'] for q in cases).values())=={8}
    source=root/'runs/logistics-cpt-20260903/private/handbook8e_cpt_4096.jsonl';assert digest(source)==BOOK_SHA
    import json
    sources={r['record_id']:r['text'] for r in map(json.loads,source.read_text().splitlines())}
    for q in cases:
        text=sources[q['source_id']];assert hashlib.sha256(text.encode()).hexdigest()==q['source_hash']
        units=source_units(text);assert q['reference_units']=={i:units[i] for i in q['source_ids']}
    out.mkdir(parents=True,exist_ok=False);write(out/'cases.private.json',cases)
    write(out/'protocol.safe.json',dict(items=40,models=MODEL_PATHS,cases_sha256=digest(out/'cases.private.json'),source_sample_sha256=digest(path),generations_per_model=80,generations_total=160,max_output_tokens=512,temperature=0,seed=1024,thinking=False,repeats=1,judge_repeats=2,max_api_calls=80,judge_policy_addendum=POLICY,training=False,script_sha256=digest(Path(__file__)),sampling='Frozen balanced sample: six train and two dev per kind; not population-weighted accuracy.'))

if __name__=='__main__':
    os.umask(0o077);p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','grade']);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--model',choices=list(MODEL_PATHS));p.add_argument('--api-config',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare(a.root,a.out)
    elif a.action=='run':run(a.root,a.out,a.model)
    else:grade(a.out,a.api_config)

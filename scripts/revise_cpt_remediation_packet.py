"""Preserve the first candidate packet and explicitly repair four wording ambiguities."""
import argparse
from collections import Counter
import json
from pathlib import Path
from prepare_cpt_remediation_packet import sha, validate_row

PARENT_SHA='9f764dd3c60e4cefac8319be2d48a80bfe6d3a6df41eb048fd919fb0d06bafde'
CHANGES={
    'remediation-pick_lines-multi_instruction_reconciliation': (
        'question',
        'A completed pick list has three separate instructions executed in order:',
        'R and S name two distinct physical bin locations. A completed pick list has three separate instructions executed in order:',
        'The original symbols could denote SKUs rather than physical bins; unique locations must be stated explicitly.'),
    'remediation-pick_lines-recover_instruction_count_from_events': (
        'question',
        'Every completed instruction produced at least one event, and no event belongs to two instructions.',
        'Every recorded event belongs to exactly one completed instruction; no events from unfinished instructions are included. Every completed instruction produced at least one recorded event.',
        'Completeness in the forward direction alone does not exclude events from unfinished instructions; distinct IDs must correspond exactly to completed instructions.'),
    'remediation-serial_parallel-deadline_and_labor_budget': (
        'question',
        'Three feasible process designs start at 0.',
        'Three proposed process designs start at 0; their compliance with the deadline and labor budget must be checked.',
        'Calling all designs feasible before checking the two constraints conflicts with the intended distinction between S, P and R.'),
    'remediation-batch_tradeoff-exclusive_interventions_with_fees': (
        'options',
        'J is the cheaper of the two allowed interventions.',
        'Choosing J yields a lower total batching cost, including its fee, than choosing I.',
        'Cheaper intervention could mean its fee alone; the intended comparison is the complete resulting batching cost.')}


def amended_row(row):
    changed=dict(row)
    if row['id'] not in CHANGES:return changed
    field,before,after,_=CHANGES[row['id']]
    if field=='question':
        assert row[field].count(before)==1
        changed[field]=row[field].replace(before,after)
    else:
        assert field=='options' and row[field].count(before)==1
        changed[field]=[after if option==before else option for option in row[field]]
    return changed


def revise(parent,out):
    assert sha(parent/'cases.private.jsonl')==PARENT_SHA
    manifest=json.loads((parent/'manifest.safe.json').read_text(encoding='utf-8'))
    for name,h in manifest['files'].items():assert sha(parent/name)==h
    rows=[json.loads(x) for x in (parent/'cases.private.jsonl').read_text(encoding='utf-8').splitlines()]
    changes=[]
    for i,row in enumerate(rows):
        if row['id'] in CHANGES:
            field,before,after,reason=CHANGES[row['id']]
            row=amended_row(row);rows[i]=row
            changes.append(dict(id=row['id'],field=field,before=before,after=after,reason=reason,
                answer_unchanged=True,trigger='Operator premise audit and substantive reviewer objections, not closed-book score selection'))
        validate_row(row)
    assert len(rows)==44 and len(changes)==4
    out.mkdir(exist_ok=False)
    collections=[('cases',rows)]+[(s,[r for r in rows if r['split']==s]) for s in ('train','dev','retention')]
    collections.append(('amended_cases',[r for r in rows if r['id'] in CHANGES]))
    for name,data in collections:
        (out/(name+'.private.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data),encoding='utf-8',newline='\n')
    manifest.update(files={p.name:sha(p) for p in out.glob('*.jsonl')},parent_packet_sha256=PARENT_SHA,
        revision='v3_explicit_premises_and_cost_boundary',amendments=changes,revision_code_sha256=sha(Path(__file__)),
        training_allowed=False)
    (out/'manifest.safe.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return {k:manifest[k] for k in ('files','parent_packet_sha256','amendments','training_allowed')}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(revise(a.parent,a.out),indent=2))

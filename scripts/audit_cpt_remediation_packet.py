"""Downstream exclusions and split-construction ledger; never an authoring input."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from cpt_transfer_draft import normalized_question
from prepare_cpt_remediation_packet import validate_row, sha


def records(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]


def audit(resources,packet):
    rows=records(packet/'cases.private.jsonl')
    manifest=json.loads((packet/'manifest.safe.json').read_text(encoding='utf-8'))
    for name,h in manifest['files'].items():assert sha(packet/name)==h
    for r in rows:validate_row(r)
    for split in ('train','dev','retention'):
        assert records(packet/(split+'.private.jsonl'))==[r for r in rows if r['split']==split]
    assert len({normalized_question(r) for r in rows})==len(rows)
    inventory=json.loads((resources/'llin-transfer-p1-20260916-03/exclusion_inputs.private.json').read_text())
    paths=[]
    for v in inventory:
        p=resources/v['path'];assert sha(p)==v['sha256'];paths.append(p)
    paths.extend(resources/p for p in [
        'llin-transfer-p1-20260916-03/cases.private.jsonl',
        'llin-transfer-audit-20260915-01/frozen_cases.private.jsonl',
        'llin-transfer-expansion-draft-20260917-01/filtered.private.jsonl',
        'llin-source-authoring-pilot-20260917-01/cases.private.jsonl'])
    excluded=set()
    def collect(o):
        if isinstance(o,dict):
            if isinstance(o.get('question'),str):excluded.add(normalized_question(o))
            for v in o.values():collect(v)
        elif isinstance(o,list):
            for v in o:collect(v)
    for p in paths:
        for r in records(p):collect(r)
    matches=[r['id'] for r in rows if normalized_question(r) in excluded]
    assert not matches, matches
    # Similarity only nominates pairs for scrutiny; it is not a semantic proof.
    def terms(row):
        stop=set('use the a an and of to in at is are for with by from as all correct statements source archived warehouse distribution science terminology concepts picking discussion apply'.split())
        return set(re.findall('[a-z]+',row['question'].lower()))-stop
    pairs=[]
    for d in [r for r in rows if r['split']=='dev']:
        candidates=[]
        for t in [r for r in rows if r['split']=='train']:
            x,y=terms(d),terms(t);score=len(x&y)/len(x|y)
            candidates.append((score,t['id']))
        score,nearest=max(candidates)
        pairs.append(dict(dev_id=d['id'],nearest_train_id=nearest,word_jaccard=round(score,4),
            intended_distinction=d['reasoning_requirement'],
            scope='Shared source rules and arithmetic primitives; held-out construction, not independent knowledge.'))
    return dict(packet_sha256=sha(packet/'cases.private.jsonl'),cases=len(rows),
        arithmetic_equalities_checked=sum(len(r['arithmetic_checks']) for r in rows),
        split_files_reconstructed=True,source_and_option_schema_verified=True,
        excluded_normalized_questions=len(excluded),exact_or_number_normalized_matches=matches,internal_duplicates=0,
        exclusion_inputs=[dict(path=str(p.relative_to(resources)).replace('\\','/'),sha256=sha(p)) for p in paths],
        development_structure_ledger=pairs,training_allowed=False,
        limitations='Mechanical checks and a construction ledger do not alone certify semantic independence. Current16 diagnostic tasks remain excluded from training. No sealed pilot items read. Labels still require source-by-source operator review.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resources',type=Path,required=True);p.add_argument('--packet',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=audit(a.resources,a.packet);a.out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in r.items() if k not in ('exclusion_inputs','development_structure_ledger')},indent=2))

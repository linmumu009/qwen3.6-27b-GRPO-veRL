"""Fail-closed selection of source-grounded QA using frozen target-model probes.

This gate assigns learning-opportunity categories, NOT causal training value.
It never consumes benchmark question text or benchmark labels.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path


def decision(row):
    if row.get('split') not in ('train','dev'):return 'invalid_split'
    quality=row.get('quality',{})
    required=('source_verified','standalone','answer_correct','conditions_complete',
              'no_benchmark_derivation','decontamination_pass','dedup_pass','group_split_frozen')
    if any(quality.get(k) is not True for k in required):return 'quality_not_passed'
    if not all(isinstance(row.get(k),str) and row[k].strip() for k in ('id','concept_group','source_hash')):
        return 'missing_provenance'
    if row['split']=='dev':return 'development_only'
    diagnostic=row.get('target_probe',{})
    if diagnostic.get('model')!='pure_book_CPT4x_step116':return 'wrong_or_missing_target'
    closed=diagnostic.get('closed_scores'); opened=diagnostic.get('open_scores')
    if not isinstance(closed,list) or not isinstance(opened,list) or len(closed)!=2 or len(opened)!=2:
        return 'incomplete_probe'
    if any(type(x) is not int or x not in (0,1,2) for x in closed+opened):return 'invalid_scores'
    if diagnostic.get('all_natural_stop') is not True:return 'generation_failure'
    if diagnostic.get('judge_evidence_valid') is not True:return 'unverified_judge'
    if closed==[2,2]:return 'maintenance_pool'
    if 2 in closed:return 'unstable_review'
    if opened!=[2,2]:return 'unresolved_review'
    if row.get('kind') in ('application','boundary','counterfactual','comparison'):
        if diagnostic.get('prerequisite_closed_pass') is True:return 'application_opportunity'
        return 'mixed_opportunity'
    return 'knowledge_access_opportunity'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();os.umask(0o077)
    rows=[json.loads(s) for s in a.input.read_text().splitlines()]
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate IDs')
    groups={}
    for r in rows:
        if r.get('concept_group'):
            groups.setdefault(r['concept_group'],set()).add(r.get('split'))
    if any(len(s)!=1 for s in groups.values()):raise ValueError('Concept group split leakage')
    # A manifest for inspection, deliberately NOT an automatic training export.
    a.output.mkdir(parents=True,exist_ok=False)
    selected=[{'id':r['id'],'decision':decision(r)} for r in rows]
    (a.output/'decisions.private.json').write_text(json.dumps(selected,indent=2))
    summary={'items':len(rows),'counts':dict(Counter(r['decision'] for r in selected)),
        'training_exported':False,'training_started':False,
        'value_interpretation':'Learning opportunity proxy; incremental training utility requires a controlled training comparison.'}
    (a.output/'summary.safe.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary))


if __name__=='__main__':main()

"""Downstream verification. Diagnostic text never flows back into the compiler.

Independent arithmetic implementation, not an independent human/source expert.
Lexical exclusion checks do not certify semantic novelty or authorize training.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from build_cpt_premise_tasks import FAMILIES, compile_task, prefix
from prepare_cpt_transfer_screen import SOURCE_SHA, historical_groups, sha

EXCLUSION_FILES = (
    'llin-composed-boundaries-20260916-01/cases.private.jsonl',
    'llin-coverage-expansion-20260916-01/cases.private.jsonl',
    'llin-reviewed-scenario-probe-20260916-01/cases.private.jsonl',
    'llin-transfer-screen-20260916-02/candidates.private.jsonl',
    'llin-transfer-screen-20260916-03/candidates.private.jsonl',
    'llin-transfer-draft-20260916-01/pilot_result/candidates.private.jsonl',
    'llin-transfer-constraint-pilot-20260916-01/result/candidates.private.jsonl',
    'llin-s4-probe-20260915-01/cases.private.jsonl',
    'llin-s4-probe-20260915-02/cases.private.jsonl',
    'llin-s4-transfer-diagnostic-20260915-01/cases.private.jsonl',
    'llin-s4-conditional-train-20260915-01/tasks.private.jsonl',
    'llin-s5-contrast-20260915-01/tasks.private.jsonl',
    'llin-source-reviewed-20260914-01/cases.private.jsonl',
    'llin-source-expansion-20260914-01/filtered_tasks.private.jsonl',
)


def records(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def reference_totals(family, rows):
    """Vector/filter implementation; does not call the compiler's measure()."""
    if family=='rail_speed':
        categories=[]
        for row in rows:
            kind=row['build'];speed=row['main_design_kmh']
            if kind=='ordinary_conventional':category='N'
            elif kind=='upgraded_conventional':category='N' if speed<200 else 'U'
            else:category='N' if speed<250 else 'D'
            categories.append(category)
        counts=Counter(categories)
        return {'D':counts['D'],'U':counts['U'],'N':counts['N'],'eligible':len(rows)-counts['N']}
    eligible=[r for r in rows if r['mode']=='main_rail' and not r['heritage_only']
              and (r['open_to_public_traffic'] or not r['constructed_solely_for_installation'])]
    route=sum(r['length_km'] for r in eligible)
    track=sum(sum([r['length_km']]*r['tracks']) for r in eligible)
    return dict(route=route,track=track,included=len(eligible),extra=track-route)


def verify_calculation(task):
    spec=task['premise_spec'];family=spec['family'];operation=spec['operation']
    # Also binds the option wording, shuffle, labels and trace to the frozen spec.
    assert compile_task(spec)==task,'render/proof/label mismatch'
    if operation in ('aggregate','ledger'):
        totals=reference_totals(family,spec['rows'])
        assert Counter((p['actual'],p['claimed'],p['true']) for p in task['option_proofs'])==Counter(
            (totals[k],v,totals[k]==v) for k,v in spec['claims'])
    elif operation=='inverse':
        candidates=[]
        pos,field=spec['unknown']
        for value,_ in spec['values']:
            rows=deepcopy(spec['rows']);rows[pos][field]=value
            candidates.append(reference_totals(family,rows))
        expected=Counter(json.dumps(x,sort_keys=True) for x in candidates)
        assert expected==Counter(json.dumps(p['actual'],sort_keys=True) for p in task['option_proofs'])
        for p in task['option_proofs']:
            assert p['true']==(all(p['actual'][k]==v for k,v in spec['target'].items()))
    elif operation=='minimum_cost':
        candidates=[]
        for package in spec['packages']:
            rows=deepcopy(spec['rows'])
            for pos,field,value in package['changes']:rows[pos][field]=value
            totals=reference_totals(family,rows)
            candidates.append((package['cost'],totals,all(totals[k]==v for k,v in spec['target'].items())))
        best=sorted(cost for cost,_,ok in candidates if ok)[0]
        expected=Counter((cost,json.dumps(totals,sort_keys=True),ok and cost==best) for cost,totals,ok in candidates)
        assert expected==Counter((p['cost'],json.dumps(p['actual'],sort_keys=True),p['true']) for p in task['option_proofs'])
    else:
        assert operation=='contingent'
        totals={b:reference_totals(family,rows)[spec['metric']] for b,rows in spec['branches'].items()}
        for p in task['option_proofs']:
            assert p['actual']==totals and p['true']==(p['claimed']==totals)


def normalize(question):
    return ' '.join(re.findall(r'[a-z]+|#',re.sub(r'\d+(?:\.\d+)?','#',question.lower())))


def collect_questions(value, parent=''):
    result=[]
    if isinstance(value,dict):
        identity=value.get('id',parent)
        if isinstance(value.get('question'),str):result.append((identity,value['question']))
        for item in value.values():result.extend(collect_questions(item,identity))
    elif isinstance(value,list):
        for item in value:result.extend(collect_questions(item,parent))
    return result


def verify(packet,source,excluded_paths,group_script):
    manifest=json.loads((packet/'manifest.safe.json').read_text(encoding='utf-8'))
    assert sha(source)==manifest['source_sha256']==SOURCE_SHA
    assert sha(packet/'tasks.private.jsonl')==manifest['task_sha256']
    assert sha(packet/'premises.private.jsonl')==manifest['premise_sha256']
    assert sha(Path(__file__).with_name('build_cpt_premise_tasks.py'))==manifest['compiler_sha256']
    tasks=records(packet/'tasks.private.jsonl');specs=records(packet/'premises.private.jsonl')
    assert len(tasks)==20 and len({t['id'] for t in tasks})==20
    assert [t['premise_spec'] for t in tasks]==specs
    assert manifest['training_allowed'] is False and manifest['training_ready'] is False
    for split,n in [('train',12),('dev_expression',2),('dev_conditions',4),('sealed',2)]:
        subset=[t for t in tasks if t['split']==split]
        assert len(subset)==n and subset==records(packet/(split+'.private.jsonl'))
    original=records(source);by_id={r['id']:r for r in original}
    sources=json.loads((packet/'sources.private.json').read_text(encoding='utf-8'))
    groups=historical_groups(original,group_script)
    frozen={g for g in groups.values() if int(hashlib.sha256(g.encode()).hexdigest(),16)%5==0}
    topics={t for r in original if groups[r['id']] in frozen for t in r['topics']}
    for family,ids in FAMILIES.items():
        assert [s['id'] for s in sources[family]]==ids
        for s in sources[family]:
            assert s=={k:by_id[s['id']][k] for k in ('id','title','scope','source_text','evidence','topics')}
            assert groups[s['id']] not in frozen and not set(s['topics'])&topics
    bindings=json.loads((packet/'source_bindings.private.json').read_text(encoding='utf-8'))
    assert len(bindings)==9
    for family,index,quote in bindings:assert quote in sources[family][index]['source_text']
    for task in tasks:
        assert task['training_allowed'] is False and task['model_baseline_queried'] is False
        verify_calculation(task)
    excluded=[];inputs=[]
    for path in excluded_paths:
        # Missing registered exclusion files fail closed, rather than silently shrinking the corpus.
        questions=collect_questions(records(path))
        excluded.extend((str(path),identifier,q) for identifier,q in questions)
        inputs.append(dict(path=str(path),sha256=sha(path),questions=len(questions)))
    old_signatures={normalize(q) for _,_,q in excluded}
    collisions=[t['id'] for t in tasks if normalize(t['question']) in old_signatures]
    new_signatures=Counter(normalize(t['question']) for t in tasks)
    duplicates=sum(n-1 for n in new_signatures.values() if n>1)
    # Screening aid only: no score threshold is interpreted as semantic clearance.
    nearest=[]
    for t in tasks:
        words=set(normalize(t['question'].removeprefix(prefix(t['unit']))).split())
        ranked=[]
        for path,identifier,q in excluded:
            other=set(normalize(q).split());score=len(words&other)/len(words|other)
            ranked.append((score,path,identifier))
        nearest.append(dict(id=t['id'],nearest=[dict(score=s,path=p,id=i) for s,p,i in sorted(ranked,reverse=True)[:3]]))
    summary=dict(task_sha256=manifest['task_sha256'],compiler_sha256=manifest['compiler_sha256'],
        verifier_sha256=sha(Path(__file__)),tasks_verified=len(tasks),option_calculations_verified=80,
        exact_source_bindings=9,old_dev_group_or_topic_violations=0,
        historical_exclusion_files=len(inputs),historical_question_occurrences=len(excluded),
        historical_unique_number_normalized_questions=len(old_signatures),
        number_normalized_historical_matches=collisions,within_packet_number_normalized_duplicates=duplicates,
        per_unit_splits={f:dict(Counter(t['split'] for t in tasks if t['unit']==f)) for f in FAMILIES},
        training_allowed=False,released_training_items=0,released_complete_units=0,
        independent_human_expert_review=False,semantic_novelty_certified=False,
        caveats=['Two dev_expression items use a ledger instead of prose but retain the same aggregation operator and option template; format transfer only.',
                 'Train aggregation and historical railway probes share operations and rules; lexical nonmatching is not evidence of semantic independence.',
                 'Inverse and minimum-cost designs change the inference task; reserved branch policies remain unqueried. These are candidate strata, not measured transfer.',
                 'All items use four sources in one historical group. They add no new candidate rule families or broad retention items.',
                 'The verifier is separately implemented by the same author. It verifies arithmetic and reconstruction, not independent expert agreement.'])
    return summary,dict(exclusion_inputs=inputs,nearest_question_screen=nearest)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--packet',required=True,type=Path)
    p.add_argument('--source',required=True,type=Path);p.add_argument('--private-root',type=Path,default=Path('CPT_resources'))
    p.add_argument('--group-script',type=Path,default=Path('scripts/generate_source_condition_tasks.py'))
    p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    summary,private=verify(a.packet,a.source,[a.private_root/f for f in EXCLUSION_FILES],a.group_script)
    a.out.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (a.packet/'downstream_audit.private.json').write_text(json.dumps(private,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

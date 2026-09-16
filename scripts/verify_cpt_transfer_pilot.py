"""P1 source/calculation/split release gate, separate from task construction."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from build_cpt_transfer_pilot import UNITS, RETENTION, forward, conditional, sealed, retention_task, describe
from prepare_cpt_transfer_screen import sha
from verify_cpt_premise_tasks import EXCLUSION_FILES, collect_questions, normalize


def read(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def reference(unit,r):
    # Separate predicates and arithmetic, without invoking the builder oracle.
    if unit=='speed':
        if r['build']=='upgraded':return ['N','U'][r['design']>=200]
        if r['build']=='ordinary':return 'N'
        return ['N','D'][r['design']>=250]
    if unit=='railcar':return sum([1]*r['powered']+[2]*r['pairs'])
    if unit=='derailment':return {(False,False):'neither',(False,True):'derailment',(True,False):'collision',(True,True):'collision'}[(r['collision'],r['wheels']>0)]
    if unit=='injury':return 'excluded' if r['hours']<=24 or not r['accident'] or r['suicide'] else 'included'
    if unit=='capacity':return 'excluded' if r['capacity']<20 or r['design']!='inland freight' else 'included'
    if unit=='propulsion':return 'excluded' if r['tanker'] or r['propulsion'] in ('none','auxiliary-only') else 'included'
    if unit=='finance':return {'alone':'supplier','approved':'OEM'}[r['route']]
    if unit=='outsourcing':return ['overall logistics concept','quantified required services','placement policy','tendering','offer evaluation','service and cost assessment'][r['completed']]
    if unit=='pipeline':return r['length']*int(r['active'])*int(r['national'])
    if unit=='wave':
        eligible={'G'}|({'F'} if r['fragile'] else set())|({'P'} if r['priority'] else set())
        return sorted(eligible,key=r['order'].index)[0]
    raise ValueError(unit)


def check(task):
    unit=task['unit'];split=task['split'];idx=int(task['id'].rsplit('-',1)[1])
    if split=='retention':
        assert task==retention_task(unit,idx)
        n=idx+2
        expected={'teu':40*n+17,'tkm':44*n+26,'offered':800+40*n,'pallet':800*1200*n,
            'reorder':106+9*n,'capital':1000+70*n,'moment':80+3*n,'payload':3800+100*n}.get(unit)
        if unit=='abc':expected=['very tight control and accurate records','less tight control and moderate records','simplest controls and minimal records'][idx]
        if unit=='packages':expected=['package','overpack','cargo transport unit','carrier','consignee'][idx]
        assert task['options'][task['correct_indices'][0]]==str(expected)
        return
    rebuilt=forward(unit,split,idx) if split in ('train','dev_expression') else conditional(unit,idx) if split=='dev_conditions' else sealed(unit)
    assert task==json.loads(json.dumps(rebuilt)),'task wording or label changed'
    p=task['premises'];actual=[reference(unit,r) for r in p['rows']]
    if task['operation']=='forward':
        for option in task['options']:
            name,claim=option.split(': the result is ');claim=claim[:-1]
            j=p['names'].index(name)
            assert (task['options'].index(option) in task['correct_indices'])==(str(actual[j])==claim)
    elif task['operation']=='inverse':
        for i,option in enumerate(task['options']):
            j=[describe(unit,r,1) for r in p['rows']].index(option)
            assert (i in task['correct_indices'])==(actual[j]==p['target'])
    elif task['operation']=='minimum_cost':
        best=min(c for c,v in zip(p['costs'],actual) if v==p['target'])
        for i,option in enumerate(task['options']):
            cost=int(re.search(r'Total cost (\d+)\.',option)[1])
            j=p['costs'].index(cost)
            assert (i in task['correct_indices'])==(actual[j]==p['target'] and cost==best)
    else:
        correct=f'L => {actual[0]}; R => {actual[1]}.'
        assert task['options'][task['correct_indices'][0]]==correct


def audit(packet,reviewed,source,private_root):
    from prepare_cpt_transfer_screen import SOURCE_SHA,historical_groups
    import hashlib
    manifest=json.loads((packet/'manifest.safe.json').read_text())
    assert manifest['source_sha256']==sha(source)==SOURCE_SHA
    assert manifest['builder_sha256']==sha(Path(__file__).with_name('build_cpt_transfer_pilot.py'))
    for name,digest in manifest['files'].items():assert sha(packet/name)==digest
    tasks=read(packet/'tasks.private.jsonl')
    for t in tasks:check(t)
    for split,count in manifest['counts'].items():
        rows=[t for t in tasks if t['split']==split]
        assert len(rows)==count and read(packet/(split+'.private.jsonl'))==rows
    sources=read(source);by_id={s['id']:s for s in sources}
    groups=historical_groups(sources,Path(__file__).with_name('generate_source_condition_tasks.py'))
    frozen={g for g in groups.values() if int(hashlib.sha256(g.encode()).hexdigest(),16)%5==0}
    frozen_topics={t for s in sources if groups[s['id']] in frozen for t in s['topics']}
    old=[r for r in read(reviewed) if r['split']=='train'];assert len(old)==135
    for r in old:
        assert groups[r['unit_id']] not in frozen and not set(by_id[r['unit_id']]['topics'])&frozen_topics
    for t in tasks:
        for sid in t['source_ids']:
            if t['split']=='retention':assert groups[sid] in frozen
            else:assert groups[sid] not in frozen and not set(by_id[sid]['topics'])&frozen_topics
    oldq=[];corpus=[]
    for name in EXCLUSION_FILES:
        path=private_root/name;questions=collect_questions(read(path));oldq.extend(q for _,q in questions)
        corpus.append(dict(path=name,sha256=sha(path),questions=len(questions)))
    oldnorm={normalize(q) for q in oldq}
    matches=[t['id'] for t in tasks if normalize(t['question']) in oldnorm]
    assert not matches,'historical number-normalized exact reuse'
    # Source concepts are intentionally shared. This only rejects identical tasks;
    # same rule/operator is allowed in the prospective expression stratum.
    traintext={t['question'] for t in tasks if t['split']=='train'}
    assert not any(t['question'] in traintext for t in tasks if t['split']!='train')
    assert len({t['id'] for t in tasks})==190
    result=dict(version='P1',training_allowed=True,authorization='User accepted prospective limited cross-category mechanism trial before broad-scale gate.',
        packet_sha256=sha(packet/'tasks.private.jsonl'),source_sha256=SOURCE_SHA,reviewed_sha256=sha(reviewed),
        builder_sha256=manifest['builder_sha256'],verifier_sha256=sha(Path(__file__)),
        checked_tasks=190,checked_options=760,new_training_tasks=60,new_training_exposures=180,old_retention_training_tasks=135,
        total_training_records=315,steps=105,lr=5e-7,batch=3,epochs=1,fresh_optimizer=True,
        evaluation_counts={'train':60,'dev_expression':20,'dev_conditions':20,'retention':80},sealed_unqueried=10,
        historical_exclusion_files=len(corpus),historical_question_occurrences=len(oldq),historical_exact_normalized_matches=matches,
        old_dev_training_violations=0,expert_independent_review=False,
        semantic_review='Author source review plus separate reference calculations. Explicit scopes/causal premises reviewed; source concepts shared intentionally. Parameterized training variants are not distinct knowledge groups.',
        limits=['Pilot sentinel retention: 80 tasks/10 rules/3 historical development groups, not full broad independent confirmation.',
                'Expression is near transfer with familiar fields and operator; conditions introduce inverse and least-cost selection. Both strata reported separately.',
                'Family selection used prior diagnostic outcomes, so selection bias remains; no original diagnostic item enters training.',
                'No official evaluation or C/D promotion is automatic from this pilot.'],
        decision_rule={'training':'Report exact gains/losses and family changes; if no training improvement, inspect implementation and supervision.',
                       'transfer':'Advance to broader data work only if each 20-item transfer stratum improves by at least 2 items across at least 2 families.',
                       'retention':'Require no net decline overall; report every family loss. Any net decline holds advancement for retention repair.',
                       'negative':'If both transfer strata fail the prospective signal rule, do not enlarge this exact recipe. No automatic hyperparameter search.',
                       'scope':'Exploratory screen, not statistical significance or the final +3pp target.'})
    (packet/'release.safe.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (packet/'exclusion_inputs.private.json').write_text(json.dumps(corpus,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('packet','reviewed','source'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--private-root',type=Path,default=Path('CPT_resources'));a=p.parse_args()
    print(json.dumps(audit(a.packet,a.reviewed,a.source,a.private_root),indent=2))

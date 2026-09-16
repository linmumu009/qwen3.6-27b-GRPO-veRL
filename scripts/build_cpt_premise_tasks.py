"""Source-bound task compiler: explicit premises -> truth tables -> rendered tasks.

No model generation and no diagnostic questions, predictions or labels are read.
This is a two-family construction pilot, not a complete training release.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from prepare_cpt_transfer_screen import SOURCE_SHA, historical_groups, sha

FAMILIES = {
    'rail_speed': ['llin-core-e18cbdfda5e0f7a00139', 'llin-core-ed190bcb6b41b42da590'],
    'rail_network_length': ['llin-core-4325923911580698ac94', 'llin-core-4de2e9a3634198597945'],
}
SCOPE = 'Use the Eurostat/UNECE/ITF Glossary for Transport Statistics, fifth edition (2019), for EU reporting of railway infrastructure. '
SPEED_SCOPE = ('Count only the dedicated and upgraded high-speed categories. N means neither of these two categories, '
    'not a conclusion about any other category. Main design speed is the designed capability of main segments; '
    'observed operating speed is a separate observation. Any connector is a short town-centre connecting segment. '
    'No upgraded line has a topographical, relief or town-planning speed-adaptation exception. ')
TRACK_SCOPE = ('Lengths are disjoint route-segment lengths; tracks gives the number of rail pairs on that segment. '
    'Every main-rail segment is maintained by an infrastructure manager for running trains. '
    'All assets are publicly owned and connected to the network, none is a private line or a stand-alone network, '
    'and no exclusion other than the explicitly recorded mode, original construction purpose, public-access status '
    'or exclusive heritage use applies. Present use and original construction purpose are separate facts. '
    'Urban light rail remains urban light rail even if heavy rail vehicles occasionally transit it. ')
METRICS = {
    'rail_speed': {'D':'number of dedicated high-speed lines', 'U':'number of upgraded high-speed lines',
                   'N':'number in neither of these two categories', 'eligible':'total number in the two categories'},
    'rail_network_length': {'route':'included railway route length (km)', 'track':'included cumulative track length (km)',
                            'included':'number of included segments', 'extra':'included track length minus route length (km)'},
}


def speed_row(name,build,design,observed,connector):
    return dict(name=name,build=build,main_design_kmh=design,observed_kmh=observed,connector_kmh=connector)


def track_row(name,length,tracks=1,mode='main_rail',built_installation=False,public=True,current_mine=False,heritage=False):
    return dict(name=name,length_km=length,tracks=tracks,mode=mode,constructed_solely_for_installation=built_installation,
        open_to_public_traffic=public,current_use_mine_only=current_mine,heritage_only=heritage)


def measure(family, rows):
    """Return a complete per-record decision trace as well as aggregate values."""
    if not rows or len({r['name'] for r in rows})!=len(rows):raise ValueError('empty or duplicate records')
    trace=[]
    if family=='rail_speed':
        totals=dict(D=0,U=0,N=0,eligible=0)
        for r in rows:
            if set(r)!=set(speed_row('', '', 0, 0, 0)):raise ValueError('incomplete speed premises')
            if r['build'] not in ('new_special','new_parallel','upgraded_conventional','ordinary_conventional'):raise ValueError('unknown construction')
            for k in ('main_design_kmh','observed_kmh','connector_kmh'):
                if type(r[k]) is not int or r[k]<0:raise ValueError('invalid speed')
            if max(r['observed_kmh'],r['connector_kmh'])>r['main_design_kmh']:raise ValueError('observed speed exceeds design capability')
            dedicated=r['build'] in ('new_special','new_parallel') and r['main_design_kmh']>=250
            upgraded=r['build']=='upgraded_conventional' and r['main_design_kmh']>=200
            category='D' if dedicated else 'U' if upgraded else 'N'
            totals[category]+=1;totals['eligible']+=category!='N'
            trace.append(dict(name=r['name'],category=category,construction=r['build'],main_design_kmh=r['main_design_kmh'],
                rule='new construction AND main design >=250' if category=='D' else 'conventional upgrade AND main design >=200' if category=='U' else 'neither conjunction holds'))
    elif family=='rail_network_length':
        totals=dict(route=0,track=0,included=0,extra=0)
        for r in rows:
            if set(r)!=set(track_row('',0)):raise ValueError('incomplete track premises')
            if r['mode'] not in ('main_rail','metro','tram','light_rail','water','road'):raise ValueError('unknown mode')
            if any(type(r[k]) is not bool for k in ('constructed_solely_for_installation','open_to_public_traffic','current_use_mine_only','heritage_only')):raise ValueError('missing boolean premise')
            if type(r['length_km']) is not int or r['length_km']<=0 or type(r['tracks']) is not int or r['tracks']<0:raise ValueError('invalid dimensions')
            if (r['mode'] in ('road','water'))!=(r['tracks']==0):raise ValueError('mode/rail-pair inconsistency')
            reasons=[]
            if r['mode']!='main_rail':reasons.append('excluded mode')
            if r['heritage_only']:reasons.append('solely heritage/tourist operation')
            if r['constructed_solely_for_installation'] and not r['open_to_public_traffic']:reasons.append('installation-only construction AND not open to public traffic')
            included=not reasons
            route=r['length_km'] if included else 0;track=route*r['tracks']
            totals['route']+=route;totals['track']+=track;totals['included']+=included
            trace.append(dict(name=r['name'],included=included,route_km=route,track_km=track,
                reasons=reasons or ['main railway; all registered exclusions absent']))
        totals['extra']=totals['track']-totals['route']
    else:raise ValueError('unknown family')
    return totals,trace


def render_rows(family,rows,ledger=False,missing=None):
    result=[]
    for i,r in enumerate(rows):
        def val(key):return 'UNRECORDED' if missing==(i,key) else r[key]
        if family=='rail_speed':
            construction={'new_special':'specially built new line','new_parallel':'new tracks added beside existing tracks that remain unupgraded',
                          'upgraded_conventional':'specially upgraded conventional line','ordinary_conventional':'conventional line that was neither specially built nor specially upgraded'}
            build='UNRECORDED' if missing==(i,'build') else construction[r['build']]
            fields=[build,f"main design {val('main_design_kmh')} km/h",f"observed operation {val('observed_kmh')} km/h",f"town-centre connector {val('connector_kmh')} km/h"]
        else:
            def yes(key):return 'UNRECORDED' if missing==(i,key) else ('yes' if r[key] else 'no')
            fields=[f"mode={val('mode')}",f"route length={val('length_km')} km",f"rail pairs={val('tracks')}",
                f"originally constructed solely for an industrial/agricultural installation={yes('constructed_solely_for_installation')}",
                f"open to public traffic={yes('open_to_public_traffic')}",f"current use solely mine freight={yes('current_use_mine_only')}",f"exclusively heritage/tourist operation={yes('heritage_only')}"]
        result.append(r['name']+(' | ' if ledger else ': ')+(' | ' if ledger else '; ').join(fields)+'.')
    return '\n'.join(result)


def prefix(family):return SCOPE+(SPEED_SCOPE if family=='rail_speed' else TRACK_SCOPE)


def compile_task(spec):
    family=spec['family'];operation=spec['operation'];rows=spec['rows']
    question=prefix(family);proof=[];options=[]
    if operation in ('aggregate','ledger'):
        metrics,trace=measure(family,rows)
        question+='Audit the following asset register.\n'+render_rows(family,rows,operation=='ledger')+'\nAssess the reported totals.'
        for metric,value in spec['claims']:
            options.append(f"The {METRICS[family][metric]} is {value}.")
            proof.append(dict(actual=metrics[metric],claimed=value,true=metrics[metric]==value,trace=trace))
    elif operation=='inverse':
        pos,field=spec['unknown']
        question+='One field in this register is unrecorded.\n'+render_rows(family,rows,missing=(pos,field))
        targets=spec['target'];question+='\nThe independently supplied report specifies '+', '.join(METRICS[family][k]+f' = {v}' for k,v in targets.items())+'. Select every completion consistent with all these totals.'
        for value,label in spec['values']:
            changed=deepcopy(rows);changed[pos][field]=value;actual,trace=measure(family,changed)
            options.append(label);proof.append(dict(actual=actual,target=targets,true=all(actual[k]==v for k,v in targets.items()),trace=trace))
    elif operation=='minimum_cost':
        question+='Start from this register.\n'+render_rows(family,rows)+'\nExactly one of the four packages below can be selected. Each is feasible, its listed cost is its entire cost, and all unmentioned facts remain unchanged. '
        targets=spec['target'];question+='The requirement is '+', '.join(METRICS[family][k]+f' = {v}' for k,v in targets.items())+'. Select every package meeting the requirement at the lowest cost among these four packages.'
        scores=[]
        for package in spec['packages']:
            changed=deepcopy(rows)
            for pos,field,value in package['changes']:changed[pos][field]=value
            actual,trace=measure(family,changed);ok=all(actual[k]==v for k,v in targets.items())
            scores.append((package['cost'],ok,actual,trace));options.append(package['label']+f" Total cost: {package['cost']} units.")
        cheapest=min(cost for cost,ok,_,_ in scores if ok)
        for cost,ok,actual,trace in scores:proof.append(dict(actual=actual,target=targets,feasible_for_target=ok,cost=cost,minimum_feasible_cost=cheapest,true=ok and cost==cheapest,trace=trace))
    elif operation=='contingent':
        question+='The audit has two mutually exclusive possible final records; the observed branch determines which report must be submitted.\n'
        branch_metrics={};branch_traces={}
        for branch,branch_rows in spec['branches'].items():
            question+='Branch '+branch+':\n'+render_rows(family,branch_rows)+'\n'
            branch_metrics[branch],branch_traces[branch]=measure(family,branch_rows)
        metric=spec['metric'];question+='Each option proposes the '+METRICS[family][metric]+' for both branches. Select every policy correct in both branches.'
        for policy in spec['policies']:
            options.append('; '.join(f'if branch {b}, report {v}' for b,v in policy.items())+'.')
            actual={b:m[metric] for b,m in branch_metrics.items()}
            proof.append(dict(actual=actual,claimed=policy,true=actual==policy,trace=branch_traces))
    else:raise ValueError('unregistered operation')
    assert len(options)==4 and len(set(options))==4
    order=list(range(4));random.Random(int(hashlib.sha256(spec['id'].encode()).hexdigest()[:16],16)).shuffle(order)
    options=[options[i] for i in order];proof=[proof[i] for i in order]
    answers=[i for i,p in enumerate(proof) if p['true']]
    assert 0<len(answers)<4
    return dict(id=spec['id'],unit=family,split=spec['split'],design=operation,
        question=question+'\nSelect all correct statements.',options=options,correct_indices=answers,
        premise_spec=spec,option_proofs=proof,training_allowed=False,purpose='structured_source_draft',
        model_baseline_queried=False,semantic_review='pending downstream source and split audit')


def specifications():
    specs=[]
    speed_catalog=[
        [speed_row('Aster','new_special',275,160,90),speed_row('Birch','upgraded_conventional',210,170,90),speed_row('Cedar','ordinary_conventional',270,170,90)],
        [speed_row('Dune','new_parallel',250,180,80),speed_row('Elm','new_special',249,150,70),speed_row('Fjord','upgraded_conventional',199,150,70)],
        [speed_row('Glen','upgraded_conventional',200,120,65),speed_row('Heath','upgraded_conventional',235,160,75),speed_row('Iris','new_special',240,180,85)],
        [speed_row('Juniper','new_special',300,140,60),speed_row('Kestrel','new_parallel',280,160,70),speed_row('Larch','upgraded_conventional',220,150,80),speed_row('Moor','ordinary_conventional',220,120,70)],
        [speed_row('Nacre','ordinary_conventional',300,180,90),speed_row('Opal','new_special',248,140,80),speed_row('Pine','new_parallel',252,160,85)],
        [speed_row('Quartz','new_parallel',265,110,55),speed_row('Reed','upgraded_conventional',205,110,55),speed_row('Spruce','new_special',255,110,55),speed_row('Tern','upgraded_conventional',215,110,55)],
    ]
    track_catalog=[
        [track_row('Amber',8,2),track_row('Blue',3,1,built_installation=True,public=False,current_mine=True),track_row('Copper',4,1,current_mine=True,public=False)],
        [track_row('Delta',7,3),track_row('Echo',2,1,heritage=True),track_row('Foxtrot',5,2,built_installation=True,public=True,current_mine=True)],
        [track_row('Granite',9,1,built_installation=True,public=False,current_mine=True),track_row('Hazel',4,2,mode='metro'),track_row('Indigo',6,2)],
        [track_row('Jade',6,1,mode='light_rail'),track_row('Khaki',5,3),track_row('Lilac',7,1,current_mine=True,public=False)],
        [track_row('Marble',4,2,built_installation=True,public=True),track_row('Nickel',8,1,built_installation=True,public=False),track_row('Ochre',3,1,heritage=True)],
        [track_row('Pearl',11,2),track_row('Russet',4,3),track_row('Silver',5,1,built_installation=True,public=False,current_mine=True)],
    ]
    # Values are derived only after premises are fixed. Vary the truth pattern;
    # no position or correct-answer cardinality is fixed across training items.
    masks=[(True,False,True,False),(False,True,False,False),(True,True,True,False),
           (False,True,True,False),(True,False,False,False),(True,False,True,True)]
    for family,catalog in [('rail_speed',speed_catalog),('rail_network_length',track_catalog)]:
        keys=list(METRICS[family])
        for i,rows in enumerate(catalog):
            totals,_=measure(family,rows)
            claims=[(k,totals[k]+(0 if valid else 1)) for k,valid in zip(keys,masks[i])]
            specs.append(dict(id=f'premise-v1-{family}-train-{i+1}',family=family,split='train',operation='aggregate',rows=rows,claims=claims))
        ledger=([speed_row('Upland','new_special',290,190,85),speed_row('Vale','upgraded_conventional',225,170,80),speed_row('Willow','ordinary_conventional',285,180,75),speed_row('Yew','new_special',245,170,70)] if family=='rail_speed' else
            [track_row('Teal',13,2),track_row('Umber',6,1,built_installation=True,public=True,current_mine=True),track_row('Violet',5,2,heritage=True),track_row('White',4,0,mode='water')])
        totals,_=measure(family,ledger)
        specs.append(dict(id=f'premise-v1-{family}-dev-expression',family=family,split='dev_expression',operation='ledger',rows=ledger,claims=[(k,totals[k]+(0 if i in (1,3) else 2)) for i,k in enumerate(keys)]))
    inverse_speed=[speed_row('Zinnia','new_special',260,180,90),speed_row('Acacia','upgraded_conventional',220,160,80),speed_row('Bramble','new_special',270,180,95)]
    specs.append(dict(id='premise-v1-rail_speed-dev-inverse',family='rail_speed',split='dev_conditions',operation='inverse',rows=inverse_speed,
        unknown=(2,'build'),target=dict(D=2,U=1,N=0),values=[('new_special','The unrecorded construction was a specially built new line.'),('new_parallel','It was new tracks beside existing tracks that remained unupgraded.'),('upgraded_conventional','It was a specially upgraded conventional line.'),('ordinary_conventional','It was an ordinary conventional line, neither specially built nor specially upgraded.')]))
    speed_base=[speed_row('Clover','new_special',240,180,85),speed_row('Dogwood','new_special',265,180,85)]
    specs.append(dict(id='premise-v1-rail_speed-dev-minimum',family='rail_speed',split='dev_conditions',operation='minimum_cost',rows=speed_base,target=dict(D=2),packages=[
        dict(label='Raise Clover main-segment design capability to 250 km/h.',changes=[(0,'main_design_kmh',250)],cost=8),
        dict(label='Replace Clover by a specially upgraded conventional line with main design capability 250 km/h.',changes=[(0,'build','upgraded_conventional'),(0,'main_design_kmh',250)],cost=3),
        dict(label='Raise Clover connector capability to 120 km/h only.',changes=[(0,'connector_kmh',120)],cost=1),
        dict(label='Raise Clover main-segment design capability to 270 km/h.',changes=[(0,'main_design_kmh',270)],cost=11)]))
    inverse_track=[track_row('Xanthic',5,2),track_row('Yellow',3,1),track_row('Zinc',4,1,built_installation=True,public=False)]
    specs.append(dict(id='premise-v1-rail_network_length-dev-inverse',family='rail_network_length',split='dev_conditions',operation='inverse',rows=inverse_track,
        unknown=(0,'tracks'),target=dict(route=8,track=18),values=[(1,'Xanthic has one rail pair.'),(2,'Xanthic has two rail pairs.'),(3,'Xanthic has three rail pairs.'),(4,'Xanthic has four rail pairs.')]))
    track_base=[track_row('Auburn',5,1),track_row('Bronze',4,1,built_installation=True,public=False,current_mine=True)]
    specs.append(dict(id='premise-v1-rail_network_length-dev-minimum',family='rail_network_length',split='dev_conditions',operation='minimum_cost',rows=track_base,target=dict(route=9),packages=[
        dict(label='Open Bronze to public traffic, preserving its original construction history and all other premises.',changes=[(1,'open_to_public_traffic',True)],cost=7),
        dict(label='Add a second rail pair to Auburn.',changes=[(0,'tracks',2)],cost=2),
        dict(label='Open Bronze to public traffic and add its second rail pair.',changes=[(1,'open_to_public_traffic',True),(1,'tracks',2)],cost=10),
        dict(label='Stop Bronze current mine-only use, but keep it closed to public traffic with its original construction history unchanged.',changes=[(1,'current_use_mine_only',False)],cost=1)]))
    specs.append(dict(id='premise-v1-rail_speed-sealed-policy',family='rail_speed',split='sealed',operation='contingent',rows=[],metric='D',
        branches={'L':[speed_row('Cobalt','new_parallel',260,160,70),speed_row('Denim','upgraded_conventional',240,160,70)],
                  'R':[speed_row('Cobalt','upgraded_conventional',260,160,70),speed_row('Denim','new_special',240,160,70)]},
        policies=[dict(L=1,R=0),dict(L=1,R=1),dict(L=2,R=0),dict(L=0,R=1)]))
    specs.append(dict(id='premise-v1-rail_network_length-sealed-policy',family='rail_network_length',split='sealed',operation='contingent',rows=[],metric='track',
        branches={'L':[track_row('Ecru',4,2),track_row('Fawn',6,1,built_installation=True,public=True)],
                  'R':[track_row('Ecru',4,2),track_row('Fawn',6,1,built_installation=True,public=False)]},
        policies=[dict(L=14,R=8),dict(L=10,R=4),dict(L=14,R=14),dict(L=8,R=14)]))
    return specs


def build(source,out,group_script):
    assert sha(source)==SOURCE_SHA,'source snapshot mismatch'
    rows=[json.loads(s) for s in source.read_text(encoding='utf-8').splitlines()];by_id={r['id']:r for r in rows}
    groups=historical_groups(rows,group_script)
    dev_groups={g for g in groups.values() if int(hashlib.sha256(g.encode()).hexdigest(),16)%5==0}
    dev_topics={t for r in rows if groups[r['id']] in dev_groups for t in r['topics']}
    sources={family:[{k:by_id[sid][k] for k in ('id','title','scope','source_text','evidence','topics')} for sid in ids] for family,ids in FAMILIES.items()}
    for entries in sources.values():
        for r in entries:
            assert groups[r['id']] not in dev_groups and not set(r['topics'])&dev_topics
    # These exact spans bind each implemented boundary/exclusion to the archived source.
    bindings=[('rail_speed',0,'equal to or greater than 250 km/h'),('rail_speed',1,'equal to or greater than 200 km/h'),
        ('rail_speed',0,'new tracks were added to existing tracks that are not upgraded'),
        ('rail_network_length',0,'In the context of the EU reporting'),
        ('rail_network_length',0,'Lines constructed solely to serve mines, forests or other industrial or agricultural installations and which are not open to public traffic;'),
        ('rail_network_length',0,'Lines solely used for operating touristic trains and heritage trains;'),
        ('rail_network_length',0,'Metro, Tram and Light rail urban lines are excluded.'),
        ('rail_network_length',1,'A line is made up of one or more tracks'),
        ('rail_network_length',1,'Stretches of road or water even if rolling stock is conveyed over such routes')]
    for family,i,quote in bindings:assert quote in sources[family][i]['source_text']
    specs=specifications();tasks=[compile_task(s) for s in specs]
    out.mkdir(parents=True,exist_ok=False)
    for name,data in [('sources.private.json',sources),('source_bindings.private.json',bindings)]:
        (out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name,data in [('premises.private.jsonl',specs),('tasks.private.jsonl',tasks)]:
        (out/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data),encoding='utf-8',newline='\n')
    for split in ('train','dev_expression','dev_conditions','sealed'):
        (out/(split+'.private.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in tasks if r['split']==split),encoding='utf-8',newline='\n')
    manifest=dict(source_sha256=SOURCE_SHA,task_sha256=sha(out/'tasks.private.jsonl'),premise_sha256=sha(out/'premises.private.jsonl'),
        compiler_sha256=sha(Path(__file__)),units=2,source_units=4,historical_source_groups=len({groups[sid] for ids in FAMILIES.values() for sid in ids}),
        tasks=len(tasks),split_counts=dict(Counter(t['split'] for t in tasks)),option_truth_tables=sum(len(t['option_proofs']) for t in tasks),
        source_boundaries_checked=len(bindings),official_or_diagnostic_inputs=False,model_generation_requests=0,model_evaluation_requests=0,
        structural_6_3_1_units=2,training_allowed=False,training_ready=False,
        limitations='Source-bound two-family construction pilot. The program and its premises were authored by Codex, not an independent domain expert. Source semantics, excluded-topic checks, cross-split structure and historical task similarity require downstream audit. Structured counts are not released units; no new knowledge-gap families and no performance gain are claimed. Upgraded-line speed-adaptation exceptions and private/stand-alone network ambiguity are intentionally out of scope.')
    (out/'manifest.safe.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--group-script',type=Path,default=Path('scripts/generate_source_condition_tasks.py'));a=p.parse_args()
    print(json.dumps(build(a.source,a.out,a.group_script),indent=2))

"""P1: source-only, prospective ten-family transfer pilot.

The builder never reads diagnostic/official questions, labels or predictions.
Historical probes select source families only. All generated records are frozen
before baseline inference. Retention uses frozen historical development topics.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from prepare_cpt_transfer_screen import SOURCE_SHA, historical_groups, sha

UNITS = {
 'speed':('rail_technology',['e18cbdfda5e0f7a00139','ed190bcb6b41b42da590']),
 'railcar':('rail_technology',['b5b62400c11c1cace1b4']),
 'derailment':('rail_safety',['949d2b89ca53083dc82e','13c5ab369e1068f067bc']),
 'injury':('rail_safety',['2a0c3a166cc4c53799b2']),
 'capacity':('waterways',['e5b2962d0054bfeac3a0']),
 'propulsion':('waterways',['031182c8e97a128c8c31']),
 'finance':('business',['468404e82233239a24e2']),
 'outsourcing':('business',['d3eeb424676374325e30']),
 'pipeline':('pipeline',['f300fd947c2ed1da3538','bff682970154c1118932']),
 'wave':('warehouse',['1e2b9ad61b478e1de2bf']),
}
RETENTION = {
 'teu':('transport_measurement','324abf1f754d9d4675a3'),
 'tkm':('transport_measurement','fe766088c01439a34537'),
 'offered':('transport_measurement','f1c3abeebdb07895f821'),
 'pallet':('unit_loads','fb224b7d21a54d38dc15'),
 'reorder':('inventory_demand','7894b0f1667db84a50cc'),
 'capital':('capital_stock','e883f36babe4afb746b1'),
 'abc':('inventory_demand','26f3697b59af0e73d5f8'),
 'packages':('cargo_package','ac3490e8e7a2190b5bd3'),
 'moment':('aircraft_mechanics','a0fd410c330c4ace4312'),
 'payload':('aircraft_mechanics','a61150b6574fe14d6ef7'),
}
STAGES=['overall logistics concept','quantified required services','placement policy','tendering','offer evaluation','service and cost assessment']
RULES={
 'speed':'Dedicated requires special new construction with main design capability at least 250 km/h; conventional special upgrade requires at least 200 km/h. Observed speed does not replace designed capability.',
 'railcar':'Each powered railcar counts separately, including both units sharing a tractive bogie. Non-tractive trailers do not count as tractive vehicles; driver compartments do not determine traction.',
 'derailment':'A collision-caused wheel departure is classified as collision. Otherwise at least one wheel leaving the rails is a derailment.',
 'injury':'Hospitalisation must exceed 24 hours, result from the accident injury, and not concern an attempted suicide.',
 'capacity':'The vessel must be designed for inland freight and have carrying capacity at least 20 tonnes; current load does not replace capacity or design purpose.',
 'propulsion':'Powered inland freight vessels qualify except self-propelled tanker barges. An auxiliary-only towed/pushed barge does not qualify; towing ability does not disqualify a self-propelled barge.',
 'finance':'In the cited revolving-credit framework, the supplier-alone route uses supplier financial health; OEM-approved invoices use OEM financial health as the interest-rate basis.',
 'outsourcing':'The cited six-stage framework orders concept, quantified requirements, placement policy, tendering, offer evaluation, then service/cost assessment.',
 'pipeline':'Count only active reference-period pipelines on national territory, including its seabed. Dormant or not-yet-started units do not count.',
 'wave':'Dynamics 365 released lines match the first qualifying wave template. Work templates and location directives have different downstream roles.',
}
SCOPES={
 'speed':'Use the 2019 Eurostat/UNECE/ITF railway definitions. D=dedicated high-speed, U=upgraded high-speed, N=neither of these two categories. New means specially built; parallel means new tracks beside old tracks left unupgraded; upgraded means specially upgraded conventional line; ordinary means a conventional line neither specially built nor specially upgraded. All connecting segments are short town-centre connectors. No upgraded-line topographic/relief/town-planning speed exception applies. ',
 'railcar':'Use the 2019 Eurostat/UNECE/ITF railcar statistics. Each listed vehicle body is fitted for passengers or goods. Independent powered bodies exclude the bodies in shared-bogie pairs; each pair consists of two railcar bodies with a common tractive bogie. Trailers are non-tractive. Report the number of tractive vehicles. ',
 'derailment':'Use the 2019 Eurostat/UNECE/ITF railway accident classification. A recorded collision is a qualifying train-to-train collision and, if wheels leave the rails, is its established cause. No level-crossing or other classification exception applies. Report collision, derailment, or neither of these. ',
 'injury':'Use the 2019 Eurostat/UNECE/ITF definition of person seriously injured in railway accidents. Durations are exact; a recorded accident injury is the reason for the entire hospital stay. Report included or excluded from this injury category. ',
 'capacity':'Use the 2019 Eurostat/UNECE/ITF inland waterways freight vessel definition. Capacity is rated carrying capacity in tonnes; load is the current actual load. Design purpose is recorded independently. Report included or excluded. ',
 'propulsion':'Use the 2019 Eurostat/UNECE/ITF self-propelled vessel definition. Every record is an inland freight vessel of at least 20 tonnes carrying capacity. Main propulsion means independent normal propulsion; auxiliary-only means a towed/pushed barge with only an auxiliary engine. Report included or excluded from the specified self-propelled-vessel category. ',
 'finance':'Use the cited revolving line of credit framework. In the approved route an OEM publishes invoices it commits to pay; in the alone route it gives no such approval. Both are borrowing by the supplier from the funder. Report supplier or OEM as the financial-health basis of the interest rate. Do not estimate a numeric rate. ',
 'outsourcing':'Use the six-stage sequence in the 2013 Logistics text, section 8.5, not universal stage numbering. Listed completed stages form the entire completed prefix; nothing later has been completed. Report the next stage in that framework. ',
 'pipeline':'Use the 2019 Eurostat/UNECE/ITF pipeline and pipeline-network definitions for the reporting country. All conduits have pumps/valves/control devices and meet the pipeline definition. Lengths are disjoint kilometres; national includes national seabed. Active means actual activity during the reference period. Report included kilometres for each record. ',
 'wave':'Use the cited Dynamics 365 wave-template framework. Each released line matches a fragile template F exactly when fragile, a priority template P exactly when priority, and a general template G always. The recorded order is the processing order. Report the selected template F, P or G. ',
}


def value(unit,r):
    if unit=='speed':return 'D' if r['build'] in ('new','parallel') and r['design']>=250 else 'U' if r['build']=='upgraded' and r['design']>=200 else 'N'
    if unit=='railcar':return r['powered']+2*r['pairs']
    if unit=='derailment':return 'collision' if r['collision'] else 'derailment' if r['wheels']>0 else 'neither'
    if unit=='injury':return 'included' if r['hours']>24 and r['accident'] and not r['suicide'] else 'excluded'
    if unit=='capacity':return 'included' if r['capacity']>=20 and r['design']=='inland freight' else 'excluded'
    if unit=='propulsion':return 'included' if r['propulsion']=='main' and not r['tanker'] else 'excluded'
    if unit=='finance':return 'OEM' if r['route']=='approved' else 'supplier'
    if unit=='outsourcing':return STAGES[r['completed']]
    if unit=='pipeline':return r['length'] if r['active'] and r['national'] else 0
    if unit=='wave':return next(k for k in r['order'] if k=='G' or (k=='F' and r['fragile']) or (k=='P' and r['priority']))
    raise ValueError(unit)


def describe(unit,r,style=0):
    if unit=='speed':text=f"construction={r['build']}; main design={r['design']} km/h; observed operation={r['observed']} km/h; connector={r['connector']} km/h"
    elif unit=='railcar':text=f"{r['powered']} independent powered bodies, {r['pairs']} shared-tractive-bogie pairs, {r['trailers']} non-tractive trailers; {r['cabs']} driver compartments in total"
    elif unit=='derailment':text=f"qualifying causal collision={'yes' if r['collision'] else 'no'}; wheels leaving rails={r['wheels']}"
    elif unit=='injury':text=f"hospital stay={r['hours']} hours; accident injury causing the stay={r['accident']}; attempted suicide={r['suicide']}"
    elif unit=='capacity':text=f"rated capacity={r['capacity']} tonnes; design purpose={r['design']}; current load={r['load']} tonnes"
    elif unit=='propulsion':text=f"propulsion={r['propulsion']}; self-propelled tanker barge={r['tanker']}; ability to tow another vessel={r['tow']}"
    elif unit=='finance':text=f"financing route={r['route']}; invoice face value={r['amount']} monetary units"
    elif unit=='outsourcing':text='completed stages: '+(', '.join(STAGES[:r['completed']]) or 'none')
    elif unit=='pipeline':text=f"length={r['length']} km; active={r['active']}; within national territory={r['national']}; seabed location={r['seabed']}"
    elif unit=='wave':text=f"template order={' then '.join(r['order'])}; fragile={r['fragile']}; priority={r['priority']}"
    else:raise ValueError(unit)
    if style==1:return 'The handover note records '+text.replace('; ','. It also records ')+'.'
    if style==2:return 'The reviewer received these facts ('+text+'). No other facts change.'
    return text+'.'


def catalog(unit,variant):
    n=variant+1
    if unit=='speed':return [dict(build=b,design=d,observed=100+n,connector=50+n) for b,d in [('new',250+n),('upgraded',200+n),('new',249),('parallel',270+n),('upgraded',199),('ordinary',280+n)]]
    if unit=='railcar':return [dict(powered=n+i,pairs=i%3,trailers=2+i,cabs=1+i) for i in range(6)]
    if unit=='derailment':return [dict(collision=c,wheels=w) for c,w in [(False,1),(True,1),(True,0),(False,0),(False,2),(True,3)]]
    if unit=='injury':return [dict(hours=h,accident=a,suicide=s) for h,a,s in [(24,True,False),(24+n,True,False),(30+n,True,True),(40+n,False,False),(20,True,False),(48+n,True,False)]]
    if unit=='capacity':return [dict(capacity=c,design=d,load=min(3+n,c)) for c,d in [(20,'inland freight'),(19,'inland freight'),(30+n,'inland passengers'),(40+n,'sea freight'),(25+n,'inland freight'),(18,'inland freight')]]
    if unit=='propulsion':return [dict(propulsion=p,tanker=t,tow=w) for p,t,w in [('main',False,True),('auxiliary-only',False,False),('main',True,True),('none',False,False),('main',False,False),('auxiliary-only',False,True)]]
    if unit=='finance':return [dict(route='approved' if i%2 else 'alone',amount=1000*n+200*i) for i in range(6)]
    if unit=='outsourcing':return [dict(completed=i) for i in range(6)]
    if unit=='pipeline':return [dict(length=n+3+i,active=a,national=c,seabed=s) for i,(a,c,s) in enumerate([(True,True,False),(False,True,False),(True,True,True),(True,False,False),(False,True,True),(True,False,True)])]
    if unit=='wave':return [dict(order=o,fragile=f,priority=p) for o,f,p in [('FPG',True,True),('FPG',False,True),('PFG',True,True),('PFG',False,False),('GFP',True,False),('FGP',False,True)]]
    raise ValueError(unit)


def alternative(unit,actual):
    if type(actual) is int:return actual+1
    values={'speed':['D','U','N'],'derailment':['collision','derailment','neither'],
        'injury':['included','excluded'],'capacity':['included','excluded'],'propulsion':['included','excluded'],
        'finance':['supplier','OEM'],'outsourcing':STAGES,'wave':['F','P','G']}[unit]
    return values[(values.index(actual)+1)%len(values)]


def task(unit,split,index,question,options,truth,proof,premises,operation):
    assert len(options)==4 and len(set(options))==4 and 0<sum(truth)<4
    identity=f'p1-{unit}-{split}-{index}'
    order=list(range(4));random.Random(int(hashlib.sha256(identity.encode()).hexdigest()[:16],16)).shuffle(order)
    return dict(id=identity,unit=unit,category=UNITS[unit][0],split=split,operation=operation,
        question=SCOPES[unit]+question+'\nSelect all correct options.',options=[options[i] for i in order],
        correct_indices=[j for j,i in enumerate(order) if truth[i]],proof=[proof[i] for i in order],premises=premises,
        explanation=RULES[unit]+' '+ ' '.join(proof),source_ids=['llin-core-'+s for s in UNITS[unit][1]])


def forward(unit,split,index):
    rows=catalog(unit,index+(10 if split=='dev_expression' else 0))
    chosen=[rows[(index+j)%6] for j in range(4)]
    names=[f'{chr(65+j)}{index+1}' for j in range(4)]
    style=0 if split=='train' else index+1
    lead=('Audit these four independent records.\n' if split=='train' else
          'Four managers submitted separate handovers. Judge each manager\'s reported result against that manager\'s facts.\n' if index==0 else
          'Reconcile the following four review notes. Each option is a proposed result for one note, and the notes are independent.\n')
    question=lead+'\n'.join(name+': '+describe(unit,r,style) for name,r in zip(names,chosen))
    truth=[(j+index)%4 in (0,2) for j in range(4)]
    if split=='train' and index%3==1:truth=[False,True,False,False]
    if split=='train' and index%3==2:truth=[True,True,False,True]
    actual=[value(unit,r) for r in chosen]
    claims=[v if t else alternative(unit,v) for v,t in zip(actual,truth)]
    options=[f'{name}: the result is {v}.' for name,v in zip(names,claims)]
    proofs=[f'{name} evaluates to {v}; the proposed {claim} is {"correct" if t else "incorrect"}.' for name,v,claim,t in zip(names,actual,claims,truth)]
    return task(unit,split,index,question,options,truth,proofs,dict(rows=chosen,names=names,claims=claims),'forward')


def intervention(unit):
    """Base row, target, four fully specified alternate records and total costs."""
    base=catalog(unit,2)[0]
    edits={
        'speed':[{'design':240},{'build':'upgraded'},{'observed':80},{'connector':20}],
        'railcar':[{'powered':4},{'pairs':1},{'trailers':12},{'cabs':9}],
        'derailment':[{'collision':True},{'wheels':0},{'wheels':4},{'wheels':2}],
        'injury':[{'hours':25},{'hours':27},{'hours':23},{'suicide':True}],
        'capacity':[{'capacity':19},{'design':'inland passengers'},{'load':0},{'capacity':24}],
        'propulsion':[{'propulsion':'auxiliary-only'},{'tanker':True},{'tow':False},{'propulsion':'none'}],
        'finance':[{'route':'approved','amount':8000},{'route':'approved','amount':7000},{'amount':200},{'amount':900}],
        'outsourcing':[{'completed':1},{'completed':2},{'completed':3},{'completed':4}],
        'pipeline':[{'active':False},{'national':False},{'length':10},{'seabed':True}],
        'wave':[{'order':'PFG'},{'order':'PGF'},{'priority':False},{'order':'GFP'}],
    }[unit]
    rows=[dict(base,**e) for e in edits]
    return base,value(unit,rows[0]),rows,[5,9,1,2]


def conditional(unit,index):
    base,target,rows,costs=intervention(unit)
    actual=[value(unit,r) for r in rows]
    if index==0:
        q='An audit independently establishes that the required result is '+str(target)+'. Four possible complete records are proposed. Which records are consistent with that result? All facts needed for each proposal are shown in its option.'
        opts=[describe(unit,r,1) for r in rows];truth=[v==target for v in actual]
        if len(set(opts))<4:raise ValueError('duplicate inverse proposals')
        proofs=[f'Proposal {i+1} gives {v}, required {target}.' for i,v in enumerate(actual)]
        return task(unit,'dev_conditions',index,q,opts,truth,proofs,dict(rows=rows,target=target),'inverse')
    best=min(c for c,v in zip(costs,actual) if v==target)
    q='Current complete record: '+describe(unit,base)+' Exactly one feasible package must be selected. Each option gives the entire resulting record and its entire cost in arbitrary units; there are no further costs or constraints. The required result is '+str(target)+'. Select the cheapest package or packages meeting this requirement.'
    opts=[describe(unit,r)+f' Total cost {c}.' for r,c in zip(rows,costs)]
    truth=[v==target and c==best for v,c in zip(actual,costs)]
    proofs=[f'Package {i+1} gives {v}, cost {c}; the minimum cost satisfying {target} is {best}.' for i,(v,c) in enumerate(zip(actual,costs))]
    return task(unit,'dev_conditions',index,q,opts,truth,proofs,dict(rows=rows,costs=costs,target=target),'minimum_cost')


def sealed(unit):
    rows=catalog(unit,21);left=rows[2];right=rows[4]
    a,b=value(unit,left),value(unit,right)
    policies=[(a,b),(alternative(unit,a),b),(a,alternative(unit,b)),(alternative(unit,a),alternative(unit,b))]
    q='Two mutually exclusive final records are possible. Branch L: '+describe(unit,left,1)+' Branch R: '+describe(unit,right,2)+' Select the policy that would report the correct result in either branch.'
    options=[f'L => {x}; R => {y}.' for x,y in policies]
    return task(unit,'sealed',0,q,options,[True,False,False,False],
        [f'Correct branch results are L={a}, R={b}; this policy is {"correct" if i==0 else "incorrect"}.' for i in range(4)],
        dict(rows=[left,right],policies=policies),'branch_policy')


def retention_task(unit,i):
    n=i+2
    if unit=='teu':
        expected=n*23+(n+1)*17
        q=f'Under the 2019 transport glossary, a movement log has {n} TEU moved 23 km and {n+1} TEU moved 17 km. No other movements occur. What is the total in TEU-kilometres?'
    elif unit=='tkm':
        expected=n*31+(n+2)*13
        q=f'For 2019 glossary goods-transport statistics, {n} tonnes travel 31 actual km on the declaring country\'s network and {n+2} tonnes travel 13 actual km there. A separate 90 km foreign leg is outside that country. What tonne-kilometres should this country report for these goods?'
    elif unit=='offered':
        expected=(20+n)*40
        q=f'Use the 2019 maritime glossary. A merchant vessel with cargo carrying capacity {20+n} tonnes sails a 40 km port-to-port journey carrying only {n} tonnes. What tonne-kilometres offered does this journey represent?'
    elif unit=='pallet':
        expected=n*960000
        q=f'{n} EPAL 1 Euro pallets are placed in one layer, without overlap or gaps counted as pallet footprint. Using the specified pallet type dimensions, what is their combined footprint in square millimetres?'
    elif unit=='reorder':
        expected=100+7*n+2*(3+n)
        q=f'Use the cited MITx continuous-review (s,Q) policy. Expected lead-time demand is {100+7*n}, forecast-error RMSE over lead time is {3+n}, safety factor is 2, and order size Q is 60. What is reorder point s?'
    elif unit=='capital':
        expected=1000+100*n-30*n
        q=f'For 2019 glossary IWT capital-stock reporting, the supplied gross physical infrastructure asset value is {1000+100*n} monetary units, with total accumulated depreciation {30*n}; there are no other valuation adjustments. What recommended net capital value should be reported?'
    elif unit=='abc':
        expected=['very tight control and accurate records','less tight control and moderate records','simplest controls and minimal records'][i%3]
        q=f'Inventory review batch {i+1} has already correctly assigned an item to ABC class {"ABC"[i%3]}. Under the cited ABC inventory-control definitions, which control-and-record policy belongs to this class? Do not reclassify the item from sales or price.'
    elif unit=='packages':
        expected=['package','overpack','cargo transport unit','carrier','consignee'][i%5]
        facts=['a containment box together with its contents prepared for transport','several completed packages secured together on a pallet for convenient handling','a freight container used to carry packages','the party undertaking carriage under the carriage contract','the party to whom the cargo is consigned'][i%5]
        q=f'In consignment file {i+1}, identify {facts}, using the 2014 CTU Code terminology.'
    elif unit=='moment':
        expected=(40+n)*3-20*2
        q=f'Use the cited FAA signed-arm convention. One item weighs {40+n} weight units and is 3 distance units aft of datum; another weighs 20 weight units and is 2 distance units forward of datum. In consistent units, what is their combined signed moment?'
    elif unit=='payload':
        expected=5000+100*n-900-300
        q=f'Under the cited transport-aircraft payload distinction, a trip has {5000+100*n} weight units available above empty weight under its controlling limit. Required usable fuel is 900 and crew/other non-revenue operating loads are 300. No other constraint is tighter. What maximum revenue payload fits this available weight?'
    else:raise ValueError(unit)
    if type(expected) is int:options=[str(expected),str(expected+7),str(expected+19),str(max(0,expected-1))]
    elif unit=='abc':options=[expected]+[v for v in ['very tight control and accurate records','less tight control and moderate records','simplest controls and minimal records','no stock records or controls'] if v!=expected][:3]
    else:options=[expected]+[v for v in ['package','overpack','cargo transport unit','carrier','consignee','packaging'] if v!=expected][:3]
    order=list(range(4));random.Random(5900+i+sum(map(ord,unit))).shuffle(order)
    return dict(id=f'p1-retention-{unit}-{i}',unit=unit,category=RETENTION[unit][0],split='retention',operation='retention_source_application',
        question=q+' Select the single correct option.',options=[options[j] for j in order],correct_indices=[order.index(0)],
        source_ids=['llin-core-'+RETENTION[unit][1]],proof=[f'Independent source application result: {expected}.'],
        explanation='',premises=dict(variant=i,expected=expected))


def build(source,out):
    assert sha(source)==SOURCE_SHA
    original=[json.loads(s) for s in source.read_text(encoding='utf-8').splitlines()]
    groups=historical_groups(original,Path(__file__).with_name('generate_source_condition_tasks.py'))
    frozen={g for g in groups.values() if int(hashlib.sha256(g.encode()).hexdigest(),16)%5==0}
    frozen_topics={t for r in original if groups[r['id']] in frozen for t in r['topics']}
    ids={'llin-core-'+s for _,ss in UNITS.values() for s in ss}
    ret_ids={'llin-core-'+s for _,s in RETENTION.values()}
    by_id={r['id']:r for r in original}
    for sid in ids:assert groups[sid] not in frozen and not set(by_id[sid]['topics'])&frozen_topics
    for sid in ret_ids:assert groups[sid] in frozen
    assert not ids&ret_ids
    tasks=[]
    for unit in UNITS:
        tasks.extend(forward(unit,'train',i) for i in range(6))
        tasks.extend(forward(unit,'dev_expression',i) for i in range(2))
        tasks.extend(conditional(unit,i) for i in range(2))
        tasks.append(sealed(unit))
    counts={u:10 for u in RETENTION};counts.update(abc=3,packages=5,moment=6,payload=6)
    tasks.extend(retention_task(u,i) for u in RETENTION for i in range(counts[u]))
    assert len(tasks)==190 and len({t['id'] for t in tasks})==190
    out.mkdir(parents=True,exist_ok=False)
    def write(name,rows):(out/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8',newline='\n')
    write('tasks.private.jsonl',tasks)
    write('sources.private.jsonl',[{k:by_id[s][k] for k in ('id','title','scope','source_text','evidence','topics')} for s in sorted(ids|ret_ids)])
    for split in ('train','dev_expression','dev_conditions','sealed','retention'):write(split+'.private.jsonl',[t for t in tasks if t['split']==split])
    cases=[dict(dataset='p1_'+t['split'],source_id=t['id'],category=t['unit'],question_type='multiple_choice' if len(t['correct_indices'])>1 else 'single_choice',question=t['question'],options=t['options'],expected=t['correct_indices']) for t in tasks if t['split']!='sealed']
    write('cases.private.jsonl',cases)
    manifest=dict(version='P1',source_sha256=SOURCE_SHA,builder_sha256=sha(Path(__file__)),
        files={p.name:sha(p) for p in out.glob('*.jsonl')},counts=dict(Counter(t['split'] for t in tasks)),
        training_families=len(UNITS),training_categories=dict(Counter(c for c,_ in UNITS.values())),
        training_source_groups=len({groups[s] for s in ids}),retention_source_groups=len({groups[s] for s in ret_ids}),
        retention_families=len(RETENTION),retention_independence='80 fresh tasks from 10 rules on historical dev sources excluded from all training; correlated variants, not 80 independent knowledge groups or a never-exposed benchmark.',
        hard_quality_gates=['source-backed labels and scope','old dev groups/topics excluded from training','no official or diagnostic generator inputs','fresh evaluation-only retention','frozen data before baseline','source and downstream audit before training'],
        prospective_scale_change='User approved ten-family mechanism pilot before 30-50-family broad experiment; 80 retention tasks are a pilot sentinel, not full 120+ retention confirmation.',
        training_allowed=False,release_audit_pending=True,model_queries=0,
        interpretation='One direct-SFT pilot from Step120. Not a single-factor comparison to S4/S5. Expression preserves rule/operator; conditions use inverse and least-cost decisions. No performance claim yet.')
    (out/'manifest.safe.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    print(json.dumps(build(a.source,a.out),indent=2))

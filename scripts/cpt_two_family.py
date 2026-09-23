"""Two-family task validation, option mapping and fixed screening decisions."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from evaluate_logistics_knowledge import build_messages,EvalItem,parse_answers

BASE=Path('CPT_resources/llin-two-family-20260923-01')
FORMS=('scope','competing_rules','application','counterexample')

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def freeze(p,x):
    text=json.dumps(x,ensure_ascii=False,indent=2)+'\n'
    if p.exists():assert p.read_text(encoding='utf-8')==text,'Frozen file changed'
    else:p.write_text(text,encoding='utf-8')


def validate(task,source):
    assert len(task['options'])==len(task['option_reasons'])==4
    assert len(set(x.strip().casefold() for x in task['options']))==4
    assert 0<len(task['correct_indices'])<4 and sorted(set(task['correct_indices']))==task['correct_indices']
    assert all(type(i) is int and 0<=i<4 for i in task['correct_indices'])
    assert 'select all correct statements' in task['question'].casefold()
    assert all(q in source for q in task['source_quotes']) and task['source_quotes']
    assert task['scenario_signature'] and task['reasoning_structure']


def variant(task,shift,source=None):
    order=[(i+shift)%4 for i in range(4)]
    expected=[i for i,j in enumerate(order) if j in task['correct_indices']]
    item=EvalItem(item_hash=task['id'],dataset='source_generated',source_id=task['id'],category=task['family'],
        question_type='multiple_choice',question=task['question'],options=tuple(task['options'][j] for j in order),expected=tuple(expected))
    messages=build_messages(item)
    if source:messages[1]['content']='Reference material:\n'+source+'\n\n'+messages[1]['content']
    return dict(messages=messages,expected=expected,order=order)


def screen_packet(tasks,families):
    assert len(tasks)==8 and Counter(t['family'] for t in tasks)=={f:4 for f in families}
    requests=[];groups=[]
    for t in sorted(tasks,key=lambda x:x['id']):
        validate(t,families[t['family']]['evidence_text'])
        conditions={}
        for condition in ('closed','source'):
            for shift in range(4):
                key=f'{condition}:{shift}'
                v=variant(t,shift,families[t['family']]['evidence_text'] if condition=='source' else None)
                conditions[key]=v
                requests.append(dict(id=t['id']+':'+key,group=t['id'],condition=condition,shift=shift,messages=v['messages']))
        for repeat in (1,2):
            key=f'repeat:{repeat}';v=variant(t,0);conditions[key]=v
            requests.append(dict(id=t['id']+':'+key,group=t['id'],condition='repeat',repeat=repeat,messages=v['messages']))
        groups.append(dict(id=t['id'],family=t['family'],form=t['form'],conditions=conditions))
    assert len(requests)==80
    return dict(groups=groups,requests=requests,reuse=[],training_allowed=False)


def screen_score(packet,rows):
    assert len(rows)==len(packet['requests'])==80 and len({r['id'] for r in rows})==80
    raw={r['id']:r for r in rows};ledger=[]
    for g in packet['groups']:
        flags={};invalid=truncated=0
        for key,c in g['conditions'].items():
            row=raw[g['id']+':'+key]
            assert row['messages_sha256']==digest(c['messages'])
            parsed,ok=parse_answers(row['text'],4);cut=row['finish_reason']=='length'
            flags[key]=ok and not cut and list(parsed)==c['expected']
            invalid+=not ok or cut;truncated+=cut
        ledger.append(dict(id=g['id'],family=g['family'],closed=all(flags[f'closed:{i}'] for i in range(4)),
            source=all(flags[f'source:{i}'] for i in range(4)),
            repeat_consistent=all(flags[f'repeat:{r}']==flags['closed:0'] for r in (1,2)),
            flags=flags,invalid=invalid,truncated=truncated))
    summary=[]
    for family in sorted({g['family'] for g in ledger}):
        selected=[g for g in ledger if g['family']==family]
        closed=sum(g['closed'] for g in selected);source=sum(g['source'] for g in selected)
        repeated=all(g['repeat_consistent'] for g in selected)
        passed=closed<=2 and source>=3 and source>=closed+1 and repeated
        summary.append(dict(family=family,tasks=4,closed_stable=closed,source_stable=source,repeat_consistent=repeated,passed=passed))
    return ledger,dict(families=summary,training_value_passed=all(g['passed'] for g in summary),calls=80,
        invalid=sum(g['invalid'] for g in ledger),truncated=sum(g['truncated'] for g in ledger))

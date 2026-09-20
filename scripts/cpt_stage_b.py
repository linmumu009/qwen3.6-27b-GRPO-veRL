"""Frozen Logistika screening protocol. No training or formal-set reads."""
import hashlib
import json
import random
import re
from pathlib import Path
from evaluate_logistics_knowledge import EvalItem, build_messages, parse_answers

SEED = 1024
FORMS = ('scope', 'competing_rules', 'application', 'counterexample')


def encode_prompt(tokenizer, text, output_budget=2048):
    rendered=tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=tokenizer.encode(rendered,add_special_tokens=False)
    if not isinstance(ids,list) or not ids or any(type(v) is not int or v<0 for v in ids):
        raise RuntimeError('invalid token IDs')
    if len(ids)+output_budget>8192:raise RuntimeError('prompt exceeds window; no truncation')
    return ids


def device_idle(text):
    """Allow observed driver baseline only with explicit empty process tables."""
    used=[int(v) for v in re.findall(r'(\d+)\s*/\s*65536',text)]
    empty={int(v) for v in re.findall(r'No running processes found in NPU (\d+)',text)}
    process_rows=re.search(r'^\s*\|\s*\d+\s+\d+\s*\|\s*\d+\s*\|',text,re.M)
    return len(used)==16 and max(used)<4096 and empty==set(range(8)) and not process_rows,used


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def object_of(text):
    return json.loads(re.sub(r'\s*```$', '', re.sub(r'^```(?:json)?\s*', '', text.strip())))


def author_prompt(group, form):
    return '''Write exactly ONE new source-grounded logistics screening task in English.
You receive only primary source material and an abstract capability specification.
Use the specified number of plausible, distinct options. Preserve the source's
named historical/statistical/teaching framework in the question. All conditions
needed to decide every option must be explicit. For scope: apply scope to a new
situation. For competing_rules: distinguish close rules under changed conditions.
For application: combine at least two source conditions. For counterexample:
identify a concrete exception to an overgeneralised claim. Do not ask simple name
lookup, copy source sentences, use trivial unrelated distractors, or invent facts.
Do not quote the governing rule in the question. Use a new operational scenario.
Use multi-select statements and explicitly say Select all correct statements.
Avoid above/below, references to option indices, all/none of the above. At least
one correct and one incorrect option are required. Do not supply arbitrary numbers
unless the source supports the calculation. If the design is unsupported return
{"skip_reason":"..."}. Otherwise return JSON only with question, options,
correct_indices (zero-based), option_reasons (one per option), source_quote
(one exact source span), and reasoning_structure. No task is training data.
DESIGN: '''+form+'\nSPECIFICATION:\n'+json.dumps(group, ensure_ascii=False)


def validate_task(task, group):
    if set(task) != {'question','options','correct_indices','option_reasons','source_quote','reasoning_structure'}:
        raise ValueError('task schema')
    opts = task['options']
    if not isinstance(opts,list) or not all(isinstance(o,str) and o.strip() for o in opts): raise ValueError('invalid options')
    if len(opts) != group['option_count'] or len(set(o.strip().casefold() for o in opts)) != len(opts):
        raise ValueError('option count or duplicates')
    if not all(isinstance(o,str) and o.strip() for o in opts): raise ValueError('empty option')
    key = task['correct_indices']
    if not isinstance(key,list) or not 0 < len(key) < len(opts) or any(type(k) is not int or k not in range(len(opts)) for k in key) or len(set(key)) != len(key):
        raise ValueError('invalid key')
    if len(task['option_reasons']) != len(opts) or not all(task['option_reasons']): raise ValueError('rationales')
    if len(task['source_quote']) < 20 or task['source_quote'] not in group['source_text']: raise ValueError('source quote')
    if 'select all correct' not in task['question'].casefold(): raise ValueError('missing multi-select instruction')
    text = task['question']+' '+' '.join(opts)
    if re.search(r'all of the above|none of the above|both options|option\s+[A-F0-9]|above statements|below statements',text,re.I): raise ValueError('position dependence')
    if not task['reasoning_structure'] or len(task['question'].split()) > 180 or any(len(o.split())>50 for o in opts): raise ValueError('task length/design')
    return dict(task, correct_indices=sorted(key))


def reviewer_prompt(group, task, form):
    return '''Solve and audit every option against SOURCE. No author key is supplied.
Reject unsupported facts, ambiguous labels, missing premises, absent source scope,
answer leakage, simple name lookup, or failure to test the specified operation.
Return JSON only: {"correct_indices":[0],"supported":true,"unambiguous":true,
"self_contained":true,"scope_preserved":true,"not_answer_leaking":true,
"design_satisfied":true,"option_reasons":["..."],"reason":"..."}.
SOURCE:\n'''+json.dumps(group,ensure_ascii=False)+'\nDESIGN: '+form+'\nTASK:\n'+json.dumps({k:task[k] for k in ('question','options')},ensure_ascii=False)


def review_ok(review, task):
    flags=('supported','unambiguous','self_contained','scope_preserved','not_answer_leaking','design_satisfied')
    return (all(review.get(k) is True for k in flags)
            and sorted(review.get('correct_indices',[])) == task['correct_indices']
            and len(review.get('option_reasons',[])) == len(task['options'])
            and all(review['option_reasons']))


def specs(packet, label):
    groups=[g['id'] for g in packet['groups']]
    if not groups or len(set(groups))!=len(groups): raise ValueError('group identity')
    ids=[t['id'] for t in packet['tasks']]
    if len(set(ids))!=len(ids): raise ValueError('task identity')
    if any(t['group'] not in groups for t in packet['tasks']): raise ValueError('unknown group')
    for g in groups:
        if sorted(t['form'] for t in packet['tasks'] if t['group']==g)!=sorted(FORMS): raise ValueError('incomplete group')
    result=[]
    for task in packet['tasks']:
        base=list(range(len(task['options'])))
        random.Random('stage-b-20260920:'+task['id']).shuffle(base)
        for variant in range(4):
            order=base[variant:]+base[:variant]
            result.append(dict(id=task['id'],group=task['group'],variant=variant,condition='closed',order=order,repeat=0))
    closed=list(result)
    if label=='p1':
        result += [dict(s,condition='evidence') for s in closed]
        selected=sorted(closed,key=lambda s:digest(s))[:24]
        for repeat in [1,2]: result += [dict(s,repeat=repeat) for s in selected]
    if label not in ('step120_current','p1'): raise ValueError('model label')
    return result


def messages(packet, spec):
    task=next(t for t in packet['tasks'] if t['id']==spec['id'])
    options=tuple(task['options'][j] for j in spec['order'])
    item=EvalItem(task['id'],'Logistika-screen',task['id'],task['group'],'multiple_choice',task['question'],options,())
    result=build_messages(item)
    if spec['condition']=='evidence':
        source=next(g for g in packet['groups'] if g['id']==task['group'])['source_text']
        result[1]['content']='Source evidence (no answer key):\n'+source+'\n\n'+result[1]['content']
    return result


def summarize(packet, raw, label):
    plan=specs(packet,label)
    if len(raw)!=len(plan): raise ValueError('incomplete predictions')
    tasks={t['id']:t for t in packet['tasks']};out=[]
    for s,r in zip(plan,raw):
        if any(r[k]!=v for k,v in s.items()): raise ValueError('prediction ordering')
        if r['messages_sha256'] != digest(messages(packet,s)): raise ValueError('prompt mismatch')
        t=tasks[s['id']];answer,valid=parse_answers(r['text'],len(t['options']))
        valid=valid and r['finish_reason']=='stop' and r['output_tokens']<=96
        original=sorted(s['order'][j] for j in answer) if valid else None
        out.append(dict(s,valid=valid,correct=original==t['correct_indices'],answer_in_original_order=original))
    groups=[]
    for g in packet['groups']:
        ids=[t['id'] for t in packet['tasks'] if t['group']==g['id']]
        row=dict(group=g['id'],items=len(ids))
        for condition in ['closed']+(['evidence'] if label=='p1' else []):
            row[condition+'_stable_correct']=sum(all(x['correct'] for x in out if x['id']==i and x['condition']==condition and x['repeat']==0) for i in ids)
        if label=='p1':row['training_value_screen_pass']=(len(ids)==4 and row['closed_stable_correct']<=2 and row['evidence_stable_correct']>=3 and row['evidence_stable_correct']>=row['closed_stable_correct']+1)
        groups.append(row)
    changes=[]
    for r in out:
        if r['repeat']:
            original=next(x for x in out if x['id']==r['id'] and x['variant']==r['variant'] and x['condition']=='closed' and x['repeat']==0)
            changes.append(r['answer_in_original_order']!=original['answer_in_original_order'])
    return dict(label=label,calls=len(out),invalid=sum(not r['valid'] for r in out),groups=groups,
                stability_extra_calls=len(changes),stability_answer_changes=sum(changes),training_allowed=False)

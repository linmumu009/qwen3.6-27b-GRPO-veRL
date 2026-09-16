"""Reconstruct draft provenance and reviewer gates; identify duplicates for quarantine."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from cpt_transfer_draft import generation_prompt, review_prompt, validate_task, normalized_question


def records(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def parse(text):
    s=text.strip()
    if s.startswith('```'):
        s=s.split('\n',1)[1].rsplit('```',1)[0].strip()
    value=json.loads(s)
    if not isinstance(value,dict):raise ValueError('not object')
    return value


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def review_accepts(record, task):
    try:
        review=parse(record['text']); answers=review['correct_indices']
        flags=('supported','unambiguous','self_contained','scope_preserved','not_answer_leaking','design_satisfied')
        return (record['finish_reason']=='stop' and all(review.get(k) is True for k in flags)
            and isinstance(answers,list) and 0<len(answers)<4 and len(set(answers))==len(answers)
            and all(type(v) is int and 0<=v<4 for v in answers)
            and sorted(answers)==task['correct_indices']
            and isinstance(review.get('option_reasons'),list) and len(review['option_reasons'])==4)
    except (KeyError,ValueError,TypeError):return False


def verify(packet, base, excluded_paths=()):
    requests=records(packet);by_request={r['id']:r for r in requests}
    reg=json.loads((base/'registration.safe.json').read_text());assert reg['packet_sha256']==sha(packet)
    raw=records(base/'generation.private.jsonl');assert len(raw)==len(requests)
    assert len({r['request_id'] for r in raw})==len(raw) and {r['request_id'] for r in raw}==set(by_request)
    rebuilt=[];truncated=0
    for r in raw:
        request=by_request[r['request_id']]
        assert r['prompt_text_sha256']==hashlib.sha256(generation_prompt(request).encode()).hexdigest()
        budget=4096 if request['split']=='train' else 2560
        assert r['output_tokens']<=budget and r['prompt_tokens']+budget<=8192
        truncated+=r['finish_reason']!='stop'
        if r['finish_reason']!='stop':continue
        try:
            obj=parse(r['text']);tasks=obj['tasks'];seen=set()
            if not isinstance(tasks,list) or len(tasks)>len(request['designs']):continue
            for task in tasks:
                try:
                    task=validate_task(task,request)
                    if task['design'] in seen:continue
                    seen.add(task['design'])
                    rebuilt.append(dict(id=request['id']+'-'+task['design'],request_id=request['id'],unit=request['unit'],split=request['split'],category=request['category'],task=task,training_allowed=False))
                except (ValueError,KeyError,TypeError,ZeroDivisionError,SyntaxError):pass
        except (ValueError,KeyError,TypeError):pass
    candidates=records(base/'candidates.private.jsonl');assert candidates==rebuilt
    reviews=records(base/'review.private.jsonl');assert len(reviews)==len(candidates)
    assert len({r['id'] for r in reviews})==len(reviews)
    by_review={r['id']:r for r in reviews};assert set(by_review)=={c['id'] for c in candidates}
    accepted=[]
    for c in candidates:
        r=by_review[c['id']]
        assert r['prompt_text_sha256']==hashlib.sha256(review_prompt(c['task'],by_request[c['request_id']]).encode()).hexdigest()
        assert r['output_tokens']<=1536 and r['prompt_tokens']+1536<=8192
        if review_accepts(r,c['task']):
            accepted.append(dict(c,review=parse(r['text']),training_ready=False,review_status='same_model_filtered_only'))
    filtered=records(base/'filtered.private.jsonl');assert filtered==accepted
    for split in ('train','dev','sealed','retention'):
        assert records(base/(split+'.draft.private.jsonl'))==[c for c in filtered if c['split']==split]
    signatures=defaultdict(list)
    for c in filtered:signatures[normalized_question(c['task'])].append(c)
    duplicate_groups=[[c['id'] for c in cs] for cs in signatures.values() if len(cs)>1]
    # Diagnostic text is read only by this downstream auditor, never the generator.
    excluded=set();excluded_inputs=[]
    def collect(value):
        if isinstance(value,dict):
            if isinstance(value.get('question'),str):excluded.add(normalized_question(value))
            for v in value.values():collect(v)
        elif isinstance(value,list):
            for v in value:collect(v)
    for path in excluded_paths:
        data=records(path);collect(data);excluded_inputs.append({'sha256':sha(path),'rows':len(data)})
    matches=[c['id'] for c in filtered if normalized_question(c['task']) in excluded]
    units=defaultdict(Counter)
    for c in filtered:units[c['unit']][c['split']]+=1
    complete=[u for u,n in units.items() if n['train']==6 and n['dev']==3 and n['sealed']==1]
    return dict(packet_sha256=sha(packet),raw_files={p.name:sha(p) for p in base.glob('*.jsonl')},
        generation_requests=len(raw),review_requests=len(reviews),candidate_items=len(candidates),filtered_items=len(filtered),
        filtered_counts=dict(Counter(c['split'] for c in filtered)),complete_6_3_1_units=len(complete),
        unit_counts={u:dict(n) for u,n in units.items() if not u.startswith('retention')},
        generation_truncations=truncated,review_truncations=sum(r['finish_reason']!='stop' for r in reviews),
        arithmetic_equalities_checked=sum(len(c['task']['arithmetic_checks']) for c in candidates),
        normalized_duplicate_groups=duplicate_groups,diagnostic_exact_or_number_normalized_matches=matches,
        diagnostic_exclusion_inputs=excluded_inputs,diagnostic_questions_checked=len(excluded),
        raw_provenance_verified=True,training_ready=False,training_allowed=False,
        limitations='Mechanical provenance, exact rational expressions and same-model reviewer decisions verified; scenario-to-equation correctness, semantic novelty, answer uniqueness, retention representativeness and split separation are not certified. Duplicate checks are a floor, never proof of semantic independence. Complete 6/3/1 counts are only counts of provisional drafts. No task is released for training.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--requests',type=Path,required=True);p.add_argument('--result',type=Path,required=True)
    p.add_argument('--exclude',type=Path,action='append',default=[]);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    summary=verify(a.requests,a.result,a.exclude)
    a.out.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('raw_files','unit_counts','diagnostic_exclusion_inputs')},indent=2))

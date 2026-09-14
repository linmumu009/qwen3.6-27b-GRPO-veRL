"""Local source-only task generation and blind source review; never starts training."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path('/workspace/llin-verl-grpo')
SOURCE_SHA = '5df7dc2b209938986cd7081aecdc9105fb989c9a872d572ce0a0c5a16dbbf8a4'
BASE = ROOT/'runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource'
FORMS = ('definition_discrimination', 'boundary_or_counterexample', 'application_with_new_entities')


def parse_object(text):
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError('expected object')
    return value


def group_sources(rows):
    # Merge shared source groups AND supplied topics; do not split related units
    # across development/train merely because they cite different documents.
    parent = {r['id']: r['id'] for r in rows}
    def find(k):
        while k != parent[k]:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    seen = {}
    for row in sorted(rows, key=lambda r:r['id']):
        for key in ['evidence:'+row['evidence_group']]+['topic:'+t for t in row['topics']]:
            if key in seen:
                left, right = sorted((find(row['id']), find(seen[key])))
                parent[right] = left
            seen[key] = row['id']
    return {r['id']: find(r['id']) for r in rows}


def validate_task(task, row):
    if task.get('form') not in FORMS:
        raise ValueError('unsupported form')
    question = task['question']
    options = task['options']
    correct = task['correct_indices']
    if not isinstance(question, str) or not 15 <= len(question) <= 1800:
        raise ValueError('invalid question')
    if not isinstance(options, list) or not 3 <= len(options) <= 6 or any(not isinstance(o, str) or not o.strip() for o in options):
        raise ValueError('invalid options')
    if len({o.strip().casefold() for o in options}) != len(options):
        raise ValueError('duplicate options')
    if not isinstance(correct, list) or not correct or any(type(i) is not int or not 0 <= i < len(options) for i in correct) or len(set(correct)) != len(correct) or len(correct) >= len(options):
        raise ValueError('invalid answer')
    quote = task['source_quote']
    sources = [row['source_text']]+[e['excerpt'] for e in row['evidence']]
    if not isinstance(quote, str) or len(quote) < 12 or not any(quote in s for s in sources):
        raise ValueError('support quote not found verbatim')
    if not isinstance(task['explanation'], str) or not task['explanation'].strip():
        raise ValueError('missing rationale')
    return dict(task, correct_indices=sorted(correct))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--requests', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if hashlib.sha256(a.requests.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('unregistered source requests')
    rows = [json.loads(x) for x in a.requests.read_text(encoding='utf-8').splitlines()]
    if len(rows) != 205 or len({r['id'] for r in rows}) != 205:
        raise ValueError('unexpected source release')
    groups = group_sources(rows)
    split = {k: 'dev' if int(hashlib.sha256(v.encode()).hexdigest(),16)%5 == 0 else 'train' for k,v in groups.items()}
    os.umask(0o077)
    a.out.mkdir(exist_ok=False)
    def save(name, value):
        (a.out/name).write_text(json.dumps(value, indent=2), encoding='utf-8')
    save('status.safe.json', dict(status='preparing', training_started=False))
    try:
        with (ROOT/'runs/.logistics-exam-cpt.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
            tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True, local_files_only=True)
            def encode(prompt):
                ids = tok.encode(tok.apply_chat_template([dict(role='user',content=prompt)], tokenize=False,
                              add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)
                if len(ids)+2048 > 8192:
                    raise ValueError('source request exceeds context budget')
                return dict(prompt_token_ids=ids)
            prompts = []
            for row in rows:
                source = json.dumps({k:row[k] for k in ('title','scope','source_text','evidence')}, ensure_ascii=False)
                prompts.append(encode('Create at most one rigorous closed-book multiple-choice task per form: '+', '.join(FORMS)+'. Use only facts in SOURCE. Invent independent scenarios and entities. Stay within the stated scope; do not invent company rules. The question must stand alone without quoting the source or referring to it. Use the language of the source. Include plausible distractors that test conditions, not trivia. Skip unsupported forms. Return JSON only: {"tasks":[{"form":"...","question":"...","options":["..."],"correct_indices":[0],"explanation":"minimal source-grounded reasoning","source_quote":"exact contiguous support quote"}],"skipped":[{"form":"...","reason":"..."}]}. 3-6 options; indices start at zero. SOURCE:\n'+source))
            save('registration.safe.json', dict(model=str(BASE), source_sha256=SOURCE_SHA, source_units=205,
                 groups=len(set(groups.values())), splits=dict(Counter(split.values())),
                 generation_seed=2048, review_seed=2049, max_tokens=2048, max_model_len=8192,
                 blind_review_same_model=True, training_ready=False,
                 review_limit='Same-model blind review is a filter, not independent factual certification. Semantic duplicate review and baseline probes remain required.'))
            llm = LLM(model=str(BASE), tensor_parallel_size=8, dtype='bfloat16', trust_remote_code=True,
                      max_model_len=8192, max_num_seqs=16, gpu_memory_utilization=.8, seed=2048, enforce_eager=True)
            candidates = []
            rejected = Counter()
            with (a.out/'generation.private.jsonl').open('x', encoding='utf-8') as raw:
                for start in range(0, len(rows), 8):
                    results = llm.generate(prompts[start:start+8], SamplingParams(temperature=.4, max_tokens=2048, seed=2048))
                    results = sorted(results, key=lambda r:int(r.request_id))
                    assert len(results)==len(rows[start:start+8])
                    for row, prompt, result in zip(rows[start:start+8], prompts[start:start+8], results):
                        assert list(result.prompt_token_ids)==prompt['prompt_token_ids']
                        answer=result.outputs[0]
                        raw.write(json.dumps(dict(id=row['id'], text=answer.text, finish_reason=answer.finish_reason))+'\n')
                        if answer.finish_reason!='stop':
                            rejected['generation_truncated']+=1; continue
                        try:
                            obj=parse_object(answer.text); tasks=obj['tasks']
                            if not isinstance(tasks,list) or len(tasks)>3: raise ValueError('invalid task list')
                            used=set()
                            for task in tasks:
                                try:
                                    task=validate_task(task,row)
                                    if task['form'] in used: raise ValueError('duplicate form')
                                    used.add(task['form'])
                                    candidates.append(dict(id=row['id']+'-'+task['form'], unit_id=row['id'],
                                         group=groups[row['id']], split=split[row['id']], task=task, source=row['source_text']))
                                except (ValueError,KeyError,TypeError): rejected['invalid_task']+=1
                        except (ValueError,KeyError,TypeError): rejected['invalid_generation_json']+=1
                    raw.flush()
                    save('status.safe.json',dict(status='generating',source_units_completed=min(start+8,len(rows)),candidates=len(candidates)))
            # Reviewer sees source and task, but not the proposed answer/rationale.
            seen = set()
            accepted = []
            with (a.out/'review.private.jsonl').open('x',encoding='utf-8') as raw:
                for start in range(0,len(candidates),8):
                    batch=candidates[start:start+8]
                    prompts=[encode('Independently solve and audit the task using ONLY SOURCE. Check every option, unique complete answer set, scope, ambiguity and self-contained wording. Return JSON only: {"correct_indices":[0],"supported":true,"unambiguous":true,"self_contained":true,"reason":"..."}. Do not assume unavailable rules. SOURCE:\n'+c['source']+'\nTASK:\n'+json.dumps({k:c['task'][k] for k in ('question','options')},ensure_ascii=False)) for c in batch]
                    results=sorted(llm.generate(prompts,SamplingParams(temperature=0,max_tokens=2048,seed=2049)),key=lambda r:int(r.request_id))
                    assert len(results)==len(batch)
                    for c,prompt,result in zip(batch,prompts,results):
                        assert list(result.prompt_token_ids)==prompt['prompt_token_ids']
                        answer=result.outputs[0]
                        raw.write(json.dumps(dict(id=c['id'],text=answer.text,finish_reason=answer.finish_reason))+'\n')
                        try:
                            review=parse_object(answer.text)
                            if answer.finish_reason!='stop' or any(review.get(k) is not True for k in ('supported','unambiguous','self_contained')):raise ValueError('review rejected')
                            if sorted(review['correct_indices'])!=c['task']['correct_indices']:raise ValueError('answer disagreement')
                            key=re.sub(r'\s+',' ',json.dumps([c['task']['question'],c['task']['options']],ensure_ascii=False).casefold())
                            if key in seen:raise ValueError('duplicate task')
                            seen.add(key)
                            accepted.append(dict(c,review=review,review_status='blind_model_filtered_pending_manual_semantic_audit',training_ready=False))
                        except (ValueError,KeyError,TypeError):rejected['review_or_duplicate_rejected']+=1
                    raw.flush()
                    save('status.safe.json',dict(status='reviewing',reviewed=min(start+8,len(candidates)),candidates=len(candidates),filtered=len(accepted)))
            with (a.out/'filtered_tasks.private.jsonl').open('x',encoding='utf-8') as f:
                for c in accepted:f.write(json.dumps(c,ensure_ascii=False)+'\n')
            save('summary.safe.json',dict(source_units=len(rows),candidates=len(candidates),filtered=len(accepted),
                 splits=dict(Counter(c['split'] for c in accepted)),rejected=dict(rejected),training_ready=False))
            save('status.safe.json',dict(status='generation_review_complete_manual_audit_pending',training_started=False))
    except BaseException as exc:
        save('status.safe.json',dict(status='failed',error=type(exc).__name__,detail=str(exc)))
        raise


if __name__=='__main__': main()

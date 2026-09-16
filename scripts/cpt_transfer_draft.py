"""Generate quarantined source-based train/dev/sealed/retention drafts, then blind-filter.

This program cannot start training or run official/diagnostic questions. Outputs
stay training_allowed=False even when the same-model reviewer agrees.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re

from run_cpt_transfer_screen import MODEL, LOCK, digest, parse_object, indices

DESIGN_GUIDE = {
    'single_case_application':'Apply one rule to a fully specified operating case.',
    'boundary_pair':'Compare two fully specified cases immediately on different sides of a rule boundary.',
    'explicit_exclusion':'Apply a stated scenario exception to a classification or operating choice.',
    'correct_a_report':'Choose justified corrections to a short flawed operational report.',
    'compare_two_actions':'Compare two feasible actions under an explicit common objective and assumptions.',
    'conditional_counterexample':'Choose a concrete counterexample to an operational claim, without quoting the governing rule.',
    'dev_expression_ledger':'Use a compact record ledger or event log with a new vocabulary and scenario; infer a combined report. Do not use a direct definition, boundary pair, report correction, action comparison or counterexample.',
    'dev_conditions_reverse_inference':'Infer which missing conditions can explain a given outcome from several candidate completions. Combine at least two independently meaningful source conditions; do not use any training-design form.',
    'dev_conditions_minimal_change':'Start from a specified failed objective and select the smallest feasible bundle of changes that makes it pass, with interactions and an explicit change budget; do not merely change one number across a threshold.',
    'sealed_two_stage_decision':'Use a new two-stage operating decision with an intermediate observation and explicit constraints; answers must be complete contingent policies. No direct classification, pair comparison, record ledger, reverse inference or minimal-change budget.',
    'retention_application':'Apply the source to a new nontrivial operational situation.',
    'retention_boundary':'Distinguish included and excluded cases in a new operational situation.',
    'retention_decision':'Choose a justified action under explicit assumptions in a new operational situation.',
}


def source_constraints(request):
    """Optional audited source rules; legacy prompts remain byte-for-byte stable."""
    constraints = request.get('source_constraints')
    if constraints is None:
        return None
    if set(constraints) != {'required_question_terms', 'rules'}:
        raise ValueError('constraint_schema')
    terms = constraints['required_question_terms']
    if not isinstance(terms, list) or not terms or any(not isinstance(t, str) or not t.strip() for t in terms):
        raise ValueError('constraint_terms')
    by_id = {s['id']: s for s in request['sources']}
    if not isinstance(constraints['rules'], list) or not constraints['rules']:
        raise ValueError('constraint_rules')
    for rule in constraints['rules']:
        if set(rule) != {'source_id', 'quote', 'instruction'}:
            raise ValueError('constraint_rule_schema')
        if rule['source_id'] not in by_id or not isinstance(rule['quote'], str) or len(rule['quote']) < 20 or rule['quote'] not in by_id[rule['source_id']]['source_text']:
            raise ValueError('constraint_quote')
        if not isinstance(rule['instruction'], str) or not rule['instruction'].strip():
            raise ValueError('constraint_instruction')
    return constraints


def generation_prompt(request):
    payload={'sources':[{k:s[k] for k in ('title','scope','source_text')} for s in request['sources']],
             'designs':{d:DESIGN_GUIDE[d] for d in request['designs']}}
    constraints = source_constraints(request)
    if constraints is not None:
        payload['audited_source_constraints'] = constraints
    return '''Write one new multiple-choice logistics task for EACH assigned design using only SOURCE facts, plus explicitly given scenario assumptions. Follow each design literally; different wording or numbers alone is not a new reasoning structure. If the source cannot support a design without invented facts or ambiguous choices, SKIP that design and explain why. Do not force task counts.
Preserve the named source framework and historical/statistical scope in the question. Do not present archived definitions as current law. No quoted governing rule or source evidence in the question; the question is closed book. Provide every scenario premise, unit, objective and boundary needed for a unique answer set. Never ask an undefined 'best' choice. Use 4 distinct plausible options and ask 'Select all correct statements.' Avoid all/none-of-the-above and redundant claims. Each distractor must violate a specific rule or scenario fact. Use English. Create genuinely new scenarios; no benchmark questions. Check every arithmetic step; if numerical, supply arithmetic_checks using exact rational expressions with integers and + - * / parentheses only. A fraction result is fine. Keep each question under 160 words, each option under 35 words, and rationales concise. Explanations are answer-key audit material, not model inputs.
Return JSON only: {"tasks":[{"design":"assigned key","question":"... Select all correct statements.","options":["...","...","...","..."],"correct_indices":[0],"source_quote":"an exact contiguous source span","explanation":"short verified reasoning","option_reasons":["why true/false", "..."],"arithmetic_checks":[["integer expression","expected exact expression"]],"reasoning_structure":"conditions used and steps required"}],"skipped":[{"design":"assigned key","reason":"..."}]}.
SOURCE AND ASSIGNMENTS:
'''+json.dumps(payload,ensure_ascii=False)


def validate_task(task, request):
    from cpt_composed_probe import arithmetic
    if task.get('design') not in request['designs']: raise ValueError('unassigned_design')
    if not isinstance(task.get('question'),str) or not re.search(r'\bselect all correct\b',task['question'],re.I):raise ValueError('question_scope')
    constraints = source_constraints(request)
    if constraints is not None and any(term.casefold() not in task['question'].casefold() for term in constraints['required_question_terms']):
        raise ValueError('missing_registered_scope')
    opts=task.get('options')
    if not isinstance(opts,list) or len(opts)!=4 or any(not isinstance(o,str) or not o.strip() for o in opts):raise ValueError('options')
    if len({o.strip().casefold() for o in opts})!=4:raise ValueError('duplicate_options')
    correct=indices(task.get('correct_indices'),4)
    quote=task.get('source_quote')
    if not isinstance(quote,str) or len(quote)<20 or not any(quote in s['source_text'] for s in request['sources']):raise ValueError('source_quote')
    reasons=task.get('option_reasons')
    if not isinstance(reasons,list) or len(reasons)!=4 or any(not isinstance(s,str) or not s.strip() for s in reasons):raise ValueError('option_rationale')
    if not all(isinstance(task.get(k),str) and task[k].strip() for k in ('explanation','reasoning_structure')):raise ValueError('audit_fields')
    if not isinstance(task.get('arithmetic_checks'),list):raise ValueError('arithmetic_checks')
    for check in task['arithmetic_checks']:
        if not isinstance(check,list) or len(check)!=2 or arithmetic(check[0])!=arithmetic(check[1]):raise ValueError('arithmetic_failed')
    return dict(task,correct_indices=correct)


def review_prompt(task, request):
    # No generated answer, rationale, quote, or calculation result goes to reviewer.
    payload={'sources':[{k:s[k] for k in ('title','scope','source_text')} for s in request['sources']],
             'task':{k:task[k] for k in ('question','options')},'required_design':DESIGN_GUIDE[task['design']]}
    constraints = source_constraints(request)
    if constraints is not None:
        payload['audited_source_constraints'] = constraints
    return '''Independently solve and audit this task against its archived sources. You have no proposed key. Check all options and recompute numbers. Reject unsupported facts, missing premises, vague source scope, ambiguity, an answer given away in the question, and failure to implement the required reasoning design. Do not invent a source rule to make a question work. Return JSON only: {"correct_indices":[0],"supported":true,"unambiguous":true,"self_contained":true,"scope_preserved":true,"not_answer_leaking":true,"design_satisfied":true,"option_reasons":["reason for each option"],"arithmetic_audit":"exact calculation or not numerical","reason":"overall audit"}.
INPUT:
'''+json.dumps(payload,ensure_ascii=False)


def normalized_question(task):
    return re.sub(r'\b\d+(?:\.\d+)?\b','#',re.sub(r'\s+',' ',task['question'].casefold())).strip()


def load_resume(path, expected_sha, requests):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha:raise ValueError('resume_hash')
    records=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
    order=[r for split in ('train','dev','sealed','retention') for r in requests if r['split']==split]
    if not records or len(records)>len(order):raise ValueError('resume_length')
    for record,request in zip(records,order):
        if record['request_id']!=request['id'] or record['prompt_text_sha256']!=hashlib.sha256(generation_prompt(request).encode()).hexdigest():raise ValueError('resume_lineage')
    return {r['request_id']:{k:v for k,v in r.items() if k!='request_id'} for r in records}


def main():
    import fcntl
    p=argparse.ArgumentParser();p.add_argument('--requests',type=Path,required=True);p.add_argument('--sha',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume-generation',type=Path);p.add_argument('--resume-sha');a=p.parse_args();os.umask(0o077)
    assert hashlib.sha256(a.requests.read_bytes()).hexdigest()==a.sha
    requests=[json.loads(s) for s in a.requests.read_text(encoding='utf-8').splitlines()]
    assert requests and len({r['id'] for r in requests})==len(requests)
    assert all(r['training_allowed'] is False and r['purpose']=='unreviewed_draft' for r in requests)
    if bool(a.resume_generation)!=bool(a.resume_sha):raise ValueError('resume_file_and_hash_required')
    cached=load_resume(a.resume_generation,a.resume_sha,requests) if a.resume_generation else {}
    a.out.mkdir(exist_ok=False)
    def save(n,v):(a.out/n).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
    def status(stage,**kw):save('status.safe.json',dict(status=stage,training_running=False,training_ready=False,**kw))
    status('waiting_for_lock')
    try:
        with open(LOCK,'a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            from transformers import AutoTokenizer
            from vllm import LLM,SamplingParams
            tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True)
            def encode(text,budget):
                rendered=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                ids=tok.encode(rendered,add_special_tokens=False)
                if len(ids)+budget>8192:raise ValueError('context_overflow')
                return {'prompt_token_ids':ids}
            # Validate every generation prompt before allocating the model.
            for r in requests:
                prompt=encode(generation_prompt(r),4096 if r['split']=='train' else 2560)
                if r['id'] in cached and cached[r['id']]['prompt_token_sha256']!=digest(prompt['prompt_token_ids']):raise ValueError('resume_token_mismatch')
            save('registration.safe.json',dict(packet_sha256=a.sha,requests=len(requests),model=MODEL,
                source_only=True,thinking=False,tp=8,max_model_len=8192,max_num_seqs=16,
                generation_seed=20918,generation_temperature=.4,review_seed=20919,
                generation_train_budget=4096,generation_other_budget=2560,review_budget=1536,
                official_protocol_changed=False,training_allowed=False,reused_generation_records=len(cached),resume_generation_sha256=a.resume_sha))
            status('loading_model')
            llm=LLM(model=MODEL,tensor_parallel_size=8,dtype='bfloat16',trust_remote_code=True,
                max_model_len=8192,max_num_seqs=16,gpu_memory_utilization=.8,seed=20918,enforce_eager=True)
            def run(texts,budget,seed,temp):
                prompts=[encode(t,budget) for t in texts]
                outputs=sorted(llm.generate(prompts,SamplingParams(temperature=temp,max_tokens=budget,seed=seed)),key=lambda x:int(x.request_id))
                assert len(outputs)==len(prompts)
                records=[]
                for text,prompt,output in zip(texts,prompts,outputs):
                    assert list(output.prompt_token_ids)==prompt['prompt_token_ids']
                    ans=output.outputs[0]
                    records.append(dict(text=ans.text,finish_reason=ans.finish_reason,output_tokens=len(ans.token_ids),
                        prompt_text_sha256=hashlib.sha256(text.encode()).hexdigest(),prompt_token_sha256=digest(prompt['prompt_token_ids']),prompt_tokens=len(prompt['prompt_token_ids'])))
                return records
            candidates=[];rejected=Counter();by_id={r['id']:r for r in requests}
            status('generating_drafts',requests_completed=0,requests=len(requests),candidate_items=0)
            with (a.out/'generation.private.jsonl').open('x',encoding='utf-8') as raw:
                completed=0
                for split in ('train','dev','sealed','retention'):
                    subset=[r for r in requests if r['split']==split]
                    for start in range(0,len(subset),8):
                        batch=subset[start:start+8];budget=4096 if split=='train' else 2560
                        missing=[r for r in batch if r['id'] not in cached]
                        fresh=run([generation_prompt(r) for r in missing],budget,20918,.4) if missing else []
                        results=dict(cached);results.update({r['id']:v for r,v in zip(missing,fresh)})
                        for r in batch:
                            record=results[r['id']]
                            raw.write(json.dumps(dict(request_id=r['id'],**record),ensure_ascii=False)+'\n');raw.flush()
                            try:
                                if record['finish_reason']!='stop':raise ValueError('generation_truncated')
                                obj=parse_object(record['text']);tasks=obj['tasks'];seen=set()
                                if not isinstance(tasks,list) or len(tasks)>len(r['designs']):raise ValueError('task_count')
                                for task in tasks:
                                    try:
                                        task=validate_task(task,r)
                                        if task['design'] in seen:raise ValueError('duplicate_design')
                                        seen.add(task['design'])
                                        candidates.append(dict(id=r['id']+'-'+task['design'],request_id=r['id'],unit=r['unit'],split=split,category=r['category'],task=task,training_allowed=False))
                                    except (ValueError,KeyError,TypeError,ZeroDivisionError,SyntaxError) as e:rejected['task_'+str(e)]+=1
                                rejected['skipped_or_missing_designs']+=len(r['designs'])-len(seen)
                            except (ValueError,KeyError,TypeError) as e:rejected['generation_'+str(e)]+=1
                        completed+=len(batch);status('generating_drafts',requests_completed=completed,requests=len(requests),candidate_items=len(candidates))
            (a.out/'candidates.private.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in candidates),encoding='utf-8')
            filtered=[]
            status('reviewing_drafts',reviewed=0,candidates=len(candidates),filtered=0)
            with (a.out/'review.private.jsonl').open('x',encoding='utf-8') as raw:
                for start in range(0,len(candidates),8):
                    batch=candidates[start:start+8]
                    texts=[review_prompt(c['task'],by_id[c['request_id']]) for c in batch]
                    for c,record in zip(batch,run(texts,1536,20919,0)):
                        raw.write(json.dumps(dict(id=c['id'],**record),ensure_ascii=False)+'\n');raw.flush()
                        try:
                            review=parse_object(record['text'])
                            if record['finish_reason']!='stop' or any(review.get(k) is not True for k in ('supported','unambiguous','self_contained','scope_preserved','not_answer_leaking','design_satisfied')):raise ValueError('review_rejected')
                            if indices(review.get('correct_indices'),4)!=c['task']['correct_indices']:raise ValueError('review_disagreement')
                            if not isinstance(review.get('option_reasons'),list) or len(review['option_reasons'])!=4:raise ValueError('review_missing_options')
                            filtered.append(dict(c,review=review,training_ready=False,review_status='same_model_filtered_only'))
                        except (ValueError,KeyError,TypeError) as e:rejected[str(e)]+=1
                    status('reviewing_drafts',reviewed=min(start+len(batch),len(candidates)),candidates=len(candidates),filtered=len(filtered))
            (a.out/'filtered.private.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in filtered),encoding='utf-8')
            for split in ('train','dev','sealed','retention'):
                (a.out/(split+'.draft.private.jsonl')).write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in filtered if c['split']==split),encoding='utf-8')
            save('summary.safe.json',dict(generation_requests=len(requests),candidate_items=len(candidates),filtered_items=len(filtered),
                candidate_counts=dict(Counter(c['split'] for c in candidates)),filtered_counts=dict(Counter(c['split'] for c in filtered)),
                rejected=dict(rejected),training_ready=False,training_allowed=False,
                limitations='Same-model filtered drafts only. Independent semantic, numeric, historical-source and cross-split template audit required. No baseline model has answered dev/sealed/retention items. No training was run.'))
            status('drafted_pending_independent_audit',generation_requests=len(requests),review_requests=len(candidates),filtered_items=len(filtered))
    except BaseException as e:
        status('failed',error=type(e).__name__,detail=str(e));raise


if __name__=='__main__':main()

"""Source-only generation, blind source filtering, and closed-book gap screening.

No training. No official questions or labels are read. Model-filtered labels remain
provisional until a separate semantic audit. All raw outputs are retained.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re

FORMS = ('boundary_counterexample', 'combined_conditions', 'decision_under_changed_conditions')
MODEL = '/workspace/llin-verl-grpo/runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource'
LOCK = '/workspace/llin-verl-grpo/runs/.logistics-exam-cpt.lock'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def parse_object(text):
    text = re.sub(r'^```(?:json)?\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text)
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError('expected object')
    return obj


def indices(value, count):
    if not isinstance(value, list) or not value or any(type(x) is not int or not 0 <= x < count for x in value):
        raise ValueError('invalid indices')
    if len(set(value)) != len(value) or len(value) >= count:
        raise ValueError('duplicate or all-answer indices')
    return sorted(value)


def parse_answer(text, count):
    matches = re.findall(r'\{\s*"answers"\s*:\s*\[[^\]]*\]\s*\}', text)
    if len(matches) != 1:
        return None
    try:
        return indices(json.loads(matches[0])['answers'], count)
    except (ValueError, KeyError, TypeError):
        return None


def validate_task(task, source):
    if task.get('form') not in FORMS:
        raise ValueError('unknown form')
    if not isinstance(task.get('question'), str) or not 30 <= len(task['question']) <= 2000:
        raise ValueError('question length')
    options = task.get('options')
    if not isinstance(options, list) or not 4 <= len(options) <= 6 or any(not isinstance(x, str) or not x.strip() for x in options):
        raise ValueError('invalid options')
    if len({x.strip().casefold() for x in options}) != len(options):
        raise ValueError('duplicate options')
    expected = indices(task.get('correct_indices'), len(options))
    if not isinstance(task.get('source_quote'), str) or len(task['source_quote']) < 20 or task['source_quote'] not in source['source_text']:
        raise ValueError('quote not exact source span')
    reasons = task.get('option_reasons')
    if not isinstance(reasons, list) or len(reasons) != len(options) or any(not isinstance(x, str) or not x.strip() for x in reasons):
        raise ValueError('missing option-level rationale')
    if not isinstance(task.get('explanation'), str) or not task['explanation'].strip():
        raise ValueError('missing explanation')
    return dict(task, correct_indices=expected)


def generation_prompt(row):
    source = {k: row[k] for k in ('title', 'scope', 'source_text')}
    return '''Create exactly one challenging, self-contained closed-book multiple-choice screening task for each form: boundary_counterexample, combined_conditions, decision_under_changed_conditions. Use ONLY SOURCE. If its facts cannot support a form, skip that form rather than invent a rule. Each task needs a different scenario and reasoning structure, not number substitutions or a definition with renamed nouns. Test scope, exclusions, interacting conditions, or consequences; no obscure trivia. Give all necessary scenario facts, but do not quote the governing rule or supply its answer in the question. When a definition belongs to a specific statistical or historical framework, name that framework or snapshot in the question. Do not generalize it into current law. Never use section numbers as answer cues. Use 4-6 distinct plausible options; say select all that apply if multiple answers. Distractors must violate a specific source condition, not just be unrelated concepts. Numerical tasks must be exactly calculable from provided facts; no unstated constants. Use the source language. Return JSON only:
{"tasks":[{"form":"...","question":"...","options":["..."],"correct_indices":[0],"source_quote":"verbatim contiguous source span","explanation":"minimal reasoning","option_reasons":["why option 0 holds/fails", "..."]}],"skipped":[{"form":"...","reason":"..."}]}
SOURCE:
''' + json.dumps(source, ensure_ascii=False)


def review_prompt(candidate, row):
    task = {k: candidate['task'][k] for k in ('question', 'options')}
    return '''Independently solve and audit TASK using only SOURCE. You have no proposed answer. Check every option. Reject if any required rule is missing, historical/statistical scope is unstated where material, an invented claim is needed, options overlap ambiguously, or the question supplies the governing answer. A scenario applying a known rule is acceptable; a quoted rule asking for its paraphrase is not. Recompute numbers. Return JSON only: {"correct_indices":[0],"supported":true,"unambiguous":true,"self_contained":true,"scope_preserved":true,"not_answer_leaking":true,"option_reasons":["..."],"reason":"..."}.
SOURCE:
''' + json.dumps({k: row[k] for k in ('title', 'scope', 'source_text')}, ensure_ascii=False) + '\nTASK:\n' + json.dumps(task, ensure_ascii=False)


def baseline_prompt(task):
    return 'Return only a JSON object with key "answers" containing all correct zero-based option indices.\nQuestion:\n' + task['question'] + '\nOptions:\n' + '\n'.join(f'[{i}] {x}' for i, x in enumerate(task['options']))


def load_generation_prefix(path, expected_sha, rows):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('resume fingerprint mismatch')
    cached = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
    if not cached or len(cached) > len(rows) or len(cached) % 8:
        raise ValueError('resume requires completed generation batches')
    for source, raw in zip(rows, cached):
        if raw['source_id'] != source['id'] or raw['prompt_text_sha256'] != hashlib.sha256(generation_prompt(source).encode()).hexdigest():
            raise ValueError('resume source order or prompt changed')
        if not all(k in raw for k in ('text','finish_reason','output_tokens','prompt_token_sha256','prompt_tokens')):
            raise ValueError('incomplete resume record')
    return cached


def main():
    import fcntl
    p = argparse.ArgumentParser()
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--sha', required=True)
    p.add_argument('--expected-source-count', type=int, default=64)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume-generation', type=Path)
    p.add_argument('--resume-sha')
    a = p.parse_args()
    os.umask(0o077)
    if hashlib.sha256(a.sources.read_bytes()).hexdigest() != a.sha:
        raise ValueError('source fingerprint mismatch')
    rows = [json.loads(x) for x in a.sources.read_text(encoding='utf-8').splitlines()]
    if a.expected_source_count <= 0 or len(rows) != a.expected_source_count or len({r['id'] for r in rows}) != a.expected_source_count or any(r['historical_split'] != 'train' or r['training_allowed'] is not False for r in rows):
        raise ValueError('unregistered source selection')
    if bool(a.resume_generation) != bool(a.resume_sha):
        raise ValueError('resume file and fingerprint required together')
    cached = load_generation_prefix(a.resume_generation, a.resume_sha, rows) if a.resume_generation else []
    a.out.mkdir(exist_ok=False)
    def save(name, value):
        (a.out / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    def status(stage, **kwargs):
        save('status.safe.json', dict(status=stage, training_running=False, **kwargs))
    status('waiting_for_lock')
    try:
        with open(LOCK, 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
            tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
            def encode(text, budget):
                rendered = tok.apply_chat_template([dict(role='user', content=text)], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                ids = tok.encode(rendered, add_special_tokens=False)
                if len(ids) + budget > 8192:
                    raise ValueError('context overflow')
                return dict(prompt_token_ids=ids)
            def run(texts, budget, seed, temperature=0):
                prompts = [encode(t, budget) for t in texts]
                outputs = sorted(llm.generate(prompts, SamplingParams(temperature=temperature, max_tokens=budget, seed=seed)), key=lambda x: int(x.request_id))
                if len(outputs) != len(prompts):
                    raise ValueError('output count mismatch')
                result = []
                for text, prompt, output in zip(texts, prompts, outputs):
                    if list(output.prompt_token_ids) != prompt['prompt_token_ids']:
                        raise ValueError('prompt/output alignment mismatch')
                    answer = output.outputs[0]
                    result.append(dict(text=answer.text, finish_reason=answer.finish_reason,
                        output_tokens=len(answer.token_ids), prompt_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                        prompt_token_sha256=digest(prompt['prompt_token_ids']), prompt_tokens=len(prompt['prompt_token_ids'])))
                return result
            save('registration.safe.json', dict(source_sha256=a.sha, source_units=64, model=MODEL,
                generation=dict(max_tokens=3072, seed=20916, temperature=.4),
                review=dict(max_tokens=1536, seed=20917, temperature=0),
                baseline=dict(max_tokens=96, seed=1024, temperature=0),
                tp=8, max_model_len=8192, max_num_seqs=16, thinking=False, repeats=1,
                official_protocol_changed=False, same_model_filter=True, training_ready=False,
                screening_not_independent_generalization=True,
                reused_generation_records=len(cached), resume_generation_sha256=a.resume_sha))
            status('loading_model')
            llm = LLM(model=MODEL, tensor_parallel_size=8, dtype='bfloat16', trust_remote_code=True, max_model_len=8192, max_num_seqs=16, gpu_memory_utilization=.8, seed=20916, enforce_eager=True)
            candidates, accepted, predictions = [], [], []
            rejected = Counter()
            by_id = {r['id']: r for r in rows}
            seen = set()
            with (a.out / 'generation.private.jsonl').open('x', encoding='utf-8') as raw:
                for start in range(0, len(rows), 8):
                    batch = rows[start:start+8]
                    if start < len(cached):
                        results = [{k:v for k,v in record.items() if k != 'source_id'} for record in cached[start:start+len(batch)]]
                        for row, result in zip(batch, results):
                            if result['prompt_token_sha256'] != digest(encode(generation_prompt(row),3072)['prompt_token_ids']):
                                raise ValueError('resume token fingerprint mismatch')
                    else:
                        results = run([generation_prompt(r) for r in batch], 3072, 20916, .4)
                    for row, result in zip(batch, results):
                        raw.write(json.dumps(dict(source_id=row['id'], **result), ensure_ascii=False) + '\n')
                        try:
                            if result['finish_reason'] != 'stop':
                                raise ValueError('generation_truncated')
                            obj = parse_object(result['text'])
                            tasks = obj['tasks']
                            if not isinstance(tasks, list) or len(tasks) > 3:
                                raise ValueError('invalid_task_count')
                            forms = set()
                            for task in tasks:
                                try:
                                    task = validate_task(task, row)
                                    if task['form'] in forms:
                                        raise ValueError('duplicate_form')
                                    forms.add(task['form'])
                                    signature = digest([task['question'], task['options']])
                                    if signature in seen:
                                        raise ValueError('duplicate_task')
                                    seen.add(signature)
                                    candidates.append(dict(id=row['id']+'-'+task['form'], unit_id=row['id'], group=row['group'], category=row['category'], task=task, training_allowed=False))
                                except (ValueError, KeyError, TypeError) as exc:
                                    rejected[str(exc)] += 1
                        except (ValueError, KeyError, TypeError) as exc:
                            rejected[str(exc)] += 1
                    raw.flush()
                    status('generating', sources_completed=start+len(batch), candidates=len(candidates))
            (a.out / 'candidates.private.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False)+'\n' for c in candidates), encoding='utf-8')
            with (a.out / 'review.private.jsonl').open('x', encoding='utf-8') as raw:
                for start in range(0, len(candidates), 8):
                    batch = candidates[start:start+8]
                    results = run([review_prompt(c, by_id[c['unit_id']]) for c in batch], 1536, 20917)
                    for c, result in zip(batch, results):
                        raw.write(json.dumps(dict(id=c['id'], **result), ensure_ascii=False)+'\n')
                        try:
                            review = parse_object(result['text'])
                            if result['finish_reason'] != 'stop' or any(review.get(k) is not True for k in ('supported','unambiguous','self_contained','scope_preserved','not_answer_leaking')):
                                raise ValueError('review_rejected')
                            if indices(review['correct_indices'],len(c['task']['options'])) != c['task']['correct_indices']:
                                raise ValueError('review_disagreement')
                            if not isinstance(review.get('option_reasons'), list) or len(review['option_reasons']) != len(c['task']['options']):
                                raise ValueError('review_missing_options')
                            accepted.append(dict(c, review=review, review_status='provisional_same_model_filtered', training_ready=False))
                        except (ValueError, KeyError, TypeError) as exc:
                            rejected[str(exc)] += 1
                    raw.flush()
                    status('reviewing', reviewed=start+len(batch), candidates=len(candidates), filtered=len(accepted))
            (a.out / 'filtered.private.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False)+'\n' for c in accepted), encoding='utf-8')
            with (a.out / 'baseline.private.jsonl').open('x', encoding='utf-8') as raw:
                for start in range(0, len(accepted), 16):
                    batch = accepted[start:start+16]
                    results = run([baseline_prompt(c['task']) for c in batch], 96, 1024)
                    for c, result in zip(batch, results):
                        parsed = parse_answer(result['text'], len(c['task']['options']))
                        record = dict(id=c['id'], **result, parsed=parsed,
                            provisional_correct=parsed == c['task']['correct_indices'] and result['finish_reason']=='stop')
                        predictions.append(record)
                        raw.write(json.dumps(record,ensure_ascii=False)+'\n')
                    raw.flush()
                    status('baseline_screening', completed=start+len(batch), total=len(accepted))
            save('summary.safe.json',dict(source_units=len(rows), candidates=len(candidates), filtered=len(accepted),
                screened=len(predictions), provisional_correct=sum(r['provisional_correct'] for r in predictions),
                invalid=sum(r['parsed'] is None for r in predictions), truncated=sum(r['finish_reason']!='stop' for r in predictions),
                rejected=dict(rejected), training_ready=False, manual_semantic_review_required=True))
            status('screened_pending_independent_semantic_review', requests=len(rows)+len(candidates)+len(predictions))
    except BaseException as exc:
        status('failed', error=type(exc).__name__, detail=str(exc))
        raise


if __name__ == '__main__':
    main()

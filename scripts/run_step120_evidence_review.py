"""Eight-item source-repaired evaluation, with private raw output telemetry. No training."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

from scripts.evaluate_logistics_knowledge import parse_answers
from scripts.prepare_step120_qa_diagnostic import digest, write_json
from scripts.rewrite_logistics_exam_stems import _extract_object
from scripts.run_step120_qa_diagnostic import audit_messages, evaluation_messages, majority, question_item


def add_type_hint(messages, question_type):
    messages = [dict(m) for m in messages]
    if question_type in ('single_choice', 'true_or_false'):
        hint = 'This task has exactly one answer. Return exactly one option index.'
    else:
        hint = 'This is a multiple-choice selection task. Return all and only the correct indices.'
    messages[1]['content'] += '\n\nQuestion-type clarification: ' + hint
    return messages


def make_tasks(rows, keys, references):
    tasks = []
    by_hash = {r['item_hash']: r for r in rows}
    if len(keys) != 8 or len({k['item_hash'] for k in keys}) != 8:
        raise ValueError('expected exactly eight authorized unique cases')
    if not set(references['items']) <= {k['review_id'] for k in keys}:
        raise ValueError('unknown reference review id')
    for key in keys:
        alias = key['review_id']
        row = by_hash[key['item_hash']]
        item = question_item(row)
        evidence = references['items'].get(alias)
        conditions = {'original_closed': evaluation_messages(item),
                      'original_with_book': evaluation_messages(item, row['evidence']),
                      'original_with_type_hint': add_type_hint(evaluation_messages(item), row['question_type'])}
        if evidence:
            conditions['original_with_reviewed_reference'] = evaluation_messages(item, evidence)
            conditions['original_with_reviewed_reference_type_hint'] = add_type_hint(
                evaluation_messages(item, evidence), row['question_type'])
        for condition, messages in conditions.items():
            tasks.append({'review_id': alias, 'condition': condition, 'messages': messages,
                          'expected': list(item.expected), 'option_count': len(item.options)})
    return tasks


def safe_results(results, traces, audit_traces):
    names = sorted({name for r in results.values() for name in r})
    conditions = {}
    for name in names:
        subset = [r[name] for r in results.values() if name in r]
        conditions[name] = {'items': len(subset), 'correct': sum(r['correct'] for r in subset),
                            'no_majority': sum(not r['parse_ok'] for r in subset),
                            'unstable': sum(not r['all_repeats_same'] for r in subset)}
    covered = [r for r in results.values() if 'original_with_reviewed_reference' in r]
    paired = {'items': len(covered)}
    for name in ('original_closed', 'original_with_book', 'original_with_reviewed_reference',
                 'original_with_reviewed_reference_type_hint'):
        paired[name + '_correct'] = sum(r[name]['correct'] for r in covered)
    paired['closed_wrong_reference_correct'] = sum(not r['original_closed']['correct'] and
        r['original_with_reviewed_reference']['correct'] for r in covered)
    paired['closed_correct_reference_wrong'] = sum(r['original_closed']['correct'] and
        not r['original_with_reviewed_reference']['correct'] for r in covered)
    audit_summary = {}
    for budget in (512, 1536):
        group = [r for r in audit_traces if r['budget'] == budget]
        audit_summary[str(budget)] = {'items': len(group),
            'json_objects': sum(isinstance(r['parsed'], dict) for r in group),
            'length_terminated': sum(r['finish_reason'] == 'length' for r in group)}
    return {'private_content_included': False, 'training_performed': False, 'model': 'Step120',
            'status': 'complete_eight_item_source_review', 'items': len(results), 'repeats': 3,
            'conditions': conditions, 'paired_reviewed_source_subset': paired,
            'audit_generation_budget_probe': audit_summary,
            'answer_generation_finish_reasons': dict(Counter(r['finish_reason'] for r in traces)),
            'human_expert_reviewed': 0, 'agent_reviewed': 8,
            'source_enrichment_after_gold_reveal': True,
            'limitations': ['eight selected historical errors, not an unbiased estimate',
                'source packets are agent paraphrases reviewed after gold reveal, not blind retrieval',
                'four cases withheld from repaired-reference inference for unresolved input/scope issues',
                'exact original labels and options preserved; no alternative semantic scores replace benchmark scores',
                'new audit budget probe does not establish cause of historical null audits without old raw outputs',
                'successful open-book answers do not prove missing parametric knowledge']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('refusing overwrite')
    original = args.root / 'runs/step120-qa-diagnostic-20260907'
    rows = json.loads((original / 'cases.private.json').read_text())
    review_manifest = json.loads((args.review / 'review.safe.json').read_text())
    if digest(original / 'cases.private.json') != review_manifest['source_hashes']['cases.private.json']:
        raise ValueError('original cases changed')
    if digest(args.review / 'review.blind.private.json') != review_manifest['blinded_packet_sha256']:
        raise ValueError('blinded packet changed')
    keys = json.loads((args.review / 'review.key.private.json').read_text())
    refs = json.loads((args.review / 'reviewed_references.private.json').read_text())
    tasks = make_tasks(rows, keys, refs)
    os.umask(0o077)
    args.output.mkdir(parents=True)
    write_json(args.output / 'inputs.safe.json', {'source_cases_sha256': digest(original / 'cases.private.json'),
        'key_sha256': digest(args.review / 'review.key.private.json'),
        'references_sha256': digest(args.review / 'reviewed_references.private.json'),
        'review_decision_sha256': digest(args.review / 'blind_review.private.json'), 'training_performed': False})
    status = args.output / 'status.txt'
    status.write_text('loading_original_step120\n')
    from vllm import LLM, SamplingParams
    model = args.root / 'runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource'
    llm = LLM(model=str(model), tensor_parallel_size=8, dtype='bfloat16', trust_remote_code=True,
              max_model_len=8192, max_num_seqs=64, gpu_memory_utilization=0.8, seed=1024, enforce_eager=True)
    tokenizer = llm.get_tokenizer()

    def generate(messages, budget):
        prompts = []
        for m in messages:
            rendered = tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=False)
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            if len(ids) + budget > 8192:
                raise ValueError('context limit exceeded')
            prompts.append({'prompt_token_ids': ids})
        outputs = sorted(llm.generate(prompts, SamplingParams(temperature=0, top_p=1,
                         max_tokens=budget, seed=1024)), key=lambda o: int(o.request_id))
        if len(outputs) != len(messages):
            raise ValueError('generation coverage mismatch')
        traced = []
        for output in outputs:
            choice = output.outputs[0]
            try:
                parsed = _extract_object(choice.text)
            except (ValueError, IndexError):
                parsed = None
            traced.append({'raw_text': choice.text, 'parsed': parsed,
                           'finish_reason': choice.finish_reason, 'tokens': len(choice.token_ids)})
        return traced

    predictions = {k['review_id']: {} for k in keys}
    traces = []
    for repeat in range(3):
        ordered = tasks if repeat % 2 == 0 else list(reversed(tasks))
        status.write_text(f'evaluating_repeat_{repeat + 1}\n')
        values = generate([t['messages'] for t in ordered], 96)
        for task, trace in zip(ordered, values):
            answers, valid = parse_answers(json.dumps(trace['parsed']), task['option_count'])
            predictions[task['review_id']].setdefault(task['condition'], []).append(
                {'answers': list(answers), 'parse_ok': valid})
            traces.append({**trace, 'review_id': task['review_id'], 'condition': task['condition'], 'repeat': repeat + 1})
        write_json(args.output / 'predictions.private.json', predictions)
        write_json(args.output / 'generation.private.json', traces)
    results = {k['review_id']: {} for k in keys}
    for task in tasks:
        results[task['review_id']][task['condition']] = majority(
            predictions[task['review_id']][task['condition']], task['expected'])
    write_json(args.output / 'results.private.json', results)
    # Reproduce the old audit prompt on these same eight only; retain raw output and stop causes.
    by_hash = {r['item_hash']: r for r in rows}
    messages = [audit_messages(by_hash[k['item_hash']], k['probe_and_audit']['candidate']
                 if k['probe_and_audit']['basic_pass'] else None) for k in keys]
    audit_traces = []
    for budget in (512, 1536):
        status.write_text(f'audit_budget_probe_{budget}\n')
        for key, trace in zip(keys, generate(messages, budget)):
            audit_traces.append({**trace, 'review_id': key['review_id'], 'budget': budget})
        write_json(args.output / 'audit_generation.private.json', audit_traces)
    report = safe_results(results, traces, audit_traces)
    report['results_sha256'] = digest(args.output / 'results.private.json')
    write_json(args.output / 'results.safe.json', report)
    status.write_text('complete_eight_item_source_review\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

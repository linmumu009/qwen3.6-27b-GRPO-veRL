"""Create standalone knowledge candidates; ambiguous gold information is withheld."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import re
from pathlib import Path


def risk_flags(row):
    flags = []
    question = row['question']
    answers = [row['options'][i] for i in row['expected']]
    if row['question_type'] == 'true_or_false':
        flags.append('true_false')
    if re.search(r'\b(not|except|incorrect|false|never|least)\b|不属于|错误|不正确|除外', question, re.I):
        flags.append('negative_stem')
    if any(re.search(r'\b(all|none|both)\s+(of\s+)?(the\s+)?(above|these)|\b(options?|statements?)\s+[A-Z0-9]\b|以上|上述', a, re.I) for a in answers):
        flags.append('option_dependency')
    if len(row['options']) > 10 or re.search(r'\bmatch(ing)?\b|匹配|连线', question, re.I):
        flags.append('matching_or_large_pool')
    return flags or ['ordinary']


def source_payload(row):
    return {'question': row['question'], 'question_type': row['question_type'],
            'options': row['options'], 'gold_indices_zero_based': row['expected']}


def messages(row, candidate=None):
    if candidate is None:
        instruction = (
            'Convert an exam item to a standalone factual knowledge paragraph, entailed by the question '
            'conditions and provided gold answer. Preserve entities, quantities, scope and negation. '
            'For negative questions state that the selected answer does NOT satisfy the property. '
            'For false statements write a corrected claim or minimal logically justified negation; never '
            'quote the false assertion followed by a true/false verdict. Do not invent the correct number '
            'or alternative if the false label does not determine it. Resolve all/none/both of the above '
            'using the full options. Do not present distractors as facts. Preserve every selected answer '
            'for multiple answer items. For matching give explicit pairs only if identifiable. '
            'Avoid Question:, Answer:, Statement:, option letters, or references to the exam. '
            'If insufficient information or contradictory gold, return status=withheld, text="", reason. '
            'Otherwise return JSON {"status":"candidate","text":"...","reason":"..."}. '
            'Use the source language. No extra knowledge or explanation beyond supported facts.')
        payload = source_payload(row)
    else:
        instruction = (
            'Audit a proposed standalone knowledge paragraph against exam conditions and gold options. '
            'Check ALL required keys independently. Do not reward fluency over correctness. '
            'A false label does not identify a unique corrected numerical value. Negative questions must '
            'not become positive claims. All/none/both answers require resolving options. '
            'Return JSON with literal booleans: entailed_by_gold, polarity_preserved, scope_preserved, '
            'all_gold_covered, standalone, no_unsupported_facts, plus issues array. '
            'Reject quoting the wrong statement and attaching a false verdict. '
            'This is an entailment audit, not independent external factual verification.')
        payload = {**source_payload(row), 'candidate': candidate}
    return [{'role': 'system', 'content': instruction},
            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]


def candidate_valid(value):
    if not isinstance(value, dict) or value.get('status') != 'candidate':
        return False
    text = value.get('text')
    return isinstance(text, str) and bool(text.strip()) and not re.search(
        r'(^|\n)\s*(Question|Correct answers?|Statement):|This statement is (true|false)|\b(all|none) of the above\b', text, re.I)


def audit_valid(value):
    keys = ('entailed_by_gold', 'polarity_preserved', 'scope_preserved',
            'all_gold_covered', 'standalone', 'no_unsupported_facts')
    return isinstance(value, dict) and all(value.get(k) is True for k in keys) and value.get('issues') == []


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--audit-only', action='store_true')
    args = p.parse_args()
    from scripts.prepare_logistics_exam_cpt import read_jsonl, write_json, write_jsonl_private
    rows = read_jsonl(args.source)
    if args.output.exists():
        raise ValueError('refusing overwrite')
    safe = {'source_hash': hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'items': len(rows), 'risk_counts_nonexclusive': dict(Counter(f for r in rows for f in risk_flags(r))),
        'private_content_included': False, 'historical_true_false_was_verdict_not_correction': True}
    if args.audit_only:
        write_json(args.output, safe)
        return
    from vllm import LLM
    from scripts.rewrite_logistics_exam_stems_offline import generate_json_objects
    llm = LLM(model=args.model, tensor_parallel_size=8, dtype='bfloat16', trust_remote_code=True,
              max_model_len=8192, max_num_seqs=64, gpu_memory_utilization=0.8, seed=1024, enforce_eager=True)
    tokenizer = llm.get_tokenizer()
    results = []
    for offset in range(0, len(rows), 64):
        batch = rows[offset:offset+64]
        eligible = []
        for r in batch:
            ids = tokenizer.apply_chat_template(messages(r), tokenize=True,
                add_generation_prompt=True, enable_thinking=False)
            if len(ids) + 512 <= 8192:
                eligible.append(r)
        generated = generate_json_objects(llm, tokenizer, [messages(r) for r in eligible],
            max_tokens=512, max_model_len=8192, seed=1024) if eligible else []
        by_id = {r['item_hash']: c for r,c in zip(eligible, generated)}
        candidates = [by_id.get(r['item_hash'], {'status':'withheld','text':'',
                       'reason':'full source options exceed model context'}) for r in batch]
        valid = [(r, c) for r, c in zip(batch, candidates) if candidate_valid(c)]
        valid = [(r,c) for r,c in valid if len(tokenizer.apply_chat_template(
            messages(r,c['text']), tokenize=True, add_generation_prompt=True, enable_thinking=False)) + 256 <= 8192]
        audits = generate_json_objects(llm, tokenizer, [messages(r, c['text']) for r,c in valid],
            max_tokens=256, max_model_len=8192, seed=1024) if valid else []
        audit_by_id = {r['item_hash']: a for (r,c), a in zip(valid, audits)}
        for r, c in zip(batch, candidates):
            audit = audit_by_id.get(r['item_hash'])
            results.append({'item_hash': r['item_hash'], 'dataset': r['dataset'], 'risk_flags': risk_flags(r),
                'source_question_sha256': hashlib.sha256(r['question'].encode()).hexdigest(),
                'candidate': c, 'audit': audit,
                'automatic_entailment_passed': candidate_valid(c) and audit_valid(audit),
                'human_reviewed': False, 'independent_fact_checked': False})
        write_jsonl_private(args.output.with_suffix('.partial.jsonl'), results)
    write_jsonl_private(args.output, results)
    passed = [r for r in results if r['automatic_entailment_passed']]
    safe.update({'automatic_entailment_passed': len(passed), 'withheld': len(results)-len(passed),
        'passed_risk_counts_nonexclusive': dict(Counter(f for r in passed for f in r['risk_flags'])),
        'human_reviewed': 0, 'independent_fact_checked': 0,
        'review_limitation': 'same local model generated and audited; no guarantee of semantic or factual correctness',
        'training_launched': False, 'private_candidates_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()})
    write_json(args.output.with_suffix('.safe.json'), safe)
    print(json.dumps(safe), flush=True)


if __name__ == '__main__':
    main()

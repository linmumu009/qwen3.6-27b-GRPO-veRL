"""Run a bounded, diagnostic-only evidence pilot on existing frozen cases.

This does not change the official benchmark protocol or produce training data.
The no-evidence prompt must match the recorded baseline token fingerprint.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_logistics_knowledge import EvalItem, build_messages, parse_answers

ROOT = Path('/workspace/llin-verl-grpo')
SOURCE = ROOT/'runs/logistics-cpt-diagnostics-20260904/private/public_eval/frozen_cases_source.jsonl'
BASE = ROOT/'runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource'
BASELINE = ROOT/'runs/cpt-controlled-storage-20260910/llin/cpt-controlled-20260910/curve8x/eval_epoch_0'
CASE_SHA = 'b652b2108cb552346df11d005c15ff3137c50a756a7b24eb35302683ec33ed99'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def messages_for(case, evidence):
    item = EvalItem(case['item_hash'], case['dataset'], str(case['source_id']), case['category'],
                    case['question_type'], case['question'], tuple(case['options']), tuple(case['expected']))
    messages = build_messages(item)
    if evidence is not None:
        messages[-1]['content'] += '\n\nReference material for this diagnostic:\n' + evidence
    return messages


def strip_section_ids(text):
    """Remove glossary heading identifiers only; preserve facts and body numbers."""
    return re.sub(r'(?m)^[A-Z]\.[IVX]+(?:/[IVX]+)?[-.]\d+(?:\.\d+)?[ \t]+', '', text)


def main():
    import fcntl
    p = argparse.ArgumentParser()
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--packet-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--section-id-control', action='store_true',
                   help='Two-case, four-request paired diagnostic of glossary heading IDs')
    a = p.parse_args()
    os.umask(0o077)
    out = a.out.resolve()
    if out.parent != Path('/opt') or not out.name.startswith('llin-transfer-evidence-'):
        raise ValueError('independent named diagnostic output required')
    packet_bytes = a.packet.read_bytes()
    if digest(packet_bytes) != a.packet_sha256 or digest(SOURCE.read_bytes()) != CASE_SHA:
        raise ValueError('input fingerprint changed')
    packet = [json.loads(line) for line in packet_bytes.decode().splitlines()]
    case_count, request_count = (2, 4) if a.section_id_control else (13, 37)
    if len(packet) != case_count or len({r['item_hash'] for r in packet}) != case_count:
        raise ValueError('unexpected reviewed case count')
    source_rows = [json.loads(line) for line in SOURCE.read_text().splitlines()]
    cases = {c['item_hash']: c for c in source_rows}
    protocol = json.loads((BASELINE/'protocol.safe.json').read_text())
    baseline_hashes = dict(zip((c['item_hash'] for c in source_rows), protocol['prompt_hashes']))
    specs = []
    for row in packet:
        if row['training_allowed'] is not False or row['review_status'] != 'source_sufficient_unique_answer':
            raise ValueError('unreviewed packet')
        case = cases[row['item_hash']]
        if row['derived_answer'] != case['expected']:
            raise ValueError('independent derivation and frozen gold disagree')
        if a.section_id_control:
            original = row['source_page']
            stripped = strip_section_ids(original)
            if original == stripped or row['short_evidence'] not in original:
                raise ValueError('section control must change heading IDs on a supported original page')
            specs.extend([(row['item_hash'], 'source_page', original),
                          (row['item_hash'], 'source_page_without_section_ids', stripped)])
            continue
        specs.append((row['item_hash'], 'closed', None))
        specs.append((row['item_hash'], 'short_evidence', row['short_evidence']))
        if row['source_page'] is not None:
            if row['short_evidence'] not in row['source_page']:
                raise ValueError('excerpt not contained in original page')
            specs.append((row['item_hash'], 'source_page', row['source_page']))
    if len(specs) != request_count:
        raise ValueError('request budget changed')
    out.mkdir(exist_ok=False)
    def save(name, obj):
        (out/name).write_text(json.dumps(obj, indent=2)+'\n')
    save('status.safe.json', dict(status='waiting_for_resource_lock', training_started=False))
    try:
        with (ROOT/'runs/.logistics-exam-cpt.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
            tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
            prompts = []
            for key, condition, evidence in specs:
                # Match the historical evaluator exactly; newer tokenizers can
                # return BatchEncoding for tokenize=True instead of a token list.
                rendered = tokenizer.apply_chat_template(messages_for(cases[key], evidence), tokenize=False,
                            add_generation_prompt=True, enable_thinking=False)
                tokens = tokenizer.encode(rendered, add_special_tokens=False)
                if len(tokens) + 96 > 8192:
                    raise ValueError('prompt overflow; truncation prohibited')
                fingerprint = digest(json.dumps(tokens).encode())
                if condition == 'closed' and fingerprint != baseline_hashes[key]:
                    raise ValueError('no-evidence baseline prompt drift')
                prompts.append(dict(prompt_token_ids=tokens))
            save('registration.safe.json', dict(requests=request_count, cases=case_count, seed=1024, temperature=0,
                max_tokens=96, max_model_len=8192, tp=8, max_num_seqs=32, repeats=1,
                packet_sha256=a.packet_sha256, source_sha256=CASE_SHA, model=str(BASE),
                original_prompt_hashes_verified=not a.section_id_control,
                section_id_control=a.section_id_control,
                max_input_tokens=max(len(p['prompt_token_ids']) for p in prompts),
                training_started=False, official_score_changed=False,
                limitation='Selected known errors; evidence contains answer-bearing definitions; not training benefit or generalization.'))
            save('status.safe.json', dict(status='loading_model', training_started=False))
            llm = LLM(model=str(BASE), tensor_parallel_size=8, dtype='bfloat16', trust_remote_code=True,
                max_model_len=8192, max_num_seqs=32, gpu_memory_utilization=.8, seed=1024, enforce_eager=True)
            save('status.safe.json', dict(status='inference', requests=request_count, training_started=False))
            outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=96, seed=1024))
            outputs = sorted(outputs, key=lambda x: int(x.request_id))
            if len(outputs) != len(specs):
                raise ValueError('missing outputs')
            totals = defaultdict(lambda: dict(n=0, correct=0, invalid=0, truncated=0))
            with (out/'predictions.private.jsonl').open('x') as handle:
                for (key, condition, _), prompt, output in zip(specs, prompts, outputs):
                    if list(output.prompt_token_ids) != prompt['prompt_token_ids']:
                        raise ValueError('prompt/output mismatch')
                    answer = output.outputs[0]
                    parsed, ok = parse_answers(answer.text, len(cases[key]['options']))
                    valid = ok and answer.finish_reason == 'stop'
                    correct = valid and list(parsed) == sorted(cases[key]['expected'])
                    row = dict(item_hash=key, dataset=cases[key]['dataset'], condition=condition,
                        prediction=answer.text, parsed=list(parsed), valid=valid, correct=correct,
                        finish_reason=answer.finish_reason, output_tokens=len(answer.token_ids),
                        prompt_sha256=digest(json.dumps(prompt['prompt_token_ids']).encode()))
                    handle.write(json.dumps(row)+'\n')
                    t = totals[cases[key]['dataset']+'/'+condition]
                    t['n'] += 1
                    t['correct'] += correct
                    t['invalid'] += not valid
                    t['truncated'] += answer.finish_reason == 'length'
            save('result.safe.json', dict(totals=dict(totals), training_started=False,
                diagnostic_only=True, predictions_sha256=digest((out/'predictions.private.jsonl').read_bytes())))
            save('status.safe.json', dict(status='completed_pending_independent_verification', training_started=False))
    except BaseException as exc:
        save('status.safe.json', dict(status='failed', error=type(exc).__name__+': '+str(exc), training_started=False))
        raise


if __name__ == '__main__':
    main()

"""Frozen open-response probes on four reviewed topics; inference only, no self-grading."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from scripts.prepare_step120_qa_diagnostic import digest, write_json

TOPICS = {'R02', 'R05', 'R06', 'R08'}
KINDS = {'definition', 'distinction', 'application'}


def validate_protocol(protocol):
    if protocol.get('training_allowed') is not False or protocol.get('repeats') != 3:
        raise ValueError('invalid no-training/repeat contract')
    tasks = protocol['tasks']
    if len(tasks) != 12 or len({t['id'] for t in tasks}) != 12:
        raise ValueError('expected twelve unique probes')
    if Counter((t['review_id'], t['kind']) for t in tasks) != Counter(
        (topic, kind) for topic in TOPICS for kind in KINDS
    ):
        raise ValueError('expected four topics by three kinds')
    if protocol.get('max_tokens') != 512 or not isinstance(protocol.get('system'), str):
        raise ValueError('unexpected generation contract')
    for task in tasks:
        if not isinstance(task['question'], str) or not task['question'].strip():
            raise ValueError('empty question')
        if not isinstance(task['criteria'], list) or len(task['criteria']) != 3 or not all(
            isinstance(c, str) and c.strip() for c in task['criteria']
        ):
            raise ValueError('expected three fixed semantic criteria')


def messages_for(protocol, task):
    # Explicit allowlist: never forward task dictionaries with rubrics or source references.
    return [{'role': 'system', 'content': protocol['system']},
            {'role': 'user', 'content': task['question']}]


def response_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def review_groups(responses):
    groups = {}
    for r in responses:
        key = (r['task_id'], response_hash(r['text']))
        if key not in groups:
            groups[key] = {'task_id': r['task_id'], 'response_sha256': key[1],
                           'text': r['text'], 'repeats': [], 'finish_reasons': []}
        groups[key]['repeats'].append(r['repeat'])
        groups[key]['finish_reasons'].append(r['finish_reason'])
    return list(groups.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('refusing overwrite')
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    validate_protocol(protocol)
    os.umask(0o077)
    args.output.mkdir(parents=True)
    write_json(args.output / 'protocol.private.json', protocol)
    frozen = digest(args.output / 'protocol.private.json')
    write_json(args.output / 'manifest.safe.json', {'private_content_included': False,
        'training_performed': False, 'topics': 4, 'probes': 12, 'repeats': 3,
        'protocol_sha256': frozen, 'input_file_sha256': digest(args.protocol),
        'rubric_frozen_before_model_load': True, 'reference_text_in_prompts': False,
        'model': 'Step120', 'max_tokens': 512, 'max_model_len': 8192,
        'max_num_seqs': 64, 'temperature': 0, 'seed': 1024, 'enable_thinking': False})
    status = args.output / 'status.txt'
    status.write_text('loading_original_step120\n')
    from vllm import LLM, SamplingParams
    model = args.root / 'runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource'
    llm = LLM(model=str(model), tensor_parallel_size=8, dtype='bfloat16', trust_remote_code=True,
              max_model_len=8192, max_num_seqs=64, gpu_memory_utilization=0.8, seed=1024, enforce_eager=True)
    tokenizer = llm.get_tokenizer()
    responses = []
    for repeat in range(1, 4):
        status.write_text(f'closed_knowledge_repeat_{repeat}\n')
        tasks = list(protocol['tasks'])
        if repeat == 2:
            tasks.reverse()
        elif repeat == 3:
            tasks = tasks[4:] + tasks[:4]
        prompts = []
        for task in tasks:
            text = tokenizer.apply_chat_template(messages_for(protocol, task), tokenize=False,
                       add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) + 512 > 8192:
                raise ValueError('context limit exceeded')
            prompts.append({'prompt_token_ids': ids})
        outputs = sorted(llm.generate(prompts, SamplingParams(temperature=0, top_p=1,
                          max_tokens=512, seed=1024)), key=lambda o: int(o.request_id))
        if len(outputs) != len(tasks):
            raise ValueError('generation coverage mismatch')
        for task, out in zip(tasks, outputs):
            value = out.outputs[0]
            responses.append({'task_id': task['id'], 'repeat': repeat, 'text': value.text,
                              'finish_reason': value.finish_reason, 'tokens': len(value.token_ids)})
        write_json(args.output / 'responses.private.json', responses)
    if frozen != digest(args.output / 'protocol.private.json'):
        raise ValueError('frozen protocol changed')
    groups = review_groups(responses)
    write_json(args.output / 'answers.for_review.private.json', groups)
    safe = {'private_content_included': False, 'training_performed': False,
            'status': 'inference_complete_pending_semantic_review', 'responses': len(responses),
            'probes': 12, 'topics': 4, 'unique_task_response_texts': len(groups),
            'finish_reasons': dict(Counter(r['finish_reason'] for r in responses)),
            'protocol_sha256': frozen, 'responses_sha256': digest(args.output / 'responses.private.json'),
            'scoring_performed': False}
    write_json(args.output / 'generation.safe.json', safe)
    status.write_text('inference_complete_pending_semantic_review\n')
    print(json.dumps(safe), flush=True)


if __name__ == '__main__':
    main()

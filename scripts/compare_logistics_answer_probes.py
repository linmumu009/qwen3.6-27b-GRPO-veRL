"""Pair answer-only probes without exporting private item records."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.probe_logistics_exam_answers import summarize


def compare(baseline, candidate):
    if baseline['source_hashes'] != candidate['source_hashes']:
        raise ValueError('source mismatch')
    def index(payload):
        rows = payload['rows']
        indexed = {(r['arm'], r['item_hash']): r for r in rows}
        if len(indexed) != len(rows):
            raise ValueError('duplicate items')
        return indexed
    left, right = index(baseline), index(candidate)
    if set(left) != set(right):
        raise ValueError('item coverage mismatch')
    results = {}
    a, b = summarize(baseline['rows']), summarize(candidate['rows'])
    for arm in a:
        keys = [key for key in left if key[0] == arm]
        for key in keys:
            for field in ('answer_tokens', 'prefix_tokens', 'boundary_crossing', 'generation_censored'):
                if left[key][field] != right[key][field]:
                    raise ValueError(f'tokenization mismatch: {field}')
        results[arm] = {'baseline': a[arm], 'candidate': b[arm],
            'relative_answer_nll_reduction_percent': 100*(a[arm]['answer_nll']-b[arm]['answer_nll'])/a[arm]['answer_nll'],
            'items_answer_nll_improved': sum(right[k]['answer_nll_sum'] < left[k]['answer_nll_sum'] for k in keys),
            'greedy_improved': sum(not left[k]['greedy_gold_token_prefix_exact'] and right[k]['greedy_gold_token_prefix_exact'] for k in keys),
            'greedy_regressed': sum(left[k]['greedy_gold_token_prefix_exact'] and not right[k]['greedy_gold_token_prefix_exact'] for k in keys)}
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    paths = {name: args.root / f'{name}.json' for name in ('step120','direct','rewritten')}
    payloads = {name: json.loads(path.read_text()) for name,path in paths.items()}
    output = {'contract':'exam-answer-paired-comparison-v1','private_content_included':False,
        'input_hashes':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()},
        'direct_vs_step120':compare(payloads['step120'],payloads['direct']),
        'rewritten_vs_step120':compare(payloads['step120'],payloads['rewritten'])}
    packed_paths = {name: args.root / f'{name}.packed.safe.json' for name in paths}
    packed = {name:json.loads(path.read_text()) for name,path in packed_paths.items()}
    for name, value in packed.items():
        if value.get('contract') != 'exam-exact-packed-answer-nll-v1' or set(value['results']) != {'direct','rewritten'}:
            raise ValueError('packed contract or arm mismatch')
        if value['model'] != name or value['parquet_hashes'] != packed['step120']['parquet_hashes']:
            raise ValueError('packed model or corpus mismatch')
        for arm, row in value['results'].items():
            for field in ('items','answer_tokens','scored_total_tokens'):
                if row[field] != packed['step120']['results'][arm][field]:
                    raise ValueError('packed token coverage mismatch')
    output['packed_training_context'] = {name:value['results'] for name,value in packed.items()}
    output['packed_parquet_hashes'] = packed['step120']['parquet_hashes']
    output['packed_input_hashes'] = {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in packed_paths.items()}
    target = args.root / 'answer-comparison.safe.json'
    if target.exists():
        raise ValueError('refusing overwrite')
    target.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()

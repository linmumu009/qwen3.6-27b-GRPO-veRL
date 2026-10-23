"""Merge the two evaluation replicas and retain only a strictly better HF model."""
import argparse
import json
from pathlib import Path
import re
import shutil

from scripts.evaluate_logistics_knowledge import build_safe_result, write_private_rows, parse_answers
from scripts.run_vllm_logistics_mcq import load_items
from scripts.prepare_logistics_mcq_fit import SOURCE_SHA256


def owned_model(run, path):
    path = Path(path)
    if path.is_symlink() or path.resolve().parent != (run / 'hf').resolve() or not re.fullmatch(r'step_\d+', path.name):
        raise ValueError('refusing cleanup outside this run HF candidates')
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for shard in range(2):
        path = args.run / 'private' / f'step_{args.step}.shard{shard}.jsonl'
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    items = load_items(args.cases)
    lookup = {row['item_hash']: row for row in rows}
    if len(rows) != 1672 or len(lookup) != 1672 or set(lookup) != {item.item_hash for item in items}:
        raise ValueError('replica evaluation coverage mismatch')
    for item in items:
        row = lookup[item.item_hash]
        parsed, valid = parse_answers(row['prediction'], len(item.options))
        if row['dataset'] != item.dataset or row['expected'] != list(item.expected) or row.get('error'):
            raise ValueError('evaluation metadata/request mismatch')
        if bool(row['correct']) != (valid and parsed == item.expected):
            raise ValueError('evaluation score mismatch')
    result = build_safe_result(rows, model=f'wrong_text_step_{args.step}', endpoint_label='m05-tp8-dp2',
        concurrency=128, input_hashes={'cases_jsonl': SOURCE_SHA256}, elapsed_sec=0,
        chat_template_disable_thinking=True)
    result['request']['max_output_tokens'] = 128
    result['runtime'] = dict(tensor_parallel_size=8, data_parallel_size=2, seed=1024)
    result['independent_test'] = False
    safe = args.run / 'safe' / f'step_{args.step}.json'
    safe.write_text(json.dumps(result, indent=2) + '\n')
    write_private_rows(args.run / 'private' / f'step_{args.step}.jsonl', rows)
    history_path = args.run / 'selection.safe.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else {'evaluations': []}
    if any(row['step'] == args.step for row in history['evaluations']):
        raise ValueError('already selected this step')
    best = history.get('best')
    entry = dict(step=args.step, model=str(args.model), correct=result['correct'], items=result['items'],
                 accuracy=result['accuracy'], by_dataset=result['by_dataset'],
                 parse_failures=result['parse_failures'])
    better = best is None or entry['correct'] > best['correct']
    discard = best['model'] if better and best is not None and best['step'] > 0 else None
    if better:
        history['best'] = entry
    elif args.step > 0:
        discard = str(args.model)
    history['evaluations'].append(entry)
    history['selection_rule'] = 'maximum total correct across all 1672; ties keep the earlier model'
    history['independent_test'] = False
    temporary = history_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(history, indent=2) + '\n')
    temporary.replace(history_path)
    link = args.run / 'best_model'
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise ValueError('unexpected best_model directory')
    link.symlink_to(history['best']['model'], target_is_directory=True)
    if discard:
        shutil.rmtree(owned_model(args.run, discard))
    print(json.dumps({'evaluated': entry, 'best_step': history['best']['step'], 'improved': better}))


if __name__ == '__main__':
    main()

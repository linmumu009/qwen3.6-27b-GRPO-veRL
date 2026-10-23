"""Reuse the verified preceding best evaluation as a protected continuation baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from scripts.evaluate_logistics_knowledge import PROMPT_VERSION
from scripts.prepare_logistics_mcq_fit import SOURCE_SHA256
from scripts.prepare_logistics_wrong_text_sft import read_predictions
from scripts.logistics_wrong_text_selection import best_link_target
from scripts.run_vllm_logistics_mcq import load_items


def validate_parent(parent, cases, base_model):
    if (parent / 'status.txt').read_text().strip() != 'complete':
        raise ValueError('parent training is incomplete')
    history = json.loads((parent / 'selection.safe.json').read_text())
    best = history['best']
    if Path(best['model']).resolve() != base_model.resolve():
        raise ValueError('initial model does not match preceding best')
    if best != max(history['evaluations'], key=lambda entry: entry['correct']):
        raise ValueError('parent best selection is inconsistent')
    summary = json.loads((parent / 'safe' / f"step_{best['step']}.json").read_text())
    if hashlib.sha256(cases.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('evaluation source fingerprint changed')
    if summary['input_sha256'] != {'cases_jsonl': SOURCE_SHA256} or summary['prompt_version'] != PROMPT_VERSION:
        raise ValueError('baseline evaluation source/protocol changed')
    request = summary['request']
    if request['temperature'] != 0 or request['max_output_tokens'] != 128 or request['enable_thinking']:
        raise ValueError('baseline generation settings changed')
    runtime = summary['runtime']
    if any(runtime.get(key) != expected for key, expected in
           [('tensor_parallel_size', 8), ('data_parallel_size', 2), ('seed', 1024)]):
        raise ValueError('baseline execution settings changed')
    raw = parent / 'private' / f"step_{best['step']}.jsonl"
    items = load_items(cases)
    indexed = read_predictions(raw, items)
    if len(indexed) != 1672 or sum(row['correct'] for row in indexed.values()) != best['correct']:
        raise ValueError('baseline scores or coverage changed')
    if sum(not row['parse_ok'] for row in indexed.values()) != best['parse_failures']:
        raise ValueError('baseline parse failure count changed')
    return best, summary, raw


def main():
    parser = argparse.ArgumentParser()
    for key in ('run', 'parent', 'cases', 'base-model'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if (args.run / 'selection.safe.json').exists() or (args.run / 'best_model').exists():
        raise ValueError('refusing to replace an initialized selection')
    best, summary, raw = validate_parent(args.parent, args.cases, args.base_model)
    entry = {**best, 'step': 0, 'model': str(args.base_model)}
    shutil.copy2(raw, args.run / 'private/step_0.jsonl')
    summary['model'] = 'continuation_starting_best'
    (args.run / 'safe/step_0.json').write_text(json.dumps(summary, indent=2) + '\n')
    history = dict(evaluations=[entry], best=entry, independent_test=False,
        selection_rule='maximum total correct across all 1672; ties keep the earlier model')
    (args.run / 'selection.safe.json').write_text(json.dumps(history, indent=2) + '\n')
    (args.run / 'best_model').symlink_to(best_link_target(args.run, args.base_model), target_is_directory=True)
    receipt = dict(private_content_included=False, reused=True, new_inference_calls=0,
        parent_run=str(args.parent), parent_step=best['step'], items=1672, correct=best['correct'],
        prediction_file_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        optimizer_state_reused=False, model_initialization=str(args.base_model))
    (args.run / 'baseline_reuse.safe.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()

"""Compare only frozen same-item official-format MCQ scores; no private text output."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.evaluate_logistics_knowledge import PROMPT_VERSION, build_safe_result, load_private_rows
from scripts.prepare_logistics_mcq_fit import SOURCE_SHA256
from scripts.run_vllm_logistics_mcq import load_items
from scripts.prepare_logistics_reviewed8 import REVIEWED


def compare_scores(baseline, candidate):
    if baseline['input_sha256'] != candidate['input_sha256'] or baseline['prompt_version'] != candidate['prompt_version']:
        raise ValueError('evaluation source or prompt mismatch')
    def indexed(report):
        rows={r['item_hash']:r for r in report['rows']}
        if len(rows)!=report['items'] or len(rows)!=len(report['rows']):
            raise ValueError('duplicate or missing result items')
        return rows
    left,right=indexed(baseline),indexed(candidate)
    if set(left)!=set(right) or not left:
        raise ValueError('paired coverage mismatch')
    for key in left:
        if any(left[key][field]!=right[key][field] for field in ('dataset','category','question_type','choice_count','expected_count')):
            raise ValueError('paired metadata mismatch')
    def group(keys):
        a=sum(bool(left[k]['correct']) for k in keys)
        b=sum(bool(right[k]['correct']) for k in keys)
        return {'items':len(keys),'baseline_correct':a,'candidate_correct':b,
            'baseline_accuracy':a/len(keys),'candidate_accuracy':b/len(keys),
            'delta_accuracy_points':100*(b-a)/len(keys),'net_correct':b-a,
            'improved_0_to_1':sum(not left[k]['correct'] and bool(right[k]['correct']) for k in keys),
            'regressed_1_to_0':sum(bool(left[k]['correct']) and not right[k]['correct'] for k in keys)}
    return {'overall':group(sorted(left)), 'by_dataset':{
        dataset:group(sorted(k for k in left if left[k]['dataset']==dataset))
        for dataset in sorted({r['dataset'] for r in left.values()})}}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--phase',choices=['masks','final'],required=True)
    args=p.parse_args()
    root=Path('/workspace/llin-verl-grpo')
    baseline_path=root/'runs/logistics-cpt-diagnostics-20260904/private/public_eval/step120.majority.jsonl'
    rows=list(load_private_rows(baseline_path).values())
    source=baseline_path.with_name('frozen_cases_source.jsonl')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=SOURCE_SHA256:
        raise ValueError('frozen source changed')
    items={item.item_hash:item for item in load_items(source)}
    if len(rows)!=1672 or {row['item_hash'] for row in rows}!=set(items):
        raise ValueError('baseline coverage mismatch')
    for row in rows:
        item=items[row['item_hash']]
        if (row['prompt_version']!=PROMPT_VERSION or row['chat_template_disable_thinking'] is not True
            or tuple(row['expected'])!=item.expected or tuple(row['options'])!=item.options
            or row['question']!=item.question):
            raise ValueError('historical baseline does not match frozen protocol')
        if bool(row['correct'])!=(bool(row['parse_ok']) and tuple(row['parsed'])==item.expected):
            raise ValueError('historical baseline correctness mismatch')
    baseline=build_safe_result(rows,model='step120',endpoint_label='historical_same_protocol',concurrency=64,
        input_hashes={'cases_jsonl':SOURCE_SHA256},elapsed_sec=0,chat_template_disable_thinking=True)
    names=['mask8_all','mask8_answer'] + (['sft_step418','sft_step836'] if args.phase=='final' else [])
    candidates={name:json.loads((args.run/f'safe/{name}.majority.json').read_text()) for name in names}
    for name,report in candidates.items():
        if report['items']!=1672 or report['aggregation']['repeats']!=3:
            raise ValueError('incomplete official evaluation')
        raw=json.loads((args.run/f'safe/{name}.json').read_text())
        if (raw['request']['max_output_tokens']!=96 or raw['runtime']['tensor_parallel_size']!=8
            or raw['request']['chat_template_disable_thinking'] is not True
            or raw['input_sha256']!={'cases_jsonl':SOURCE_SHA256}):
            raise ValueError('candidate runtime protocol mismatch')
    result={'private_content_included':False,'independent_test':False,'phase':args.phase,
        'objective':'maximize SC-bench and LogistikaBench same-item official-format scores',
        'baseline_source_sha256':hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        'models':{name:{key:report[key] for key in ('items','correct','accuracy','parse_failures','by_dataset')}
                  for name,report in {'step120':baseline,**candidates}.items()},
        'vs_step120':{name:compare_scores(baseline,report) for name,report in candidates.items()}}
    result['reviewed8_in_official_format']={name:{
        'items':sum(r['item_hash'] in REVIEWED for r in report['rows']),
        'correct':sum(bool(r['correct']) for r in report['rows'] if r['item_hash'] in REVIEWED)}
        for name,report in {'step120':baseline,**candidates}.items()}
    if args.phase=='final':
        eligible=['step120']+[name for name in names if all(
            group['net_correct']>=0 for group in result['vs_step120'][name]['by_dataset'].values())]
        result['best_without_dataset_regression']=max(eligible,key=lambda name:result['models'][name]['correct'])
        result['selection_rule']='highest total correct among candidates with neither dataset below Step120; ties keep earlier baseline/candidate'
    output=args.run/f'safe/{args.phase}-comparison.safe.json'
    if output.exists():
        raise ValueError('refusing overwrite')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

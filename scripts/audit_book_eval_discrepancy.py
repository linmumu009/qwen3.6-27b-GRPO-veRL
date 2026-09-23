"""Reconcile archived evaluations offline; never overwrite external scores."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

BASE = Path('CPT_resources/llin-book-eval-investigation-20260923-01')
TAGS = ('llin-cpt-book-1ep', 'llin-step120-opensource', 'llin-grpo-step120', 'qwen36-27b')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    # Unicode option labels include line-separator characters inside JSON strings.
    return [json.loads(line) for line in path.read_text(encoding='utf-8').split('\n') if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(s):
    return ' '.join(s.split())


def old_score(response, gold):
    predicted = ''.join(sorted(set(re.findall('[A-J]', response))))
    expected = ''.join(sorted(set(re.findall('[A-J]', gold))))
    return bool(expected) and predicted == expected, predicted


def paired(before, after):
    assert set(before) == set(after)
    gains = sum(not before[k] and after[k] for k in before)
    losses = sum(before[k] and not after[k] for k in before)
    return dict(n=len(before), before_correct=sum(before.values()), after_correct=sum(after.values()),
                gains=gains, losses=losses, net=gains-losses,
                delta_percentage_points=100*(gains-losses)/len(before))


def main():
    manifest = read(BASE/'download_manifest.private.json')
    for name, entry in manifest.items():
        if not name.endswith('/derived_predictions') and not entry.get('redacted'):
            assert sha(BASE/name) == entry['sha256'], name
    data = rows(BASE/'data/LogistikaBench_train.jsonl')
    assert len(data) == 1446 and all((d['extract_method'],d['score_method']) == ('answer_or_raw','exact_match') for d in data)
    our_raw = {label: rows(BASE/'ours'/f'{label}.jsonl') for label in ('cpt','step120')}
    our_log = {label: [r for r in raw if r['dataset']=='LogistikaBench'] for label,raw in our_raw.items()}
    lookup = {norm(r['question']+'\n'+'\n'.join(f'{chr(65+i)}. {v}' for i,v in enumerate(r['options']))):r for r in our_log['cpt']}
    assert len(lookup) == 1446
    linked = [lookup[norm(d['task_input'])] for d in data]
    assert len({r['item_hash'] for r in linked}) == 1446
    assert all(d['golden_answer'] == ''.join(chr(65+i) for i in sorted(r['expected'])) for d,r in zip(data,linked))
    exact_inputs = sum(d['task_input']==r['question']+'\n'+'\n'.join(f'{chr(65+i)}. {v}' for i,v in enumerate(r['options'])) for d,r in zip(data,linked))
    table = []; scores = {}; prediction_maps = {}; transitions = {}
    for tag in TAGS:
        folder = BASE/tag
        predictions = {r['task_id']:r for r in rows(folder/'predictions.jsonl')}
        current = {r['task_id']:r for r in rows(folder/'scores.jsonl')}
        expected_ids = {f'LogistikaBench_{i:04d}' for i in range(1446)}
        assert len(predictions)==len(current)==1446 and set(predictions)==set(current)==expected_ids
        summary = read(folder/'summary.json')
        assert sum(r['correct'] for r in current.values()) == summary['correct']
        assert abs(summary['accuracy']-summary['correct']/1446)<1e-12
        raw_exact_disagreement = 0
        for i,d in enumerate(data):
            key=f'LogistikaBench_{i:04d}';p=predictions[key];s=current[key]
            assert json.loads(p['prompt']) == [dict(role='system',content=d['system_prompt']),dict(role='user',content=d['task_input'])]
            assert s['golden_answer']==d['golden_answer']
            assert (s['extract_method'],s['score_method'])==('answer_or_raw','exact_match')
            exact=bool(p['response'].strip()) and p['response'].strip()==d['golden_answer'].strip()
            raw_exact_disagreement += exact!=s['correct']
            # For the main comparison, all answers can be verified by raw exact match.
            if tag in TAGS[:2]:
                assert p['response'].strip()==s['extracted_answer'] and exact==s['correct']
        record=dict(tag=tag,n=1446,current_correct=summary['correct'],current_percent=100*summary['accuracy'],
                    offline_legacy_rule_correct=sum(old_score(predictions[k]['response'],r['golden_answer'])[0] for k,r in current.items()),
                    current_extract_fail=sum(not r['extracted_answer'] for r in current.values()),
                    raw_exact_vs_current_disagreements=raw_exact_disagreement,
                    finish_reasons=dict(Counter(p['finish_reason'] for p in predictions.values())),
                    all_prompts_equal_to_frozen_processed_data=True)
        backup=folder/'scores_pre_dual_unify.jsonl'
        if backup.exists():
            old={r['task_id']:r for r in rows(backup)}
            assert len(old)==1446 and set(old)==expected_ids
            for key,r in old.items():
                correct,extracted=old_score(predictions[key]['response'],r['golden_answer'])
                assert correct==r['correct'] and extracted==r['extracted_answer']
                assert r['golden_answer']==current[key]['golden_answer']
            change=paired({k:r['correct'] for k,r in old.items()},{k:r['correct'] for k,r in current.items()})
            gains=[k for k in old if not old[k]['correct'] and current[k]['correct']]
            change['gains_with_empty_old_extraction']=sum(not old[k]['extracted_answer'] for k in gains)
            change['gains_on_269_option_items']=sum(len(linked[int(k.rsplit('_',1)[1])]['options'])==269 for k in gains)
            transitions[tag]=change
            record.update(old_correct=sum(r['correct'] for r in old.values()),old_extract_fail=sum(not r['extracted_answer'] for r in old.values()))
        derived=read(folder/'derived_30720_summary.json')
        assert derived['derived_from_8k'] is True
        assert manifest[tag+'/derived_predictions']['sha256']==manifest[tag+'/predictions.jsonl']['sha256']
        record.update(derived_30720_correct=derived['correct'],derived_predictions_identical=True,
                      derived_summary_stale=derived['correct']!=summary['correct'])
        table.append(record);scores[tag]={k:r['correct'] for k,r in current.items()};prediction_maps[tag]=predictions
    ours={}
    for label,raw in our_raw.items():
        assert len(raw)==1672
        for r in raw:
            answer=json.loads(r['prediction'])
            assert set(answer)=={'answers'}
            assert (sorted(set(answer['answers']))==sorted(r['expected']))==r['correct']
        summary=read(BASE/'ours'/f'{label}.safe.json')
        assert sum(r['correct'] for r in raw)==summary['correct']
        ours[label]={r['item_hash']:r['correct'] for r in our_log[label]}
    primary=paired(scores[TAGS[1]],scores[TAGS[0]])
    private=[]
    for i,r in enumerate(linked):
        key=f'LogistikaBench_{i:04d}'
        private.append(dict(task_id=key,item_hash=r['item_hash'],option_count=len(r['options']),
                            external_baseline_correct=scores[TAGS[1]][key],external_cpt_correct=scores[TAGS[0]][key],
                            our_baseline_correct=ours['step120'][r['item_hash']],our_cpt_correct=ours['cpt'][r['item_hash']]))
    (BASE/'linked_ledger.private.json').write_text(json.dumps(private,indent=2)+'\n',encoding='utf-8')
    cpt=table[0]['current_correct'];baseline_old=table[1]['old_correct'];baseline_new=table[1]['current_correct']
    result=dict(status='complete_offline_verified',new_model_calls=0,remote_mutations=0,
                dataset='LogistikaBench',items=1446,model_export_manifest=read(BASE/'model/llin_export_manifest.json'),
                runs=table,scoring_only_transitions=transitions,external_current_primary=primary,
                our_september3_primary=paired(ours['step120'],ours['cpt']),
                apparent_mixed_scoring_delta_pp=100*(cpt-baseline_old)/1446,
                baseline_rescoring_delta_pp=100*(baseline_new-baseline_old)/1446,
                dataset_reconciliation=dict(all_1446_questions_options_and_labels_matched=True,exact_formatted_inputs=exact_inputs,whitespace_only=1446-exact_inputs),
                main_2892_predictions_independently_exact_rescored=True,
                original_native_and_grpo_summaries_are_not_pure_exact_string_scoring=True,
                original_scores_and_remote_files_unchanged=True,
                input_fingerprints=manifest,private_linked_ledger_sha256=sha(BASE/'linked_ledger.private.json'),
                limitations=['Did not observe the table viewed by the user’s supervisor; mixed-scoring comparison reproduces the reported magnitude',
                             'Serving versions and MTP settings differ; current aggregate is not a strict controlled causal estimate',
                             'Current model identity is tied by saved path and export metadata, not a historical full-weight hash',
                             'Derived 30K outputs are copies, not additional generations'])
    Path('docs/book_cpt_eval_investigation_20260923.safe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['runs','scoring_only_transitions','external_current_primary','our_september3_primary','dataset_reconciliation']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

import copy
import json
import pytest

from scripts.prepare_step120_evidence_review import build_packet, evidence_gate_reason, reconcile, select_review
from scripts.run_step120_qa_diagnostic import summarize
from tests.test_step120_qa_diagnostic import example_row, example_probe


def artifacts():
    row = example_row()
    audit = {k: True for k in ('basic_factual', 'basic_relevant', 'basic_unique_answer',
                              'basic_gold_supported', 'basic_not_original_restatement')}
    audit.update(original_answerable=True, original_answers=[0], original_quote=example_probe()['quote'])
    probe = {'candidate': example_probe(), 'audit': audit, 'basic_pass': True, 'evidence_pass': True}
    from scripts.run_step120_qa_diagnostic import question_item, majority
    predictions, result = {}, {}
    for condition in ('original_closed', 'original_with_book', 'basic_closed', 'basic_with_book'):
        item = question_item(row, probe['candidate'] if condition.startswith('basic') else None)
        answers = list(item.expected) if condition.startswith('basic') else [1]
        predictions[condition] = [{'parse_ok': True, 'answers': answers} for _ in range(3)]
        result[condition] = majority(predictions[condition], item.expected)
    key = row['item_hash']
    return [row], {key: probe}, {key: predictions}, {key: result}


def test_reconcile_recomputes_and_rejects_mutation():
    rows, probes, predictions, results = artifacts()
    report = summarize(rows, probes, results)
    reconcile(rows, probes, predictions, results, report)
    broken = copy.deepcopy(results)
    broken[rows[0]['item_hash']]['original_closed']['correct'] = True
    with pytest.raises(ValueError, match='majority'):
        reconcile(rows, probes, predictions, broken, report)
    predictions[rows[0]['item_hash']]['basic_closed'].pop()
    with pytest.raises(ValueError, match='three repeats'):
        reconcile(rows, probes, predictions, results, report)


def test_packet_blinds_labels_and_keeps_safe_output_text_free():
    rows, probes, _, results = artifacts()
    blind, key, template, safe = build_packet(rows, probes, results)
    assert safe['application_candidates'] == 1
    assert safe['independently_reviewed_items'] == 0
    assert safe['candidate_all_four_conditions_stable'] == 1
    assert set(blind[0]) == {'review_id', 'question', 'options', 'question_type', 'reference_excerpts'}
    assert 'expected' in key[0] and 'results' in key[0]
    assert template[0]['status'] == 'pending'
    assert rows[0]['question'] not in json.dumps(safe)
    assert rows[0]['evidence'][0]['text'] not in json.dumps(safe)


def test_gates_separate_quote_and_answer_disagreement():
    rows, probes, _, _ = artifacts()
    row = rows[0]
    probe = copy.deepcopy(probes[row['item_hash']])
    assert evidence_gate_reason(row, probe) == 'automatic_pass_not_independently_verified'
    probe['audit']['original_answers'] = [1]
    assert evidence_gate_reason(row, probe) == 'blind_reference_answer_disagrees_with_gold'
    probe['audit']['original_quote'] = 'invented'
    assert evidence_gate_reason(row, probe) == 'verbatim_quote_gate_failed'
    probe['audit']['original_answers'] = [True]
    assert evidence_gate_reason(row, probe) == 'invalid_answer_indices'


def test_sample_is_deterministic_unique_and_bounded():
    rows, probes, _, results = artifacts()
    template = rows[0]
    for i in range(20):
        row = {**template, 'item_hash': f'{i:064x}', 'category': str(i % 3)}
        rows.append(row)
        probes[row['item_hash']] = {'basic_pass': False, 'evidence_pass': False}
        results[row['item_hash']] = {}
    selected, candidates = select_review(rows, probes, results)
    assert len(selected) == len({r['item_hash'] for r in selected}) == 8
    assert len(candidates) == 1
    assert select_review(list(reversed(rows)), probes, results) == (selected, candidates)


def test_rerun_preserves_questions_and_never_injects_gold():
    from scripts.run_step120_evidence_review import make_tasks, add_type_hint
    rows = [{**example_row(), 'item_hash': str(i)} for i in range(8)]
    keys = [{'item_hash': str(i), 'review_id': f'R{i:02d}'} for i in range(8)]
    refs = {'items': {'R00': [{'text': 'Synthetic independently published reference.'}]}}
    tasks = make_tasks(rows, keys, refs)
    assert len(tasks) == 26
    assert all(rows[0]['question'] in t['messages'][1]['content'] for t in tasks)
    assert all('expected' not in json.dumps(t['messages']) for t in tasks)
    assert sum(t['condition'] == 'original_with_reviewed_reference' for t in tasks) == 1
    assert 'exactly one' in add_type_hint([{'content': ''}, {'content': ''}], 'single_choice')[1]['content']
    assert 'exactly one' not in add_type_hint([{'content': ''}, {'content': ''}], 'multiple_choice')[1]['content']
    with pytest.raises(ValueError, match='eight'):
        make_tasks(rows, keys[:7], refs)


def test_rerun_summary_keeps_paired_denominator_and_no_raw_text():
    from scripts.run_step120_evidence_review import safe_results
    score = {'correct': False, 'parse_ok': True, 'all_repeats_same': True}
    result = {'R01': {c: dict(score) for c in ('original_closed', 'original_with_book',
             'original_with_reviewed_reference', 'original_with_reviewed_reference_type_hint')},
             'R02': {'original_closed': dict(score), 'original_with_book': dict(score)}}
    result['R01']['original_with_reviewed_reference']['correct'] = True
    safe = safe_results(result, [{'finish_reason': 'stop', 'raw_text': 'PRIVATE'}],
                        [{'budget': 512, 'parsed': None, 'finish_reason': 'length', 'raw_text': 'PRIVATE'}])
    assert safe['paired_reviewed_source_subset']['items'] == 1
    assert safe['paired_reviewed_source_subset']['closed_wrong_reference_correct'] == 1
    assert safe['audit_generation_budget_probe']['512']['length_terminated'] == 1
    assert 'PRIVATE' not in json.dumps(safe)

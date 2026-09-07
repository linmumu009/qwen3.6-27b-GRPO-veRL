"""Reconcile diagnostic artifacts and freeze a private, blinded review packet; no inference."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

from scripts.prepare_step120_qa_diagnostic import digest, write_json
from scripts.run_step120_qa_diagnostic import (
    basic_pass, evidence_pass, majority, question_item, summarize, valid_probe,
)

CONDITIONS = ('original_closed', 'original_with_book', 'basic_closed', 'basic_with_book')


def reconcile(rows, probes, predictions, results, report):
    keys = {r['item_hash'] for r in rows}
    if len(keys) != len(rows) or any(set(x) != keys for x in (probes, predictions, results)):
        raise ValueError('artifact identity mismatch')
    for row in rows:
        key = row['item_hash']
        probe = probes[key]
        basic = valid_probe(probe['candidate'], row) and basic_pass(probe['audit'])
        covered = evidence_pass(probe['audit'], row)
        if probe['basic_pass'] is not basic or probe['evidence_pass'] is not covered:
            raise ValueError('stored eligibility mismatch')
        eligible = set(CONDITIONS if basic else CONDITIONS[:2])
        if set(results[key]) != eligible or set(predictions[key]) != set(CONDITIONS):
            raise ValueError('condition coverage mismatch')
        for name in CONDITIONS:
            votes = predictions[key][name]
            if name not in eligible:
                if votes:
                    raise ValueError('ineligible predictions present')
                continue
            if len(votes) != 3:
                raise ValueError('expected three repeats')
            item = question_item(row, probe['candidate'] if name.startswith('basic') else None)
            for vote in votes:
                if type(vote['parse_ok']) is not bool or not isinstance(vote['answers'], list):
                    raise ValueError('invalid prediction schema')
                if any(type(a) is not int or a not in range(len(item.options)) for a in vote['answers']):
                    raise ValueError('invalid answer index')
            if majority(votes, item.expected) != results[key][name]:
                raise ValueError('stored majority mismatch')
    recomputed = summarize(rows, probes, results)
    if any(report.get(k) != v for k, v in recomputed.items()):
        raise ValueError('safe summary mismatch')
    return recomputed


def evidence_gate_reason(row, probe):
    """Mutually exclusive structural gates, NOT semantic root-cause attribution."""
    audit = probe.get('audit')
    if not isinstance(audit, dict):
        return 'audit_missing_or_invalid'
    if audit.get('original_answerable') is not True:
        return 'audit_did_not_assert_sufficiency'
    indices = audit.get('original_answers')
    if not isinstance(indices, list) or not indices or any(
        type(x) is not int or x not in range(len(row['options'])) for x in indices
    ):
        return 'invalid_answer_indices'
    quote = audit.get('original_quote')
    if not isinstance(quote, str) or len(quote.strip()) < 20 or not any(
        quote in p['text'] for p in row['evidence']
    ):
        return 'verbatim_quote_gate_failed'
    if tuple(sorted(set(indices))) != tuple(row['expected']):
        return 'blind_reference_answer_disagrees_with_gold'
    return 'automatic_pass_not_independently_verified'


def is_candidate(row, probe, result):
    return (row['cohort'] == 'original_error' and probe['evidence_pass']
            and probe['basic_pass'] and result['basic_closed']['correct']
            and result['basic_with_book']['correct'] and not result['original_with_book']['correct'])


def select_review(rows, probes, results, limit=8):
    candidates = sorted((r for r in rows if is_candidate(r, probes[r['item_hash']],
                        results[r['item_hash']])), key=lambda r: r['item_hash'])
    if len(candidates) > limit:
        raise ValueError('candidate count exceeds review budget')
    used = {r['item_hash'] for r in candidates}
    # Outcome-independent systematic selection among remaining historical errors.
    # Sorted dataset/category/type supports diversity, not population representativeness.
    pool = sorted((r for r in rows if r['cohort'] == 'original_error' and r['item_hash'] not in used),
                  key=lambda r: (r['dataset'], r['category'], r['question_type'], r['item_hash']))
    count = min(limit - len(candidates), len(pool))
    extra = [pool[int((i + 0.5) * len(pool) / count)] for i in range(count)]
    selected = sorted(candidates + extra, key=lambda r: r['item_hash'])
    return selected, used


def build_packet(rows, probes, results):
    selected, candidates = select_review(rows, probes, results)
    blind, key = [], []
    for index, row in enumerate(selected, 1):
        alias = f'R{index:02d}'
        item_hash = row['item_hash']
        blind.append({'review_id': alias, 'question': row['question'], 'options': row['options'],
                      'question_type': row['question_type'], 'reference_excerpts': row['evidence']})
        key.append({'review_id': alias, 'item_hash': item_hash, 'expected': row['expected'],
                    'selection': 'application_candidate' if item_hash in candidates else 'systematic_error_sample',
                    'probe_and_audit': probes[item_hash], 'results': results[item_hash]})
    template = [{'review_id': b['review_id'], 'status': 'pending', 'reviewer': None,
                 'source_sufficient_for_all_options': None, 'source_derived_answers': None,
                 'option_evidence': [], 'missing_facts': [], 'source_scope_checks': [],
                 'basic_probe_prerequisite_coverage': None, 'notes': ''} for b in blind]
    candidate_results = [results[k] for k in candidates]
    safe = {'private_content_included': False, 'training_performed': False,
            'status': 'prepared_pending_authorized_content_review',
            'review_items': len(selected), 'application_candidates': len(candidates),
            'additional_systematic_errors': len(selected) - len(candidates),
            'selected_by_dataset': dict(Counter(r['dataset'] for r in selected)),
            'candidate_all_four_conditions_stable': sum(all(r[c]['all_repeats_same'] for c in CONDITIONS)
                                                         for r in candidate_results),
            'evidence_gate_counts_all': dict(Counter(evidence_gate_reason(r, probes[r['item_hash']]) for r in rows)),
            'evidence_gate_counts_errors': dict(Counter(evidence_gate_reason(r, probes[r['item_hash']])
                                                       for r in rows if r['cohort'] == 'original_error')),
            'independently_reviewed_items': 0, 'new_evidence_verified_items': 0,
            'new_evaluation_performed': False,
            'limitations': ['structural gate reasons are not semantic diagnoses',
                            'candidate selection is outcome dependent; extra sample is systematic not random',
                            'blinded packet excludes gold, predictions, old audit, and selection reason',
                            'reference enrichment and rerun are gated on content review; no new evidence claimed']}
    return blind, key, template, safe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing overwrite')
    names = ('cases.private.json', 'probes.private.json', 'progress.private.json',
             'results.private.json', 'diagnostic.safe.json', 'cohort.safe.json')
    values = {name: json.loads((args.run / name).read_text(encoding='utf-8')) for name in names}
    hashes = {name: digest(args.run / name) for name in names}
    if hashes['cases.private.json'] != values['cohort.safe.json']['cases_sha256']:
        raise ValueError('case hash mismatch')
    if hashes['results.private.json'] != values['diagnostic.safe.json']['results_sha256']:
        raise ValueError('result hash mismatch')
    if values['diagnostic.safe.json']['manifest'] != values['cohort.safe.json']:
        raise ValueError('manifest mismatch')
    rows, probes, results = (values[n] for n in ('cases.private.json', 'probes.private.json', 'results.private.json'))
    reconcile(rows, probes, values['progress.private.json'], results, values['diagnostic.safe.json'])
    blind, key, template, safe = build_packet(rows, probes, results)
    safe['source_hashes'] = hashes
    safe['all_majorities_and_summary_recomputed'] = True
    os.umask(0o077)
    args.output.mkdir(parents=True)
    for name, payload in (('review.blind.private.json', blind), ('review.key.private.json', key),
                          ('review.decisions.private.json', template)):
        write_json(args.output / name, payload)
    safe['blinded_packet_sha256'] = digest(args.output / 'review.blind.private.json')
    write_json(args.output / 'review.safe.json', safe)
    print(json.dumps(safe, ensure_ascii=False))


if __name__ == '__main__':
    main()

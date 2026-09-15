"""Independently reparse a bounded evidence diagnostic; never modify official scores."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_logistics_knowledge import parse_answers
from scripts.audit_cpt_transfer import CASE_SHA, read_rows, require, sha


def verify(root, packet_path, result_dir):
    source = root/'frozen_cases.private.jsonl'
    require(sha(source) == CASE_SHA, 'frozen source changed')
    source_rows = read_rows(source)
    cases = {r['item_hash']: r for r in source_rows}
    packet_rows = read_rows(packet_path)
    packet = {r['item_hash']: r for r in packet_rows}
    require(len(packet) == len(packet_rows), 'duplicate packet cases')
    registration = json.loads((result_dir/'registration.safe.json').read_text())
    section_control = registration.get('section_id_control', False)
    count, requests = (2, 4) if section_control else (13, 37)
    require(len(packet) == count and sha(packet_path) == registration['packet_sha256'], 'packet mismatch')
    for field, value in dict(requests=requests, cases=count, repeats=1, seed=1024, temperature=0,
            max_tokens=96, max_model_len=8192, tp=8, max_num_seqs=32, source_sha256=CASE_SHA).items():
        require(registration[field] == value, 'diagnostic protocol changed: '+field)
    require(registration['training_started'] is False and registration['official_score_changed'] is False,
            'unexpected experiment scope')
    expected = set()
    for key, row in packet.items():
        require(row['training_allowed'] is False and row['derived_answer'] == cases[key]['expected'], 'unreviewed case')
        if row['source_page'] is not None:
            require(row['short_evidence'] in row['source_page'], 'evidence not a source excerpt')
            require(hashlib.sha256(row['source_page'].encode()).hexdigest() == row['source_text_sha256'],
                    'source page hash mismatch')
        conditions = (['source_page', 'source_page_without_section_ids'] if section_control else
            ['closed', 'short_evidence'] + (['source_page'] if row['source_page'] is not None else []))
        expected.update((key, condition) for condition in conditions)
    rows = read_rows(result_dir/'predictions.private.jsonl')
    require(len(rows) == requests and {(r['item_hash'], r['condition']) for r in rows} == expected,
            'duplicate/missing diagnostic response')
    baseline = json.loads((root/'step120/protocol.safe.json').read_text())
    require(registration['model'] == baseline['model'], 'starting model changed')
    prompt_hashes = dict(zip((r['item_hash'] for r in source_rows), baseline['prompt_hashes']))
    totals = defaultdict(lambda: dict(n=0, correct=0, invalid=0, truncated=0))
    item_results = defaultdict(dict)
    for row in rows:
        case = cases[row['item_hash']]
        parsed, ok = parse_answers(row['prediction'], len(case['options']))
        valid = ok and row['finish_reason'] == 'stop'
        correct = valid and list(parsed) == sorted(case['expected'])
        require(row['parsed'] == list(parsed) and row['valid'] == valid and row['correct'] == correct,
                'raw response disagrees with saved result')
        require(row['dataset'] == case['dataset'] and 0 <= row['output_tokens'] <= 96, 'output budget/dataset mismatch')
        if row['condition'] == 'closed':
            require(row['prompt_sha256'] == prompt_hashes[row['item_hash']], 'closed prompt mismatch')
        group = totals[row['dataset']+'/'+row['condition']]
        group['n'] += 1
        group['correct'] += correct
        group['invalid'] += not valid
        group['truncated'] += row['finish_reason'] == 'length'
        item_results[row['item_hash']][row['condition']] = correct
    saved = json.loads((result_dir/'result.safe.json').read_text())
    require(saved['totals'] == dict(totals), 'aggregate mismatch')
    require(saved['predictions_sha256'] == sha(result_dir/'predictions.private.jsonl'), 'prediction fingerprint mismatch')
    return dict(date='2026-09-15', status='completed_independently_reparsed', requests=requests, unique_cases=count,
        knowledge_groups=len({r['knowledge_unit_id'] for r in packet.values()}), totals=dict(totals),
        predictions_sha256=saved['predictions_sha256'], packet_sha256=sha(packet_path),
        no_evidence_prompt_hashes_match_frozen_baseline=not section_control, section_id_control=section_control,
        official_score_changed=False, training_started=False,
        item_results=[dict(item_hash=k, conditions=v) for k, v in sorted(item_results.items())],
        limitations=['Selected known errors, not a representative sample or independent test.',
            'Answer-bearing definitions test evidence use, not closed-book learning.',
            'Short glossary evidence equals the historical extracted unit; no extraction loss demonstrated.',
            'Single deterministic round; no training-seed or general/Agent retention inference.'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--safe-output', type=Path, required=True)
    a = p.parse_args()
    require(not a.safe_output.exists(), 'refuse overwrite')
    result = verify(a.input, a.packet, a.results)
    a.safe_output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result['totals']))


if __name__ == '__main__':
    main()

"""Reconstruct the frozen draft run and attach explicit operator semantic decisions.

Read-only with respect to inputs. Exclusions are downstream only. No training release.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from cpt_transfer_draft import validate_task
from verify_cpt_transfer_draft import verify, parse, records


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(resources, decisions):
    packet = resources / 'llin-transfer-expansion-sources-20260917-02/requests.private.jsonl'
    base = resources / 'llin-transfer-expansion-draft-20260917-01'
    exclusions = []
    for entry in json.loads((resources / 'llin-transfer-p1-20260916-03/exclusion_inputs.private.json').read_text()):
        path = resources / entry['path']
        assert sha(path) == entry['sha256']
        exclusions.append(path)
    # cases contains the 180 opened pilot items; sealed.private.jsonl is never read.
    exclusions.extend(resources / p for p in (
        'llin-transfer-p1-20260916-03/cases.private.jsonl',
        'llin-transfer-p1-20260916-03/train.private.jsonl',
        'llin-transfer-audit-20260915-01/frozen_cases.private.jsonl'))
    result = verify(packet, base, exclusions)
    requests = {r['id']: r for r in records(packet)}
    failures = Counter()
    count_only = []
    parsed_tasks = parsed_requests = parse_failures = truncated = 0
    for raw in records(base / 'generation.private.jsonl'):
        if raw['finish_reason'] != 'stop':
            truncated += 1
            continue
        try:
            tasks = parse(raw['text'])['tasks']
            assert isinstance(tasks, list)
        except (ValueError, KeyError, AssertionError):
            parse_failures += 1
            continue
        parsed_requests += 1
        for task in tasks:
            parsed_tasks += 1
            request = requests[raw['request_id']]
            try:
                validate_task(task, request)
            except (ValueError, KeyError, TypeError, ZeroDivisionError, SyntaxError) as exc:
                failures[str(exc)] += 1
                relaxed = dict(request)
                relaxed.pop('answer_cardinalities', None)
                try:
                    validate_task(task, relaxed)
                except (ValueError, KeyError, TypeError, ZeroDivisionError, SyntaxError):
                    continue
                count_only.append(raw['request_id'] + '-' + task['design'])
    filtered = records(base / 'filtered.private.jsonl')
    assert len(decisions) == len(filtered)
    assert {d['id'] for d in decisions} == {c['id'] for c in filtered}
    assert all(d['decision'] in ('retain_seed', 'revise_or_reject') and d['reason'] for d in decisions)
    result.update(
        semantic_review=dict(reviewer='Codex operator source-by-source review; not a human or external expert adjudication',
            decisions=decisions, counts=dict(Counter(d['decision'] for d in decisions)),
            seed_meaning='Factually usable starting material only; not a training release or a validated transfer test.'),
        rejection_accounting=dict(parsed_requests=parsed_requests, parse_failures=parse_failures,
            truncated_requests=truncated, parsed_tasks=parsed_tasks,
            first_failure_per_parsed_task=dict(failures),
            cardinality_only_structural_failures=len(count_only),
            cardinality_only_ids=count_only,
            note='First failures partition parsed tasks only. The original skipped_or_missing_designs overlaps failures. Removing cardinality is a counterfactual structural diagnostic, not acceptance.'),
        filtered_train_answer_cardinalities=dict(Counter(len(c['task']['correct_indices']) for c in filtered if c['split']=='train')),
        decision='Do not train this run. Keep all original decisions. Qualify a small source-authored replacement packet before any larger expansion.',
        semantic_review_input_sha256=sha(base / 'filtered.private.jsonl'))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--resources', type=Path, required=True)
    p.add_argument('--decisions', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.resources, json.loads(a.decisions.read_text(encoding='utf-8')))
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('filtered_counts', 'diagnostic_questions_checked', 'training_allowed')}))

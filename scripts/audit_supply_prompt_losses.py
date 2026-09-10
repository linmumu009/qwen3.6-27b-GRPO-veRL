"""Read-only private-result audit; emit aggregate counts only (no question text)."""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/workspace/llin-verl-grpo')
RUN = ROOT / 'runs/supply-chain-full-prompt-comparison-20260909-01'
CASES = ROOT / 'runs/logistics-cpt-diagnostics-20260904/private/public_eval/frozen_cases_source.jsonl'


def majority(rows, gold):
    assert len(rows) == 3 and {r['repeat'] for r in rows} == {0, 1, 2}
    counts = Counter(tuple(r['parsed']) if r['valid'] else None for r in rows)
    answer, count = counts.most_common(1)[0]
    return count >= 2 and answer == tuple(sorted(gold))


def loss_kind(rows):
    """Descriptive, mutually exclusive observed states, not causal attribution."""
    valid = [tuple(r['parsed']) for r in rows if r['valid']]
    if valid and Counter(valid).most_common(1)[0][1] >= 2:
        return 'valid_wrong_majority'
    if sum(not r['valid'] for r in rows) >= 2:
        return 'invalid_majority'
    return 'no_answer_majority'


def main():
    import hashlib
    assert hashlib.sha256(CASES.read_bytes()).hexdigest() == 'b652b2108cb552346df11d005c15ff3137c50a756a7b24eb35302683ec33ed99'
    cases = [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]
    index = {r['item_hash']: r for r in cases}
    assert len(index) == len(cases) == 1672
    groups = defaultdict(list)
    seen = set()
    for shard in range(2):
        assert json.loads((RUN / f'shard{shard}/progress.safe.json').read_text())['status'] == 'complete'
        for line in (RUN / f'shard{shard}/predictions.private.jsonl').read_text().splitlines():
            r = json.loads(line)
            key = (r['item_hash'], r['condition'], r['repeat'])
            assert key not in seen
            seen.add(key)
            case = index[r['item_hash']]
            assert r['dataset'] == case['dataset'] and r['shard'] == shard
            assert r['condition'] in ('original96', 'guided2048')
            assert r['correct'] == (r['valid'] and r['parsed'] == sorted(case['expected']))
            assert not (r['truncated'] and r['valid'])
            groups[key[:2]].append(r)
    assert len(seen) == 10032 and len(groups) == 3344
    result = []
    for dataset in ('all', 'SC-bench-knowledge', 'LogistikaBench'):
        out = Counter()
        for item, case in index.items():
            if dataset != 'all' and case['dataset'] != dataset:
                continue
            a, b = (groups[item, label] for label in ('original96', 'guided2048'))
            before, after = (majority(rows, case['expected']) for rows in (a, b))
            out.update(n=1, before=int(before), after=int(after))
            if not before and after:
                out['gains'] += 1
            if before and not after:
                out['losses'] += 1
                out[loss_kind(b)] += 1
                out['loss_with_any_truncation'] += int(any(r['truncated'] for r in b))
            out['guided_invalid_without_truncation'] += sum(not r['valid'] and not r['truncated'] for r in b)
        result.append(dict(dataset=dataset, **out))
    summary = json.loads((RUN / 'summary.safe.json').read_text())
    for row, original in zip(result, summary['table']):
        for key in ('dataset', 'n', 'before', 'after', 'gains', 'losses'):
            assert row[key] == original[key]
    print(json.dumps(dict(requests=len(seen), score_and_summary_checks='passed', table=result,
        limitation='Uses recorded parsed/valid flags; no independent text-parser or prompt-to-output alignment audit. Loss states are descriptive, not causes.'), indent=2))


if __name__ == '__main__':
    main()

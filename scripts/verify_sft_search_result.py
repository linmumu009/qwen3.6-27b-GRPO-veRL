"""Independently reparse every frozen benchmark answer; emit aggregate evidence only."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/workspace/llin-verl-grpo')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', type=Path, required=True)
    a = p.parse_args()
    from run_logistics_strategy_diagnostic import parse_answers
    source = ROOT/'runs/logistics-cpt-diagnostics-20260904/private/public_eval/frozen_cases_source.jsonl'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == 'b652b2108cb552346df11d005c15ff3137c50a756a7b24eb35302683ec33ed99'
    source_rows = [json.loads(x) for x in source.read_text().splitlines()]
    cases = {r['item_hash']: r for r in source_rows}
    assert len(cases) == len(source_rows) == 1672

    def verify(directory):
        prediction_file = directory/'predictions.private.jsonl'
        rows = [json.loads(x) for x in prediction_file.read_text().splitlines()]
        assert len(rows) == 5016
        groups = defaultdict(list)
        for r in rows:
            c = cases[r['item_hash']]
            parsed, ok = parse_answers(r['prediction'], len(c['options']))
            valid = ok and not r['truncated']
            assert list(parsed) == r['parsed'] and valid == r['valid']
            assert (valid and sorted(parsed) == sorted(c['expected'])) == r['correct']
            assert r['dataset'] == c['dataset']
            groups[r['item_hash']].append(r)
        assert set(groups) == set(cases)
        scores = {}
        for key, group in groups.items():
            assert len(group) == 3 and {r['repeat'] for r in group} == {0, 1, 2}
            votes = Counter(tuple(r['parsed']) if r['valid'] else None for r in group)
            answer, n = votes.most_common(1)[0]
            scores[key] = n >= 2 and answer is not None and list(answer) == sorted(cases[key]['expected'])
        saved = json.loads((directory/'scores.safe.json').read_text())
        assert set(saved) == set(scores)
        assert all(saved[k]['correct'] == v for k, v in scores.items())
        return scores, dict(requests=len(rows), invalid=sum(not r['valid'] for r in rows),
                    truncated=sum(r['truncated'] for r in rows),
                    predictions_sha256=hashlib.sha256(prediction_file.read_bytes()).hexdigest())

    baseline = ROOT/'runs/cpt-controlled-storage-20260910/llin/cpt-controlled-20260910/curve8x/eval_epoch_0'
    before, before_audit = verify(baseline)
    after, after_audit = verify(a.candidate)
    protocols = [json.loads((d/'protocol.safe.json').read_text()) for d in (baseline, a.candidate)]
    assert {k:v for k,v in protocols[0].items() if k != 'model'} == {k:v for k,v in protocols[1].items() if k != 'model'}
    table = []
    for dataset in ('SC-bench-knowledge', 'LogistikaBench'):
        keys = [k for k,c in cases.items() if c['dataset'] == dataset]
        table.append(dict(dataset=dataset, n=len(keys), before=sum(before[k] for k in keys),
                     after=sum(after[k] for k in keys), gains=sum(not before[k] and after[k] for k in keys),
                     losses=sum(before[k] and not after[k] for k in keys)))
    print(json.dumps(dict(table=table, baseline=before_audit, candidate=after_audit,
                     protocol_equal=True, raw_outputs_reparsed=True, model_promoted=False)))


if __name__ == '__main__':
    main()

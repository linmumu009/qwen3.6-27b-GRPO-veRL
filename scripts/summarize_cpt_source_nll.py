"""Verify frozen source NLL outputs and publish text-free descriptive aggregates."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    return dict(tokens=len(values), nll_sum=sum(values), nll=sum(values)/len(values))


def summarize(root, cases_path, registration_path):
    reg = read(registration_path)
    assert sha(cases_path) == reg['cases_sha256']
    assert read(root/'registration.safe.json') == reg
    assert read(root/'status.safe.json')['state'] == 'scoring_complete_analysis_pending'
    cases = read(cases_path)
    assert len(cases) == 6 and len({c['unit_id'] for c in cases}) == 6
    for case in cases:
        assert hashlib.sha256(case['body'].encode()).hexdigest() == case['body_sha256']
        assert len(case['input_ids'])-case['answer_start'] == case['loss_tokens']
    report = dict(id=reg['id'], models={}, paired_content_delta={},
                  interpretation=reg['interpretation'], promotion=False,
                  registration_sha256=sha(registration_path))
    masks = {}
    requests = discarded = 0
    for model in reg['models']:
        folder = root/model
        complete = read(folder/'completed.safe.json')
        reserved = read(folder/'request.reserved.json')
        result_path = folder/'token_nll.private.json'
        assert sha(result_path) == complete['result_sha256']
        assert reserved['cases_sha256'] == reg['cases_sha256']
        assert complete['requests'] == reserved['requests'] == 6
        assert complete['scored_tokens'] == reserved['scored_tokens'] == 357
        assert complete['training_steps'] == 0
        records = read(result_path)
        assert len(records) == 6
        units = []
        pooled = {k: [] for k in ('content', 'heading', 'all_supervised', 'eos')}
        for case, record in zip(cases, records):
            assert case['unit_id'] == record['unit_id']
            assert case['body_sha256'] == record['body_sha256']
            vals, mask = record['token_nll'], record['content_mask']
            assert len(vals) == case['loss_tokens'] and len(mask) == len(vals)-1
            assert all(type(x) is bool for x in mask) and any(mask) and not all(mask)
            assert all(math.isfinite(v) and v >= -1e-5 for v in vals)
            previous = masks.setdefault(case['unit_id'], mask)
            assert mask == previous, 'Token partitions differ between models'
            partitions = dict(content=[v for v, b in zip(vals[:-1], mask) if b],
                              heading=[v for v, b in zip(vals[:-1], mask) if not b],
                              all_supervised=vals, eos=[vals[-1]])
            unit = dict(unit_id=case['unit_id'], body_sha256=case['body_sha256'])
            for part, values in partitions.items():
                unit[part] = stats(values)
                pooled[part].extend(values)
                if part != 'eos':
                    assert record[part]['tokens'] == len(values)
                    for key in ('nll_sum', 'nll'):
                        assert math.isclose(record[part][key], unit[part][key], abs_tol=1e-9)
            assert record['eos_nll'] == vals[-1]
            assert 0 <= record['discarded_decode_tokens'] <= 1
            units.append(unit)
        assert len(pooled['all_supervised']) == 357
        count = sum(r['discarded_decode_tokens'] for r in records)
        assert count == complete['discarded_decode_tokens'] <= 6
        requests += complete['requests']
        discarded += count
        report['models'][model] = dict(units=units, result_sha256=sha(result_path),
            aggregates={part: dict(**stats(values), macro_nll=sum(u[part]['nll'] for u in units)/6)
                        for part, values in pooled.items()})
    assert requests == 24 and discarded <= 24
    report['counts'] = dict(scoring_requests=requests, scored_tokens=1428,
                            discarded_decode_tokens=discarded, free_answer_calls=0,
                            training_steps=0, formal_reruns=0)
    for baseline in ('step120', 'p1', 'R'):
        k = report['models']['K']
        b = report['models'][baseline]
        report['paired_content_delta']['K_minus_'+baseline] = dict(
            macro=k['aggregates']['content']['macro_nll']-b['aggregates']['content']['macro_nll'],
            token_weighted=k['aggregates']['content']['nll']-b['aggregates']['content']['nll'],
            units=[dict(unit_id=u['unit_id'], delta=u['content']['nll']-v['content']['nll'])
                   for u, v in zip(k['units'], b['units'])])
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = summarize(a.root, a.cases, a.registration)
    a.out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(counts=result['counts'], deltas=result['paired_content_delta']), indent=2))

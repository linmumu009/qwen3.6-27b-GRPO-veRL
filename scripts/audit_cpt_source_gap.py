"""Offline linkage of frozen fact coverage, reference token NLL and free answers."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from tokenizers import Tokenizer
from summarize_cpt_source_nll import summarize


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def overlap_tokens(offsets, ranges):
    return sorted({i for i, (a, b) in enumerate(offsets)
                   for start, end in ranges if a < end and b > start})


def common_prefix(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def audit(project, tokenizer_path):
    private = project/'CPT_resources'
    nll = private/'llin-source-nll-20260922-01'
    recall = private/'llin-source-recall-20260922-01'
    gap = private/'llin-source-gap-20260922-01'
    reg = read(project/'docs/cpt_source_gap_registration_20260922.safe.json')
    assert sha(gap/'fact_spans.private.json') == reg['fact_spans_sha256']
    summarize(nll/'results', nll/'cases.private.json',
              project/'docs/cpt_source_nll_registration_20260922.safe.json')
    cases = read(nll/'cases.private.json')
    spans = read(gap/'fact_spans.private.json')
    probes = read(recall/'probes.private.json')
    assert len(cases) == len(spans) == len(probes) == 6
    review = recall/'blind-review'
    freeze, frozen_ratings = read(review/'freeze.safe.json'), read(review/'ratings_frozen.safe.json')
    assert sha(recall/'probes.private.json') == freeze['probes_sha256']
    assert sha(review/'identity_key.private.json') == freeze['key_sha256']
    assert sha(review/'ratings.private.json') == frozen_ratings['ratings_sha256']
    identity = read(review/'identity_key.private.json')
    ratings = read(review/'ratings.private.json')
    assert len(ratings) == 72 and len({r['blind_id'] for r in ratings}) == 72
    coverage = {}
    for r in ratings:
        key = identity[r['blind_id']]
        join = (key['model'], key['unit_id'], key['repeat'])
        assert join not in coverage
        coverage[join] = r
    tokenizer_hash = sha(tokenizer_path)
    for m in reg['models']:
        reserved = read(nll/'results'/m/'request.reserved.json')
        assert reserved['identity']['metadata_sha256']['tokenizer.json'] == tokenizer_hash
    tok = Tokenizer.from_file(str(tokenizer_path))
    token_results, predictions = {}, {}
    for m in reg['models']:
        token_results[m] = {r['unit_id']: r for r in read(nll/'results'/m/'token_nll.private.json')}
        path = recall/'results'/(m+'_direct')/'predictions.private.jsonl'
        assert sha(path) == freeze['raw_sha256'][m]
        rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
        assert len(rows) == 18 and len({(r['unit_id'], r['repeat']) for r in rows}) == 18
        predictions[m] = rows
    result = dict(id=reg['id'], registration=reg, tokenizer_sha256=tokenizer_hash,
                  frozen_ratings_sha256=frozen_ratings['ratings_sha256'], units=[])
    for case, mapping, probe in zip(cases, spans, probes):
        uid = case['unit_id']
        assert uid == mapping['unit_id'] == probe['unit_id']
        assert case['body_sha256'] == mapping['body_sha256'] == probe['source_body_sha256']
        encoded = tok.encode(case['body'], add_special_tokens=False)
        assert encoded.ids == case['input_ids'][case['answer_start']:-1]
        source_heading, source_body = case['body'].split('\n\n', 1)
        start_content = len(source_heading)+2
        unit = dict(unit_id=uid, body_sha256=case['body_sha256'], facts=[], free_answers={})
        assert len(mapping['facts']) == len(probe['required'])
        for i, fact in enumerate(mapping['facts']):
            assert fact['fact_index'] == i and fact['required'] == probe['required'][i]
            assert all(start_content <= a < b <= len(case['body']) for a, b in fact['ranges'])
            indices = overlap_tokens(encoded.offsets, fact['ranges'])
            assert indices
            item = dict(fact_index=i, char_ranges=fact['ranges'], token_indices=indices,
                        implied_not_separate_text=fact['implied_not_separate_text'], models={})
            for m in reg['models']:
                values = [token_results[m][uid]['token_nll'][j] for j in indices]
                facts_by_repeat = [coverage[(m, uid, r)]['facts'] for r in range(3)]
                assert all(len(f) == len(mapping['facts']) for f in facts_by_repeat)
                item['models'][m] = dict(nll=math.fsum(values)/len(values),
                    first_reference_token_nll=values[0], max_reference_token_nll=max(values),
                    supported_repeats=sum(f[i] for f in facts_by_repeat))
            item['K_minus_p1'] = item['models']['K']['nll']-item['models']['p1']['nll']
            item['K_minus_R'] = item['models']['K']['nll']-item['models']['R']['nll']
            unit['facts'].append(item)
        for m in reg['models']:
            rows = sorted((r for r in predictions[m] if r['unit_id'] == uid), key=lambda r: r['repeat'])
            assert [r['repeat'] for r in rows] == [0, 1, 2]
            records = []
            for r in rows:
                parts = r['prediction'].split('\n\n', 1)
                # The observed K format contains a heading. Other styles are not forced into this schema.
                headed = len(parts) == 2 and m == 'K'
                lexical = common_prefix(source_body, parts[1]) if headed else None
                score = coverage[(m, uid, r['repeat'])]
                records.append(dict(repeat=r['repeat'], finish_reason=r['finish_reason'],
                    output_tokens=r['output_tokens'], source_supervised_tokens=case['loss_tokens'],
                    source_heading_exact=(parts[0] == source_heading) if headed else None,
                    body_common_prefix_chars=lexical,
                    all_facts_supported=all(score['facts']), supported_facts=sum(score['facts']),
                    forbidden_inference=score['forbidden_inference']))
            unit['free_answers'][m] = records
        result['units'].append(unit)
    assert len(result['units']) == 6 and sum(len(u['facts']) for u in result['units']) == 26
    facts = [f for u in result['units'] for f in u['facts']]
    result['summary'] = dict(facts=26, new_model_calls=0, training_steps=0,
        K_missing_all_repeats=sum(f['models']['K']['supported_repeats'] == 0 for f in facts),
        K_supported_all_repeats=sum(f['models']['K']['supported_repeats'] == 3 for f in facts),
        K_nll_lower_than_p1=sum(f['K_minus_p1'] < 0 for f in facts),
        K_nll_lower_than_R=sum(f['K_minus_R'] < 0 for f in facts),
        missing_facts_K_nll_lower_than_R=sum(f['K_minus_R'] < 0 and f['models']['K']['supported_repeats'] == 0 for f in facts),
        K_natural_stops=sum(r['finish_reason'] == 'stop' for u in result['units'] for r in u['free_answers']['K']),
        K_exact_headings=sum(r['source_heading_exact'] for u in result['units'] for r in u['free_answers']['K']))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, default=Path('.'))
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.project, a.tokenizer)
    a.out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(result['summary'], indent=2))

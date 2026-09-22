"""Verify context diagnostic, blind response review, then reveal frozen results."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import secrets


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, data):
    assert not p.exists(), 'Refusing to overwrite frozen artifact: '+str(p)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def verify(base, registration):
    reg = read(registration)
    cases_path = base/'cases.private.json'
    assert sha(cases_path) == reg['cases_sha256']
    cases = {c['id']: c for c in read(cases_path)}
    assert len(cases) == 8
    root = base/'results'
    assert read(root/'registration.safe.json') == reg
    assert read(root/'status.safe.json')['state'] == 'complete_review_pending'
    data, hashes = {}, {}
    for model in reg['models']:
        data[model] = {}
        for kind in ('nll','response'):
            folder = root/model
            done = read(folder/(kind+'.completed.safe.json'))
            reserved = read(folder/(kind+'.reserved.json'))
            path = folder/(kind+'.private.json')
            assert done['requests'] == reserved['requests'] == 8
            assert sha(path) == done['result_sha256']
            assert reserved['cases_sha256'] == reg['cases_sha256']
            rows = read(path)
            assert len(rows) == 8 and {r['id'] for r in rows} == set(cases)
            assert sum(r['output_tokens'] for r in rows) == done['output_tokens']
            for r in rows:
                c = cases[r['id']]
                assert all(r[k] == c[k] for k in ('unit_id','condition','fact_index'))
                assert 0 <= r['output_tokens'] <= (1 if kind == 'nll' else 384)
                if kind == 'nll':
                    values = r['token_nll']
                    assert len(values)-1 == len(c['answer_ids']) == r['answer_tokens']
                    assert all(math.isfinite(v) and v >= -1e-5 for v in values)
                    assert math.isclose(math.fsum(values[:-1])/len(c['answer_ids']),r['answer_nll'],abs_tol=1e-10)
                    assert r['first_token_nll'] == values[0] and r['eos_nll'] == values[-1]
            data[model][kind] = {r['id']: r for r in rows}
            hashes[model+'/'+kind] = sha(path)
    return reg, cases, data, hashes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['blind','freeze','summarize'])
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    reg, cases, data, hashes = verify(a.base, a.registration)
    review = a.base/'blind-review'
    review.mkdir(exist_ok=True)
    if a.mode == 'blind':
        records, key = [], {}
        for model, results in data.items():
            for cid, r in results['response'].items():
                blind_id = secrets.token_hex(12)
                c = cases[cid]
                key[blind_id] = dict(model=model, case_id=cid)
                records.append(dict(blind_id=blind_id, question=c['messages'][-1]['content'],
                    required=c['required'], reference=c['target_sentence'],
                    response=r['prediction'], finish_reason=r['finish_reason']))
        secrets.SystemRandom().shuffle(records)
        save(review/'review.private.json', records)
        save(review/'key.private.json', key)
        save(review/'freeze.safe.json', dict(cases_sha256=reg['cases_sha256'], raw_sha256=hashes,
             review_sha256=sha(review/'review.private.json'), key_sha256=sha(review/'key.private.json')))
        print('16 records blinded; model and context key withheld')
        return
    frozen = read(review/'freeze.safe.json')
    assert frozen['raw_sha256'] == hashes and frozen['cases_sha256'] == reg['cases_sha256']
    assert sha(review/'review.private.json') == frozen['review_sha256']
    assert sha(review/'key.private.json') == frozen['key_sha256']
    ratings_path = review/'ratings.private.json'
    ratings = read(ratings_path)
    assert len(ratings) == 16 and len({r['blind_id'] for r in ratings}) == 16
    assert {r['blind_id'] for r in ratings} == {r['blind_id'] for r in read(review/'review.private.json')}
    for r in ratings:
        assert all(type(r[k]) is bool for k in ('supported','contradiction','uncertain'))
        assert r['reason'].strip()
    if a.mode == 'freeze':
        save(review/'ratings_frozen.safe.json', dict(ratings_sha256=sha(ratings_path), items=16,
             reviewer='Single Codex review, model and context labels hidden; no independent human review',
             criterion='Complete target fact supported by response, without contradiction or uncertainty; truncation reported separately'))
        print('Ratings frozen',sha(ratings_path))
        return
    assert sha(ratings_path) == read(review/'ratings_frozen.safe.json')['ratings_sha256']
    key = read(review/'key.private.json')
    scored = {}
    for r in ratings:
        who = key[r['blind_id']]
        scored[(who['model'], who['case_id'])] = r
    report = dict(id=reg['id'], ratings_sha256=sha(ratings_path), raw_sha256=hashes,
                  models={}, response_calls=16, nll_requests=16, training_steps=0,
                  formal_reruns=0, promotion=False)
    units = sorted({c['unit_id'] for c in cases.values()})
    for model in reg['models']:
        rows = []
        for uid in units:
            row = dict(unit_id=uid, conditions={})
            for condition in ('source','self'):
                cid = uid+'-'+condition
                nll, response = data[model]['nll'][cid], data[model]['response'][cid]
                rating = scored[(model,cid)]
                row['conditions'][condition] = dict(answer_nll=nll['answer_nll'],
                    first_token_nll=nll['first_token_nll'], eos_nll=nll['eos_nll'],
                    answer_tokens=nll['answer_tokens'], response_tokens=response['output_tokens'],
                    truncated=response['finish_reason'] == 'length',
                    supported=rating['supported'], contradiction=rating['contradiction'],
                    uncertain=rating['uncertain'],
                    passed=rating['supported'] and not rating['contradiction'] and not rating['uncertain'],
                    prompt_tokens=len(cases[cid]['prompt_ids']))
            row['self_minus_source_nll'] = row['conditions']['self']['answer_nll']-row['conditions']['source']['answer_nll']
            rows.append(row)
        report['models'][model] = dict(units=rows,
            macro_self_minus_source_nll=sum(r['self_minus_source_nll'] for r in rows)/4,
            passes={c:sum(r['conditions'][c]['passed'] for r in rows) for c in ('source','self')})
    report['discarded_nll_decode_tokens'] = sum(r['output_tokens'] for v in data.values() for r in v['nll'].values())
    report['response_tokens'] = sum(r['output_tokens'] for v in data.values() for r in v['response'].values())
    assert report['discarded_nll_decode_tokens'] <= 16 and report['response_tokens'] <= 6144
    report['difference_of_context_effect_K_minus_R'] = report['models']['K']['macro_self_minus_source_nll']-report['models']['R']['macro_self_minus_source_nll']
    assert a.out is not None
    save(a.out, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

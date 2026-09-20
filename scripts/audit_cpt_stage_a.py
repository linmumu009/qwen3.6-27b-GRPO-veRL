"""Stage A descriptive audit. No training, model calls, or benchmark relabelling.

The review JSON is a human-readable semantic assessment, not a classifier.
Source-ID matches identify training candidates; absence never proves non-exposure.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import zipfile

from audit_cpt_transfer import CASE_SHA, read_rows, require, sha, verify_predictions, verify_protocols
from prepare_cpt_p4_cumulative import OLD_MESSAGES_SHA
from run_logistics_strategy_diagnostic import messages_for
from verify_cpt_pilot_result import independently_parse


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def audit(root, review_path, private_out):
    resources = root / 'CPT_resources'
    snapshot = resources / 'llin-transfer-audit-20260915-01/frozen_cases.private.jsonl'
    require(sha(snapshot) == CASE_SHA, 'case snapshot changed')
    all_cases = read_rows(snapshot)
    cases = {r['item_hash']: r for r in all_cases}
    require(len(cases) == len(all_cases) == 1672, 'case identity mismatch')
    dirs = {
        'p1': resources / 'llin-transfer-formal-20260917-01/p1',
        'p4': resources / 'llin-transfer-p4-run-20260918-01/formal5016',
    }
    verify_protocols({m: read(d / 'protocol.safe.json') for m, d in dirs.items()}, CASE_SHA, len(cases))
    scores, raw_hashes = {}, {}
    for model, directory in dirs.items():
        raw_path = directory / 'predictions.private.jsonl'
        raw = read_rows(raw_path)
        scores[model] = verify_predictions(cases, raw, read(directory / 'scores.safe.json'))
        for r in raw:
            parsed, valid = independently_parse(r['prediction'], len(cases[r['item_hash']]['options']))
            require(parsed == r['parsed'] and (valid and not r['truncated']) == r['valid'], 'parser disagreement')
        raw_hashes[model] = dict(sha256=sha(raw_path), requests=len(raw))
    common = {k for k in cases if not scores['p1'][k]['correct'] and not scores['p4'][k]['correct']}
    ordered = sorted((cases[k] for k in common), key=lambda c: (c['dataset'] != 'SC-bench-knowledge', c['category'], c['item_hash']))
    require(len(ordered) == 269, 'unexpected common error count')
    source_path = resources / 'llin-cpt-sft-search-20260912/source_generation_requests.private.jsonl'
    archive_path = resources / 'llin-knowledge-complete-20260911/core/sources.private.jsonl'
    books_path = resources / 'llin-knowledge-expansion-20260911/book_pages.private.jsonl'
    train_path = resources / 'llin-transfer-p4-data-20260918-01/train.messages.private.jsonl'
    sources, archive, books, train = read_rows(source_path), read_rows(archive_path), read_rows(books_path), read_rows(train_path)
    archive = {r['key']: r for r in archive}
    books = {r['id']: r for r in books}
    review = read(review_path)
    for field, path in [('source_catalog_sha256', source_path), ('archive_sha256', archive_path), ('book_pages_sha256', books_path), ('train_sha256', train_path)]:
        require(review[field] == sha(path), field + ' mismatch')
    require(len(train) == 363, 'training count mismatch')
    prefix = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in train[:315]).encode()
    require(hashlib.sha256(prefix).hexdigest() == OLD_MESSAGES_SHA, 'P1 prefix mismatch')
    require(len(review['items']) == 269, 'incomplete review')
    require({a['item_hash'] for a in review['items']} == common, 'review identity mismatch')
    require(len({a['item_hash'] for a in review['items']}) == 269, 'duplicate review item')
    web_path = resources / 'llin-stage-a-20260920-01/official_web_read.private.json'
    require(sha(web_path) == review['official_web_read_sha256'], 'official web archive mismatch')
    primary_snapshots = {}
    for reference in review.get('primary_source_snapshots', []):
        path = root / reference['path']
        require(sha(path) == reference['sha256'], 'primary snapshot mismatch')
        primary_snapshots[reference['id']] = path.read_text(encoding='utf-8')
    zip_path = root / 'datasets/SC-bench-main.zip'
    require(sha(zip_path) == review['sc_zip_sha256'], 'SC archive mismatch')
    with zipfile.ZipFile(zip_path) as z:
        sop_hashes = {n: hashlib.sha256(z.read(n)).hexdigest() for n in review['sc_context_members']}

    ledger, private = [], []
    for a in review['items']:
        c = cases[a['item_hash']]
        require(ordered[a['review_index_zero_based']]['item_hash'] == a['item_hash'], 'review order mismatch')
        require(a['status'] in ['A', 'B', 'C', 'D'], 'unrecognised review status')
        require(a['note'] and a['capability_group'] and a['dedup_group'], 'empty review field')
        refs, source_ids = [], []
        for i in a['source_rows']:
            require(type(i) is int and 0 <= i < len(sources), 'bad catalog row')
            source = sources[i]
            source_ids.append(source['id'])
            for e in source['evidence']:
                r = archive[e['key']]
                require(hashlib.sha256(r['text'].encode()).hexdigest() == r['sha256'] == e['sha256'], 'source evidence hash mismatch')
                refs.append(dict(key=e['key'], sha256=r['sha256'], url=r.get('url'),
                                 kind='primary_book_extraction' if e['key'].startswith('book:') else 'primary_code_list' if e['key'].startswith('iata-code-list:') else 'summary_requires_primary_check'))
        book_refs = []
        for identity in a['book_ids']:
            b = books[identity]
            book_refs.append(dict(id=identity, source_file=b['source_file'], locator=b['locator'], sha256=hashlib.sha256(b['text'].encode()).hexdigest()))
        candidate_rows = [i for i, r in enumerate(train) if any(s in r['id'] for s in source_ids)]
        checked = a['training_rows']
        require(all(type(i) is int and 0 <= i < len(train) for i in checked), 'bad training row')
        require(a['training_coverage'] != 'rule_present' or checked, 'rule assertion without actual messages')
        if a['status'] == 'A':
            require(refs or book_refs or a.get('official_url'), 'source sufficient claim without source')
            require(a['input_sufficiency'] == 'sufficient_in_reviewed_scope' and a['label_uniqueness'] == 'supported_by_review', 'A lacks explicit input/label review')
            if refs and all(r['kind'] == 'summary_requires_primary_check' for r in refs):
                require(a.get('official_url'), 'summary promoted to primary')
            if a.get('primary_snapshot_id'):
                require(a['primary_snapshot_id'] in primary_snapshots, 'missing primary snapshot')
                require(a['official_url'] in primary_snapshots[a['primary_snapshot_id']], 'primary URL not in snapshot')
                require(all(token in primary_snapshots[a['primary_snapshot_id']] for token in a['primary_required_spans']), 'primary evidence span missing')
        prompt = messages_for(c, 'original')[0]
        require(all(isinstance(m['content'], str) for m in prompt), 'unexpected multimodal content')
        norm = lambda x: re.sub(r'\s+', ' ', x).strip().casefold()
        overlaps = [[j for j, o in enumerate(c['options']) if j != g and norm(o) == norm(c['options'][g])] for g in c['expected']]
        require(a['status'] != 'A' or not any(overlaps), 'A includes duplicate gold label')
        entry = dict(a, dataset=c['dataset'], category=c['category'], option_count=len(c['options']),
                     source_ids=source_ids, source_references=refs, book_references=book_refs,
                     training_source_id_candidate_rows=candidate_rows,
                     training_message_ids=[train[i]['id'] for i in checked],
                     gold_has_casefold_equivalent_candidate=any(overlaps),
                     p1_repeat_unstable=scores['p1'][c['item_hash']]['repeat_unstable'],
                     p4_repeat_unstable=scores['p4'][c['item_hash']]['repeat_unstable'])
        ledger.append(entry)
        private.append(dict(review=entry, case=c, scores={m: s[c['item_hash']] for m, s in scores.items()},
                            training_messages=[train[i] for i in sorted(set(checked + candidate_rows))]))
    summary = []
    for dataset, gap in [('SC-bench-knowledge', 4), ('LogistikaBench', 31)]:
        subset = [x for x in ledger if x['dataset'] == dataset]
        confirmed = [x for x in subset if x['status'] == 'A']
        dedup = len({r['dedup_group'] for r in confirmed})
        summary.append(dict(dataset=dataset, items=len(subset), statuses=dict(sorted(Counter(x['status'] for x in subset).items())),
                            confirmed_items=len(confirmed), confirmed_deduplicated_items=dedup,
                            confirmed_capability_groups=len({x['capability_group'] for x in confirmed}),
                            confirmed_with_rule_supervision=sum(x['training_coverage'] == 'rule_present' for x in confirmed),
                            training_coverage=dict(Counter(x['training_coverage'] for x in subset)),
                            unknown_items=sum(x['status'] in ['B','C'] for x in subset),
                            disputed_items=sum(x['status'] == 'D' for x in subset), required_net_gain=gap,
                            space_gate='necessary_count_only' if dedup >= gap else 'insufficient_current_verified_sources'))
    private_out.mkdir(parents=True, exist_ok=True)
    write(private_out / 'ledger.private.json', private)
    return dict(schema_version=1, date='2026-09-20', stage='A_basic_review_complete',
                method='Codex semantic review; no independent adjudication; counts describe current verified sources, not an upper bound on repairability',
                status_definitions=review['status_definitions'], datasets=summary,
                common_error_categories=dict(Counter(x['category'] for x in ledger)),
                raw_predictions=raw_hashes, formal_cases_sha256=CASE_SHA,
                source_catalog_sha256=sha(source_path), archive_sha256=sha(archive_path),
                training_sha256=sha(train_path), p1_prefix_sha256=OLD_MESSAGES_SHA,
                review_sha256=sha(review_path), sc_context_member_sha256=sop_hashes,
                **({'primary_source_snapshots': review['primary_source_snapshots']} if review.get('primary_source_snapshots') else {}),
                long_list_items=sum(x['option_count'] == 269 for x in ledger),
                confirmed_long_list_items=sum(x['option_count'] == 269 and x['status'] == 'A' for x in ledger),
                gold_casefold_overlap_items=sum(x['gold_has_casefold_equivalent_candidate'] for x in ledger),
                model_calls=0, training_runs=0, official_scoring_changed=False,
                next_action='source_remediation_before_stage_B', items=ledger)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--review', type=Path, default=Path('docs/cpt_stage_a_review_20260920.safe.json'))
    p.add_argument('--out', type=Path, default=Path('docs/cpt_stage_a_result_20260920.safe.json'))
    p.add_argument('--private-out', type=Path, default=Path('CPT_resources/llin-stage-a-20260920-01'))
    args = p.parse_args()
    result = audit(args.root, args.review, args.private_out)
    write(args.out, result)
    print(json.dumps({k:v for k,v in result.items() if k != 'items'}, ensure_ascii=False, indent=2))

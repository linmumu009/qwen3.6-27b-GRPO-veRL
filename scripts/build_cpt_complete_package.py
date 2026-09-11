"""Assemble all-book CPT prose and reviewed additions without benchmark labels.

Readiness here is a data contract, not proof of exhaustive factual correctness or
held-out benchmark validity. Inputs and private outputs never belong in Git.
"""
import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from build_cpt_reviewed_core import EOS, TOKENIZER_SHA, build, digest, read_jsonl, write_json

BASE_SHA = '898dada4de8c7274ee0b18bf03f685998bd9cf25739af2502f1eed67c3cb9733'


def checked_row(row, tokenizer, origin, source_keys):
    text = row['text']
    sha = digest(text.encode('utf-8'))
    if sha != row['text_sha256'] or row['tokenizer_sha256'] != TOKENIZER_SHA:
        raise ValueError('Text or tokenizer fingerprint mismatch')
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if not ids or len(ids) + 1 > 4096 or EOS in ids:
        raise ValueError('Empty, oversized or EOS-containing content')
    if any(mark in text for mark in ('<|im_start|>', '<|im_end|>', '\ufffd')):
        raise ValueError('Damaged or chat-formatted prose')
    if row['content_tokens'] != len(ids) or row['token_count'] != len(ids) + 1:
        raise ValueError('Token budget mismatch')
    if not source_keys:
        raise ValueError('Missing provenance')
    return dict(id=row['id'], text=text, text_sha256=sha,
                content_tokens=len(ids), token_count=len(ids)+1, eos_token_id=EOS,
                tokenizer_sha256=TOKENIZER_SHA, origin=origin, source_keys=source_keys,
                original_record_ids=[row['id']], topics=row.get('topics', []))


def deduplicate(rows):
    result, seen = [], {}
    for row in rows:
        key = row['text_sha256']
        if key in seen:
            existing = seen[key]
            if existing['text'] != row['text']:
                raise ValueError('Text digest collision')
            for field in ('source_keys', 'original_record_ids', 'topics'):
                existing[field] = sorted(set(existing[field] + row[field]))
            continue
        seen[key] = row
        result.append(row)
    return result


def assemble(base, core, tokenizer_path, output):
    if digest((base / 'train.parquet').read_bytes()) != BASE_SHA:
        raise ValueError('Unexpected all-13-book baseline')
    if digest(tokenizer_path.read_bytes()) != TOKENIZER_SHA:
        raise ValueError('Unexpected tokenizer')
    core_summary = build(core, tokenizer_path, version='llin-knowledge-core-v2')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    baseline = pq.read_table(base / 'train.parquet').to_pylist()
    catalog = read_jsonl(base / 'sources.jsonl')
    expected_sources = {r['source_id'] for r in catalog}
    if len(expected_sources) != 13 or {r['source_id'] for r in baseline} != expected_sources:
        raise ValueError('All 13 source families must be represented')
    supplement = read_jsonl(core / 'train.jsonl')
    rows = [checked_row(r, tokenizer, 'original_13_books', ['book-source:'+r['source_id']]) for r in baseline]
    rows += [checked_row(r, tokenizer, 'reviewed_supplement', r['source_keys']) for r in supplement]
    unique = deduplicate(rows)
    retained = {k.removeprefix('book-source:') for r in unique for k in r['source_keys'] if k.startswith('book-source:')}
    if retained != expected_sources:
        raise ValueError('Source lost during deduplication')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in unique), encoding='utf-8', newline='\n')
    pq.write_table(pa.Table.from_pylist(unique), output / 'train.parquet')
    if pq.read_table(output / 'train.parquet').to_pylist() != unique:
        raise ValueError('Parquet round-trip failure')
    # Keep block/page evidence separate from the trainer-facing text column.
    provenance = [dict(id=r['id'], source_id=r['source_id'], source_file=r['source_file'],
                       source_pages=r['source_pages'], block_ids=r['block_ids']) for r in baseline]
    (output / 'baseline_provenance.private.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in provenance), encoding='utf-8', newline='\n')
    baseline_text = '\n'.join(r['text'] for r in baseline)
    excerpt_overlap = sum(u['body'] in baseline_text for u in read_jsonl(core / 'reviewed_units.private.jsonl'))
    summary = dict(version='llin-knowledge-complete-v1', date='2026-09-11',
                   baseline_records=len(baseline), supplement_records=len(supplement),
                   records=len(unique), exact_duplicates_removed=len(rows)-len(unique),
                   original_source_families=len(retained),
                   content_tokens=sum(r['content_tokens'] for r in unique),
                   sequence_tokens=sum(r['token_count'] for r in unique),
                   maximum_sequence_tokens=max(r['token_count'] for r in unique),
                   supplement_content_tokens=core_summary['content_tokens'],
                   reviewed_topics=core_summary['topics_with_selected_units'],
                   supplemental_bodies_already_in_baseline=excerpt_overlap,
                   overlap_policy='Retain title/scope/evidence-restored units; no repeated sampling. Exact full-text dedup only; not semantic dedup.',
                   benchmark_labels_imported_by_builder=False,
                   benchmark_same_source_exclusion=False,
                   training_started=False, all_300_cases_resolved=False,
                   readiness='Local data contract validated; factual review applies to selected supplement assertions, not every baseline sentence.',
                   inputs={'baseline_parquet_sha256':BASE_SHA, 'tokenizer_sha256':TOKENIZER_SHA,
                           'core_parquet_sha256':digest((core/'train.parquet').read_bytes())},
                   checks=['13 source families retained', 'full retokenization without truncation',
                           'single EOS appended by text loader', 'provenance retained', 'Parquet round trip',
                           'reviewed-core evidence gate'])
    summary['files'] = {name:digest((output/name).read_bytes()) for name in
                        ('baseline_provenance.private.jsonl', 'train.jsonl', 'train.parquet')}
    write_json(output / 'verification.safe.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('base', 'core', 'tokenizer', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.base, args.core, args.tokenizer, args.output), indent=2))

"""Build a small, explicitly reviewed CPT supplement; never auto-approve candidates."""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer

TOKENIZER_SHA = '06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523'
EOS = 248046


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def validate_unit(unit, sources, tokenizer):
    if unit.get('review_status') != 'reviewed_selected_assertions':
        raise ValueError('Unreviewed unit')
    if not unit.get('title') or not unit.get('scope') or not unit.get('evidence') or not unit.get('topics'):
        raise ValueError('Missing title, scope, evidence or topic')
    for evidence in unit['evidence']:
        source = sources[evidence['key']]
        if digest(source['text'].encode('utf-8')) != source['sha256']:
            raise ValueError('Source text does not match its fingerprint')
        if source['sha256'] != evidence['sha256']:
            raise ValueError('Source fingerprint changed')
        if 'excerpt' in evidence and evidence['excerpt'] not in source['text']:
            raise ValueError('Evidence span absent from source')
        if unit['kind'] == 'source_excerpt' and evidence.get('excerpt') != unit['body']:
            raise ValueError('Excerpt changed without editorial review')
    text = unit['title'] + '\n\n' + unit['scope'] + '\n\n' + unit['body']
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if not ids or len(ids) + 1 > 4096 or EOS in ids:
        raise ValueError('Invalid content length or embedded EOS; do not truncate knowledge units')
    if any(x in text for x in ['<|im_start|>', '<|im_end|>', '\ufffd']):
        raise ValueError('Chat or damaged-text marker in CPT prose')
    return text, ids + [EOS]


def build(root, tokenizer_path):
    if digest(tokenizer_path.read_bytes()) != TOKENIZER_SHA:
        raise ValueError('Unexpected tokenizer')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    sources = {x['key']: x for x in read_jsonl(root / 'sources.private.jsonl')}
    units = read_jsonl(root / 'reviewed_units.private.jsonl')
    rows, seen = [], set()
    for unit in units:
        text, ids = validate_unit(unit, sources, tokenizer)
        sha = digest(text.encode('utf-8'))
        if sha in seen:
            raise ValueError('Duplicate reviewed unit; resolve provenance before building')
        seen.add(sha)
        rows.append(dict(id=unit['id'], text=text, text_sha256=sha, input_ids=ids,
                         content_tokens=len(ids)-1, token_count=len(ids), eos_token_id=EOS,
                         tokenizer_sha256=TOKENIZER_SHA, topics=unit['topics'],
                         training_ready=True, source_keys=[e['key'] for e in unit['evidence']]))
    (root / 'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8', newline='\n')
    pq.write_table(pa.Table.from_pylist(rows), root / 'train.parquet')
    assert pq.read_table(root / 'train.parquet').to_pylist() == rows
    md = ['# 已核对的CPT核心补充语料', '', '本文件只包含逐条选定的断言。来源全文没有因此获得整体审核通过；主题命中不代表错题已解决。', '']
    for unit, row in zip(units, rows):
        refs = [sources[e['key']] for e in unit['evidence']]
        md.extend(['## '+unit['title'], '', unit['scope'], '', unit['body'], '',
                   '来源：'+'；'.join(s['label']+' '+s.get('url','') for s in refs), '',
                   '形式：'+unit['kind']+'；'+unit['review_note'], ''])
    (root / '核心知识文档.md').write_text('\n'.join(md), encoding='utf-8', newline='\n')
    summary = dict(version='llin-knowledge-core-v1', records=len(rows),
                   content_tokens=sum(r['content_tokens'] for r in rows),
                   sequence_tokens=sum(r['token_count'] for r in rows),
                   max_sequence_tokens=max(r['token_count'] for r in rows),
                   topics_with_selected_units=len({t for r in rows for t in r['topics']}),
                   source_records=len(sources), tokenizer_sha256=TOKENIZER_SHA,
                   training_started=False, scope='reviewed supplement, not exhaustive 300-case coverage',
                   checks=['explicit assertion review', 'source fingerprints', 'exact excerpt membership',
                           'title and scope retained', 'unique prose', '4096 token limit without truncation',
                           'single terminal EOS', 'JSONL/Parquet round trip'])
    summary['files'] = {p.name: digest(p.read_bytes()) for p in sorted(root.iterdir()) if p.is_file() and p.name != 'verification.safe.json'}
    write_json(root / 'verification.safe.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.tokenizer), ensure_ascii=False, indent=2))

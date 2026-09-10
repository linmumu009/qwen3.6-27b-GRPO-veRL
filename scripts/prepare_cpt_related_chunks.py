"""Reorganize only the authorized book; no benchmark inputs or generated facts."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import os

from scripts.prepare_logistics_book_cpt import build_blocks, chunk_blocks

SOURCE_SHA = 'b2b4ed0156dea4076f86894478c8db2643ee05f25fb8af6bdf4622ee70b93bd1'
TOKENIZER_SHA = '06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523'
STOP = set('the and for that with this from are can have will which their these they not also into more such other used use may than has been all but when some any each its those most there should would must within between over under only both one two three often many about through out per very well however based different example figure table chapter introduction'.split())
STOP.update('jan feb mar apr jun jul aug sep oct nov dec january february march april june july august september october november december nos'.split())


def vectors(texts):
    counts = [Counter(w for w in re.findall(r'[a-z]{3,}', t.lower()) if w not in STOP) for t in texts]
    df = Counter(w for c in counts for w in c)
    result = []
    for count in counts:
        weighted = {w: (1 + math.log(n)) * math.log(1 + len(texts) / df[w]) for w, n in count.items()}
        norm = math.sqrt(sum(v*v for v in weighted.values()))
        result.append({w: v/norm for w, v in weighted.items()} if norm else {})
    return result


def cosine(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(value*b.get(word, 0) for word, value in a.items())


def render(units, indices):
    return '\n\n'.join(f'[Chapter {units[i]["chapter"]}]\n{units[i]["text"]}' for i in indices)


def pack_related(units, token_count, groups=116, limit=4095):
    if len(units) < groups:
        raise ValueError('Not enough intact units for matched sample count')
    lengths = [token_count(render(units, [i])) for i in range(len(units))]
    if max(lengths) > limit:
        raise ValueError('An intact source unit exceeds the context limit')
    features = vectors([u['text'] for u in units])
    order = sorted(range(len(units)), key=lambda i: (-lengths[i], i))
    bins = [[i] for i in order[:groups]]
    sizes = [lengths[i[0]] for i in bins]
    for index in order[groups:]:
        candidates = []
        for g, members in enumerate(bins):
            if sizes[g] + lengths[index] + 3 > limit:
                continue
            cross = [j for j in members if units[j]['chapter'] != units[index]['chapter']]
            score = max((cosine(features[index], features[j]) for j in cross), default=0)
            same = [j for j in members if units[j]['chapter'] == units[index]['chapter']]
            # Weak lexical matches should retain natural same-chapter context when possible.
            priority = 2 if score >= .10 else (1 if same else 0)
            relevance = score if priority == 2 else (-min(abs(index-j) for j in same) if same else score)
            candidates.append((priority, relevance, -sizes[g], -g))
        if not candidates:
            raise ValueError('Cannot fit complete source units within matched sample budget')
        for _, _, _, neg_group in sorted(candidates, reverse=True):
            group = -neg_group
            trial = bins[group] + [index]
            size = token_count(render(units, trial))
            if size <= limit:
                bins[group] = trial
                sizes[group] = size
                break
        else:
            raise ValueError('Exact tokenization exceeds matched context budget')
    if Counter(i for group in bins for i in group) != Counter(range(len(units))):
        raise ValueError('Source coverage or duplication changed')
    return bins, features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    os.umask(0o077)
    if a.out.exists():
        raise FileExistsError('Immutable preparation output already exists')
    assert hashlib.sha256(a.source.read_bytes()).hexdigest() == SOURCE_SHA
    assert hashlib.sha256((a.model/'tokenizer.json').read_bytes()).hexdigest() == TOKENIZER_SHA
    from transformers import AutoTokenizer
    import pandas as pd
    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    blocks, cleaning = build_blocks(a.source.read_text(encoding='utf-8'))
    old = chunk_blocks(blocks, tokenizer, 4095)
    original = pd.read_parquet(a.original)
    assert [r['text'] for r in old] == original['text'].tolist(), 'Original book reconstruction mismatch'
    # Preserve intact paragraphs; combine consecutive same-chapter paragraphs to ~1600 tokens.
    units, pending = [], []
    def flush():
        if pending:
            indices = list(pending)
            units.append(dict(chapter=blocks[indices[0]].chapter, block_indices=indices,
                              source_line_start=blocks[indices[0]].source_line_start,
                              source_line_end=blocks[indices[-1]].source_line_end,
                              text='\n\n'.join(blocks[i].text for i in indices)))
            pending.clear()
    for i, block in enumerate(blocks):
        if pending and (block.chapter != blocks[pending[0]].chapter or
                        count('\n\n'.join([*(blocks[j].text for j in pending), block.text])) > 1600):
            flush()
        pending.append(i)
    flush()
    bins, features = pack_related(units, count)
    rows, pairs = [], []
    for index, group in enumerate(bins):
        text = render(units, group)
        rows.append(dict(record_id=f'related-{index:05d}', text=text, token_count=count(text),
                         unit_ids=group, text_sha256=hashlib.sha256(text.encode()).hexdigest()))
        cross = [(cosine(features[i], features[j]), i, j) for pos, i in enumerate(group)
                 for j in group[pos+1:] if units[i]['chapter'] != units[j]['chapter']]
        if cross:
            score, i, j = max(cross)
            common = sorted(features[i].keys() & features[j].keys(),
                            key=lambda w: -(features[i][w]*features[j][w]))[:12]
            pairs.append(dict(group=index, unit_a=i, unit_b=j, cosine=score, shared_terms=common))
    original_tokens = sum(count(r['text']) for r in old)
    tokens = sum(r['token_count'] for r in rows)
    coverage = Counter(i for group in bins for u in group for i in units[u]['block_indices'])
    assert coverage == Counter(range(len(blocks)))
    budget_ok = abs(tokens-original_tokens)/original_tokens <= .01
    mechanism_ok = sum(p['cosine'] >= .10 for p in pairs) >= 58
    a.out.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(a.out/'related.private.parquet', index=False)
    (a.out/'units.private.json').write_text(json.dumps(units), encoding='utf-8')
    (a.out/'pairs.private.json').write_text(json.dumps(pairs), encoding='utf-8')
    # Stratified fixed audit of weaker, middle, and stronger links, not outcome-selected.
    ordered = sorted(pairs, key=lambda r: (r['cosine'], r['group']))
    selected = [ordered[round(i*(len(ordered)-1)/11)] for i in range(12)] if len(ordered) >= 12 else ordered
    (a.out/'review.private.json').write_text(json.dumps([dict(p, a=units[p['unit_a']], b=units[p['unit_b']]) for p in selected]), encoding='utf-8')
    summary = dict(source_sha256=SOURCE_SHA, tokenizer_sha256=TOKENIZER_SHA, blocks=len(blocks),
        units=len(units), records=len(rows), original_content_tokens=original_tokens,
        related_content_tokens=tokens, original_sequence_tokens=original_tokens+116,
        related_sequence_tokens=tokens+116, relative_token_difference=(tokens-original_tokens)/original_tokens,
        min_tokens=min(r['token_count'] for r in rows), max_tokens=max(r['token_count'] for r in rows),
        cross_chapter_groups=len(pairs), cross_chapter_cosine_at_least_point1=sum(p['cosine']>=.10 for p in pairs),
        complete_unique_block_coverage=True, original_reconstruction_exact=True,
        paragraph_text_edited=False, additional_text='Chapter markers and separators only',
        budget_gate=budget_ok, lexical_mechanism_gate=mechanism_ok, semantic_review='pending',
        attention_audit='pending', training_allowed=False, api_calls=0,
        parquet_sha256=hashlib.sha256((a.out/'related.private.parquet').read_bytes()).hexdigest())
    (a.out/'summary.safe.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()

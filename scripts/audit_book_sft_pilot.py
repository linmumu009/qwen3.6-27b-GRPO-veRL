"""Local lexical quarantine only; never sends benchmark material to an API."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from build_book_sft_api_pilot import words


def grams(text, n=12):
    seq = words(text)
    return set(tuple(seq[i:i+n]) for i in range(len(seq)-n+1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--benchmark', type=Path, required=True)
    a = p.parse_args()
    os.umask(0o077)
    raw = a.benchmark.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != 'b652b2108cb552346df11d005c15ff3137c50a756a7b24eb35302683ec33ed99':
        raise ValueError('Unexpected frozen benchmark hash')
    bench = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    exact = set()
    index = set()
    for item in bench:
        texts = [item['question']] + [x for x in item['options'] if isinstance(x, str)]
        exact.add(tuple(words(item['question'])))
        for text in texts:
            index.update(grams(text))
    rows = [json.loads(line) for line in (a.pilot / 'candidates.private.jsonl').read_text().splitlines()]
    issues = Counter()
    flags = []
    for row in rows:
        issues.update(row.get('errors', []))
        if row['status'] != 'auto_pass':
            continue
        qa = row['qa']
        if tuple(words(qa['question'])) in exact or (grams(qa['question']) | grams(qa['answer'])) & index:
            flags.append(row['id'])
    (a.pilot / 'overlap_flags.private.json').write_text(json.dumps(flags))
    accepted = [row for row in rows if row['status'] == 'auto_pass' and row['id'] not in flags]
    with (a.pilot / 'review_queue.private.jsonl').open('x') as f:
        for row in accepted:
            f.write(json.dumps({'id': row['id'], 'source_id': row['source_id'], 'chapter': row['chapter'],
                                'kind': row['kind'], **row['qa']}) + '\n')
    summary = json.loads((a.pilot / 'summary.safe.json').read_text())
    summary.update(benchmark_overlap_checked=True, benchmark_sha256=digest,
        benchmark_rows=len(bench), overlap_quarantined=len(flags), review_queue_count=len(accepted),
        structural_issue_counts=dict(issues), lexical_rule='Normalized exact question or shared 12-word span with benchmark question/options.',
        overlap_limitation='Lexical screen only; neither semantic decontamination proof nor independent test-set status.',
        no_benchmark_content_sent_to_api=True)
    (a.pilot / 'audit.safe.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

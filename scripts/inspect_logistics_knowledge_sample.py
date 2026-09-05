"""Display a bounded, deterministic private sample for substantive agent review."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--candidates', type=Path, required=True)
    p.add_argument('--count', type=int, default=8)
    args = p.parse_args()
    if not 1 <= args.count <= 16:
        raise ValueError('review sample must contain 1 to 16 items')
    source = {r['item_hash']:r for r in map(json.loads,args.source.read_text().splitlines())}
    rows = [r for r in map(json.loads,args.candidates.read_text().splitlines()) if r['automatic_entailment_passed']]
    selected = []
    for flag in ('true_false','negative_stem','option_dependency','matching_or_large_pool','ordinary'):
        selected.extend(next(([r] for r in rows if flag in r['risk_flags'] and r not in selected), []))
    selected.extend(r for r in rows if r not in selected)
    for r in selected[:args.count]:
        s = source[r['item_hash']]
        print(json.dumps({'item_hash':r['item_hash'],'flags':r['risk_flags'],'question':s['question'],
            'gold_texts':[s['options'][i] for i in s['expected']],
            'candidate':r['candidate']['text']},ensure_ascii=False))


if __name__ == '__main__':
    main()

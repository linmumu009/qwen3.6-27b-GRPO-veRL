"""Freeze matched per-update token budgets using lengths only, without changing data."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random


def match_batches(lengths_a, lengths_b, *, seed, batch_size=4, iterations=160000):
    if len(lengths_a) != len(lengths_b) or len(lengths_a) % batch_size:
        raise ValueError('Equal complete batches are required')
    if min([*lengths_a, *lengths_b]) < 1:
        raise ValueError('Positive supervised lengths required')
    ratio = sum(lengths_b)/sum(lengths_a)
    if abs(ratio-1) > .01:
        raise ValueError('Total token budget differs by more than 1%')
    rng = random.Random(seed)
    n = len(lengths_a)
    permutations = [list(range(n)), list(range(n))]
    for p in permutations:
        rng.shuffle(p)
    lengths = [lengths_a, lengths_b]
    sums = [[sum(values[i] for i in p[j:j+batch_size]) for j in range(0, n, batch_size)]
            for values, p in zip(lengths, permutations)]
    residuals = [b-ratio*a for a, b in zip(*sums)]
    best = sum(d*d for d in residuals)
    best_permutations = [p.copy() for p in permutations]
    for step in range(iterations):
        arm = rng.randrange(2)
        x, y = rng.sample(range(n), 2)
        gx, gy = x//batch_size, y//batch_size
        if gx == gy:
            continue
        p, values = permutations[arm], lengths[arm]
        delta = values[p[y]] - values[p[x]]
        dr = delta * (1 if arm else -ratio)
        rx, ry = residuals[gx], residuals[gy]
        change = (rx+dr)**2 + (ry-dr)**2 - rx**2 - ry**2
        temperature = 20000 * (1-step/iterations)**4 + .05
        if change <= 0 or rng.random() < math.exp(-change/temperature):
            p[x], p[y] = p[y], p[x]
            sums[arm][gx] += delta
            sums[arm][gy] -= delta
            residuals[gx] += dr
            residuals[gy] -= dr
            loss = sum(d*d for d in residuals)
            if loss < best:
                best = loss
                best_permutations = [v.copy() for v in permutations]
    budgets = [[sum(v[i] for i in p[j:j+batch_size]) for j in range(0, n, batch_size)]
               for v, p in zip(lengths, best_permutations)]
    differences = [(b-a)/a for a, b in zip(*budgets)]
    if max(map(abs, differences)) > .01:
        raise ValueError(f'Per-step matching failed: max difference {max(map(abs, differences)):.6f}')
    for p in best_permutations:
        assert sorted(p) == list(range(n)), 'Dropped or repeated sample'
    return dict(original=best_permutations[0], related=best_permutations[1],
                original_supervised_tokens=budgets[0], related_supervised_tokens=budgets[1],
                max_relative_difference=max(map(abs, differences)), seed=seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--related', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    import pandas as pd
    paths = {'original': a.original, 'related': a.related}
    lengths = {k: pd.read_parquet(v)['token_count'].astype(int).tolist() for k, v in paths.items()}
    assert len(lengths['original']) == len(lengths['related']) == 116
    epochs = [match_batches(lengths['original'], lengths['related'], seed=20260910+epoch)
              for epoch in range(4)]
    result = dict(schema_version=1, epochs=epochs, records=116, batch_size=4,
                  total_epochs=4, steps_per_epoch=29, source_content_included=False,
                  lengths=lengths, dataset_sha256={k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in paths.items()},
                  rule='Length-only deterministic paired permutations; all samples once per epoch in both arms',
                  max_relative_step_difference=max(e['max_relative_difference'] for e in epochs))
    with a.out.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ('epochs', 'lengths')}))


if __name__ == '__main__':
    main()

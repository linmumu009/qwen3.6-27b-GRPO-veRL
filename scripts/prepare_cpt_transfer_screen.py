"""Freeze source-only candidates while preserving the historical development split."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

SOURCE_SHA = '5df7dc2b209938986cd7081aecdc9105fb989c9a872d572ce0a0c5a16dbbf8a4'
SELECTION = {
    'rail_infrastructure_and_vehicles': [50, 51, 52, 131, 132, 133, 113, 114],
    'rail_safety': [102, 147, 148, 149, 153, 155, 157, 161],
    'road_vehicles': [59, 61, 63, 64, 116, 118, 119, 164],
    'waterway_vessels_and_clearance': [66, 68, 120, 123, 124, 125, 129, 199],
    'pipeline': [97, 98, 99, 100, 101, 106, 128],
    'warehouse_operations': [73, 77, 78, 166, 168, 169, 178, 196],
    'planning_and_quantitative_rules': [3, 9, 11, 89, 91, 92, 192, 181],
    'supply_chain_management': [79, 108, 171, 174, 176, 179, 194, 197, 202],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def historical_groups(rows, script):
    # The original module imports Linux-only fcntl. Execute its pure grouping
    # function unchanged on Windows, instead of implementing a different split.
    tree = ast.parse(Path(script).read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'group_sources')
    ns = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(script), 'exec'), ns)
    return ns['group_sources'](rows)


def prepare(source, group_script, out):
    if sha(source) != SOURCE_SHA:
        raise ValueError('historical source snapshot changed')
    rows = [json.loads(x) for x in Path(source).read_text(encoding='utf-8').splitlines()]
    groups = historical_groups(rows, group_script)
    dev_groups = {g for g in groups.values() if int(hashlib.sha256(g.encode()).hexdigest(), 16) % 5 == 0}
    dev_topics = {t for r in rows if groups[r['id']] in dev_groups for t in r['topics']}
    selected = []
    for category, indices in SELECTION.items():
        for index in indices:
            r = rows[index]
            if groups[r['id']] in dev_groups or set(r['topics']) & dev_topics:
                raise ValueError('frozen development source/topic selected')
            selected.append({k: r[k] for k in ('id', 'title', 'scope', 'source_text', 'evidence', 'topics')} | {
                'group': groups[r['id']], 'category': category,
                'historical_split': 'train', 'purpose': 'screen_only', 'training_allowed': False,
            })
    assert len(selected) == len({r['id'] for r in selected}) == 64
    counts = Counter(r['category'] for r in selected)
    assert max(counts.values()) <= len(selected) / 4
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    packet = out / 'sources.private.jsonl'
    packet.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected), encoding='utf-8', newline='\n')
    manifest = dict(
        source_sha256=SOURCE_SHA, packet_sha256=sha(packet), group_script_sha256=sha(group_script),
        candidate_source_units=len(selected), historical_source_groups=len({r['group'] for r in selected}),
        categories=dict(counts), frozen_dev_groups=len(dev_groups), frozen_dev_topics=sorted(dev_topics),
        requested_probes_per_source=3, requested_probes=192,
        source_fields_only=True, official_qa_inputs=False, training_ready=False,
        source_ids=[r['id'] for r in selected],
        limitations='Candidate source units are not yet independent validated gap units. Same-model generation/review is only a filter. Manual semantic and numerical review remains required. Screening probes are never training, transfer-dev, or sealed confirmation items.',
    )
    (out / 'manifest.safe.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--group-script', type=Path, default=Path('scripts/generate_source_condition_tasks.py'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.group_script, a.out), indent=2))

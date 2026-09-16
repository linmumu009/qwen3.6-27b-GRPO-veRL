"""Freeze previously unscreened source excerpts; never load official questions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from prepare_cpt_transfer_screen import SOURCE_SHA, historical_groups, sha

SELECTION = {
    'warehouse_operations': [74, 75, 76, 167, 170, 12],
    'planning_and_quantitative_rules': [2, 90, 203, 204],
    'supply_chain_management': [27, 88, 94, 95],
    'enterprise_and_warehouse_systems': [6, 29, 40, 42],
    'waterway_vessels_and_clearance': [180, 200, 126],
    'rail_infrastructure_and_safety': [134, 151, 159],
}


def prepare(source, previous, group_script, out):
    assert sha(source) == SOURCE_SHA, 'historical source snapshot changed'
    rows = [json.loads(x) for x in source.read_text(encoding='utf-8').splitlines()]
    old = [json.loads(x) for x in previous.read_text(encoding='utf-8').splitlines()]
    old_ids = {x['id'] for x in old}
    old_text = {x['source_text'] for x in old}
    groups = historical_groups(rows, group_script)
    dev_groups = {g for g in groups.values() if int(hashlib.sha256(g.encode()).hexdigest(), 16) % 5 == 0}
    dev_topics = {t for x in rows if groups[x['id']] in dev_groups for t in x['topics']}
    selected = []
    for category, indices in SELECTION.items():
        for index in indices:
            r = rows[index]
            assert r['id'] not in old_ids and r['source_text'] not in old_text
            assert groups[r['id']] not in dev_groups and not set(r['topics']) & dev_topics
            selected.append({k: r[k] for k in ('id', 'title', 'scope', 'source_text', 'evidence', 'topics')} | {
                'group': groups[r['id']], 'category': category,
                'historical_split': 'train', 'purpose': 'screen_only', 'training_allowed': False,
            })
    assert len(selected) == len({x['id'] for x in selected}) == 24
    counts = Counter(x['category'] for x in selected)
    assert max(counts.values()) <= len(selected) / 4
    out.mkdir(parents=True, exist_ok=False)
    packet = out / 'sources.private.jsonl'
    packet.write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in selected), encoding='utf-8', newline='\n')
    manifest = dict(source_sha256=SOURCE_SHA, previous_packet_sha256=sha(previous),
        packet_sha256=sha(packet), source_units=24, categories=dict(counts),
        historical_source_groups=len({x['group'] for x in selected}),
        frozen_dev_groups=len(dev_groups), frozen_dev_topics=sorted(dev_topics),
        official_qa_inputs=False, training_ready=False, requested_probes=72,
        limitations='Unscreened excerpts, not necessarily new independent rules. Forecast and customs excerpts overlap conceptually and require consolidation after semantic review. Short definitions may not support three forms; skips are retained. Historical scopes remain fixed. All probes excluded from later training/dev/sealed sets.')
    (out/'manifest.safe.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--group-script', type=Path, default=Path('scripts/generate_source_condition_tasks.py'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.previous, a.group_script, a.out), indent=2))

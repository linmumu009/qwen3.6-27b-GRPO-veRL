"""Build source-only draft requests without reading any diagnostic question or answer."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from prepare_cpt_transfer_screen import SOURCE_SHA, historical_groups, sha

DESIGNS = {
    'train': ['single_case_application', 'boundary_pair', 'explicit_exclusion',
              'correct_a_report', 'compare_two_actions', 'conditional_counterexample'],
    'dev': ['dev_expression_ledger', 'dev_conditions_reverse_inference', 'dev_conditions_minimal_change'],
    'sealed': ['sealed_two_stage_decision'],
    'retention': ['retention_application', 'retention_boundary', 'retention_decision'],
}


def prepare(source, registry, group_script, out):
    assert sha(source) == SOURCE_SHA
    rows = [json.loads(s) for s in source.read_text(encoding='utf-8').splitlines()]
    groups = historical_groups(rows, group_script)
    dev_groups = {g for g in groups.values() if int(hashlib.sha256(g.encode()).hexdigest(),16)%5 == 0}
    dev_topics = {t for r in rows if groups[r['id']] in dev_groups for t in r['topics']}
    by_id = {r['id']: r for r in rows}
    # The registry's question IDs, baseline outcomes and diagnoses are deliberately
    # discarded. Only source membership and a broad category cross this boundary.
    metadata = json.loads(registry.read_text(encoding='utf-8'))
    families = [{'source_ids': f['source_ids'], 'category': f['category']} for f in metadata['families']]
    if not any('llin-core-4abc2a4cad028703b936' in f['source_ids'] for f in families):
        families.append({'source_ids':['llin-core-4abc2a4cad028703b936'], 'category':'warehouse'})
    used = {sid for f in families for sid in f['source_ids']}
    target_topics = {t for sid in used for t in by_id[sid]['topics']}
    def clean(sid):
        r=by_id[sid]
        assert groups[sid] not in dev_groups and not set(r['topics']) & dev_topics
        return {k:r[k] for k in ('id','title','scope','source_text','evidence','topics')} | {'group':groups[sid]}
    requests=[]
    for i,f in enumerate(families):
        for split in ('train','dev','sealed'):
            requests.append(dict(id=f'unit-{i+1:03d}-{split}',unit=f'unit-{i+1:03d}',split=split,
                category=f['category'],designs=DESIGNS[split],sources=[clean(s) for s in f['source_ids']],
                training_allowed=False,purpose='unreviewed_draft'))
    retention=[r for r in rows if r['id'] not in used and groups[r['id']] not in dev_groups
               and not set(r['topics']) & (dev_topics|target_topics)]
    # Collapse exact source-body repeats. Translations and semantic duplicates
    # still require consolidation before these become retention evaluation units.
    seen=set()
    retention=[r for r in retention if not (r['source_text'] in seen or seen.add(r['source_text']))]
    for i,r in enumerate(retention):
        requests.append(dict(id=f'retention-{i+1:03d}',unit=f'retention-{i+1:03d}',split='retention',
            category='retention_'+r['topics'][0],designs=DESIGNS['retention'],sources=[clean(r['id'])],
            training_allowed=False,purpose='unreviewed_draft'))
    out.mkdir(parents=True,exist_ok=False)
    packet=out/'requests.private.jsonl'
    packet.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in requests),encoding='utf-8',newline='\n')
    counts=Counter()
    for r in requests:counts[r['split']]+=len(r['designs'])
    manifest=dict(packet_sha256=sha(packet),source_sha256=SOURCE_SHA,registry_sha256=sha(registry),
        generation_requests=len(requests),requested_tasks=dict(counts),candidate_units=len(families),
        categories=dict(Counter(f['category'] for f in families)),retention_source_units=len(retention),
        source_groups=len({s['group'] for r in requests for s in r['sources']}),
        frozen_dev_groups=len(dev_groups),frozen_dev_topics=sorted(dev_topics),
        generator_inputs='source fields and abstract task-design assignments only',
        diagnostic_questions_in_generator=False,official_qa_in_generator=False,
        training_ready=False,training_allowed=False,
        limitations='Requested counts are not produced or accepted counts. Check the current candidate count, category distribution and registered 30-50 rule gate separately. Generation and same-model blind review only produce drafts. Every item, cross-split reasoning structure and arithmetic still require independent audit. Retention excludes target topics but shared historical source groups and broad semantic relations remain possible. No output is a training release or certified independent holdout.')
    (out/'manifest.safe.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True);p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--group-script',type=Path,default=Path('scripts/generate_source_condition_tasks.py'))
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.source,a.registry,a.group_script,a.out),indent=2))

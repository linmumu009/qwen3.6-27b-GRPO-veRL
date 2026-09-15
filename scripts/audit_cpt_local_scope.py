"""Audit registered knowledge scope; never exports official question text."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

def audit(root):
    files = {
        'cases': root/'llin-knowledge-complete-20260911/case_disposition.private.jsonl',
        'local_tasks': root/'llin-s5-contrast-20260915-01/tasks.private.jsonl',
        'reviewed': root/'llin-source-reviewed-20260914-01/reviewed_tasks.private.jsonl',
    }
    rows = {k: [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()] for k,p in files.items()}
    cases, tasks, reviewed = (rows[k] for k in ('cases','local_tasks','reviewed'))
    assert len(cases)==300 and len({x['item_hash'] for x in cases})==300
    groups={x['group'] for x in tasks}
    linked=lambda c: set(c['selected_assertion_units']+c['topic_related_units'])
    matched=[c for c in cases if linked(c)&groups]
    by_dataset=collections.Counter(c['dataset'] for c in matched)
    train={x['unit_id'] for x in reviewed if x['split']=='train'}
    dev={x['unit_id'] for x in reviewed if x['split']=='dev'}
    assert not train&dev
    broad=[c for c in cases if linked(c)&train]
    return {
        'hypothesis':'The four local S5 families cover enough registered error knowledge to account directly for both +3pp targets.',
        'method':'Join existing case-to-unit registration to local task source group; no keyword guessing or official-label training.',
        'cases':len(cases), 'local_unique_tasks':len(tasks), 'local_source_groups':len(groups),
        'local_families':dict(collections.Counter(t['family'] for t in tasks)),
        'local_registered_case_links':dict(by_dataset),
        'optimistic_direct_link_gain_pp':{name:100*by_dataset[name]/n for name,n in [('SC-bench-knowledge',226),('LogistikaBench',1446)]},
        'needed_net_correct_gain':{'SC-bench-knowledge':7,'LogistikaBench':44},
        'existing_broad_training_unit_links':dict(collections.Counter(c['dataset'] for c in broad)),
        'error_topic_counts':dict(collections.Counter(t for c in cases for t in c['topics'])),
        'case_status_counts':dict(collections.Counter(c['status'] for c in cases)),
        'finding':'Direct registered scope is too narrow to justify another four-family-only training run as the main +3pp strategy.',
        'limitations':['Links are historical partial knowledge registrations, not exhaustive semantic coverage or a hard bound on indirect transfer.', 'The 300 errors are the frozen historical cohort, not the latest S5 error set.', 'Baseline retention can change on other items; optimistic direct-link gain is not a performance prediction.'],
        'next_test':'Freeze a broader source-only transfer probe across multiple existing non-dev training topics, with novel applications and explanation-vs-answer-only paired inference. Review source support before running; exclude official QA/rewrites and frozen dev. Do not repeat the completed 22-case S5 diagnostic.',
        'training_running':False,
        'input_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in files.items()},
    }

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=audit(a.root)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['local_registered_case_links','optimistic_direct_link_gain_pp','existing_broad_training_unit_links']},ensure_ascii=False))

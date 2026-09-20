"""Reproduce source follow-up without mutating the initial Stage A audit."""
from collections import Counter
import copy
import json
from pathlib import Path
import zipfile

from audit_cpt_stage_a import audit, read, write
from audit_cpt_transfer import require, sha

ROOT = Path('.')
PRIVATE = ROOT / 'CPT_resources/llin-stage-a-20260920-02'
SNAPSHOTS = [
    dict(id='primary_passages', path=str(PRIVATE / 'primary_passages.private.json').replace('\\', '/'),
         sha256='49e70c66c8f9ebb0e6e15945e2f71e27c84ad128c116d54786649e9d0e5f0a40', kind='web_reader_response'),
    dict(id='paper_context', path=str(PRIVATE / 'paper_context.private.json').replace('\\', '/'),
         sha256='db7fed8800d398eb8c41ef950028a094b1c0025f369c33563941801af9638ee6', kind='web_reader_response'),
]
RELEASE_SHA = '8667ad6a20b11e99e1a3db35ce27640715ca3f7fa15cb6de7517ca42a3d514d1'
MODAL = 'https://transportgeography.org/contents/chapter5/intermodal-transportation-containerization/intermodalism-multimodalism-transmodalism/'
UNCTAD = 'https://unctad.org/node/3989'


def run():
    old_path = Path('docs/cpt_stage_a_review_20260920.safe.json')
    review = copy.deepcopy(read(old_path))
    review['primary_source_snapshots'] = SNAPSHOTS
    review['parent_review_sha256'] = sha(old_path)
    changes = []
    for i in [147, 155, 199, 221, 222]:
        row = review['items'][i]
        require(row['review_index_zero_based'] == i and row['status'] == 'B', 'unexpected parent review')
        gvc = i in [147, 155]
        row.update(status='A', input_sufficiency='sufficient_in_reviewed_scope', label_uniqueness='supported_by_review',
                   official_url=UNCTAD if gvc else MODAL, primary_snapshot_id='primary_passages',
                   official_locator='WIR 2020 Key Messages, printed page xiii, trajectories (1)-(4)' if gvc else 'Definition paragraphs 1, 2 and 4',
                   primary_required_spans=['Diversification', 'Replication', 'rebundling'] if gvc else ['separate ticket', 'single ticket', 'same mode'],
                   note='已核验 UNCTAD 2020 原始报告摘要中的四轨迹比较；仅解释该历史分析框架，不声称预测已实现。' if gvc else '已核验原作者教材中的运输方式与合同维度定义；仅在该教材术语框架内放行，不外推为所有法规的通用定义。',
                   training_search_status='no_source_id_or_named_concept_match; paraphrased exposure remains unestablished')
        changes.append(dict(item_hash=row['item_hash'], review_index_zero_based=i, before='B', after='A',
                            capability_group=row['capability_group'], url=row['official_url'], locator=row['official_locator']))
    # Do not promote the ambiguous unspecific terminology case.
    require(review['items'][220]['status'] == 'D', 'ambiguity silently promoted')
    review_path = Path('docs/cpt_stage_a_followup_review_20260920.safe.json')
    write(review_path, review)
    result = audit(ROOT, review_path, PRIVATE)
    write(Path('docs/cpt_stage_a_followup_result_20260920.safe.json'), result)
    release = PRIVATE / 'sc_release.private.zip'
    require(sha(release) == RELEASE_SHA, 'downloaded release changed')
    tables, pointers = [], Counter()
    with zipfile.ZipFile(release) as z, zipfile.ZipFile('datasets/SC-bench-main.zip') as old:
        files = [n for n in z.namelist() if not n.endswith('/')]
        old_files = [n for n in old.namelist() if not n.endswith('/')]
        require(set(files) == set(old_files), 'SC release file set changed')
        require(all(z.read(n) == old.read(n) for n in files), 'SC release content changed')
        for name in files:
            if name.endswith('clean_final_clean.jsonl'):
                rows = [json.loads(s) for s in z.read(name).decode('utf-8').splitlines() if s.strip()]
                pointers.update(r['source_file'] for r in rows)
                fields = sorted({k for r in rows for k in r['output']})
                require(not {'context','document','source_text','background'} & set(fields), 'unexpected background field')
                tables.append(dict(member=name, records=len(rows), output_fields=fields))
        missing = [p for p in pointers if not any(n.endswith('/'+p) for n in files)]
        commit = z.comment.decode()
    require(sum(t['records'] for t in tables) == 226, 'SC question count mismatch')
    require(missing == ['desensitization_results.jsonl'], 'source pointer status changed')
    train = [json.loads(s) for s in Path('CPT_resources/llin-transfer-p4-data-20260918-01/train.messages.private.jsonl').read_text(encoding='utf-8').splitlines()]
    terms = ['transmodal','intermodal','multimodal','reshor','replication','diversification','regionalization']
    matches = [i for i,r in enumerate(train) if any(t in r['messages'][1]['content'].lower() for t in terms)]
    require(matches == [], 'named concept training matches changed')
    summary = dict(date='2026-09-20', parent_review_sha256=sha(old_path), changes=changes,
                   sc_release=dict(url='https://github.com/Damon-GSY/SC-bench', commit=commit, sha256=RELEASE_SHA,
                                   exact_member_match_with_local=True, files=len(files), tables=tables,
                                   source_pointers=dict(pointers), missing_members=missing,
                                   paper_version='arXiv:2602.07342v1', paper_qa_count=435,
                                   warning='Paper and public snapshot counts differ; this is not a validated complete reproduction of the paper.'),
                   datasets=result['datasets'], primary_source_snapshots=SNAPSHOTS,
                   training_check=dict(records=363, named_concept_match_rows=matches, exhaustive_semantic_absence_proven=False),
                   model_calls=0, training_runs=0, next_action='SC source acquisition; no two-arm training',
                   result_sha256=sha(Path('docs/cpt_stage_a_followup_result_20260920.safe.json')))
    write(Path('docs/cpt_stage_a_followup_summary_20260920.safe.json'), summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['changes','sc_release','primary_source_snapshots']}, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    run()

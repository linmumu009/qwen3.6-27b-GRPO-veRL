"""Apply explicit human-review hold; verify new replacements without leaking text."""
import argparse
import json
from pathlib import Path
from build_targeted_book_groups import digest
from repair_book_question_independence import valid_candidate

HUMAN_HOLDS={
    'taskgroup-002-discrimination':'Closed task still depends on source-specific operating policy; rubric requires reporting what the absent text does not state.',
    'taskgroup-003-discrimination':'Answer asserts absent ferrywagon design/cargo restrictions without supporting evidence.',
    'taskgroup-017-discrimination':'Still requires exactly two essential characteristics without defining the framework, despite explicit repair constraint.',
    'taskgroup-039-application':'Source example of three to six months becomes typical duration; rubric additionally requires an unasked historical-data point.',
    'taskgroup-052-definition':'Why-question rubric requires incidental numerical averages not requested and conflates sales turnover with cost of sales.',
    'taskgroup-072-discrimination':'Typical wall-supported and rail-mounted designs become essential dependencies; source itself mentions rubber-tyred gantries.',
}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    first=a.root/'runs/book-task-repair-20260908-01'
    hold=dict(status='superseded_human_review_hold',automatic_passes=1,usable_after_human_review=0,reason='Source possibility was strengthened into guaranteed instant complete identification; no old candidate may be promoted.',training_ready=False)
    (first/'human_review_hold.safe.json').write_text(json.dumps(hold,indent=2))
    out=a.root/'runs/book-task-repair-20260908-02'
    summary=json.loads((out/'summary.safe.json').read_text())
    if summary['status']=='repairing':raise ValueError('Still running')
    parents=json.loads((a.root/'runs/book-task-aligned-value-20260908-01/cases.private.json').read_text())
    original={q['id']:q for q in parents}
    candidates=[json.loads(x) for x in (out/'accepted_replacements.private.jsonl').read_text().splitlines()]
    seen=set()
    for q in candidates:
        assert q['id'] not in seen;seen.add(q['id']);old=original[q['parent_id']]
        assert all(q[k]==old[k] for k in ('split','chapter','source_id','source_hash','concept_group','kind','concept'))
        assert q['split']=='train' and q['target_probe_status']=='pending_new_scores_required'
        assert set(q['reference_units'])==set(q['source_ids'])
        assert valid_candidate(dict(q,reason='verified'),q['reference_units'])
    retained=[q for q in candidates if q['parent_id'] not in HUMAN_HOLDS]
    with (out/'reviewed_replacements.private.jsonl').open('w') as f:
        for q in retained:f.write(json.dumps(q)+'\n')
    value=dict(api_accepted_questions=len(candidates),human_review_held_questions=len(candidates)-len(retained),reviewed_candidate_questions=len(retained),reviewed_modes={m:sum(q['mode']==m for q in retained) for m in ('closed','evidence')},human_holds=HUMAN_HOLDS,duplicate_ids=0,split_changes=0,old_scores_inherited=False,training_ready=False,target_probe_completed=False,replacement_sha256=digest(out/'accepted_replacements.private.jsonl'),reviewed_replacement_sha256=digest(out/'reviewed_replacements.private.jsonl'),first_round_hold=hold,authoritative_candidate_file='reviewed_replacements.private.jsonl',draft_groups_not_promoted=True)
    (out/'integrity.safe.json').write_text(json.dumps(value,indent=2));print(json.dumps(value,indent=2))

if __name__=='__main__':main()

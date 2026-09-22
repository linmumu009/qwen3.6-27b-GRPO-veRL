"""Publish safe counts after frozen blind review and root source/overlap review."""
from collections import Counter
from pathlib import Path
import json
from cpt_logistika_independent import BASE, FORMS, compare, read, sha


def summarize(base=BASE):
    compare(base)
    result=read(base/'comparison.safe.json')
    review=read(base/'review.private.json')
    root=read(base/'root_group_review.private.json')
    overlap=read(base/'overlap.private.json')
    writer=read(base/'writer_input.private.json')
    author=read(base/'author.private.json')
    if root['reviewer_input_sha256']!=sha(base/'reviewer_input.private.json'):
        raise ValueError('Root review input mismatch')
    if root['author_answers_seen'] is not False or root['independent_reviewer_output_seen'] is not False:
        raise ValueError('Root pre-comparison review declaration missing')
    groups={g['id'] for g in writer['groups']}
    decisions={r['group']:r for r in root['records']}
    if len(decisions)!=len(root['records']) or set(decisions)!=groups:
        raise ValueError('Missing or repeated group decision')
    if len(overlap['records'])!=32 or {r['id'] for r in overlap['records']}!={r['id'] for r in result['records']}:
        raise ValueError('Overlap task scope mismatch')
    status=[]
    for group in sorted(groups):
        records=[r for r in result['records'] if r['group']==group]
        if sorted(r['form'] for r in records)!=sorted(FORMS):raise ValueError('Incomplete group')
        structural=all(r['evidence_agreement_pass'] for r in records)
        decision=decisions[group]
        status.append(dict(group=group,evidence_agreement_tasks=sum(r['evidence_agreement_pass'] for r in records),
            evidence_agreement_complete_group=structural,root_complete_group_released=decision['complete_group_released'],
            diagnostic_released=structural and decision['complete_group_released'],
            root_reason=decision['blocking_reason'],root_note=decision['note'],
            root_blocking_task_id=decision['task_id'],historical_reference_id=decision['reference_old_task_id']))
    released=sum(r['diagnostic_released'] for r in status)
    author_by_id={r['id']:r for r in author['records']}
    resolved_agreement=0
    unresolved=0
    option_judgments=0
    for record in review['records']:
        evidence=record['option_evidence']; option_judgments+=len(evidence)
        if any(e['relation']=='insufficient' for e in evidence):
            unresolved+=1
            continue
        reviewer_key=sorted(e['option_index'] for e in evidence if e['relation']=='entailed')
        author_key=sorted(e['option_index'] for e in author_by_id[record['id']]['option_evidence'] if e['relation']=='entailed')
        resolved_agreement+=reviewer_key==author_key
    # This closure summarizes zero-release results. A positive release requires its
    # separate input/protocol freeze and actual diagnostic execution, not a guessed count.
    if released or root['diagnostic_calls_released']!=0:
        raise ValueError('Positive release requires a diagnostic execution stage')
    fingerprints={name:sha(base/name) for name in (
        'registration.private.json','writer_input.private.json','author.private.json',
        'reviewer_input.private.json','review.private.json','review_freeze.private.json',
        'root_group_review.private.json','overlap_corpus.private.json','overlap.private.json','comparison.safe.json')}
    output=dict(date='2026-09-22',id=base.name,status='closed_no_complete_quality_group',
        author_method='History-free model subagent; source-only input by instruction',
        reviewer_method='Separate history-free model subagent; author evidence withheld until review freeze',
        human_expert_review=False,os_enforced_file_isolation=False,
        author_records=len(author['records']),author_skips=sum('skip_reason' in r for r in author['records']),
        option_judgments=option_judgments,resolved_label_agreement_tasks=resolved_agreement,
        tasks_with_insufficient_options=unresolved,
        blind_quality_pass_tasks=sum(r['quality_pass'] is True for r in review['records']),
        evidence_agreement_pass_tasks=sum(r['evidence_agreement_pass'] for r in result['records']),
        evidence_agreement_complete_groups=sum(r['evidence_agreement_complete_group'] for r in status),
        final_quality_complete_groups=released,
        root_group_rejection_counts=dict(Counter(r['root_reason'] for r in status)),groups=status,
        lexical=dict(files=len(overlap['inputs']),records=sum(r['extracted_records'] for r in overlap['inputs']),
            tasks_with_hits=sum(bool(r['hits']) for r in overlap['records']),threshold=overlap['threshold'],
            limitation=overlap['limitation']),fingerprints=fingerprints,
        actual_student_model_calls=0,actual_training_runs=0,actual_formal_eval_calls=0,
        draft_revisions=0,maximum_unused_diagnostic_calls=432,
        old_results_preserved=True,stage_c_passed=False,training_allowed=False,
        next_step='Do not expand automatic authoring on unchanged sources. Need a materially different judgment operation with source support and pre-authoring feasibility review.')
    path=Path('docs/cpt_logistika_independent_results_20260922.safe.json')
    path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:output[k] for k in ('blind_quality_pass_tasks','evidence_agreement_pass_tasks',
        'evidence_agreement_complete_groups','final_quality_complete_groups','actual_student_model_calls')},ensure_ascii=False))
    return output


if __name__=='__main__':summarize()

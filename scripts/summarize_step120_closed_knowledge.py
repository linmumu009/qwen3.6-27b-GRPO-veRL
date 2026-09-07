"""Apply externally reviewed semantic decisions; never infer correctness from keywords."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.prepare_step120_qa_diagnostic import digest, write_json
from scripts.run_step120_closed_knowledge import response_hash, validate_protocol


def verdict(response, decision):
    if not response['text'].strip() or response['finish_reason'] != 'stop':
        return 'unscorable'
    if decision['material_contradiction']:
        return 'conflict'
    values = decision['criteria']
    if all(v == 'met' for v in values):
        return 'full'
    if any(v == 'met' for v in values):
        return 'partial'
    return 'incorrect'


def summarize(protocol, responses, decisions):
    validate_protocol(protocol)
    tasks = {t['id']: t for t in protocol['tasks']}
    expected = {(k, r) for k in tasks for r in (1, 2, 3)}
    if len(responses) != 36 or {(r['task_id'], r['repeat']) for r in responses} != expected:
        raise ValueError('response coverage mismatch')
    by_text = {(r['task_id'], response_hash(r['text'])) for r in responses}
    judged = {(d['task_id'], d['response_sha256']): d for d in decisions}
    if len(judged) != len(decisions) or set(judged) != by_text:
        raise ValueError('decision coverage/hash mismatch')
    for key, d in judged.items():
        if len(d['criteria']) != len(tasks[key[0]]['criteria']) or any(
            v not in ('met', 'not_met', 'unclear') for v in d['criteria']
        ):
            raise ValueError('invalid criterion decision')
        if type(d['material_contradiction']) is not bool or type(d['alternative_convention']) is not bool:
            raise ValueError('boolean flags required')
        if type(d.get('ancillary_factual_issue', False)) is not bool:
            raise ValueError('boolean ancillary flag required')
        if not isinstance(d.get('rationale'), str) or not d['rationale'].strip():
            raise ValueError('semantic review rationale required')
    scored = []
    for r in responses:
        d = judged[(r['task_id'], response_hash(r['text']))]
        scored.append({'task_id': r['task_id'], 'repeat': r['repeat'], 'status': verdict(r, d),
                       'criteria_met': sum(v == 'met' for v in d['criteria']),
                       'criteria_unclear': sum(v == 'unclear' for v in d['criteria']),
                       'alternative_convention': d['alternative_convention'],
                       'ancillary_factual_issue': d.get('ancillary_factual_issue', False)})
    question_results = {}
    for task_id, task in tasks.items():
        group = [s for s in scored if s['task_id'] == task_id]
        raw = [r for r in responses if r['task_id'] == task_id]
        question_results[task_id] = {'review_id': task['review_id'], 'kind': task['kind'],
            'full_repeats': sum(s['status'] == 'full' for s in group),
            'pass': sum(s['status'] == 'full' for s in group) >= 2,
            'statuses': dict(Counter(s['status'] for s in group)),
            'all_texts_identical': len({r['text'] for r in raw}) == 1,
            'alternative_convention_repeats': sum(s['alternative_convention'] for s in group),
            'flagged_extra_factual_issue_repeats': sum(s['ancillary_factual_issue'] for s in group),
            'conservative_pass_excluding_flagged_extra_errors': sum(
                s['status'] == 'full' and not s['ancillary_factual_issue'] for s in group) >= 2,
            'unclear_criterion_judgments': sum(s['criteria_unclear'] for s in group)}
    by_kind = {}
    for kind in sorted({t['kind'] for t in tasks.values()}):
        group = [q for q in question_results.values() if q['kind'] == kind]
        by_kind[kind] = {'questions': len(group), 'passed': sum(q['pass'] for q in group)}
    topics = {}
    for topic in sorted({t['review_id'] for t in tasks.values()}):
        group = [q for q in question_results.values() if q['review_id'] == topic]
        topics[topic] = {'questions': len(group), 'passed': sum(q['pass'] for q in group),
                         'all_three_passed': all(q['pass'] for q in group)}
    return {'private_content_included': False, 'training_performed': False, 'model': 'Step120',
        'status': 'complete_agent_semantic_review', 'topics': 4, 'questions': 12, 'responses': 36,
        'question_passed': sum(q['pass'] for q in question_results.values()),
        'conservative_question_passed_excluding_flagged_extra_errors': sum(
            q['conservative_pass_excluding_flagged_extra_errors'] for q in question_results.values()),
        'flagged_extra_factual_issue_responses': sum(s['ancillary_factual_issue'] for s in scored),
        'extra_error_sensitivity_is_posthoc_not_primary_scoring': True,
        'topics_all_three_passed': sum(t['all_three_passed'] for t in topics.values()),
        'response_statuses': dict(Counter(s['status'] for s in scored)),
        'by_kind': by_kind, 'by_topic': topics, 'question_results': question_results,
        'reviewer': 'Codex agent, not human domain expert; rubric frozen before inference',
        'human_expert_reviewed': 0,
        'limitations': ['four outcome-selected topics, twelve related probes, not twelve independent topics',
            'named-concept questions and simpler scenarios provide cues and differ from original MCQ format',
            'three deterministic repeats check stability, not independent statistical confidence',
            'source-specific taxonomy agreement is distinct from universal factual correctness',
            'primary full means frozen core criteria satisfied, not a certification of every extra factual claim',
            'conservative score excludes observed flagged extra errors; not an exhaustive all-fact audit',
            'passing a probe does not prove all original-question reasoning prerequisites are present',
            'missing/unclear criteria mean not demonstrated in this answer, not proven absent in model parameters',
            'historical original MCQ and current free-response probes are not a controlled format-only ablation']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--decisions', type=Path, required=True)
    args = p.parse_args()
    target = args.run / 'semantic_results.safe.json'
    if target.exists():
        raise ValueError('refusing overwrite')
    protocol = json.loads((args.run / 'protocol.private.json').read_text())
    responses = json.loads((args.run / 'responses.private.json').read_text())
    generation = json.loads((args.run / 'generation.safe.json').read_text())
    decision_file = json.loads(args.decisions.read_text())
    for name, file in (('protocol_sha256', 'protocol.private.json'), ('responses_sha256', 'responses.private.json')):
        if digest(args.run / file) != generation[name] or decision_file[name] != generation[name]:
            raise ValueError('frozen input hash mismatch')
    result = summarize(protocol, responses, decision_file['decisions'])
    result['protocol_sha256'] = generation['protocol_sha256']
    result['responses_sha256'] = generation['responses_sha256']
    result['decisions_sha256'] = digest(args.decisions)
    write_json(target, result)
    (args.run / 'status.txt').write_text('complete_agent_semantic_review\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

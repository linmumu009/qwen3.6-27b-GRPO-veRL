import copy
import json
import pytest

from scripts.run_step120_closed_knowledge import (KINDS, TOPICS, messages_for, response_hash,
                                                review_groups, validate_protocol)
from scripts.summarize_step120_closed_knowledge import summarize, verdict


def protocol():
    tasks = [{'id': str(i), 'review_id': topic, 'kind': kind, 'question': 'Synthetic question',
              'criteria': ['RUBRIC-SECRET'] * 3} for i, (topic, kind) in enumerate(
                  (t, k) for t in sorted(TOPICS) for k in sorted(KINDS))]
    return {'tasks': tasks, 'training_allowed': False, 'repeats': 3, 'max_tokens': 512,
            'system': 'Answer briefly.', 'sources': {'SECRET-SOURCE': 'not forwarded'}}


def example_decision(task_id='0', text='Synthetic answer'):
    return {'task_id': task_id, 'response_sha256': response_hash(text), 'criteria': ['met'] * 3,
            'material_contradiction': False, 'alternative_convention': False, 'rationale': 'Synthetic rationale'}


def test_frozen_contract_and_prompt_allowlist():
    p = protocol()
    validate_protocol(p)
    actual = messages_for(p, p['tasks'][0])
    assert actual == [{'role': 'system', 'content': 'Answer briefly.'},
                      {'role': 'user', 'content': 'Synthetic question'}]
    assert 'SECRET' not in json.dumps(actual)
    for field, value in [('training_allowed', True), ('repeats', 1), ('max_tokens', 1024)]:
        with pytest.raises(ValueError):
            validate_protocol({**p, field: value})
    broken = copy.deepcopy(p)
    broken['tasks'][0]['kind'] = 'other'
    with pytest.raises(ValueError, match='topics'):
        validate_protocol(broken)


def test_semantic_verdict_and_incomplete_generation():
    response = {'text': 'answer', 'finish_reason': 'stop'}
    decision = example_decision()
    assert verdict(response, decision) == 'full'
    assert verdict({**response, 'finish_reason': 'length'}, decision) == 'unscorable'
    assert verdict(response, {**decision, 'material_contradiction': True}) == 'conflict'
    assert verdict(response, {**decision, 'criteria': ['met', 'unclear', 'not_met']}) == 'partial'
    assert verdict(response, {**decision, 'criteria': ['not_met'] * 3}) == 'incorrect'


def test_exact_text_review_dedup_does_not_merge_different_probes():
    responses = [{'task_id': t, 'repeat': r, 'text': 'same', 'finish_reason': 'stop'}
                 for t in ('a', 'b') for r in (1, 2, 3)]
    groups = review_groups(responses)
    assert len(groups) == 2 and all(g['repeats'] == [1, 2, 3] for g in groups)


def test_summary_keeps_semantic_majorities_and_rejects_missing_or_forged_grades():
    p = protocol()
    responses = [{'task_id': t['id'], 'repeat': r, 'text': 'Synthetic answer', 'finish_reason': 'stop'}
                 for t in p['tasks'] for r in (1, 2, 3)]
    decisions = [example_decision(t['id']) for t in p['tasks']]
    responses[0]['finish_reason'] = 'length'
    result = summarize(p, responses, decisions)
    assert result['question_passed'] == 12 and result['topics_all_three_passed'] == 4
    assert result['response_statuses'] == {'unscorable': 1, 'full': 35}
    flagged = copy.deepcopy(decisions)
    flagged[0]['ancillary_factual_issue'] = True
    conservative = summarize(p, responses, flagged)
    assert conservative['question_passed'] == 12
    assert conservative['conservative_question_passed_excluding_flagged_extra_errors'] == 11
    assert conservative['flagged_extra_factual_issue_responses'] == 3
    assert all(v['questions'] == 4 for v in result['by_kind'].values())
    assert 'Synthetic answer' not in json.dumps(result) and 'RUBRIC-SECRET' not in json.dumps(result)
    with pytest.raises(ValueError, match='coverage'):
        summarize(p, responses[:-1], decisions)
    with pytest.raises(ValueError, match='coverage/hash'):
        summarize(p, responses, decisions[:-1])
    bad = copy.deepcopy(decisions)
    bad[0]['material_contradiction'] = 'false'
    with pytest.raises(ValueError, match='boolean'):
        summarize(p, responses, bad)

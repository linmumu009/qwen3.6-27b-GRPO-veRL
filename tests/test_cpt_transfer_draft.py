import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cpt_transfer_draft import generation_prompt, review_prompt, validate_task
from queue_cpt_coverage_probe import predecessor_ready
from cpt_transfer_draft import load_resume
from verify_cpt_transfer_draft import verify
import hashlib
import json


@pytest.fixture
def source_request():
    return {'designs':['single_case_application'],'sources':[{
        'title':'Test rule','scope':'Archived test scope',
        'source_text':'Operating cost includes all specified cost components.'}],
        'diagnostic_question':'PRIVATE_DIAGNOSTIC_CANARY','prediction':'PRIVATE_PREDICTION_CANARY'}


@pytest.fixture
def task():
    return dict(design='single_case_application',question='Which total applies? Select all correct statements.',
        options=['five','six','seven','eight'],correct_indices=[0],
        source_quote='Operating cost includes all specified cost components.',
        explanation='SECRET_KEY_EXPLANATION',option_reasons=['SECRET_KEY_REASON']*4,
        arithmetic_checks=[['9-3-1','5']],reasoning_structure='sum specified components')


def test_generator_allowlist(source_request):
    text=generation_prompt(source_request)
    assert 'PRIVATE_DIAGNOSTIC_CANARY' not in text and 'PRIVATE_PREDICTION_CANARY' not in text
    assert 'Operating cost includes' in text


def test_source_constraints_bind_quote_and_scope(source_request, task):
    source_request['sources'][0]['id'] = 'source-1'
    source_request['source_constraints'] = {
        'required_question_terms': ['Archived test scope'],
        'rules': [{'source_id': 'source-1',
                   'quote': source_request['sources'][0]['source_text'],
                   'instruction': 'Name the scope and include every cost component.'}],
    }
    assert 'audited_source_constraints' in generation_prompt(source_request)
    assert 'SECRET_KEY_' not in review_prompt(task, source_request)
    with pytest.raises(ValueError, match='missing_registered_scope'):
        validate_task(task, source_request)
    task['question'] = 'Archived test scope: ' + task['question']
    assert validate_task(task, source_request)['correct_indices'] == [0]
    source_request['source_constraints']['rules'][0]['quote'] = 'Invented rule that is not in the source.'
    with pytest.raises(ValueError, match='constraint_quote'):
        generation_prompt(source_request)


def test_reviewer_is_blind_to_generated_key(source_request,task):
    text=review_prompt(task,source_request)
    assert 'SECRET_KEY_' not in text
    assert 'arithmetic_checks' not in text
    assert 'PRIVATE_DIAGNOSTIC_CANARY' not in text


def test_valid_numeric_task(source_request,task):
    assert validate_task(task,source_request)['correct_indices']==[0]


def test_expansion_all_correct_is_opt_in_and_reviewer_blind(source_request,task):
    task['correct_indices']=[0,1,2,3]
    with pytest.raises(ValueError):validate_task(task,source_request)
    source_request.update(allow_all_correct=True,answer_cardinalities={'single_case_application':4},
                          authoring_scope='ARCHIVED_SCOPE_CONSTRAINT')
    assert validate_task(task,source_request)['correct_indices']==[0,1,2,3]
    assert 'requested_correct_option_count_by_design' in generation_prompt(source_request)
    blind=review_prompt(task,source_request)
    assert 'ARCHIVED_SCOPE_CONSTRAINT' in blind
    assert 'requested_correct_option_count_by_design' not in blind
    assert 'answer_cardinalities' not in blind and 'SECRET_KEY_' not in blind
    source_request['answer_cardinalities']['single_case_application']=3
    with pytest.raises(ValueError,match='answer_cardinality_mismatch'):validate_task(task,source_request)


def test_all_correct_still_rejects_duplicate_or_boolean_indices(source_request,task):
    source_request['allow_all_correct']=True
    for value in ([0,1,2,2],[0,1,2,True],[0,1,2,4]):
        task['correct_indices']=value
        with pytest.raises(ValueError):validate_task(task,source_request)


def test_independent_review_verifier_all_correct_opt_in(task):
    from verify_cpt_transfer_draft import review_accepts
    task['correct_indices']=[0,1,2,3]
    answer=dict(correct_indices=[0,1,2,3],option_reasons=['supported']*4,
                **{k:True for k in ('supported','unambiguous','self_contained','scope_preserved','not_answer_leaking','design_satisfied')})
    record=dict(text=json.dumps(answer),finish_reason='stop')
    assert not review_accepts(record,task)
    assert review_accepts(record,task,allow_all=True)
    answer['correct_indices']=[0,1,2,True];record['text']=json.dumps(answer)
    assert not review_accepts(record,task,allow_all=True)


def test_source_scope_does_not_change_legacy_prompt(source_request,task):
    before=generation_prompt(source_request)
    source_request['authoring_scope']='KEEP_SOURCE_SCOPE'
    assert 'KEEP_SOURCE_SCOPE' in generation_prompt(source_request)
    assert 'KEEP_SOURCE_SCOPE' in review_prompt(task,source_request)
    del source_request['authoring_scope']
    assert generation_prompt(source_request)==before


def test_embedded_select_all_instruction_keeps_original_text(source_request,task):
    task['question']='Under this archived scope, select all correct statements about these records.'
    assert validate_task(task,source_request)['question']==task['question']


def test_resume_binds_original_request_and_prompt(tmp_path,source_request):
    source_request.update(id='unit-001-train',split='train')
    record=dict(request_id=source_request['id'],text='ORIGINAL OUTPUT',prompt_text_sha256=hashlib.sha256(generation_prompt(source_request).encode()).hexdigest())
    file=tmp_path/'generation.jsonl';file.write_text(json.dumps(record)+'\n')
    digest=hashlib.sha256(file.read_bytes()).hexdigest()
    assert load_resume(file,digest,[source_request])[source_request['id']]['text']=='ORIGINAL OUTPUT'
    with pytest.raises(ValueError):load_resume(file,'0'*64,[source_request])
    changed=copy.deepcopy(source_request);changed['sources'][0]['scope']='different scope'
    with pytest.raises(ValueError):load_resume(file,digest,[changed])


@pytest.mark.parametrize('change',[
    {'arithmetic_checks':[['48*9+12*29','756']]},
    {'arithmetic_checks':[["__import__('os').getcwd()",'0']]},
    {'design':'sealed_two_stage_decision'},
    {'correct_indices':[True]},
    {'source_quote':'A made-up governing rule that is not in the source.'},
    {'options':['same','same','other','different']},
])
def test_reject_bad_data(source_request,task,change):
    task=copy.deepcopy(task);task.update(change)
    with pytest.raises((ValueError,SyntaxError)):validate_task(task,source_request)


def test_queue_does_not_launch_on_failure_or_partial_output():
    assert not predecessor_ready({'status':'reviewing_drafts'})
    assert not predecessor_ready({'status':'completed_pending_verification'})
    assert predecessor_ready({'status':'drafted_pending_independent_audit'})
    with pytest.raises(RuntimeError):predecessor_ready({'status':'failed'})


def test_verifier_rebuilds_raw_candidates_and_detects_tampering(tmp_path,source_request,task):
    source_request.update(id='u-train',unit='u',split='train',category='test',training_allowed=False)
    packet=tmp_path/'requests.jsonl';packet.write_text(json.dumps(source_request)+'\n')
    def write(name,rows):
        (tmp_path/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    def raw(text,prompt):
        return dict(text=text,finish_reason='stop',output_tokens=100,prompt_tokens=100,
            prompt_text_sha256=hashlib.sha256(prompt.encode()).hexdigest())
    (tmp_path/'registration.safe.json').write_text(json.dumps({'packet_sha256':hashlib.sha256(packet.read_bytes()).hexdigest()}))
    write('generation.private.jsonl',[dict(request_id='u-train',**raw(json.dumps({'tasks':[task]}),generation_prompt(source_request)))])
    candidate=dict(id='u-train-single_case_application',request_id='u-train',unit='u',split='train',category='test',task=task,training_allowed=False)
    write('candidates.private.jsonl',[candidate])
    review=dict(correct_indices=[0],option_reasons=['ok']*4,**{k:True for k in ('supported','unambiguous','self_contained','scope_preserved','not_answer_leaking','design_satisfied')})
    write('review.private.jsonl',[dict(id=candidate['id'],**raw(json.dumps(review),review_prompt(task,source_request)))])
    filtered=dict(candidate,review=review,training_ready=False,review_status='same_model_filtered_only')
    write('filtered.private.jsonl',[filtered]);write('train.draft.private.jsonl',[filtered])
    for split in ('dev','sealed','retention'):write(split+'.draft.private.jsonl',[])
    result=verify(packet,tmp_path)
    assert result['filtered_items']==1 and result['training_allowed'] is False
    candidate['task']['correct_indices']=[1]
    write('candidates.private.jsonl',[candidate])
    with pytest.raises(AssertionError):verify(packet,tmp_path)

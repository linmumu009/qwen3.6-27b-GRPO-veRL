import pytest
from scripts.evaluate_logistics_knowledge import _make_item, PROMPT_VERSION
from scripts.prepare_step120_qa_diagnostic import choose_cohort, retrieve
from scripts.run_step120_qa_diagnostic import (
    task_payload, valid_probe, basic_pass, evidence_pass, majority, question_item, evaluation_messages, summarize)


def make_item(number):
    return _make_item(dataset='synthetic',source_id=str(number),category='cat',question_type='single_choice',
        question=f'Synthetic question {number}',options=['alpha','beta'],expected=[0])


def test_cohort_keeps_all_historical_errors_and_distinct_controls():
    items=[make_item(i) for i in range(8)]
    baseline={item.item_hash:{**item.__dict__,'prompt_version':PROMPT_VERSION,
        'chat_template_disable_thinking':True,'parse_ok':True,'correct':i<5,
        'parsed':[0] if i<5 else [1]} for i,item in enumerate(items)}
    errors,controls=choose_cohort(items,baseline,controls=2)
    assert len(errors)==3 and len(controls)==2
    assert len({r.item_hash for r,_,_ in controls})==2
    assert not {r.item_hash for r in errors}&{r.item_hash for r,_,_ in controls}
    assert all(exact for _,_,exact in controls)
    assert choose_cohort(items,baseline,controls=2)==(errors,controls)
    baseline[items[0].item_hash]['correct']=False
    with pytest.raises(ValueError,match='correctness'):
        choose_cohort(items,baseline,controls=2)


def example_row():
    return {**make_item(1).__dict__,'cohort':'original_error',
        'evidence':[{'text':'A synthetic reference states that alpha is the permitted answer.'}]}


def example_probe():
    return {'status':'candidate','question':'What is permitted?',
        'options':['alpha','beta','gamma','delta'],'answer':0,
        'quote':'alpha is the permitted answer.'}


def test_authoring_payload_never_contains_gold_or_cohort():
    payload=task_payload(example_row())
    assert set(payload)=={'question','options','reference_excerpts'}
    assert 'expected' not in payload and 'cohort' not in payload


def test_retrieval_is_gold_blind_and_keeps_source_offsets():
    windows=[{'record':0,'start_token':0,'text':'alpha inventory stock'},
             {'record':0,'start_token':288,'text':'alpha inventory stock'},
             {'record':1,'start_token':0,'text':'beta transport'}]
    found=retrieve('inventory',['alpha','beta'],windows)
    assert found[0]['record']==0
    assert len([r for r in found if r['record']==0])==1
    assert retrieve('zzzz',[],windows)==[]


def test_probe_requires_real_quote_and_unique_options():
    probe=example_probe()
    assert valid_probe(probe,example_row())
    assert not valid_probe({**probe,'quote':'invented unsupported evidence'},example_row())
    assert not valid_probe({**probe,'answer':True},example_row())
    assert not valid_probe({**probe,'options':['alpha']*4},example_row())


def test_evidence_sufficiency_is_provisional_and_gold_checked_after_review():
    audit={'original_answerable':True,'original_answers':[0],
        'original_quote':example_probe()['quote']}
    assert evidence_pass(audit,example_row())
    assert not evidence_pass({**audit,'original_answers':[1]},example_row())
    assert not evidence_pass({**audit,'original_answers':[True]},example_row())
    assert not evidence_pass({**audit,'original_answerable':'true'},example_row())
    assert not basic_pass({})


def test_rotated_probe_gold_and_common_system_prompt():
    row=example_row()
    item=question_item(row,example_probe())
    assert item.options[item.expected[0]]=='alpha'
    closed=evaluation_messages(item)
    opened=evaluation_messages(item,row['evidence'])
    assert closed[0]==opened[0]
    assert closed[1]['content'] in opened[1]['content']


def test_majority_tie_is_not_best_repeat():
    assert not majority([{'parse_ok':True,'answers':[i]} for i in (0,1,2)],[0])['correct']
    result=majority([{'parse_ok':True,'answers':[i]} for i in (0,0,1)],[0])
    assert result['correct'] and not result['all_repeats_same']


def test_summary_does_not_call_unverified_evidence_a_reasoning_failure():
    row=example_row()
    score={'correct':False,'parse_ok':True,'all_repeats_same':True}
    results={row['item_hash']:{'original_closed':score,'original_with_book':score}}
    probes={row['item_hash']:{'basic_pass':False,'evidence_pass':False}}
    report=summarize([row],probes,results)
    group=report['cohorts']['original_error']
    assert group['evidence_not_auto_verified']==1
    assert group['provisional_application_review_candidates']==0
    assert group['provisional_application_candidate_denominator']==0
    assert report['training_performed'] is False

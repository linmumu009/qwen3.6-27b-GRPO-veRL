import copy
import hashlib
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_cpt_p1_authoring_compare import summarize, specs, closed_prompt


def fixture():
    rows=[dict(id='one',split='dev',question='Question',options=['a','b','c','d'],correct_indices=[0,2])]
    raw=[]
    for s in specs(rows):
        raw.append(dict(id=s['id'],variant=s['variant'],order=s['order'],text=json.dumps({'answers':s['expected']}),
            finish_reason='stop',output_tokens=10,prompt_tokens=20,prompt_token_sha256='same-tokens',
            prompt_text_sha256=hashlib.sha256(closed_prompt(rows[0],s).encode()).hexdigest()))
    return rows,raw,copy.deepcopy(raw)


def test_same_correct_answers_are_stable_after_mapping():
    rows,raw,base=fixture();r=summarize(rows,raw,base)
    assert r['both_orders_correct']==1 and r['counts']['gains']==r['counts']['losses']==0
    assert r['option_order_changed_answer_ids']==[]


def test_detect_regression_and_invalid_output_without_repair():
    rows,raw,base=fixture();raw[0]['text']='{"answers":[true]}'
    r=summarize(rows,raw,base)
    assert r['counts']['losses']==r['counts']['invalid']==1
    assert r['both_orders_correct']==0 and r['training_allowed'] is False


def test_token_and_option_order_mismatch_fail_closed():
    for field,value in [('prompt_token_sha256','different'),('order',[3,2,1,0])]:
        rows,raw,base=fixture();raw[0][field]=value
        with pytest.raises(AssertionError):summarize(rows,raw,base)


def test_truncated_answer_does_not_receive_credit():
    rows,raw,base=fixture();raw[1]['finish_reason']='length'
    r=summarize(rows,raw,base)
    assert r['counts']['truncated']==1 and r['counts']['p1_correct']==1

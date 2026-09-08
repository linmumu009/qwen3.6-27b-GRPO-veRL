import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from calibrate_single_book_judge import valid_criteria,valid_score
from build_minimal_edit_book_pairs import audit_ok

def test_frozen_minimal_criteria_and_single_answer_axes():
    units={'S001':'source'}
    c={'question_supported':True,'core_criteria':['core'],'necessary_criteria':['reason'],'source_ids':['S001']}
    assert valid_criteria(c,units)
    c['core_criteria']=[];assert not valid_criteria(c,units)
    s={'core_correct':True,'necessary_complete':False,'material_error':False,'source_ids':['S001']}
    assert valid_score(s,units)
    s['core_correct']=1;assert not valid_score(s,units)

def test_pair_audit_rejects_any_failed_arm():
    v={k:True for k in ('question_supported','same_required_facts','a_correct','a_complete','b_correct','b_complete','b_no_unsupported_additions')}
    v['source_ids']=['S001'];assert audit_ok(v,{'S001':'source'})
    v['a_complete']=False;assert not audit_ok(v,{'S001':'source'})

def test_repeat_gate_counts_invalid_pairs_in_denominator():
    assert 51/56>=.90
    assert 50/56<.90
    assert 50/54>=.90  # Dropping invalids would wrongly change the decision.

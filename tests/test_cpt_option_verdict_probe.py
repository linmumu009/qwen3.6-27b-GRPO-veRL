import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_cpt_option_verdict_probe import parse


def test_consistent_full_answer():
    assert parse('{"option_truth":[true,true,true,true],"answers":[0,1,2,3]}')==([0,1,2,3],True)


def test_contradicting_indices_are_not_silently_corrected():
    assert parse('{"option_truth":[true,false,true,false],"answers":[0,1,2,3]}')==([0,1,2,3],False)


def test_enforce_true_boolean_values_and_output_order():
    for text in ('{"option_truth":[1,0,1,0],"answers":[0,2]}',
                 '{"answers":[0,2],"option_truth":[true,false,true,false]}',
                 '{"option_truth":[true,false],"answers":[0]}'):
        assert parse(text)==(None,None)


def test_invalid_answer_set_stays_invalid():
    assert parse('{"option_truth":[true,false,true,false],"answers":[0,0,2]}')==(None,None)

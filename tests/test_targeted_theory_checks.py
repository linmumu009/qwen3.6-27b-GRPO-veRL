import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from audit_pilot_judge_stability import valid

def test_semantic_judge_requires_independent_boolean_axes():
    a={'id':'A','core_correct':True,'necessary_complete':False,'material_error':False,'source_ids':['S001']}
    v={'question_supported':True,'answers':[a,dict(a,id='B')]}
    assert valid(v,{'S001':'source'})
    v['answers'][1]['core_correct']=1
    assert not valid(v,{'S001':'source'})

def test_nll_is_inference_only_with_exact_training_prefix():
    root=Path(__file__).parents[1]
    s=(root/'scripts/measure_targeted_pilot_nll.py').read_text()
    assert 'prompt_logprobs=1' in s and 'prompt_messages(r,0)' in s
    assert 'content_nll_sum' in s and 'eos_nll' in s
    assert "actual[r['id']]['answer_start']" in s
    launcher=(root/'scripts/run_targeted_pilot_nll.sh').read_text()
    assert 'flock -n 9' in launcher and 'torchrun' not in launcher

import sys
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from diagnose_book_sft_metrics import transitions
from grade_book_sft_diagnosis import validate


def test_transitions_partition():
    a={i:{'correct':x} for i,x in enumerate([True,True,False,False])}
    b={i:{'correct':x} for i,x in enumerate([True,False,True,False])}
    assert transitions(a,b)=={'both_correct':1,'regressed':1,'improved':1,'both_wrong':1}


def test_blind_grade_validation():
    v={'question_answerable':True,'reference_supported':True,
       'A':{'score':2,'reason':'supported','source_ids':['S001']},
       'B':{'score':0,'reason':'contradicts','source_ids':['S001']}}
    assert validate(v,{'S001':'source'})
    assert not validate(v,{})
    v['A']['score']=True
    assert not validate(v,{'S001':'source'})


def test_saved_diagnostic_evidence_reconciles():
    docs=Path(__file__).parents[1]/'docs'
    read=lambda name:json.loads((docs/name).read_text(encoding='utf-8'))
    g=read('book_sft_diagnosis_grading_20260907.safe.json')
    assert sum(g['status'].values())==g['items']==186
    assert sum(g['paired'].values())==sum(g['full_correct_transitions'].values())==g['scored_items']==183
    for arm in ('cpt','sft'):
        assert sum(g['scores'][arm].values())==183
        c=read(f'book_sft_diagnosis_{arm}_concise_20260907.safe.json')
        assert c['finish_reasons']=={'stop':186}
    assert g['scores']['sft']['2']-g['scores']['cpt']['2']==49-12==37
    m=read('book_sft_diagnosis_metrics_20260907.safe.json')
    assert sum(m['transitions'].values())==m['items']==1672
    assert sum(s['net'] for s in m['segments'])==m['sft_correct']-m['baseline_correct']==-26
    a=read('book_sft_diagnosis_20260907.artifact.json')
    assert sum(r['net'] for r in a['snapshot']['datasets']['drivers'])==-26
    assert sorted(r['net'] for r in a['snapshot']['datasets']['drivers'])==sorted(s['net'] for s in m['segments'] if s['net'])
    for source in a['manifest']['sources']:
        assert (docs.parent/source['path']).is_file()

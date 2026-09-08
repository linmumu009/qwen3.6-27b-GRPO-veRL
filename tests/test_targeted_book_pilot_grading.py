import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from grade_targeted_book_pilot import valid,summarize,MODELS,CONDITIONS,LABELS


def test_review_requires_twelve_distinct_supported_scores():
    v={'question_valid':True,'rubric_valid':True,'answers':[
        {'id':k,'score':2,'reason':'supported','source_ids':['S001']} for k in LABELS]}
    assert valid(v,{'S001':'source'})
    v['answers'][0]['score']=True
    assert not valid(v,{'S001':'source'})
    v['answers'][0]['score']=2;v['answers'][-1]['id']='A'
    assert not valid(v,{'S001':'source'})


def test_common_denominator_and_maintenance_separation():
    rows=[{'id':'a','split':'dev','concept_group':'g'},
          {'id':'b','split':'dev','concept_group':'g'},
          {'id':'c','split':'train','concept_group':'h'}]
    scores={m+':'+c:2 for m in MODELS for c in CONDITIONS}
    scores['baseline:closed0']=1
    results=[{'id':'a','status':'scored','scores':scores},
             {'id':'b','status':'invalid_review'},
             {'id':'c','status':'scored','scores':scores}]
    out=summarize(rows,results)
    dev=out['development']
    assert dev['items']==2 and dev['scored']==1
    assert dev['paired_closed_both']['step10']=={'gain':1}
    assert dev['metrics']['step10']['closed']['answer_count']==2
    assert out['in_training_maintenance']['items']==1

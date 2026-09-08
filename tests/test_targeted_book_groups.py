import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from build_targeted_book_groups import valid_group,audit_pass,KINDS
from probe_targeted_book_groups import prompt_messages
from grade_targeted_book_groups import valid_verdict
import io
import json
import build_targeted_book_groups as builder
from organize_targeted_book_pools import pool


def test_group_requires_variants_and_all_three_kinds():
    g={'concept':'Rule','questions':[{'kind':k,'question':f'What {k}?','variant':f'Explain {k}.',
       'answer':'Supported answer.','rubric':['Necessary point'],'source_ids':['S001']} for k in KINDS]}
    assert valid_group(g,{'S001':'evidence'})
    assert not valid_group(g,{})
    g['questions'][0]['variant']=g['questions'][0]['question']
    assert not valid_group(g,{'S001':'evidence'})


def test_audit_fails_closed():
    a={'same_concept':True,'topic_supported':True,'questions':[dict(kind=k,
        standalone=True,variants_equivalent=True,answer_correct=True,source_complete=True,
        rubric_fair=True,kind_valid=True,source_ids=['S001'],issues=[]) for k in KINDS]}
    assert audit_pass(a,{'S001':'evidence'})
    a['questions'][1]['variants_equivalent']='true'
    assert not audit_pass(a,{'S001':'evidence'})


def test_probe_prompt_excludes_answers_and_rubric():
    r={'question':'Question one?','variant':'Equivalent question?', 'answer':'SECRET_ANSWER','rubric':['SECRET_RUBRIC']}
    messages=prompt_messages(r,0)
    text=str(messages)
    assert 'SECRET_' not in text
    assert messages[1]['content']=='Question one?'
    assert 'Reference material' in prompt_messages(r,1,'Only evidence')[1]['content']


def test_four_answer_grading_requires_complete_ids():
    v={'question_valid':True,'rubric_valid':True,'answers':[{'id':x,'score':2,'reason':'supported','source_ids':['S001']} for x in 'ABCD']}
    assert valid_verdict(v,{'S001':'source'})
    v['answers'][0]['id']='B';assert not valid_verdict(v,{'S001':'source'})


def test_api_json_mode_has_explicit_json_instruction(monkeypatch):
    def fake(req,timeout):
        payload=json.loads(req.data)
        assert 'json' in ' '.join(m['content'] for m in payload['messages']).casefold()
        assert payload['response_format']=={'type':'json_object'}
        return io.BytesIO(b'{"ok":true}')
    monkeypatch.setattr(builder.urllib.request,'urlopen',fake)
    assert builder.call_api({'base_url':'https://dashscope.aliyuncs.com/compatible-mode/v1','api_key':'unit-test-not-a-secret'},'Instruction',{})=={'ok':True}


def test_pool_does_not_export_dev_as_training_opportunity():
    assert pool({'decision':'mixed_opportunity','split':'train'})=='opportunity'
    assert pool({'decision':'mixed_opportunity','split':'dev'})=='quarantine'
    assert pool({'decision':'development_only','split':'dev'})=='development'

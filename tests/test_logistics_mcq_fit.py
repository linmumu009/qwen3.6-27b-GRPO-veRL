import pytest
from scripts.evaluate_logistics_knowledge import _make_item, build_messages
from scripts.prepare_logistics_mcq_fit import encode_item
from scripts.qwen36_mcq_answer_dataset import validate_record
from scripts.summarize_logistics_mcq_fit import compare_scores


class Tokenizer:
    eos_token_id=7
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs==dict(tokenize=False,add_generation_prompt=True,enable_thinking=False)
        self.messages=messages
        return 'PREFIX'
    def encode(self,text,**kwargs):
        return [ord(c) for c in text]


def test_exact_eval_messages_options_and_multi_answer_json():
    item=_make_item(dataset='synthetic',source_id='1',category='test',question_type='multiple_choice',
                    question='Choose two.',options=['first','second','third'],expected=[0,2])
    tokenizer=Tokenizer()
    row=encode_item(item,tokenizer)
    assert tokenizer.messages==build_messages(item)
    start=row['answer_start']
    assert row['input_ids'][:start]==list(map(ord,'PREFIX'))
    assert ''.join(map(chr,row['input_ids'][start:-1]))=='{"answers":[0,2]}'
    assert row['input_ids'][-1]==7
    mask=validate_record(row['input_ids'],start,7,4096)
    assert mask[:start]==[0]*start and all(mask[start:]) and mask[0]==0


def test_no_truncation_or_empty_answer_mask():
    with pytest.raises(ValueError):
        validate_record([10,11,7],2,7,4096)
    with pytest.raises(ValueError):
        validate_record([10,11,7],1,7,2)
    with pytest.raises(ValueError):
        validate_record([10,11,8],1,7,4096)


def test_paired_scores_require_same_source_coverage_and_metadata():
    row=dict(item_hash='x',dataset='synthetic',category='t',question_type='single_choice',
             choice_count=2,expected_count=1,correct=False)
    base={'input_sha256':{'cases_jsonl':'fixed'},'prompt_version':'v1','items':1,'rows':[row]}
    candidate={**base,'rows':[{**row,'correct':True}]}
    assert compare_scores(base,candidate)['overall']['net_correct']==1
    with pytest.raises(ValueError,match='source or prompt'):
        compare_scores(base,{**candidate,'prompt_version':'v2'})
    with pytest.raises(ValueError,match='metadata'):
        compare_scores(base,{**candidate,'rows':[{**row,'choice_count':3}]})
    with pytest.raises(ValueError,match='duplicate'):
        compare_scores(base,{**candidate,'rows':[row,row]})

import pytest
from pathlib import Path
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


def test_pipeline_pins_export_bridge_and_checks_resolution():
    script=(Path(__file__).parents[1]/'scripts/run_logistics_mcq_fit_pipeline.sh').read_text()
    assert 'BRIDGE=${ROOT}/reference/Megatron-Bridge-de93536e/src' in script
    assert '${ROOT}:${BRIDGE}:${ROOT}/runtime:/verl:' in script
    assert 'origin.is_relative_to(expected)' in script
    assert script.index('origin.is_relative_to(expected)') < script.index('for step in 418 836')


def test_recovery_skips_training_and_preserves_original_log():
    script=(Path(__file__).parents[1]/'scripts/run_logistics_mcq_fit_pipeline.sh').read_text()
    assert 'POSTTRAIN_ONLY=${POSTTRAIN_ONLY:-false}' in script
    assert '! -e "${RUN}/recovery.log"' in script
    assert 'exec >"${RUN}/recovery.log"' in script
    guarded=script.split('if [[ "${POSTTRAIN_ONLY}" == false ]]; then\nfor arm',1)[1]
    fresh,recovery=guarded.split('\nelse\n',1)
    assert 'run_logistics_mcq_fit_train.sh' in fresh
    assert 'run_logistics_mcq_fit_train.sh' not in recovery
    assert '--checkpoint-exposure 1 --checkpoint-exposure 2' in recovery


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

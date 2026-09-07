import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from produce_book_sft_2000 import duplicate
from audit_book_sft_pilot import grams
from prepare_book_sft_training import encode_messages


def test_cross_batch_dedup():
    text='a warehouse holds goods until they are needed by customers'
    tokens=text.split(); g=grams(text,5)
    assert duplicate(tokens,g,[(tokens,g)])
    assert not duplicate(['unrelated','concept'],set(),[(tokens,g)])


def test_training_is_book_only_and_separate_dev():
    source=(Path(__file__).parents[1]/'scripts/run_book_sft_production_train.sh').read_text()
    assert 'SOURCE=${BOOK}/checkpoints/global_step_116/model/dist_ckpt' in source
    assert 'data.val_files=${OUT}/data/dev.parquet' in source
    assert 'trainer.total_epochs=1' in source
    assert 'prepare_logistics_mcq_fit.py' not in source
    assert '/opt/llin-book-sft-2000-20260907' in source


def test_template_mapping_compatibility():
    class Tokenizer:
        eos_token_id=99
        def apply_chat_template(self,messages,**kw):
            assert kw['tokenize'] is False and kw['enable_thinking'] is False
            return 'prompt'
        def encode(self,text,**kw):
            assert kw['add_special_tokens'] is False
            return [1,2] if text=='prompt' else [3,4]
    ids,start=encode_messages([{'role':'user','content':'question'},{'role':'assistant','content':'answer'}],Tokenizer())
    assert ids==[1,2,3,4,99] and start==2

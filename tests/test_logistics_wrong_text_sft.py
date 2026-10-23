import json
from pathlib import Path

import pytest

from scripts.evaluate_logistics_knowledge import _make_item, parse_answers
from scripts.prepare_logistics_wrong_text_sft import encode, read_predictions, select_errors, training_example
from scripts.logistics_wrong_text_selection import owned_model
from scripts.train_logistics_wrong_text_sft import evaluation_environment


def item(source='1', expected=(0, 2)):
    return _make_item(dataset='test', source_id=source, category='test', question_type='multiple_choice',
        question=f'Choose every suitable action, case {source}.', options=['Move by rail.', 'Delay the order.', 'Use a warehouse.'],
        expected=list(expected))


def test_full_correct_texts_and_shuffled_indices_agree():
    case = item()
    messages, answer, order = training_example(case)
    assert sorted(order) == [0, 1, 2] and order != [0, 1, 2]
    parsed, valid = parse_answers(answer, 3)
    assert valid and {order[index] for index in parsed} == set(case.expected)
    texts = json.loads(answer.split('Correct option text(s):\n')[1])
    assert set(texts) == {'Move by rail.', 'Use a warehouse.'}
    assert all(option in messages[1]['content'] for option in case.options)
    assert training_example(case) == (messages, answer, order)


def test_union_includes_native_only_and_step120_only_errors():
    cases = [item('1'), item('2'), item('3')]
    baseline = {case.item_hash: {'correct': index != 0} for index, case in enumerate(cases)}
    native = {case.item_hash: {'correct': index != 1} for index, case in enumerate(cases)}
    assert select_errors(cases, baseline, native) == cases[:2]


def test_prediction_reader_rejects_false_correctness_and_missing_coverage(tmp_path):
    case = item()
    row = dict(item_hash=case.item_hash, dataset=case.dataset, question=case.question,
        options=list(case.options), expected=list(case.expected), prediction='{"answers":[0]}', correct=True)
    path = tmp_path/'rows.jsonl'
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='correctness'):
        read_predictions(path, [case])
    row['correct'] = False
    path.write_text(json.dumps(row)+'\n')
    assert not read_predictions(path, [case])[case.item_hash]['correct']
    with pytest.raises(ValueError, match='coverage'):
        read_predictions(path, [case, item('2')])


def test_all_answer_text_and_eos_receive_supervision():
    class Tokenizer:
        eos_token_id = 1
        def apply_chat_template(self, messages, **kwargs):
            return 'PREFIX'
        def encode(self, text, **kwargs):
            return list(text.encode())
    row = encode(item(), Tokenizer())
    assert row['answer_start'] == 6 and row['input_ids'][-1] == 1
    text = bytes(row['input_ids'][row['answer_start']:-1]).decode()
    assert 'Move by rail.' in text and 'Use a warehouse.' in text


def test_cleanup_is_limited_to_owned_candidate_models(tmp_path):
    assert owned_model(tmp_path, tmp_path/'hf/step_50') == tmp_path/'hf/step_50'
    for path in [tmp_path/'step_50', tmp_path/'hf/../step_50', tmp_path/'hf/baseline']:
        with pytest.raises(ValueError):
            owned_model(tmp_path, path)


def test_child_evaluator_does_not_join_training_process_group():
    env = evaluation_environment({'RANK': '0', 'WORLD_SIZE': '16', 'MASTER_PORT': '12',
        'HCCL_FOO': 'x', 'TORCHELASTIC_FOO': 'x', 'WRONG_TEXT_RUN': 'run', 'PYTHONPATH': 'code'})
    assert env == {'WRONG_TEXT_RUN': 'run', 'PYTHONPATH': 'code'}


def test_checkpoint_evaluation_offloads_then_restores_same_training_state(tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    import scripts.train_logistics_wrong_text_sft as entry
    events = []
    torch = ModuleType('torch')
    dist = ModuleType('torch.distributed')
    dist.barrier = lambda: events.append('barrier')
    dist.broadcast_object_list = lambda error, src: None
    torch.distributed = dist
    memory = ModuleType('verl.utils.memory_utils')
    memory.aggressive_empty_cache = lambda **kwargs: events.append('empty_cache')
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'torch.distributed', dist)
    monkeypatch.setitem(sys.modules, 'verl.utils.memory_utils', memory)
    monkeypatch.setenv('WRONG_TEXT_RUN', str(tmp_path))
    monkeypatch.setattr(entry.subprocess, 'run', lambda *args, **kwargs: events.append('evaluate'))

    class Trainer:
        rank = 0
        total_training_steps = 103
        def _build_ckpt_handler(self):
            self.ckpt_handler = SimpleNamespace(save_checkpoint=lambda step: events.append(('save', step)))
            self.training_client = SimpleNamespace(to=lambda device, **kwargs: events.append(('to', device)))
    entry.install_evaluation_hook(Trainer)
    trainer = Trainer()
    trainer._build_ckpt_handler()
    client = trainer.training_client
    trainer.ckpt_handler.save_checkpoint(step=50)
    assert trainer.training_client is client
    assert events.index(('save', 50)) < events.index(('to', 'cpu')) < events.index('evaluate') < events.index(('to', 'device'))
    events.clear()
    trainer.ckpt_handler.save_checkpoint(step=103)
    assert ('to', 'cpu') in events and ('to', 'device') not in events

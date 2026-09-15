import ast
import os
from types import SimpleNamespace

import pytest

from llin_verl.math_smoke_reward import compute_score
from scripts.patch_verl_vllm_device_namespace import patch
from scripts.patch_verl_vllm_dp_weight_sync import NEW


@pytest.mark.parametrize('parent,child,expected', [
    ('8,9,10,11,12,13,14,15', '8,9,10,11', [0, 1, 2, 3]),
    ('8,9,10,11,12,13,14,15', '12,13,14,15', [4, 5, 6, 7]),
    ('0,1,2,3,4,5,6,7', '4,5,6,7', [4, 5, 6, 7]),
    ('1,3,5,7', '5,7', [2, 3]),
])
def test_rollout_namespace(tmp_path, monkeypatch, parent, child, expected):
    base = tmp_path / 'workers/rollout/vllm_rollout'
    base.mkdir(parents=True)
    server = base / 'vllm_async_server.py'
    utils = base / 'utils.py'
    server.write_text('        os.environ[get_visible_devices_keyword()] = cuda_visible_devices\n')
    utils.write_text('def resolve(worker_local_rank, parallel_config):\n' + NEW)
    patch(tmp_path)
    saved = utils.read_text()
    patch(tmp_path)
    assert saved == utils.read_text()
    assert 'VERL_ROLLOUT_LOCAL_DEVICES' in server.read_text()
    ns = {'os': os}
    exec(compile(ast.parse(saved), '<patched-helper>', 'exec'), ns)
    monkeypatch.setenv('VERL_ROLLOUT_LOCAL_DEVICES', parent)
    monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES', child)
    cfg = SimpleNamespace(tensor_parallel_size=len(expected))
    assert [ns['resolve'](i, cfg) for i in range(len(expected))] == expected
    monkeypatch.setenv('VERL_ROLLOUT_LOCAL_DEVICES', '999')
    with pytest.raises(ValueError, match='namespace'):
        ns['resolve'](0, cfg)


@pytest.mark.parametrize('answer,gold,score', [
    (r'\boxed{2}', '2', 1),
    (r'\boxed{-\frac{1}{2}}', '-0.5', 1),
    (r'\boxed{-\frac{1}{2}}', '0.5', 0),
    (r'\boxed{50\%}', '0.5', 1),
    (r'\fbox{3} then \boxed{2}', '2', 1),
    (r'\boxed{2} then \fbox{3}', '2', 0),
    ('', '2', 0),
    ('I considered 2 but do not know', '2', 0),
    (r'\boxed{3}', '2', 0),
])
def test_framework_reward_contract(answer, gold, score):
    assert compute_score(data_source='gsm8k', solution_str=answer,
                         ground_truth={'golden_answer': gold}, extra_info={}) == score

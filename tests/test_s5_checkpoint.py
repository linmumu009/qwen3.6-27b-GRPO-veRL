import json
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('s5_gate', Path(__file__).parents[1]/'scripts/check_s5_checkpoint.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fixture_checkpoint(tmp_path):
    m = dict(save_contents=['model', 'extra'], global_step=77, world_size=16,
             contents={k: dict(path=v) for k,v in [('model','model/dist_ckpt'),('rng_state','extra/dist_ckpt')]})
    for v in m['contents'].values():
        d = tmp_path/v['path']; d.mkdir(parents=True)
        (d/'.metadata').write_bytes(b'metadata')
        (d/'shard.distcp').write_bytes(b'shard')
    (tmp_path/'data_0.pt').write_bytes(b'loader')
    (tmp_path/'ckpt_contents.json').write_text(json.dumps(m))
    return m


def test_model_only_is_explicitly_not_optimizer_resumable(tmp_path):
    fixture_checkpoint(tmp_path)
    assert mod.checkpoint_gate(tmp_path)['optimizer_resume_available'] is False


@pytest.mark.parametrize('fault', ['missing_rng', 'outside', 'optimizer', 'wrong_step'])
def test_reject_incomplete_or_wrong_contract(tmp_path, fault):
    m = fixture_checkpoint(tmp_path)
    if fault == 'missing_rng': (tmp_path/'extra/dist_ckpt/shard.distcp').unlink()
    elif fault == 'outside': m['contents']['model']['path'] = '../other'
    elif fault == 'optimizer': m['contents']['optimizer'] = dict(path='optimizer/dist_ckpt')
    else: m['global_step'] = 2
    (tmp_path/'ckpt_contents.json').write_text(json.dumps(m))
    with pytest.raises(ValueError): mod.checkpoint_gate(tmp_path)

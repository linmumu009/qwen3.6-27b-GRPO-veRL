import json

import pytest

from scripts.run_logistics_cpt_curve_8x import checkpoint_gate, majority, paired


def test_invalid_majority_and_missing_repeat_do_not_pass():
    rows = [dict(repeat=i, parsed=[0], valid=i == 0, correct=i == 0) for i in range(3)]
    assert not majority(rows)
    with pytest.raises(ValueError):
        majority([rows[0], rows[0], rows[1]])


def test_paired_tracks_regressions_and_rejects_identity_mismatch():
    assert paired({'a': True, 'b': False}, {'a': False, 'b': True}) == dict(
        n=2, before=1, after=1, gains=1, losses=1)
    with pytest.raises(ValueError):
        paired({'a': True}, {'b': True})


def test_checkpoint_requires_optimizer_extra_and_loader(tmp_path):
    manifest = dict(save_contents=['model', 'optimizer', 'extra'], contents={})
    for key in manifest['save_contents']:
        target = tmp_path / key / 'dist_ckpt'
        target.mkdir(parents=True)
        (target / '.metadata').touch()
        (target / '__0_0.distcp').write_bytes(b'state')
        manifest['contents']['rng_state' if key == 'extra' else key] = dict(path=f'{key}/dist_ckpt')
    manifest['contents']['lr_scheduler'] = dict(path='optimizer/dist_ckpt')
    (tmp_path / 'ckpt_contents.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='dataloader'):
        checkpoint_gate(tmp_path)
    (tmp_path / 'data_0.pt').write_bytes(b'loader')
    assert checkpoint_gate(tmp_path) == manifest
    (tmp_path / 'optimizer/dist_ckpt/.metadata').unlink()
    with pytest.raises(ValueError, match='optimizer'):
        checkpoint_gate(tmp_path)

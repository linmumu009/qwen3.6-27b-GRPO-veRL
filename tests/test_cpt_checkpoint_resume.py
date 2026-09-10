from copy import deepcopy

import pytest

from scripts.cpt_checkpoint_resume import normalize_epoch_boundary


def test_boundary_normalization_preserves_original_state():
    state = dict(_iterator_finished=False, _num_yielded=29, _sampler_iter_yielded=29,
                 _sampler_iter_state=dict(samples_yielded=116))
    before = deepcopy(state)
    result = normalize_epoch_boundary(state, global_step=29, steps_per_epoch=29, records=116)
    assert result['_iterator_finished'] is True and state == before


def test_mid_epoch_state_remains_unfinished():
    state = dict(_iterator_finished=False, _num_yielded=2)
    assert normalize_epoch_boundary(state, global_step=2, steps_per_epoch=29, records=116) == state


def test_inconsistent_boundary_never_silently_skips_data():
    with pytest.raises(ValueError, match='claimed complete epoch'):
        normalize_epoch_boundary(dict(_iterator_finished=False, _num_yielded=28),
                                 global_step=29, steps_per_epoch=29, records=116)

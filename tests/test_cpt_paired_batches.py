import pytest

from scripts.prepare_cpt_paired_batches import match_batches
from scripts.run_cpt_paired_trainer import validate_plan


def test_length_only_match_preserves_complete_coverage_and_budget():
    lengths = [100, 110, 200, 210, 300, 310, 400, 410]
    result = match_batches(lengths, list(reversed(lengths)), seed=17, iterations=10000)
    assert sorted(result['original']) == sorted(result['related']) == list(range(8))
    assert result['max_relative_difference'] <= .01
    assert result == match_batches(lengths, list(reversed(lengths)), seed=17, iterations=10000)


def test_large_total_budget_difference_is_rejected():
    with pytest.raises(ValueError, match='Total token'):
        match_batches([100]*8, [110]*8, seed=0)


def test_runtime_plan_rejects_budget_metadata_mismatch():
    p = dict(records=116, batch_size=4, lengths=dict(original=[10]*116, related=[10]*116))
    e = dict(original=list(range(116)), related=list(range(116)),
             original_supervised_tokens=[40]*29, related_supervised_tokens=[40]*29)
    p['epochs'] = [e for _ in range(4)]
    validate_plan(p, 'original')
    p['lengths']['related'][0] = 11
    with pytest.raises(AssertionError, match='metadata'):
        validate_plan(p, 'related')

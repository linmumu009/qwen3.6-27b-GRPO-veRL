from scripts.estimate_48k_capacity import CapacityInputs, estimate


def test_48k_capacity_fits_validated_topology_with_explicit_headroom():
    result = estimate(CapacityInputs())

    assert result["training"]["expected_to_fit"] is True
    assert result["training"]["planning_peak_gib"] < 45
    assert result["training"]["headroom_gib"] > 15

    rollout = result["rollout_per_tp_rank"]
    assert 0.8 < rollout["cache_per_48k_sequence"]["total_gib"] < 1.0
    assert 14 < rollout["cache_for_max_active_sequences_gib"] < 15
    assert rollout["expected_to_fit"] is True
    assert result["verdict"]["probe_sequence_tokens"] == [8192, 16384, 32768, 49152]


def test_capacity_rejects_unrealistically_small_hbm_budget():
    result = estimate(CapacityInputs(usable_hbm_gib=32.0))

    assert result["verdict"]["expected_to_fit"] is False

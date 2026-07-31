from pathlib import Path

import pytest

from scripts.analyze_trajectory_comparison import (
    describe,
    parse_driver_steps,
    pearson_correlation,
    summarize_timeout_bounds,
)


def test_describe_uses_interpolated_quantiles():
    summary = describe([1, 2, 3, 4])

    assert summary["n"] == 4
    assert summary["median"] == 2.5
    assert summary["p90"] == pytest.approx(3.7)
    assert summary["p95"] == pytest.approx(3.85)


def test_parse_driver_steps_extracts_rollout_tail_metrics(tmp_path: Path):
    driver_log = tmp_path / "driver.log"
    driver_log.write_text(
        "step:12 - training/global_step:12 - "
        "response_length/mean:np.float64(1792.3125) - "
        "response_length/max:np.float64(4096.0) - "
        "response_length/clip_ratio:np.float64(0.0625) - "
        "response/aborted_ratio:np.float64(0.0) - "
        "timing_s/agent_loop/generate_sequences/mean:np.float64(132.1419) - "
        "timing_s/agent_loop/generate_sequences/max:np.float64(440.2329) - "
        "timing_s/agent_loop/slowest/response_length:np.float64(4096.0)\n",
        encoding="utf-8",
    )

    steps = parse_driver_steps(driver_log)

    assert steps == [
        {
            "step": 12.0,
            "response_mean": 1792.3125,
            "response_max": 4096.0,
            "response_clip_ratio": 0.0625,
            "aborted_ratio": 0.0,
            "generate_mean_s": 132.1419,
            "generate_max_s": 440.2329,
            "slowest_response_tokens": 4096.0,
        }
    ]


def test_timeout_bounds_are_conservative_batch_level_lower_bounds():
    steps = [
        {"step": 1.0, "generate_max_s": 190.0},
        {"step": 2.0, "generate_max_s": 250.0},
        {"step": 3.0, "generate_max_s": 440.0},
    ]

    bounds = summarize_timeout_bounds(steps, [180, 300])

    assert bounds[0]["batches_with_at_least_one_timeout"] == 3
    assert bounds[0]["minimum_trajectories_affected"] == 3
    assert bounds[1]["batches_with_at_least_one_timeout"] == 1
    assert bounds[1]["exact_trajectories_affected"] is None
    assert pearson_correlation([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)

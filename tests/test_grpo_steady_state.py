from pathlib import Path

from scripts.analyze_grpo_steady_state import (
    parse_step_metrics,
    read_final_cache_counters,
)
from scripts.monitor_npu_utilization import StepTracker, parse_usage
from scripts.monitor_vllm_cache_metrics import find_endpoint, parse_metrics


def test_parse_npu_smi_two_chip_usage() -> None:
    text = """
    Aicore Usage Rate(%)            : 71
    HBM Usage Rate(%)               : 42
    HBM Bandwidth Usage Rate(%)     : 33
    NPU Utilization(%)              : 70
    Chip ID                         : 0
    Aicore Usage Rate(%)            : 69
    HBM Usage Rate(%)               : 43
    HBM Bandwidth Usage Rate(%)     : 31
    NPU Utilization(%)              : 68
    Chip ID                         : 1
    """
    assert parse_usage(text) == [
        {
            "aicore_pct": 71,
            "hbm_usage_pct": 42,
            "hbm_bandwidth_pct": 33,
            "npu_util_pct": 70,
            "chip": 0,
        },
        {
            "aicore_pct": 69,
            "hbm_usage_pct": 43,
            "hbm_bandwidth_pct": 31,
            "npu_util_pct": 68,
            "chip": 1,
        },
    ]


def test_step_tracker_reads_only_appended_metrics(tmp_path: Path) -> None:
    log = tmp_path / "driver.log"
    log.write_text("step:1 - training/global_step:1\n", encoding="utf-8")
    tracker = StepTracker(log)
    assert tracker.update() == 1
    with log.open("a", encoding="utf-8") as handle:
        handle.write("step:2 - training/global_step:2\n")
    assert tracker.update() == 2


def test_parse_numpy_wrapped_step_metrics() -> None:
    line = (
        "step:3 - training/global_step:3 "
        "- actor/loss:np.float64(-0.125) "
        "- timing_s/step:12.5"
    )
    assert parse_step_metrics(line) == [
        {
            "step": 3.0,
            "training/global_step": 3.0,
            "actor/loss": -0.125,
            "timing_s/step": 12.5,
        }
    ]


def test_find_vllm_endpoint_and_parse_cache_metrics(tmp_path: Path) -> None:
    log = tmp_path / "driver.log"
    log.write_text(
        "LLMServerManager: ['192.168.202.4:44865']\n",
        encoding="utf-8",
    )
    assert find_endpoint(log) == "192.168.202.4:44865"

    metrics = parse_metrics(
        """
# HELP vllm:prefix_cache_hits Prefix cache hits
vllm:prefix_cache_hits{engine="0"} 120
vllm:prefix_cache_queries{engine="0"} 200
vllm:num_requests_running{engine="0"} 4
vllm:prompt_tokens_total{engine="0"} 8192
"""
    )
    assert metrics == {
        'vllm:prefix_cache_hits{engine="0"}': 120.0,
        'vllm:prefix_cache_queries{engine="0"}': 200.0,
        'vllm:prompt_tokens_total{engine="0"}': 8192.0,
    }


def test_read_final_cache_counters_aggregates_both_engines(tmp_path: Path) -> None:
    cache_log = tmp_path / "cache.jsonl"
    cache_log.write_text(
        "\n".join(
            [
                '{"timestamp":"1","metrics":{}}',
                (
                    '{"timestamp":"2","metrics":{'
                    '"vllm:prefix_cache_hits_total{engine=\\"0\\"}":120,'
                    '"vllm:prefix_cache_queries_total{engine=\\"0\\"}":200,'
                    '"vllm:prefix_cache_hits_total{engine=\\"1\\"}":80,'
                    '"vllm:prefix_cache_queries_total{engine=\\"1\\"}":300,'
                    '"vllm:prefix_cache_hits_total_created{engine=\\"0\\"}":99'
                    "}}"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert read_final_cache_counters(cache_log) == {
        "engines": 2,
        "hits": 200,
        "queries": 500,
        "hit_rate_pct": 40.0,
    }

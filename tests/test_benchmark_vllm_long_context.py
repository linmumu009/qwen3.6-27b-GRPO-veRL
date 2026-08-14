import json
import statistics
from pathlib import Path

from scripts.benchmark_vllm_long_context import build_cases, summarize_rows


def test_build_cases_keeps_48k_and_rejects_64k_at_current_limit() -> None:
    cases = build_cases([2048, 49152, 65536], output_tokens=256, max_model_len=49152)

    assert cases[0].supported is True
    assert cases[0].prompt_tokens == 1792
    assert cases[1].supported is True
    assert cases[1].prompt_tokens == 48896
    assert cases[2].supported is False
    assert cases[2].reason == "exceeds_max_model_len"


def test_summarize_rows_uses_medians_and_preserves_capacity_boundary() -> None:
    rows = [
        {
            "target_total_tokens": 4096,
            "prompt_tokens": 3840,
            "output_tokens": 256,
            "status": "ok",
            "ttft_s": 2.0,
            "decode_s": 4.0,
            "total_s": 6.0,
            "decode_tps": 60.0,
            "e2e_tps": 40.0,
            "prompt_tokens_per_ttft_s": 1920.0,
        },
        {
            "target_total_tokens": 4096,
            "prompt_tokens": 3840,
            "output_tokens": 256,
            "status": "ok",
            "ttft_s": 4.0,
            "decode_s": 6.0,
            "total_s": 10.0,
            "decode_tps": 40.0,
            "e2e_tps": 25.6,
            "prompt_tokens_per_ttft_s": 960.0,
        },
        {
            "target_total_tokens": 65536,
            "prompt_tokens": 65280,
            "output_tokens": 256,
            "status": "unsupported",
            "reason": "exceeds_max_model_len",
        },
    ]

    summary = summarize_rows(rows)

    assert summary[0]["n"] == 2
    assert summary[0]["ttft_s_median"] == 3.0
    assert summary[0]["decode_tps_median"] == 50.0
    assert summary[1]["status"] == "unsupported"
    assert summary[1]["reason"] == "exceeds_max_model_len"


def test_recorded_step120_results_are_complete_and_stable() -> None:
    result_path = (
        Path(__file__).parents[1]
        / "docs"
        / "long_context_decode_benchmark_step120_20260814.json"
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    valid = [row for row in payload["rows"] if row["status"] == "ok"]
    unsupported = [
        row for row in payload["rows"] if row["status"] == "unsupported"
    ]

    assert payload["status"] == "complete"
    assert len(valid) == 14
    assert all(row["output_tokens"] == 256 for row in valid)
    assert unsupported == [
        {
            "output_tokens": 256,
            "prompt_tokens": 65280,
            "reason": "exceeds_max_model_len",
            "repeat": None,
            "status": "unsupported",
            "target_total_tokens": 65536,
        }
    ]

    by_target = {
        target: [
            row["decode_tps"]
            for row in valid
            if row["target_total_tokens"] == target
        ]
        for target in {row["target_total_tokens"] for row in valid}
    }
    assert all(len(values) == 2 for values in by_target.values())
    assert max(
        (max(values) - min(values)) / statistics.median(values)
        for values in by_target.values()
    ) < 0.02

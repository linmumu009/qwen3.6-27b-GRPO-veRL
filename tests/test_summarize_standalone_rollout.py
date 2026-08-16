import csv
import json
from pathlib import Path

from scripts.standalone_rollout_shards import shard_path, write_jsonl_atomic
from scripts.summarize_standalone_rollout import summarize


def test_summary_reports_shape_scheduler_lengths_and_npu_without_content(tmp_path: Path):
    (tmp_path / "standalone_contract.json").write_text(
        json.dumps({
            "tasks": 2,
            "samples_per_task": 2,
            "task_batch_size": 2,
            "max_response_tokens": 10,
            "max_num_seqs_per_dp_engine": 3,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "context_tokens": 49152,
            "trajectory_timeout_seconds": 900,
            "trajectory_telemetry": {
                "contract": "llin-pi-trajectory-telemetry-v1",
            },
        }),
        encoding="utf-8",
    )
    rows = [
        {"source_task_index": task, "sample_index": sample, "response_tokens": task + sample + 8, "runtime_error": False}
        for task in range(2)
        for sample in range(2)
    ]
    rows[0]["trajectory_timeout"] = True
    rows[0]["trajectory_abort_acknowledged_count"] = 1
    rows[0]["trajectory_abort_physical_request_count"] = 1
    rows[0]["trajectory_abort_error_count"] = 0
    for index, row in enumerate(rows):
        row.update({
            "trajectory_telemetry_contract": "llin-pi-trajectory-telemetry-v1",
            "trajectory_queue_wait_available": True,
            "trajectory_queue_wait_seconds": index + 0.5,
            "trajectory_generation_seconds": index + 4.0,
            "trajectory_tool_seconds": index + 1.0,
            "trajectory_execution_seconds": index + 6.0,
            "trajectory_total_seconds": index + 6.5,
            "trajectory_overhead_seconds": 1.0,
        })
    rows[0]["trajectory_timeout_partial_response_tokens"] = 321
    rows[0]["trajectory_timeout_partial_generation_tokens"] = 300
    write_jsonl_atomic(shard_path(tmp_path, 0, 2), rows)
    (tmp_path / "driver.log").write_text(
        "data.truncation=error\n"
        "Running: 3 reqs, Waiting: 2 reqs\nRunning: 2 reqs, Waiting: 0 reqs\n",
        encoding="utf-8",
    )
    with (tmp_path / "npu_utilization.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["aicore_pct", "npu_util_pct", "hbm_usage_pct", "hbm_bandwidth_pct"])
        writer.writeheader()
        writer.writerow({"aicore_pct": 80, "npu_util_pct": 90, "hbm_usage_pct": 70, "hbm_bandwidth_pct": 60})

    result = summarize(tmp_path)

    assert result["completed_rows"] == 4
    assert result["trajectory_timeout_rows"] == 1
    assert result["trajectory_timeout_seconds"] == 900
    assert result["timeout_abort_acknowledged_count"] == 1
    assert result["timeout_abort_physical_request_count"] == 1
    assert result["timeout_abort_error_count"] == 0
    assert result["scheduler"]["waiting_max"] == 2
    assert result["scheduler"]["running_latest"] == 2
    assert result["scheduler"]["waiting_latest"] == 0
    assert result["scheduler"]["at_sequence_cap_samples"] == 1
    assert result["response_tokens"]["at_budget_rows"] == 1
    telemetry = result["trajectory_telemetry"]
    assert telemetry["contract"] == "llin-pi-trajectory-telemetry-v1"
    assert telemetry["rows"] == 4
    assert telemetry["queue_wait_available_rows"] == 4
    assert telemetry["timing_seconds"]["generation"]["p50"] == 5.0
    assert telemetry["timing_seconds"]["tool_execution"]["max"] == 4.0
    assert telemetry["timing_seconds"]["total"]["count"] == 4
    assert telemetry["timeout_partial_response_tokens"]["max"] == 321
    assert telemetry["timeout_partial_generation_tokens"]["max"] == 300
    assert result["npu"]["aicore_mean"] == 80
    assert result["npu"]["recent_aicore_mean"] == 80
    assert result["driver_error_markers"]["context_or_truncation"] == 0
    assert result["contains_prompts_gold_sql_task_ids_tool_outputs_or_server_paths"] is False

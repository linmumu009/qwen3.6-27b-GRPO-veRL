import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.analyze_multisandbox_dwh_rollout import analyze


def test_analyze_scores_final_only_and_selects_only_mixed_groups(tmp_path: Path):
    dataset = tmp_path / "dataset.parquet"
    rows = []
    for index, expected in enumerate((10, 20)):
        rows.append({
            "prompt": [{"role": "user", "content": f"q{index}"}],
            "reward_model": {"ground_truth": {"answer_type": "numeric", "expected_value_json": str(expected)}},
            "extra_info": {
                "verifier_id": f"v:{index}",
                "instruction_sha256": f"hash{index}",
                "source_version": "v1",
            },
        })
    pq.write_table(pa.Table.from_pylist(rows), dataset)
    shards = tmp_path / "shards"
    shards.mkdir()
    outputs = [
        {"source_task_index": 0, "sample_index": 0, "output": "assistant\n最终答案 10", "response_tokens": 5, "runtime_error": False},
        {"source_task_index": 0, "sample_index": 1, "output": "assistant\n最终答案 9", "response_tokens": 6, "runtime_error": False},
        {"source_task_index": 1, "sample_index": 0, "output": "", "response_tokens": 0, "runtime_error": False, "trajectory_timeout": True, "trajectory_abort_acknowledged_count": 1, "trajectory_abort_physical_request_count": 1, "trajectory_abort_error_count": 0},
        {"source_task_index": 1, "sample_index": 1, "output": "assistant\n最终答案 0", "response_tokens": 8, "runtime_error": False},
    ]
    (shards / "tasks_00000_00002.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs),
        encoding="utf-8",
    )

    summary = analyze(dataset, shards, tmp_path / "analysis", expected_tasks=2, samples_per_task=2)

    assert summary["bucket_counts"] == {"mixed": 1, "timed_out": 1}
    assert summary["correct_trajectories"] == 1
    assert summary["mixed_screening_rows"] == 1
    assert summary["timeout_trajectories"] == 1
    assert summary["evaluable_trajectories"] == 3
    assert summary["timeout_abort_acknowledged_count"] == 1
    assert summary["timeout_abort_physical_request_count"] == 1
    assert summary["timeout_abort_error_count"] == 0
    selected = pq.read_table(tmp_path / "analysis" / "mixed_groups.sensitive.parquet").to_pylist()
    assert len(selected) == 1
    assert selected[0]["extra_info"]["verifier_id"] == "v:0"
    assert summary["training_allowed"] is False

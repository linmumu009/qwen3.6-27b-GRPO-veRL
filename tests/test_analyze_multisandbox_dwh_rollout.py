import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.analyze_multisandbox_dwh_rollout import analyze


def write_dataset(path: Path, expected_values: tuple[int, ...]) -> None:
    rows = []
    for index, expected in enumerate(expected_values):
        rows.append({
            "prompt": [{"role": "user", "content": f"q{index}"}],
            "reward_model": {
                "ground_truth": {
                    "answer_type": "numeric",
                    "expected_value_json": str(expected),
                    "verification_sql": f"select {expected}",
                }
            },
            "extra_info": {
                "verifier_id": f"v:{index}",
                "instruction_sha256": f"hash{index}",
                "source_version": "v1",
                "difficulty_level": index + 1,
                "training_allowed": False,
            },
        })
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_analyze_scores_final_only_and_selects_only_mixed_groups(tmp_path: Path):
    dataset = tmp_path / "dataset.parquet"
    write_dataset(dataset, (10, 20))
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
    assert summary["difficulty_correct_count_histogram"]["1"]["1"] == 1
    assert summary["difficulty_bucket_counts"]["2"] == {"timed_out": 1}
    assert summary["response_token_distribution"] == {
        "count": 4,
        "mean": 4.75,
        "p50": 5,
        "p90": 8,
        "p95": 8,
        "p99": 8,
        "max": 8,
    }
    assert summary["response_token_histogram"] == {"0": 1, "5": 1, "6": 1, "8": 1}
    assert summary["response_token_histogram_by_difficulty"]["1"] == {"5": 1, "6": 1}
    review_rows = [
        json.loads(line)
        for line in (tmp_path / "analysis" / "mixed_review_queue.sensitive.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(review_rows) == 1
    assert review_rows[0]["review_decision"]["verdict"] == "needs_review"
    assert [row["final_answer"] for row in review_rows[0]["trajectory_final_answers"]] == [
        "最终答案 10",
        "最终答案 9",
    ]
    assert "output" not in review_rows[0]["trajectory_final_answers"][0]


def test_partial_analysis_uses_only_complete_shards_and_fails_closed_on_runtime_error(
    tmp_path: Path,
):
    dataset = tmp_path / "dataset.parquet"
    write_dataset(dataset, (10, 20, 30))
    shards = tmp_path / "shards"
    shards.mkdir()
    outputs = [
        {
            "source_task_index": 0,
            "sample_index": 0,
            "output": "assistant\n最终答案 10",
            "response_tokens": 5,
            "runtime_error": False,
        },
        {
            "source_task_index": 0,
            "sample_index": 1,
            "output": "assistant\n最终答案 9",
            "response_tokens": 6,
            "runtime_error": True,
        },
        {
            "source_task_index": 1,
            "sample_index": 0,
            "output": "assistant\n最终答案 20",
            "response_tokens": 5,
            "runtime_error": False,
        },
        {
            "source_task_index": 1,
            "sample_index": 1,
            "output": "assistant\n最终答案 19",
            "response_tokens": 6,
            "runtime_error": False,
        },
    ]
    (shards / "tasks_00000_00002.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs),
        encoding="utf-8",
    )

    summary = analyze(
        dataset,
        shards,
        tmp_path / "analysis",
        expected_tasks=3,
        samples_per_task=2,
        allow_partial=True,
    )

    assert summary["partial"] is True
    assert summary["tasks"] == 2
    assert summary["expected_tasks"] == 3
    assert summary["complete_shard_ranges"] == [{"start": 0, "stop": 2}]
    assert summary["bucket_counts"] == {"mixed": 1, "runtime_error": 1}
    assert summary["mixed_screening_rows"] == 1
    assert summary["runtime_error_or_timeout_fail_closed"] is True
    selected = pq.read_table(tmp_path / "analysis" / "mixed_groups.sensitive.parquet").to_pylist()
    assert [row["extra_info"]["verifier_id"] for row in selected] == ["v:1"]


def test_strict_table_analysis_rejects_number_dump_and_keeps_timeout_out_of_wrong_count(
    tmp_path: Path,
):
    dataset = tmp_path / "dataset.parquet"
    task = {
        "prompt": [{"role": "user", "content": "q"}],
        "reward_model": {
            "ground_truth": {
                "answer_type": "table",
                "expected_value_json": json.dumps(
                    [{"category": "A", "value": 10}, {"category": "B", "value": 20}]
                ),
                "verification_sql": "select category, value from t",
                "abs_tol": 1e-3,
                "rel_tol": 1e-5,
            }
        },
        "extra_info": {
            "verifier_id": "v:strict",
            "instruction_sha256": "strict-hash",
            "source_version": "v23",
            "difficulty_level": 3,
            "training_allowed": False,
        },
    }
    pq.write_table(pa.Table.from_pylist([task]), dataset)
    shards = tmp_path / "shards"
    shards.mkdir()
    outputs = [
        {
            "source_task_index": 0,
            "sample_index": 0,
            "output": "assistant\n|类别|数值|\n|---|---:|\n|A|10|\n|B|20|",
            "response_tokens": 20,
            "runtime_error": False,
        },
        {
            "source_task_index": 0,
            "sample_index": 1,
            "output": "assistant\n10, 20",
            "response_tokens": 2,
            "runtime_error": False,
        },
    ]
    (shards / "tasks_00000_00001.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs),
        encoding="utf-8",
    )

    summary = analyze(
        dataset,
        shards,
        tmp_path / "analysis",
        expected_tasks=1,
        samples_per_task=2,
        strict_table=True,
    )

    assert summary["outcome_contract"] == "banded-v2-strict-table-v1"
    assert summary["bucket_counts"] == {"mixed": 1}
    result = json.loads(
        (tmp_path / "analysis" / "per_task.sensitive.jsonl").read_text(encoding="utf-8")
    )
    assert result["correct_count"] == 1
    assert result["completed_count"] == 2
    assert result["outcome_contract"] == "banded-v2-strict-table-v1"

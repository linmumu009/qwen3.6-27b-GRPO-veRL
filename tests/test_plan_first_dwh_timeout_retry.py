import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.analyze_multisandbox_dwh_rollout import analyze
from scripts.plan_first_dwh_timeout_retry import (
    MAX_CONTEXT_TOKENS,
    MAX_RESPONSE_TOKENS,
    TIMEOUT_SECONDS,
    merge_retries,
    prepare_retry_dataset,
)


def dataset_row(index: int) -> dict:
    return {
        "prompt": [{"role": "user", "content": f"prompt {index}"}],
        "reward_model": {
            "ground_truth": {
                "answer_type": "numeric",
                "expected_value_json": str(index + 10),
            }
        },
        "extra_info": {
            "source_version": "test",
            "verifier_id": f"test:{index}",
            "instruction_sha256": f"instruction-{index}",
            "comparison_split": "training_candidate",
            "difficulty_band": 1,
            "training_allowed": False,
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def original_row(task: int, sample: int, *, timeout: bool = False) -> dict:
    return {
        "source_task_index": task,
        "sample_index": sample,
        "output": "" if timeout else f"assistant\n最终：{task + 10}",
        "response_tokens": 0 if timeout else 5,
        "trajectory_timeout": timeout,
        "runtime_error": False,
    }


def test_prepare_and_merge_retry_replaces_only_exact_timeout_slot(tmp_path: Path):
    dataset = tmp_path / "dataset.parquet"
    pq.write_table(pa.Table.from_pylist([dataset_row(0), dataset_row(1)]), dataset)
    original_shards = tmp_path / "original"
    write_jsonl(
        original_shards / "tasks_00000_00002.jsonl",
        [
            original_row(0, 0),
            original_row(0, 1, timeout=True),
            original_row(1, 0),
            original_row(1, 1),
        ],
    )
    retry_dataset = tmp_path / "retry.parquet"
    manifest = prepare_retry_dataset(
        dataset,
        original_shards,
        retry_dataset,
        tmp_path / "manifest.json",
        arm_label="native",
        expected_tasks=2,
        samples_per_task=2,
        expected_timeouts=1,
    )
    assert manifest["retry_rows"] == 1
    assert manifest["timeout_seconds"] == TIMEOUT_SECONDS == 1800
    assert manifest["max_response_tokens"] == MAX_RESPONSE_TOKENS == 90112
    assert manifest["max_context_tokens"] == MAX_CONTEXT_TOKENS == 94208
    retry = pq.read_table(retry_dataset).to_pylist()[0]
    assert retry["extra_info"]["retry_original_task_index"] == 0
    assert retry["extra_info"]["retry_original_sample_index"] == 1

    retry_shards = tmp_path / "retry_shards"
    write_jsonl(
        retry_shards / "tasks_00000_00001.jsonl",
        [
            {
                "source_task_index": 0,
                "sample_index": 0,
                "output": "assistant\n最终：10",
                "response_tokens": 8,
                "trajectory_timeout": False,
                "runtime_error": False,
            }
        ],
    )
    reconciled = tmp_path / "reconciled"
    merged = merge_retries(
        original_shards,
        retry_dataset,
        retry_shards,
        reconciled,
        expected_tasks=2,
        samples_per_task=2,
    )
    assert merged["resolved_timeout_slots"] == 1
    assert merged["remaining_timeout_slots"] == 0
    rows = [
        json.loads(line)
        for line in (reconciled / "shards" / "tasks_00000_00002.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [(row["source_task_index"], row["sample_index"]) for row in rows] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert rows[1]["replaced_original_timeout"] is True
    assert rows[0].get("replaced_original_timeout") is None
    outcome = analyze(
        dataset,
        reconciled / "shards",
        reconciled / "outcomes",
        expected_tasks=2,
        samples_per_task=2,
    )
    assert outcome["timeout_trajectories"] == 0
    assert outcome["correct_trajectories"] == 4


def test_prepare_rejects_wrong_expected_timeout_count(tmp_path: Path):
    dataset = tmp_path / "dataset.parquet"
    pq.write_table(pa.Table.from_pylist([dataset_row(0)]), dataset)
    shards = tmp_path / "shards"
    write_jsonl(
        shards / "tasks_00000_00001.jsonl",
        [original_row(0, 0, timeout=True), original_row(0, 1)],
    )
    try:
        prepare_retry_dataset(
            dataset,
            shards,
            tmp_path / "retry.parquet",
            tmp_path / "manifest.json",
            arm_label="native",
            expected_tasks=1,
            samples_per_task=2,
            expected_timeouts=2,
        )
    except ValueError as exc:
        assert "expected 2 timeouts, got 1" in str(exc)
    else:
        raise AssertionError("expected timeout count mismatch")

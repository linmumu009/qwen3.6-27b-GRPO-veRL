import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.normalize_runtime_parity_outputs import (
    messages_to_solution,
    normalize_pi,
    normalize_verl,
    pi_api_error_count,
    pi_error_summary,
)


def test_messages_to_solution_preserves_final_assistant_boundary():
    solution = messages_to_solution(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "wrong 9", "tool_calls": [{"id": "x"}]},
            {"role": "tool", "content": "10"},
            {"role": "assistant", "content": "final 10"},
        ]
    )
    assert solution.endswith("assistant\nfinal 10")


def test_normalize_verl_recomputes_outcome_and_sample_indexes(tmp_path: Path):
    prompt = [{"role": "system", "content": "sys"}, {"role": "user", "content": "question"}]
    dataset = tmp_path / "data.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "prompt": prompt,
                    "reward_model": {
                        "ground_truth": {
                            "answer_type": "numeric",
                            "expected_value_json": "10",
                        }
                    },
                }
            ]
        ),
        dataset,
    )
    validation = tmp_path / "0.jsonl"
    validation.write_text(
        "\n".join(
            json.dumps(
                {
                    "input": "user\nquestion\nassistant\n",
                    "output": output,
                    "runtime_error": index == 1,
                }
            )
            for index, output in enumerate(("assistant\nfinal 10", "assistant\nfinal 9"))
        )
        + "\n",
        encoding="utf-8",
    )
    rows = normalize_verl(dataset, validation)
    assert [row["sample_index"] for row in rows] == [0, 1]
    assert [row["final_answer_correct"] for row in rows] == [1, 0]
    assert [row["runtime_error"] for row in rows] == [False, True]


def test_pi_api_errors_are_detected_without_reading_visible_content(tmp_path: Path):
    trajectory = tmp_path / "trajectory.jsonl"
    trajectory.write_text(
        json.dumps(
            {
                "type": "message_end",
                "message": {"role": "assistant", "stopReason": "error", "content": []},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert pi_api_error_count(trajectory) == 1


def test_pi_recovered_context_overflow_is_audited_but_not_fatal(tmp_path: Path):
    trajectory = tmp_path / "trajectory.jsonl"
    events = [
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "stopReason": "error",
                "errorMessage": "This model's maximum context length is 49152 tokens.",
                "content": [],
            },
        },
        {
            "type": "compaction_end",
            "reason": "overflow",
            "aborted": False,
            "willRetry": True,
            "result": {"estimatedTokensAfter": 12345},
        },
    ]
    trajectory.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    assert pi_error_summary(trajectory) == {
        "assistant_error_events": 1,
        "context_overflow_events": 1,
        "recovered_context_overflows": 1,
        "fatal_api_errors": 0,
    }
    assert pi_api_error_count(trajectory) == 0


def test_pi_unrecovered_context_overflow_remains_fatal(tmp_path: Path):
    trajectory = tmp_path / "trajectory.jsonl"
    trajectory.write_text(
        json.dumps(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "stopReason": "error",
                    "errorMessage": "maximum context length is 49152 tokens",
                    "content": [],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert pi_api_error_count(trajectory) == 1


def test_pi_transient_non_overflow_error_followed_by_success_is_not_fatal(tmp_path: Path):
    trajectory = tmp_path / "trajectory.jsonl"
    events = [
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "stopReason": "error",
                "errorMessage": "transient provider failure",
                "content": [],
            },
        },
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "stopReason": "stop",
                "content": [{"type": "text", "text": "final answer"}],
            },
        },
    ]
    trajectory.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    assert pi_error_summary(trajectory)["assistant_error_events"] == 1
    assert pi_api_error_count(trajectory) == 0


def test_pi_gnu_timeout_exit124_is_marked_as_timeout(tmp_path: Path):
    prompt = [{"role": "system", "content": "sys"}, {"role": "user", "content": "question"}]
    dataset = tmp_path / "data.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "prompt": prompt,
                    "reward_model": {
                        "ground_truth": {
                            "answer_type": "numeric",
                            "expected_value_json": "10",
                        }
                    },
                }
            ]
        ),
        dataset,
    )
    from scripts.normalize_runtime_parity_outputs import task_key

    key = task_key(prompt)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "task_key": key,
                "sample_index": 0,
                "status": "exit124",
                "duration_seconds": 900,
                "trajectory_file": "timed-out.jsonl",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    converter = tmp_path / "converter.py"
    converter.write_text("def parse_trajectory(path, unused):\n    return {'messages': []}\n", encoding="utf-8")

    result = normalize_pi(dataset, manifest, tmp_path, converter)

    assert result[0]["timeout"] is True
    assert result[0]["runtime_error"] is True

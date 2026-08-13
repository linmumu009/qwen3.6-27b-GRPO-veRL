import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.normalize_runtime_parity_outputs import (
    messages_to_solution,
    normalize_verl,
    pi_api_error_count,
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
            json.dumps({"input": "user\nquestion\nassistant\n", "output": output})
            for output in ("assistant\nfinal 10", "assistant\nfinal 9")
        )
        + "\n",
        encoding="utf-8",
    )
    rows = normalize_verl(dataset, validation)
    assert [row["sample_index"] for row in rows] == [0, 1]
    assert [row["final_answer_correct"] for row in rows] == [1, 0]
    assert all(row["runtime_error"] is False for row in rows)


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

from __future__ import annotations

from pathlib import Path
import sqlite3

from scripts.prepare_state_conditioned_repair_sft import build_state_conditioned_row
from scripts.teacher_forced_component_masks import (
    assistant_mask_from_ranges,
    normalize_assistant_turn_indices,
)


def _database(path: Path) -> Path:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE shipments(category TEXT, amount INTEGER)")
    connection.executemany(
        "INSERT INTO shipments VALUES (?, ?)",
        [("A", 1), ("A", 2), ("B", 10)],
    )
    connection.commit()
    connection.close()
    return path


def test_selective_assistant_mask_excludes_error_context_turn():
    ranges = [(2, 5), (8, 12), (15, 18)]
    selected = normalize_assistant_turn_indices([1, 2], len(ranges))
    mask = assistant_mask_from_ranges(20, ranges, selected)

    assert sum(mask[2:5]) == 0
    assert sum(mask[8:12]) == 4
    assert sum(mask[15:18]) == 3


def test_state_conditioned_row_uses_observed_wrong_result_as_zero_loss_context(
    tmp_path: Path,
):
    database = _database(tmp_path / "logistics.sqlite")
    correction_command = (
        "sqlite3 -json /workspace/logistics.sqlite "
        "\"SELECT SUM(amount) AS value FROM shipments WHERE category = 'A'\""
    )
    teacher = {
        "task_id": "task_000001",
        "sample_id": "repair-sft-task_000001",
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "sum A"},
            {
                "role": "assistant",
                "content": "query",
                "tool_calls": [
                    {
                        "id": "teacher",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": {"command": correction_command},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "teacher", "content": '[{"value":3}]'},
            {"role": "assistant", "content": "3"},
        ],
    }
    wrong_command = (
        "sqlite3 -json /workspace/logistics.sqlite "
        "\"SELECT SUM(amount) AS value FROM shipments\""
    )
    rollout = [
        {
            "role": "assistant",
            "content": {"analysis": "try all rows"},
            "tool_calls": [
                {
                    "id": "wrong",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": {"command": wrong_command},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "wrong", "content": '[{"value":13}]'},
    ]
    truth = {
        "answer_type": "numeric",
        "expected": 3,
        "verification_sql": "SELECT SUM(amount) AS value FROM shipments WHERE category = 'A'",
    }

    row, evidence = build_state_conditioned_row(
        teacher=teacher,
        rollout_messages=rollout,
        truth=truth,
        database=database,
    )

    assert [message["role"] for message in row["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert row["supervised_assistant_turn_indices"] == [1, 2]
    assert row["error_context_assistant_turn_index"] == 0
    assert row["messages"][3]["content"] == '[{"value":13}]'
    assert row["messages"][2]["content"] == '{"analysis":"try all rows"}'
    assert row["messages"][4]["tool_calls"][0]["id"] == "call_recover_000001"
    assert row["messages"][5]["tool_call_id"] == "call_recover_000001"
    assert evidence["correction_verified_or_equivalent"] is True

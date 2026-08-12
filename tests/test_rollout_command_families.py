from scripts.analyze_rollout_command_families import command_family, summarize


def call(call_id: str, name: str, arguments: dict[str, str]) -> dict:
    return {"id": call_id, "function": {"name": name, "arguments": arguments}}


def test_command_family_distinguishes_sqlite_and_alternatives() -> None:
    assert (
        command_family("sqlite3 -json /workspace/logistics.sqlite 'SELECT 1'")
        == "recognized_readonly_sqlite"
    )
    assert command_family("python3 -c 'print(1)'") == "python"
    assert command_family("sqlite3 /workspace/logistics.sqlite 'DELETE FROM x'") == (
        "sqlite_unparsed_or_non_readonly"
    )
    assert command_family("cd /workspace && find . -type f") == "find"


def test_summary_is_payload_free_and_counts_observation() -> None:
    rows = {
        "task_1": [
            {
                "role": "assistant",
                "tool_calls": [
                    call(
                        "sql",
                        "bash",
                        {"command": "sqlite3 /workspace/logistics.sqlite 'SELECT 1'"},
                    ),
                    call("read", "read", {"path": "/workspace/schema.txt"}),
                ],
            },
            {"role": "tool", "tool_call_id": "sql", "content": "1"},
        ],
        "task_2": [
            {
                "role": "assistant",
                "tool_calls": [call("py", "bash", {"command": "python3 -c 'print(1)'"})],
            }
        ],
    }
    result = summarize(rows)
    assert result["rows"] == 2
    assert result["tool_call_name_counts"] == {"bash": 2, "read": 1}
    assert result["bash_command_family_counts"] == {
        "python": 1,
        "recognized_readonly_sqlite": 1,
    }
    assert result["rows_with_recognized_readonly_sqlite"] == 1
    assert result["observed_tool_calls"] == 1
    assert result["unobserved_tool_calls"] == 2
    assert not result["contains_raw_commands_sql_prompts_or_tool_outputs"]


def test_summary_accepts_serialized_tool_arguments() -> None:
    result = summarize(
        {
            "task_1": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        call(
                            "sql",
                            "bash",
                            '{"command":"sqlite3 /workspace/logistics.sqlite \'SELECT 1\'"}',
                        )
                    ],
                }
            ]
        }
    )
    assert result["rows_with_recognized_readonly_sqlite"] == 1


def test_reused_call_ids_are_scoped_per_row() -> None:
    result = summarize(
        {
            "task_1": [
                {
                    "role": "assistant",
                    "tool_calls": [call("shared", "read", {"path": "a"})],
                },
                {"role": "tool", "tool_call_id": "shared", "content": "a"},
            ],
            "task_2": [
                {
                    "role": "assistant",
                    "tool_calls": [call("shared", "read", {"path": "b"})],
                }
            ],
        }
    )
    assert result["observed_tool_calls"] == 1
    assert result["unobserved_tool_calls"] == 1
    assert result["rows_with_unobserved_tool_calls"] == 1

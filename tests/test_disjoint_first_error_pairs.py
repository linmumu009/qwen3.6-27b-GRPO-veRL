import sqlite3
from pathlib import Path

from scripts.analyze_repair_sft_free_run_divergence import sql_from_command
from scripts.prepare_disjoint_first_error_pairs import build_pairs


def make_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute("create table metric(id integer, amount integer)")
    connection.executemany("insert into metric values (?, ?)", [(1, 5), (2, 15)])
    connection.commit()
    connection.close()


def replay_row(task: str) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": "boss-system"},
            {"role": "user", "content": "统计总额"},
        ],
        "reward_model": {
            "ground_truth": {
                "task_id": task,
                "environment_id": "sft/v15",
                "answer_type": "numeric",
                "expected_value_json": "20",
                "verification_sql": "SELECT SUM(amount) FROM metric",
                "task_family": "dwh",
            }
        },
    }


def rollout(task: str, sql: str, output: str) -> list[dict]:
    call_id = f"call_{task}"
    return [
        {"role": "system", "content": "boss-system"},
        {"role": "user", "content": "统计总额"},
        {
            "role": "assistant",
            "content": "查询",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": {
                            "command": f"sqlite3 -json /workspace/logistics.sqlite '{sql}'"
                        },
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": output},
        {"role": "assistant", "content": "完成"},
    ]


def boss_contract() -> dict:
    return {
        "system_prompt": "boss-system",
        "tools": [
            {
                "type": "function",
                "function": {"name": "bash", "description": "run", "parameters": {}},
            }
        ],
    }


def test_builds_actual_wrong_vs_verified_pair_at_identical_observed_state(tmp_path):
    database = tmp_path / "logistics.sqlite"
    make_database(database)
    task = "task_000001"
    wrong_sql = "SELECT amount FROM metric ORDER BY id LIMIT 1"

    rows, evidence, exclusions = build_pairs(
        replay_rows=[replay_row(task)],
        rollout_messages={task: rollout(task, wrong_sql, '[{"amount":5}]')},
        database=database,
        boss_contract=boss_contract(),
        minimum_pairs=1,
    )

    assert len(rows) == 2
    assert len(evidence) == 1
    assert exclusions == {}
    chosen, rejected = rows
    assert chosen["messages"][:4] == rejected["messages"][:4]
    assert chosen["messages"][3]["content"] == '[{"amount":5}]'
    chosen_sql = sql_from_command(
        chosen["messages"][4]["tool_calls"][0]["function"]["arguments"]["command"]
    )
    rejected_sql = sql_from_command(
        rejected["messages"][4]["tool_calls"][0]["function"]["arguments"]["command"]
    )
    assert chosen_sql == "SELECT SUM(amount) FROM metric"
    assert rejected_sql == wrong_sql
    assert rows[0]["candidate_label"] == "chosen"
    assert rows[1]["candidate_label"] == "rejected"


def test_excludes_correct_or_equivalent_first_query_and_no_query(tmp_path):
    database = tmp_path / "logistics.sqlite"
    make_database(database)
    correct = "task_correct"
    missing = "task_missing"
    no_query = [
        {"role": "system", "content": "boss-system"},
        {"role": "user", "content": "统计总额"},
        {"role": "assistant", "content": "不知道"},
    ]

    rows, evidence, exclusions = build_pairs(
        replay_rows=[replay_row(correct), replay_row(missing)],
        rollout_messages={
            correct: rollout(correct, "SELECT SUM(amount) FROM metric", '[{"SUM(amount)":20}]'),
            missing: no_query,
        },
        database=database,
        boss_contract=boss_contract(),
        minimum_pairs=1,
    )

    assert rows == []
    assert evidence == []
    assert exclusions == {
        "first_query_correct_or_equivalent": 1,
        "no_readonly_query": 1,
    }


def test_excludes_generated_sql_without_observed_tool_result(tmp_path):
    database = tmp_path / "logistics.sqlite"
    make_database(database)
    task = "task_unobserved"
    messages = rollout(task, "SELECT amount FROM metric LIMIT 1", '[{"amount":5}]')
    messages.pop(3)

    rows, evidence, exclusions = build_pairs(
        replay_rows=[replay_row(task)],
        rollout_messages={task: messages},
        database=database,
        boss_contract=boss_contract(),
        minimum_pairs=1,
    )

    assert rows == []
    assert evidence == []
    assert exclusions == {"first_readonly_query_without_observed_tool_result": 1}

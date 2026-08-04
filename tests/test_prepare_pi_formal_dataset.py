import sqlite3
from pathlib import Path

from scripts.prepare_pi_formal_dataset import (
    SYSTEM_PROMPT,
    build_training_record,
    select_split,
    validate_candidates,
)


def make_environment(root: Path, version: str) -> None:
    database = root / "sft" / version / "logistics.sqlite"
    database.parent.mkdir(parents=True)
    connection = sqlite3.connect(database)
    connection.execute("create table fact_metric(category text, value real)")
    connection.executemany("insert into fact_metric values (?, ?)", [("华东", 12.5), ("华南", 8.0)])
    connection.commit()
    connection.close()


def manifest_row(version: str, task_id: str, instruction: str, expected: float = 20.5) -> dict:
    return {
        "task_id": task_id,
        "v": version,
        "type": "dwh",
        "instruction": instruction,
        "expected_tables": ["fact_metric"],
        "gold_answer": {
            "answer_type": "numeric",
            "value": expected,
            "verification_sql": "SELECT SUM(value) FROM fact_metric",
        },
    }


def test_validate_candidates_executes_gold_sql_and_rejects_bad_label(tmp_path):
    make_environment(tmp_path, "v1")
    rows = [
        manifest_row("v1", "good", "正确任务"),
        manifest_row("v1", "bad", "错误标签", 999.0),
    ]

    valid, rejected = validate_candidates(rows, tmp_path, Path("manifest.jsonl"))

    assert [row["task_id"] for row in valid] == ["good"]
    assert rejected["gold_result_mismatch"] == 1


def test_split_selection_blocks_task_and_instruction_leakage():
    candidates = [
        {"verifier_id": f"sft/v:{i}", "task_id": str(i), "instruction_sha256": f"h{i}"}
        for i in range(4)
    ]
    blocked_ids = {"0"}
    blocked_hashes = {"h1"}

    selected = select_split(candidates, 2, "seed", blocked_ids, blocked_hashes)

    assert {row["task_id"] for row in selected} == {"2", "3"}
    assert blocked_ids == {"0", "2", "3"}


def test_training_record_uses_full_pi_contract():
    verifier = {
        "verifier_id": "sft/v:task",
        "task_id": "task",
        "environment_id": "sft/v",
        "instruction": "问题",
        "instruction_sha256": "abc",
        "required_tables": ["fact_metric"],
        "gold": {
            "answer_type": "numeric",
            "value": 20.5,
            "verification_sql": "SELECT SUM(value) FROM fact_metric",
            "abs_tol": 1e-3,
            "rel_tol": 1e-5,
        },
    }

    record = build_training_record(verifier, "train", 7)

    assert record["agent_name"] == "pi_agent"
    assert record["prompt"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert record["extra_info"]["tool_selection"] == ["bash", "read", "write", "edit"]
    assert set(record["extra_info"]["tools_kwargs"]) == {"bash", "read", "write", "edit"}
    assert record["reward_model"]["ground_truth"]["verification_sql"].startswith("SELECT")
    assert record["reward_model"]["ground_truth"]["expected_value_json"] == "20.5"


def test_source_system_prompt_is_preserved_and_fallback_names_workspace(tmp_path):
    make_environment(tmp_path, "v1")
    source = manifest_row("v1", "source", "任务")
    source["system_prompt"] = "老板原始 system"
    fallback = manifest_row("v1", "fallback", "另一任务")

    values, rejected = validate_candidates([source, fallback], tmp_path, Path("manifest.jsonl"))

    assert not rejected
    assert values[0]["system_prompt"] == "老板原始 system"
    assert values[0]["system_prompt_source"] == "source"
    assert values[1]["system_prompt"] == SYSTEM_PROMPT
    assert values[1]["system_prompt_source"] == "fallback"
    assert "/workspace/logistics.sqlite" in SYSTEM_PROMPT

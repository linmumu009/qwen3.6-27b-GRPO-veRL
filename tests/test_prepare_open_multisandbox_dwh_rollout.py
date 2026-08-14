import json
from pathlib import Path
import sqlite3

import pyarrow.parquet as pq

from scripts.prepare_open_multisandbox_dwh_rollout import prepare


def make_source(path: Path) -> None:
    path.mkdir()
    database = path / "logistics.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE fact_metric(category TEXT, value REAL)")
    connection.executemany("INSERT INTO fact_metric VALUES (?, ?)", [("A", 2.0), ("B", 1.0)])
    connection.commit()
    connection.close()
    (path / "schema_dictionary.md").write_text("fact_metric(category, value)\n", encoding="utf-8")
    tasks = []
    for level in range(1, 6):
        for offset in range(100):
            instruction = f"角色{level}查看对象{offset}，按数值高到低给出前5项。"
            value = [{"category": "A", "value": 2.0}, {"category": "B", "value": 1.0}]
            sql = f"SELECT category, value FROM fact_metric ORDER BY value DESC -- {level}-{offset}"
            tasks.append({
                "task_id": f"t{level}_{offset}",
                "task_type": f"family_{level}",
                "source_sandbox_version": "v15",
                "difficulty_level": level,
                "natural_language_instruction": instruction,
                "instruction_role": f"role_{level}",
                "semantic_anchors": [f"对象{offset}"],
                "expected_tables": ["fact_metric"],
                "gold_answer": {
                    "answer_type": "table",
                    "value": value,
                    "verification_sql": sql,
                },
                "query_plan": {"feature_counts": {"essential_joins": 2, "evidence_steps": level, "derived_metrics": 0, "temporal_comparisons": 0, "business_openness": level}},
                "validation": {"result_sha256": __import__("scripts.audit_open_multisandbox_dwh", fromlist=["canonical_hash"]).canonical_hash(value)},
                "instruction_generation": {"semantic_validation_passed": True},
                "generation_contract": "source-v1",
                "training_allowed": False,
            })
    (path / "dwh_tasks.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in tasks), encoding="utf-8")
    (path / "generation_summary.json").write_text(json.dumps({"contract": "source-v1"}), encoding="utf-8")


def test_prepare_builds_balanced_private_partitions(tmp_path: Path) -> None:
    source = tmp_path / "source"
    make_source(source)
    tasks_path = source / "dwh_tasks.jsonl"
    tasks = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines()]
    tasks[0]["gold_answer"]["value"][0]["value"] = 2.01
    tasks[0]["validation"]["result_sha256"] = __import__(
        "scripts.audit_open_multisandbox_dwh", fromlist=["canonical_hash"]
    ).canonical_hash(tasks[0]["gold_answer"]["value"])
    tasks_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in tasks),
        encoding="utf-8",
    )
    manifest = prepare(source, tmp_path / "sandboxes", tmp_path / "out", seed="s")
    assert manifest["tasks"] == 500
    assert manifest["partition_tasks"] == {"m05": 250, "m06": 250}
    assert manifest["runtime_gold_replay"]["adjusted_tasks"] == 1
    assert manifest["runtime_gold_replay"]["maximum_abs_diff"] < 0.011
    assert manifest["partition_difficulty_level_counts"]["m05"] == {str(level): 50 for level in range(1, 6)}
    assert manifest["partition_difficulty_level_counts"]["m06"] == {str(level): 50 for level in range(1, 6)}
    first = pq.read_table(tmp_path / "out" / "open_multisandbox_dwh_m05.sensitive.parquet").to_pylist()
    second = pq.read_table(tmp_path / "out" / "open_multisandbox_dwh_m06.sensitive.parquet").to_pylist()
    assert len(first) == len(second) == 250
    assert {row["extra_info"]["global_index"] for row in first}.isdisjoint(
        {row["extra_info"]["global_index"] for row in second}
    )
    assert all(row["extra_info"]["training_allowed"] is False for row in first + second)
    assert all(
        json.loads(row["reward_model"]["ground_truth"]["expected_value_json"])[0]["value"] == 2.0
        for row in first + second
    )
    runtime = tmp_path / "sandboxes" / "sft" / "source_runtime"
    assert sorted(path.name for path in runtime.iterdir()) == ["documents", "logistics.sqlite", "schema_dictionary.md"]

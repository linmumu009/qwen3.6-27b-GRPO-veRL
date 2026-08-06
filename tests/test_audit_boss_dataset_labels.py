import json
import sqlite3

from scripts.audit_boss_dataset_labels import audit_rows


def test_audit_separates_sql_self_consistency_from_semantic_quality(tmp_path):
    database = tmp_path / "sft" / "v1" / "logistics.sqlite"
    database.parent.mkdir(parents=True)
    connection = sqlite3.connect(database)
    connection.execute("create table metric(value real)")
    connection.execute("insert into metric values (3.5)")
    connection.commit()
    connection.close()
    row = {
        "prompt": [{"role": "system", "content": "s"}, {"role": "user", "content": "查询最新指标"}],
        "reward_model": {
            "ground_truth": {
                "verifier_id": "v1:task_1",
                "task_id": "task_1",
                "environment_id": "sft/v1",
                "answer_type": "numeric",
                "expected_value_json": json.dumps(3.5),
                "verification_sql": "SELECT SUM(value) FROM metric",
            }
        },
        "extra_info": {"instruction_sha256": "i1", "gold_sha256": "g1"},
    }
    review = {
        "task_id": "task_1",
        "instruction": "查询最新指标",
        "gold": {"answer_type": "numeric", "verification_sql": "SELECT SUM(value) FROM metric"},
        "source_instruction_in_current_task_definition": False,
    }

    report = audit_rows({"train": [row], "val": [], "test": []}, tmp_path, [review])

    assert report["passed_mechanical_gate"] is True
    assert report["expected_value_matches_sql"] == 1
    assert report["semantic_warning_rows"] == 1
    assert report["split_summary"]["train"]["semantic_warning_rows"] == 1
    assert report["semantic_warning_counts"]["latest_instruction_without_temporal_sql"] == 1
    assert report["human_semantic_review_coverage"] == "not established by this audit"

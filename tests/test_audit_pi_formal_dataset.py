import sqlite3

from scripts.audit_pi_formal_dataset import audit_rows
from scripts.prepare_pi_formal_dataset import SYSTEM_PROMPT, build_training_record


def verifier(version: str, task: str, instruction_hash: str) -> dict:
    return {
        "verifier_id": f"sft/{version}:{task}",
        "task_id": task,
        "environment_id": f"sft/{version}",
        "instruction": f"问题 {task}",
        "instruction_sha256": instruction_hash,
        "required_tables": ["metric"],
        "gold": {
            "answer_type": "numeric",
            "value": 7,
            "verification_sql": "SELECT value FROM metric",
            "abs_tol": 1e-3,
            "rel_tol": 1e-5,
        },
    }


def test_audit_rows_accepts_isolated_splits_and_detects_task_leakage(tmp_path):
    rows = {}
    for index, (split, version, task) in enumerate(
        [("train", "v1", "a"), ("val", "v2", "b"), ("test", "v3", "c")]
    ):
        database = tmp_path / "sft" / version / "logistics.sqlite"
        database.parent.mkdir(parents=True)
        connection = sqlite3.connect(database)
        connection.execute("create table metric(value integer)")
        connection.execute("insert into metric values (7)")
        connection.commit()
        connection.close()
        rows[split] = [build_training_record(verifier(version, task, f"h-{task}"), split, index)]

    errors, detail = audit_rows(rows, tmp_path)
    assert errors == []
    assert detail["train"]["rows"] == 1

    rows["test"][0]["reward_model"]["ground_truth"]["task_id"] = "a"
    errors, _ = audit_rows(rows, tmp_path)
    assert any("task_id leakage" in error for error in errors)


import json
import sqlite3

from scripts.profile_boss_sandbox_catalog import canonical_hash, profile_catalog


def write_version(root, version, rows):
    directory = root / version
    directory.mkdir(parents=True)
    database = directory / "logistics.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE shipments (status TEXT, amount REAL)")
    connection.executemany(
        "INSERT INTO shipments VALUES (?, ?)",
        [("done", 10.0), ("done", 15.0), ("open", 4.0)],
    )
    connection.commit()
    connection.close()
    (directory / "dwh_tasks.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def task(task_id, instruction, *, sql="SELECT SUM(amount) FROM shipments", value=29.0):
    return {
        "task_id": task_id,
        "task_type": "dwh",
        "task_category": "answerable",
        "scenario_type": "single_turn",
        "business_domain": "logistics",
        "difficulty": "medium",
        "natural_language_instruction": instruction,
        "expected_tables": ["shipments"],
        "gold_answer": {
            "answer_type": "numeric",
            "value": value,
            "verification_sql": sql,
        },
        "answerability_label": {"is_answerable": True, "reason": "database"},
        "validation": {
            "checked_against_database": True,
            "expected_result_exists": True,
            "validation_method": "query",
        },
        "_qa_status": "passed",
    }


def test_profile_catalog_executes_sql_and_exports_only_high_precision_candidates(tmp_path):
    write_version(
        tmp_path,
        "v1",
        [
            task("a", "请计算所有运单金额总和。"),
            task("b", "请分析当前物流问题并给出建议。"),
            task("c", "请给出已完成运单数量。", sql="", value=2),
        ],
    )

    summary, candidates = profile_catalog(tmp_path, execute_sql=True)

    assert summary["sandbox_versions"] == 1
    assert summary["rows"] == 3
    assert summary["elapsed_seconds"] >= 0
    assert summary["high_precision_candidate_rows"] == 1
    assert summary["versions"][0]["verification_sql_executable_rows"] == 2
    assert summary["versions"][0]["gold_result_match_rows"] == 2
    assert [row["task_id"] for row in candidates] == ["a"]
    assert candidates[0]["globally_unique_instruction"] is True
    assert candidates[0]["gold_sha256"] == canonical_hash(candidates[0]["gold"])
    assert summary["contains_prompts_sql_answers_task_ids_tool_outputs_or_server_paths"] is False


def test_profile_catalog_marks_cross_version_prompt_reuse(tmp_path):
    repeated = "请计算所有运单金额总和。"
    write_version(tmp_path, "v1", [task("a", repeated)])
    write_version(tmp_path, "v2", [task("a", repeated)])

    summary, candidates = profile_catalog(tmp_path, execute_sql=True)

    assert summary["cross_version_task_id_duplicate_groups"] == 1
    assert summary["cross_version_instruction_duplicate_groups"] == 1
    assert summary["high_precision_candidate_rows"] == 2
    assert summary["high_precision_unique_instruction_rows"] == 0
    assert all(row["instruction_version_count"] == 2 for row in candidates)


def test_profile_catalog_denies_mutating_verification_sql(tmp_path):
    write_version(
        tmp_path,
        "v1",
        [task("a", "请给出运单金额总和。", sql="DELETE FROM shipments", value=29.0)],
    )

    summary, candidates = profile_catalog(tmp_path, execute_sql=True)

    assert candidates == []
    assert summary["versions"][0]["verification_sql_executable_rows"] == 0
    connection = sqlite3.connect(tmp_path / "v1" / "logistics.sqlite")
    try:
        assert connection.execute("SELECT COUNT(*) FROM shipments").fetchone()[0] == 3
    finally:
        connection.close()


def test_profile_catalog_requires_answerable_category_and_passed_qa(tmp_path):
    out_of_scope = task("a", "请给出运单金额总和。")
    out_of_scope["task_category"] = "out_of_scope"
    failed_qa = task("b", "请给出全部运单金额合计。")
    failed_qa["_qa_status"] = "failed"
    write_version(tmp_path, "v1", [out_of_scope, failed_qa])

    summary, candidates = profile_catalog(tmp_path, execute_sql=True)

    assert candidates == []
    exclusions = summary["versions"][0]["exclusion_counts"]
    assert exclusions["not_answerable_task_category"] == 1
    assert exclusions["qa_not_passed"] == 1

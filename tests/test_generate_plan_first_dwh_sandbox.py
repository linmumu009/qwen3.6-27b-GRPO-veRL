from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts.generate_plan_first_dwh_sandbox import (
    DEFAULT_VERSION,
    TECHNICAL_INSTRUCTION_RE,
    build_plans,
    compile_sql,
    generate,
    render_instruction,
    verify_existing,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return generate(tmp_path_factory.mktemp("dwh-plan-first"), DEFAULT_VERSION)


def test_plan_first_rendering_is_complete_and_unique() -> None:
    plans = build_plans()
    assert len(plans) == 300
    assert len({compile_sql(plan) for plan in plans}) == 300
    assert len({render_instruction(plan) for plan in plans}) == 300
    assert {plan.difficulty_band for plan in plans} == set(range(1, 7))


def test_band_four_filters_warehouse_type_without_fixing_grouped_warehouse() -> None:
    plans = [plan for plan in build_plans() if plan.difficulty_band == 4]

    assert len(plans) == 50
    assert all(plan.group_sql == "w.warehouse_name" for plan in plans)
    assert all(
        any(
            item.sql.startswith("w.warehouse_type =")
            and item.description.startswith("发货仓类型为“")
            for item in plan.filters
        )
        for plan in plans
    )
    assert all(
        not any(item.sql.startswith("w.warehouse_name =") for item in plan.filters)
        for plan in plans
    )


def test_generated_sandbox_contract_and_balance(generated: Path) -> None:
    tasks = _read_jsonl(generated / "dwh_tasks.jsonl")
    assert len(tasks) == 300
    assert all(task["business_domain"] == "logistics" for task in tasks)
    assert all(task["scenario_type"] == "dwh_query" for task in tasks)
    assert all(task["_qa_status"] == "passed" for task in tasks)
    assert all(task["answerability_label"]["is_answerable"] is True for task in tasks)
    assert all(task["validation"]["checked_against_database"] is True for task in tasks)
    assert all(task["validation"]["expected_result_exists"] is True for task in tasks)
    assert all(task["validation"]["semantic_source"] == "query_plan" for task in tasks)
    assert all(task["instruction_style"] == "mixed_company_roles_natural_language_v1" for task in tasks)
    assert all(
        TECHNICAL_INSTRUCTION_RE.search(task["natural_language_instruction"]) is None
        for task in tasks
    )
    assert all("请查询" not in task["natural_language_instruction"] for task in tasks)
    roles = {task["instruction_role"] for task in tasks}
    assert {
        "company_owner",
        "finance",
        "data_analyst",
        "operations",
        "warehouse_manager",
        "regional_manager",
        "procurement",
        "customer_service",
        "planning",
        "sales",
        "general_employee",
    } <= roles
    assert {band: sum(task["difficulty_band"] == band for task in tasks) for band in range(1, 7)} == {
        band: 50 for band in range(1, 7)
    }
    summary = json.loads((generated / "generation_summary.json").read_text(encoding="utf-8"))
    assert summary["environment_id"] == f"sft/{DEFAULT_VERSION}"
    assert summary["external_api_used"] is False
    assert summary["training_allowed"] is False
    assert summary["instruction_style"] == "mixed_company_roles_natural_language_v1"
    assert len(summary["instruction_role_counts"]) >= 11
    assert set(summary["files"]) == {
        "logistics.sqlite",
        "dwh_tasks.jsonl",
        "schema_dictionary.md",
        "rollout_calibration.json",
    }


def test_every_gold_sql_replays_exactly(generated: Path) -> None:
    tasks = _read_jsonl(generated / "dwh_tasks.jsonl")
    connection = sqlite3.connect(generated / "logistics.sqlite")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    try:
        for task in tasks:
            rows = [dict(row) for row in connection.execute(task["gold_answer"]["verification_sql"])]
            gold = task["gold_answer"]
            if gold["answer_type"] == "numeric":
                assert rows == [{"value": gold["value"]}]
            else:
                assert rows == gold["value"]
    finally:
        connection.close()


def test_structural_difficulty_rises_by_band(generated: Path) -> None:
    tasks = _read_jsonl(generated / "dwh_tasks.jsonl")
    expected = {
        1: (0, 1, 0, 0, 0),
        2: (0, 3, 0, 0, 0),
        3: (0, 2, 1, 0, 0),
        4: (1, 3, 1, 0, 1),
        5: (2, 4, 1, 0, 1),
        6: (3, 5, 1, 1, 1),
    }
    for task in tasks:
        features = task["query_plan"]["feature_counts"]
        observed = (
            features["joins"],
            features["filters"],
            features["group_by"],
            features["having"],
            features["top_k"],
        )
        assert observed == expected[task["difficulty_band"]]
    assert all(
        task["query_plan"]["feature_counts"]["derived_metric"] == 1
        for task in tasks
        if task["difficulty_band"] == 6
    )


def test_calibration_requires_rollout_evidence(generated: Path) -> None:
    calibration = json.loads((generated / "rollout_calibration.json").read_text(encoding="utf-8"))
    assert calibration["pilot"]["total_pilot_rollouts"] == 192
    assert calibration["target_success_rate"] == {"minimum": 0.2, "maximum": 0.8}
    assert "API" not in (generated / "schema_dictionary.md").read_text(encoding="utf-8")


def test_exact_directory_verifier_replays_all_tasks(generated: Path) -> None:
    result = verify_existing(generated)
    assert result["task_count"] == 300
    assert result["gold_replay_rows"] == 300
    assert result["file_hashes_verified"] == 4
    assert result["database_integrity_check"] == "ok"

from __future__ import annotations

import sqlite3
from pathlib import Path

from llin_verl.grounded_trajectory_reward import (
    REWARD_CONTRACT,
    capture_deterministic_composition,
    compute_grounded_trajectory_reward,
)
from llin_verl.outcome_gated_contract import evidence_binding_hash


def make_database(tmp_path: Path) -> Path:
    root = tmp_path / "sft" / "v1"
    root.mkdir(parents=True)
    database = root / "logistics.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE fact_metric(id INTEGER, category TEXT, value REAL)")
        connection.executemany(
            "INSERT INTO fact_metric VALUES (?, ?, ?)",
            [(1, "a", 10.0), (2, "b", 20.0)],
        )
    return database


def truth(*, table: bool = False) -> dict:
    value = {
        "environment_id": "sft/v1",
        "answer_type": "table" if table else "numeric",
        "expected_value": [["a", 10.0], ["b", 20.0]] if table else 30.0,
        "verification_sql": (
            "SELECT category, value FROM fact_metric ORDER BY id"
            if table
            else "SELECT SUM(value) FROM fact_metric"
        ),
        "required_tables": ["fact_metric"],
        "must_use_fields": ["value"],
        "evidence_plan": {"ordered": table, "measure": "value"},
        "abs_tol": 1e-3,
        "rel_tol": 1e-5,
    }
    value["process_evidence_binding_sha256"] = evidence_binding_hash(value)
    return value


def event(sql: str, *, ok: bool = True, response: str = "30.0") -> dict:
    return {
        "name": "bash",
        "arguments": {"command": f'sqlite3 /workspace/logistics.sqlite "{sql}"'},
        "ok": ok,
        "response_preview": response,
        "response_truncated": False,
        "observed_tool_response": True,
        "call_parse_valid": True,
        "source": "runtime_structured_pi_workspace",
    }


def extra(database: Path, events: list[dict], *, protocol: bool = True) -> dict:
    request_id = "req-unit"
    environment_id = "sft/v1"
    bound_events = []
    wrapper_events = []
    for index, source_event in enumerate(events):
        value = dict(source_event)
        value.setdefault("source", "runtime_structured_pi_workspace")
        value.setdefault("command_origin", "model")
        value.setdefault("workspace_alias", "/workspace")
        value.setdefault("workspace_request_id", request_id)
        value.setdefault("environment_id", environment_id)
        bound_events.append(value)
        wrapper_events.append(
            {
                "name": value.get("name"),
                "source": "runtime_wrapper",
                "model_event_index": index,
                "workspace_request_id": request_id,
                "environment_id": environment_id,
                "assigned_workspace_root": "/private/run/req-unit",
                "canonical_target": "",
                "target_path_classification": (
                    "runtime_mapped_from_model_workspace_alias"
                ),
                "ok": bool(value.get("ok")),
            }
        )
    return {
        "pi_tool_events": bound_events,
        "pi_runtime_wrapper_events": wrapper_events,
        "pi_tool_log_present": True,
        "pi_tool_protocol_complete": protocol,
        "pi_tool_event_contract": "runtime-captured-structured-tool-events-v3",
        "pi_workspace_request_id": request_id,
        "pi_environment_id": environment_id,
        "pi_workspace_released": True,
        "pi_reward_database_path": str(database),
        "pi_reward_database_root": str(database.parent.parent),
        "trajectory_timeout": False,
        "runtime_error": False,
        "pi_tool_event_source": "runtime_structured_pi_workspace",
    }


def score(database: Path, answer: str, events: list[dict], *, ground_truth=None, protocol=True):
    return compute_grounded_trajectory_reward(
        "dwh",
        answer,
        ground_truth or truth(),
        extra(database, events, protocol=protocol),
    )


def test_direct_grounded_correct_is_pass(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    result = score(database, "最终答案是 30。", [event("SELECT SUM(value) FROM fact_metric")])
    assert result["judge_state"] == "PASS"
    assert result["train_mask"] == 1.0
    assert result["success"] == 1.0
    assert result["evidence_route"] == "direct"
    assert result["reward_contract"] == REWARD_CONTRACT


def test_correct_no_tool_guess_and_wrong_sql_guess_are_fail(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    guessed = score(database, "最终答案是 30。", [])
    wrong_sql = score(
        database,
        "最终答案是 30。",
        [event("SELECT SUM(value) + 1 FROM fact_metric", response="31")],
    )
    for result in (guessed, wrong_sql):
        assert result["judge_state"] == "FAIL"
        assert result["train_mask"] == 1.0
        assert result["success"] == 0.0
        assert result["guess_correct_blocked"] == 1.0


def test_correct_evidence_wrong_final_is_fail(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    result = score(database, "最终答案是 31。", [event("SELECT SUM(value) FROM fact_metric")])
    assert result["judge_state"] == "FAIL"
    assert result["final_state"] == "FAIL"
    assert result["train_mask"] == 1.0


def test_later_unrelated_one_by_one_query_does_not_kill_support(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    result = score(
        database,
        "最终答案是 30。",
        [
            event("SELECT SUM(value) FROM fact_metric"),
            event("SELECT COUNT(*) FROM sqlite_master", response="1"),
        ],
    )
    assert result["judge_state"] == "PASS"
    assert result["supporting_event_indices"] == [0]


def test_replayable_multiquery_python_composition_is_pass(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    command = """python3 - <<'PY'
import sqlite3
con = sqlite3.connect('/workspace/logistics.sqlite')
a = con.execute('SELECT SUM(value) FROM fact_metric').fetchone()[0]
b = con.execute('SELECT 0').fetchone()[0]
result = a + b
print(result)
PY"""
    trace = capture_deterministic_composition(command, "30.0")
    assert trace and trace["replayable"] is True
    composed = {
        "name": "bash",
        "arguments": {"command": command},
        "ok": True,
        "response_preview": "30.0",
        "response_truncated": False,
        "observed_tool_response": True,
        "call_parse_valid": True,
        "source": "runtime_structured_pi_workspace",
        "composition_trace": trace,
    }
    result = score(database, "最终答案是 30。", [composed])
    assert result["judge_state"] == "PASS"
    assert result["evidence_route"] == "composed"


def test_unsupported_plausible_composition_is_unknown(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    command = """python3 - <<'PY'
import sqlite3
con = sqlite3.connect('/workspace/logistics.sqlite')
rows = con.execute('SELECT value FROM fact_metric').fetchall()
print('custom=' + str(rows))
PY"""
    result = score(
        database,
        "最终答案是 30。",
        [{
            "name": "bash",
            "arguments": {"command": command},
            "ok": True,
            "response_preview": "custom=[(10.0,), (20.0,)]",
            "response_truncated": False,
            "observed_tool_response": True,
            "call_parse_valid": True,
        }],
    )
    assert result["judge_state"] == "UNKNOWN"
    assert result["train_mask"] == 0.0


def test_missing_response_and_ambiguous_final_are_unknown(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    missing = event("SELECT SUM(value) FROM fact_metric")
    missing["observed_tool_response"] = False
    missing.pop("response_preview")
    missing_result = score(database, "最终答案是 30。", [missing], protocol=False)
    ambiguous = score(
        database,
        "最终答案是 30 或 31。",
        [event("SELECT SUM(value) FROM fact_metric")],
    )
    for result in (missing_result, ambiguous):
        assert result["judge_state"] == "UNKNOWN"
        assert result["train_mask"] == 0.0


def test_unsafe_and_malformed_model_behavior_are_fail(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    unsafe = event("SELECT SUM(value) FROM fact_metric")
    unsafe["arguments"]["command"] = "rm -rf /"
    malformed = event("SELECT SUM(value) FROM fact_metric")
    malformed["call_parse_valid"] = False
    for candidate in (unsafe, malformed):
        result = score(database, "最终答案是 30。", [candidate])
        assert result["judge_state"] == "FAIL"
        assert result["train_mask"] == 1.0


def test_workspace_local_write_is_not_hard_unsafe_and_wrapper_never_supports_reward(
    tmp_path: Path,
) -> None:
    database = make_database(tmp_path)
    write_event = {
        "name": "write",
        "arguments": {"path": "/workspace/tmp/result.txt", "content": "30"},
        "ok": True,
        "response_preview": "Wrote 2 bytes",
        "response_truncated": False,
        "observed_tool_response": True,
        "call_parse_valid": True,
        "source": "runtime_structured_pi_workspace",
    }
    result = score(
        database,
        "最终答案是 30。",
        [event("SELECT SUM(value) FROM fact_metric"), write_event],
    )
    assert result["judge_state"] == "PASS"
    assert result["unsafe_tool_event_count"] == 0
    assert result["runtime_wrapper_events_excluded_from_reward"] is True
    assert result["supporting_event_indices"] == [0]


def test_host_escape_is_fail_but_missing_runtime_binding_is_unknown(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    escaped = {
        "name": "read",
        "arguments": {"path": "/etc/passwd"},
        "ok": False,
        "response_preview": "PermissionError",
        "response_truncated": False,
        "observed_tool_response": True,
        "call_parse_valid": True,
        "source": "runtime_structured_pi_workspace",
    }
    escaped_result = score(database, "最终答案是 30。", [escaped])
    assert escaped_result["judge_state"] == "FAIL"
    assert escaped_result["hard_unsafe_reason_counts"] == {"workspace_path_escape": 1}

    metadata = extra(database, [event("SELECT SUM(value) FROM fact_metric")])
    metadata["pi_workspace_request_id"] = ""
    missing_binding = compute_grounded_trajectory_reward(
        "dwh", "最终答案是 30。", truth(), metadata
    )
    assert missing_binding["judge_state"] == "UNKNOWN"
    assert missing_binding["judge_reason"] == "workspace_or_event_provenance_incomplete"


def test_shadow_parse_failure_is_unknown_not_model_fail(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    malformed = event("SELECT SUM(value) FROM fact_metric")
    malformed["call_parse_valid"] = False
    malformed["source"] = "qwen_xml_shadow_adapter"
    metadata = extra(database, [malformed])
    metadata.pop("pi_tool_event_source")
    result = compute_grounded_trajectory_reward("dwh", "最终答案是 30。", truth(), metadata)
    assert result["judge_state"] == "UNKNOWN"
    assert result["judge_reason"] == "shadow_or_unattributed_tool_parse_failure"
    assert result["malformed_source_unattributed_count"] == 1


def test_matching_result_cannot_bypass_filter_join_group_or_limit_contract(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE dim_category(category TEXT PRIMARY KEY, active INTEGER)")
        connection.executemany("INSERT INTO dim_category VALUES (?, ?)", [("a", 1), ("b", 1)])
    gold = (
        "SELECT f.category, SUM(f.value) FROM fact_metric f "
        "JOIN dim_category d ON d.category = f.category "
        "WHERE f.id >= 1 AND d.active = 1 GROUP BY f.category "
        "ORDER BY SUM(f.value) DESC LIMIT 2"
    )
    bound = {
        "environment_id": "sft/v1",
        "answer_type": "table",
        "expected_value": [["b", 20.0], ["a", 10.0]],
        "verification_sql": gold,
        "required_tables": ["fact_metric", "dim_category"],
        "must_use_fields": ["value", "active", "id", "category"],
        "evidence_plan": {
            "filters": [{"sql": "f.id >= 1"}, {"sql": "d.active = 1"}],
            "group_by": "f.category",
            "order_by": "SUM(f.value)",
            "order_direction": "DESC",
            "limit": 2,
            "feature_counts": {"joins": 1},
            "aggregation": "SUM",
        },
        "abs_tol": 1e-3,
        "rel_tol": 1e-5,
    }
    bound["process_evidence_binding_sha256"] = evidence_binding_hash(bound)
    final = "|category|value|\n|---|---|\n|b|20|\n|a|10|"
    variants = [
        gold.replace("WHERE f.id >= 1 AND d.active = 1", "WHERE d.active = 1"),
        gold.replace("d.active = 1", "d.active = 2"),
        gold.replace("ON d.category = f.category", "ON 1 = 1"),
        gold.replace("GROUP BY f.category", "GROUP BY d.active"),
        gold.replace("DESC LIMIT 2", "ASC LIMIT 2"),
        gold.replace("LIMIT 2", "LIMIT 1"),
    ]
    for variant in variants:
        result = score(database, final, [event(variant)], ground_truth=bound)
        assert result["judge_state"] != "PASS", variant
        assert result["evidence_state"] != "PASS", variant


def test_task_binding_or_database_failure_is_unknown(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    invalid = truth()
    invalid["must_use_fields"] = ["other"]
    result = score(
        database,
        "最终答案是 30。",
        [event("SELECT SUM(value) FROM fact_metric")],
        ground_truth=invalid,
    )
    assert result["judge_state"] == "UNKNOWN"
    assert result["task_reason"] == "task_binding_invalid"


def test_complete_table_route_preserves_order_semantics(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    table_truth = truth(table=True)
    sql = "SELECT category, value FROM fact_metric ORDER BY id"
    correct = score(
        database,
        "|category|value|\n|---|---|\n|a|10|\n|b|20|",
        [event(sql, response="a|10\nb|20")],
        ground_truth=table_truth,
    )
    reordered = score(
        database,
        "|category|value|\n|---|---|\n|b|20|\n|a|10|",
        [event(sql, response="a|10\nb|20")],
        ground_truth=table_truth,
    )
    assert correct["judge_state"] == "PASS"
    assert correct["evidence_route"] == "table"
    assert reordered["judge_state"] == "FAIL"

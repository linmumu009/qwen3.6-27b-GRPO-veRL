from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import pytest

from llin_verl.outcome_gated_contract import evidence_binding_hash
from llin_verl.pi_reward import compute_score_tiered_query_cost_v1
from llin_verl.tiered_query_cost_reward import (
    compute_tiered_query_cost_reward,
    efficiency_terms,
    extract_sqlite_cli_selects,
)


def fixture(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "sft" / "v1"
    root.mkdir(parents=True)
    database = root / "logistics.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE fact_metric(value REAL)")
        connection.execute("CREATE TABLE unrelated(value REAL)")
        connection.executemany("INSERT INTO fact_metric VALUES (?)", [(10.0,), (20.0,)])
    truth = {
        "environment_id": "sft/v1",
        "answer_type": "numeric",
        "expected_value": 30.0,
        "verification_sql": "SELECT SUM(value) FROM fact_metric",
        "required_tables": ["fact_metric"],
        "must_use_fields": ["value"],
        "evidence_plan": {"aggregation": "sum"},
        "abs_tol": 1e-3,
        "rel_tol": 1e-5,
    }
    truth["process_evidence_binding_sha256"] = evidence_binding_hash(truth)
    return database, truth


def event(sql: str, *, ok: bool = True, tokens: int = 100) -> dict:
    return {
        "name": "bash",
        "arguments": {"command": f"sqlite3 /workspace/logistics.sqlite '{sql}'"},
        "sql_statements": [sql],
        "ok": ok,
        "response_preview": "30" if ok else "Error: query failed",
        "response_token_count": tokens,
        "observed_tool_response": True,
        "call_parse_valid": True,
        "source": "api_multiturn_dwh_sandbox",
    }


def score(database: Path, truth: dict, answer: str, events: list[dict], **overrides):
    extra = {
        "pi_tool_events": events,
        "tool_log_present": True,
        "tool_protocol_complete": True,
        "pi_reward_database_path": str(database),
        "pi_reward_database_root": str(database.parent.parent),
        "trajectory_timeout": False,
        "runtime_error": False,
        "api_error": False,
    }
    extra.update(overrides)
    return compute_tiered_query_cost_reward("dwh", answer, truth, extra)


def test_exact_log1p_and_clip_boundaries() -> None:
    free = efficiency_terms(
        query_attempts=4,
        tool_response_tokens=4000,
        irrelevant_ratio=0,
        duplicate_ratio=0,
    )
    assert free == {"Eq": 0.0, "Et": 0.0, "Eb": 0.0, "E": 0.0}
    saturated = efficiency_terms(
        query_attempts=12,
        tool_response_tokens=16000,
        irrelevant_ratio=1,
        duplicate_ratio=1,
    )
    assert saturated == {"Eq": 1.0, "Et": 1.0, "Eb": 1.0, "E": 1.0}
    middle = efficiency_terms(
        query_attempts=5,
        tool_response_tokens=8000,
        irrelevant_ratio=0.5,
        duplicate_ratio=0.25,
    )
    assert middle["Eq"] == pytest.approx(math.log1p(1) / math.log1p(8))
    assert middle["Et"] == pytest.approx(math.log1p(1) / math.log1p(3))
    assert middle["Eb"] == pytest.approx(0.375)
    assert 0 < middle["E"] < 1


def test_sqlite_cli_select_extraction_supports_single_and_double_quotes() -> None:
    expected = ["SELECT SUM(value) FROM fact_metric"]
    assert extract_sqlite_cli_selects(
        "sqlite3 /workspace/logistics.sqlite 'SELECT SUM(value) FROM fact_metric'"
    ) == expected
    assert extract_sqlite_cli_selects(
        'sqlite3 /workspace/logistics.sqlite "SELECT SUM(value) FROM fact_metric"'
    ) == expected


def test_guess_correct_without_relevant_query_is_zero(tmp_path: Path) -> None:
    database, truth = fixture(tmp_path)
    result = score(database, truth, "Final result: 30", [])
    assert result["reward"] == 0.0
    assert result["reward_layer"] == "no_attempt"
    assert result["guess_correct_blocked"] == 1.0


def test_failed_attempt_and_successful_wrong_are_soft_tiers(tmp_path: Path) -> None:
    database, truth = fixture(tmp_path)
    failed = score(
        database,
        truth,
        "Final result: 31",
        [event("SELECT SUM(value) FROM fact_metric", ok=False)],
    )
    wrong = score(
        database,
        truth,
        "Final result: 31",
        [event("SELECT SUM(value) FROM fact_metric")],
    )
    assert failed["reward"] == pytest.approx(0.1)
    assert failed["reward_layer"] == "attempt_failed"
    assert wrong["reward"] == pytest.approx(0.2)
    assert wrong["reward"] <= 0.2


def test_correct_grounded_reward_is_at_least_point_eight(tmp_path: Path) -> None:
    database, truth = fixture(tmp_path)
    events = [event("SELECT SUM(value) FROM fact_metric", tokens=32000)]
    result = score(database, truth, "Final result: 30", events)
    assert result["E"] <= 1.0
    assert result["reward"] >= 0.8
    assert result["reward"] == pytest.approx(0.94)
    assert result["success"] == 1.0
    assert result["train_mask"] == 1.0


def test_irrelevant_and_duplicate_ratios_are_exact(tmp_path: Path) -> None:
    database, truth = fixture(tmp_path)
    events = [
        event("SELECT SUM(value) FROM fact_metric"),
        event("SELECT SUM(value) FROM fact_metric"),
        event("SELECT SUM(value) FROM unrelated"),
        event("SELECT 1"),
    ]
    result = score(database, truth, "Final result: 30", events)
    assert result["query_attempt_count"] == 4
    assert result["irrelevant_query_ratio"] == pytest.approx(0.5)
    assert result["duplicate_query_ratio"] == pytest.approx(0.25)
    assert result["E"] == pytest.approx(0.075)
    assert result["reward"] == pytest.approx(0.985)


@pytest.mark.parametrize("q,tokens", [(17, 100), (1, 32001)])
def test_hard_budget_is_zero(tmp_path: Path, q: int, tokens: int) -> None:
    database, truth = fixture(tmp_path)
    events = [
        event(
            f"SELECT SUM(value) FROM fact_metric -- {index}",
            tokens=tokens if index == 0 else 0,
        )
        for index in range(q)
    ]
    result = score(database, truth, "Final result: 30", events)
    assert result["budget_exceeded"] is True
    assert result["reward"] == 0.0
    assert result["reward_layer"] == "unsafe_or_budget"


def test_unknown_masks_timeout_missing_log_verifier_and_cost(tmp_path: Path) -> None:
    database, truth = fixture(tmp_path)
    valid = event("SELECT SUM(value) FROM fact_metric")
    cases = [
        score(database, truth, "Final result: 30", [valid], trajectory_timeout=True),
        score(database, truth, "Final result: 30", [valid], tool_log_present=False),
        score(
            database,
            truth,
            "Final result: 30",
            [{**valid, "response_token_count": None}],
        ),
    ]
    bad_truth = dict(truth)
    bad_truth["process_evidence_binding_sha256"] = "bad"
    cases.append(score(database, bad_truth, "Final result: 30", [valid]))
    for result in cases:
        assert result["judge_state"] == "UNKNOWN"
        assert result["train_mask"] == 0.0
        assert result["reward"] == 0.0


def test_online_pi_protocol_aliases_are_consumed(tmp_path: Path) -> None:
    database, truth = fixture(tmp_path)
    result = score(
        database,
        truth,
        "Final result: 30",
        [event("SELECT SUM(value) FROM fact_metric")],
        tool_log_present=None,
        tool_protocol_complete=None,
        pi_tool_log_present=True,
        pi_tool_protocol_complete=True,
    )
    assert result["judge_state"] == "PASS"
    assert result["success"] == 1.0


def test_unsafe_and_full_database_scan_are_zero(tmp_path: Path) -> None:
    database, truth = fixture(tmp_path)
    for command in ("rm -rf /", "sqlite3 /workspace/logistics.sqlite '.dump'"):
        value = event("SELECT SUM(value) FROM fact_metric")
        value["arguments"] = {"command": command}
        if ".dump" in command:
            value["sql_statements"] = []
        result = score(database, truth, "Final result: 30", [value])
        assert result["unsafe"] is True
        assert result["reward"] == 0.0


def test_verl_entrypoint_avoids_validation_reward_column_collision(tmp_path: Path) -> None:
    database, truth = fixture(tmp_path)
    extra = {
        "pi_tool_events": [event("SELECT SUM(value) FROM fact_metric")],
        "tool_log_present": True,
        "tool_protocol_complete": True,
        "pi_reward_database_path": str(database),
        "pi_reward_database_root": str(database.parent.parent),
        "trajectory_timeout": False,
        "runtime_error": False,
        "api_error": False,
    }

    result = compute_score_tiered_query_cost_v1(
        "dwh", "Final result: 30", truth, extra
    )

    assert "reward" not in result
    assert result["tiered_reward"] == result["score"]
    assert result["tiered_reward"] >= 0.8
    assert result["hard_unsafe_reason_counts"] == "{}"
    assert result["sampling_policy_version_min"] == "null"
    assert result["sampling_policy_version_max"] == "null"
    assert all(
        isinstance(value, (str, bool, int, float))
        for key, value in result.items()
        if key != "score"
    )

    # Mirror veRL's validation merge: the framework owns ``reward`` while
    # custom scalar extras are appended once per sample.
    merged: dict[str, list] = {"reward": []}
    for _ in range(32):
        merged["reward"].append(result["score"])
        for key, value in result.items():
            if key != "score":
                merged.setdefault(key, []).append(value)
    assert len(merged["reward"]) == 32
    assert len(merged["tiered_reward"]) == 32
    # veRL skips strings and computes NumPy means for every other extra.
    assert all(
        isinstance(values[0], str)
        or all(isinstance(value, (bool, int, float)) for value in values)
        for values in merged.values()
    )

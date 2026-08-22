"""Correctness-gated trajectory-level process reward for DWH GRPO.

The reward in this module is intentionally one scalar for a completed
multi-turn trajectory.  It is *not* turn-level or token-level credit
assignment.  Process evidence is accepted only from observed tool events;
claims in the final answer never create SQL, table, field, or task-fit credit.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from llin_verl.pi_reward import (
    _full_table_row_multisets_equal,
    _full_table_rows_equal,
    _full_table_value_equal,
    _normalize_full_expected_table,
    boss_numbers_match,
    execute_readonly_sql,
    extract_final_assistant_answer,
    extract_selects,
    strict_table_answer_match_semantic,
    table_order_semantics,
)
from llin_verl.pi_tool_contract import command_unsafe_reasons, extract_table_names
from llin_verl.outcome_gated_contract import evidence_binding_hash


REWARD_CONTRACT = "outcome-gated-verified-trajectory-process-v5"
MAX_PROCESS_BONUS_ALPHA = 0.10
# These components are retained as diagnostics only.  They are deliberately
# excluded from the training scalar until they demonstrate useful precision.
OBSERVED_PROCESS_WEIGHTS = {
    "sql": 0.50,
    "table": 0.15,
    "field": 0.15,
    "fit": 0.10,
    "efficiency": 0.10,
}
PROTOCOL_TOOLS = {"bash", "read", "write", "edit"}

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>]+)>", re.IGNORECASE)
_PARAMETER_RE = re.compile(
    r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.IGNORECASE | re.DOTALL
)
_TOOL_ERROR_RE = re.compile(
    r"(?:traceback|permissionerror|filenotfounderror|command timed out|"
    r"no such file|not found|\berror\s*:|exception:|exit code\s+[1-9]\d*)",
    re.IGNORECASE,
)
_MUTATING_SHELL_RE = re.compile(
    r"(?:^|[;&|()]\s*)(?:rm|mv|cp|touch|mkdir|rmdir|chmod|chown|tee|truncate|ln)\b|"
    r"\bsed\s+-[^\n]*i\b|\bperl\s+-[^\n]*i\b|"
    r"\.(?:write|write_text|write_bytes|unlink|rename|replace)\s*\(|"
    r"\bopen\s*\([^\n]{0,200},\s*['\"](?:w|a|x|\+)\b",
    re.IGNORECASE,
)
_MUTATING_SQL_STATEMENT_RE = re.compile(
    r"(?:^|[;'\"\n])\s*(?:insert|update|delete|drop|alter|create|replace|attach|detach|vacuum)\b",
    re.IGNORECASE,
)
_FULL_SCAN_RE = re.compile(r"\bselect\s+\*\s+from\b", re.IGNORECASE | re.DOTALL)
_WHERE_RE = re.compile(r"\bwhere\b", re.IGNORECASE)
_LIMIT_RE = re.compile(r"\blimit\b", re.IGNORECASE)
_FINAL_RESULT_RE = re.compile(
    r"(?:^|\n|[。.!?]\s*)(?:the\s+)?(?:final\s+(?:answer|result)|"
    r"最终(?:答案|结果)|答案)\s*(?:is|equals?|为|是|[:：=])\s*([^\n]+)",
    re.IGNORECASE,
)
_PYTHON_SQL_EXECUTOR_RE = re.compile(
    r"(?:sqlite3\.connect|\.execute\s*\(|\.executemany\s*\(|"
    r"read_sql(?:_query)?\s*\()",
    re.IGNORECASE,
)
_ACTUAL_SQL_EXECUTOR_RE = re.compile(
    r"(?:^|[;&|()]\s*)(?:python(?:3)?\b|sqlite3\b)", re.IGNORECASE
)


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_qwen_tool_events(transcript: str) -> dict[str, Any]:
    """Recover paired Qwen tool calls/results from a decoded trajectory.

    The existing 800-trajectory standalone shards predate persistence of
    ``pi_tool_events``.  They do retain Qwen's explicit function/parameter
    blocks and tool-response blocks.  This adapter is shadow-only: online
    scoring consumes runtime-captured events and never reparses answer text.
    """

    call_blocks = _TOOL_CALL_RE.findall(transcript or "")
    responses = _TOOL_RESPONSE_RE.findall(transcript or "")
    events: list[dict[str, Any]] = []
    malformed_calls = 0
    for index, block in enumerate(call_blocks):
        function_matches = _FUNCTION_RE.findall(block)
        parameters = _PARAMETER_RE.findall(block)
        parameter_names = [name.strip() for name, _ in parameters]
        valid = (
            len(function_matches) == 1
            and bool(parameters)
            and len(parameter_names) == len(set(parameter_names))
        )
        if not valid:
            malformed_calls += 1
        name = function_matches[0].strip() if len(function_matches) == 1 else ""
        arguments = {key.strip(): value.strip() for key, value in parameters}
        response = responses[index].strip() if index < len(responses) else ""
        event = {
            "name": name,
            "arguments": arguments,
            "ok": bool(response) and not bool(_TOOL_ERROR_RE.search(response)),
            "response_preview": response,
            "observed_tool_response": index < len(responses),
            "source": "qwen_xml_shadow_adapter",
            "call_parse_valid": valid,
        }
        events.append(event)

    complete = (
        bool(call_blocks)
        and len(call_blocks) == len(responses)
        and malformed_calls == 0
    )
    return {
        "events": events,
        "protocol_complete": complete,
        "tool_call_count": len(call_blocks),
        "tool_response_count": len(responses),
        "malformed_tool_call_count": malformed_calls,
        "unpaired_tool_call_count": max(0, len(call_blocks) - len(responses)),
        "unpaired_tool_response_count": max(0, len(responses) - len(call_blocks)),
    }


def _expected_value(ground_truth: dict[str, Any]) -> Any:
    value = ground_truth.get("expected_value")
    if "expected_value_json" in ground_truth:
        try:
            value = json.loads(str(ground_truth["expected_value_json"]))
        except (TypeError, json.JSONDecodeError):
            return None
    return value


def semantic_rows_equal(
    candidate: Iterable[Iterable[Any]],
    expected: Iterable[Iterable[Any]],
    *,
    ordered: bool,
    abs_tol: float,
    rel_tol: float,
) -> bool:
    left = [list(row) for row in candidate]
    right = [list(row) for row in expected]
    if ordered:
        return _full_table_rows_equal(left, right, abs_tol, rel_tol)
    return _full_table_row_multisets_equal(left, right, abs_tol, rel_tol)


def parse_unique_numeric_final(answer: str) -> dict[str, Any]:
    """Parse one unambiguous numeric conclusion from the final assistant turn.

    An explicit final-result field is preferred.  If it is absent, only the
    final non-empty line is considered.  Multiple explicit fields, multiple
    numbers in the selected field, or conflicting candidates fail closed.
    Numbers mentioned earlier in the reasoning can therefore never make a
    later wrong conclusion pass.
    """

    from llin_verl.pi_reward import extract_answer_numbers

    explicit = [match.group(1).strip() for match in _FINAL_RESULT_RE.finditer(answer or "")]
    mode = "explicit_final_result" if explicit else "last_nonempty_line"
    if explicit:
        candidates = explicit
    else:
        lines = [line.strip() for line in (answer or "").splitlines() if line.strip()]
        candidates = lines[-1:] if lines else []
    if len(candidates) != 1:
        return {
            "value": None,
            "mode": mode,
            "ambiguous": bool(candidates),
            "ambiguity_reason": "multiple_final_result_fields" if candidates else "missing_final_result",
            "candidate_count": len(candidates),
        }
    numbers = extract_answer_numbers(candidates[0])
    if len(numbers) != 1:
        return {
            "value": None,
            "mode": mode,
            "ambiguous": bool(numbers),
            "ambiguity_reason": "multiple_numeric_candidates" if numbers else "no_numeric_candidate",
            "candidate_count": len(numbers),
        }
    return {
        "value": float(numbers[0]),
        "mode": mode,
        "ambiguous": False,
        "ambiguity_reason": "",
        "candidate_count": 1,
    }


def corrected_final_verifier(
    solution_str: str,
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    """Return corrected final correctness without consulting tool evidence."""

    answer = extract_final_assistant_answer(solution_str)
    answer_type = str(ground_truth.get("answer_type") or "")
    expected = _expected_value(ground_truth)
    abs_tol = float(ground_truth.get("abs_tol", 1e-3))
    rel_tol = float(ground_truth.get("rel_tol", 1e-5))
    ordered, order_source = table_order_semantics(ground_truth)
    match_mode = "none"
    parsed_rows = 0
    correct = False
    error = ""
    if answer_type == "numeric" and isinstance(expected, (int, float)) and not isinstance(
        expected, bool
    ):
        number = float(expected)
        if math.isfinite(number):
            numeric_final = parse_unique_numeric_final(answer)
            parsed_value = numeric_final["value"]
            correct = parsed_value is not None and math.isclose(
                parsed_value, number, abs_tol=abs_tol, rel_tol=rel_tol
            )
            match_mode = numeric_final["mode"] if correct else "none"
            if numeric_final["ambiguity_reason"]:
                error = str(numeric_final["ambiguity_reason"])
        else:
            numeric_final = {
                "value": None,
                "mode": "none",
                "ambiguous": False,
                "ambiguity_reason": "non_finite_numeric_gold",
                "candidate_count": 0,
            }
            error = "non_finite_numeric_gold"
    elif answer_type == "table" and isinstance(expected, list):
        numeric_final = {
            "value": None,
            "mode": "not_numeric",
            "ambiguous": False,
            "ambiguity_reason": "",
            "candidate_count": 0,
        }
        correct, match_mode, parsed_rows = strict_table_answer_match_semantic(
            answer,
            expected,
            abs_tol,
            rel_tol,
            ordered=ordered,
        )
    else:
        numeric_final = {
            "value": None,
            "mode": "none",
            "ambiguous": False,
            "ambiguity_reason": "unsupported_or_incomplete_gold",
            "candidate_count": 0,
        }
        error = "unsupported_or_incomplete_gold"
    return {
        "correct": bool(correct and not error),
        "answer": answer,
        "answer_type": answer_type,
        "match_mode": match_mode,
        "parsed_rows": parsed_rows,
        "table_ordered": ordered if answer_type == "table" else None,
        "table_order_source": order_source if answer_type == "table" else "not_table",
        "numeric_value": numeric_final["value"],
        "numeric_parse_mode": numeric_final["mode"],
        "numeric_parse_ambiguous": bool(numeric_final["ambiguous"]),
        "numeric_parse_ambiguity_reason": numeric_final["ambiguity_reason"],
        "numeric_parse_candidate_count": int(numeric_final["candidate_count"]),
        "error": error,
    }


def _resolve_database(ground_truth: dict[str, Any], extra_info: dict[str, Any]) -> Path:
    explicit = extra_info.get("pi_reward_database_path")
    root_value = extra_info.get("pi_reward_database_root") or os.environ.get(
        "PI_AGENT_SANDBOX_LOWER", "/pi_sandbox"
    )
    root = Path(str(root_value)).resolve(strict=True)
    if explicit:
        database = Path(str(explicit)).resolve(strict=True)
    else:
        environment_id = str(ground_truth.get("environment_id") or "")
        database = (root / environment_id / "logistics.sqlite").resolve(strict=True)
    database.relative_to(root)
    if not database.is_file():
        raise FileNotFoundError(database)
    return database


def _gold_sql_self_consistent(
    ground_truth: dict[str, Any],
    database: Path,
    *,
    timeout_seconds: float,
) -> tuple[bool, list[tuple[Any, ...]], str]:
    sql = str(ground_truth.get("verification_sql") or "")
    answer_type = str(ground_truth.get("answer_type") or "")
    expected = _expected_value(ground_truth)
    abs_tol = float(ground_truth.get("abs_tol", 1e-3))
    rel_tol = float(ground_truth.get("rel_tol", 1e-5))
    try:
        rows = execute_readonly_sql(
            database,
            sql,
            query_timeout_seconds=timeout_seconds,
        )
    except (ValueError, OSError, sqlite3.Error) as exc:
        return False, [], f"{type(exc).__name__}: {exc}"
    if answer_type == "numeric":
        passed = (
            isinstance(expected, (int, float))
            and not isinstance(expected, bool)
            and len(rows) == 1
            and len(rows[0]) == 1
            and _full_table_value_equal(rows[0][0], expected, abs_tol, rel_tol)
        )
    elif answer_type == "table":
        normalized = _normalize_full_expected_table(expected)
        ordered, _ = table_order_semantics(ground_truth)
        passed = normalized is not None and semantic_rows_equal(
            rows,
            normalized or [],
            ordered=ordered,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
    else:
        passed = False
    return bool(passed), rows, "" if passed else "gold_sql_result_mismatch"


def _event_schema_valid(event: Any) -> bool:
    return (
        isinstance(event, dict)
        and str(event.get("name") or "") in PROTOCOL_TOOLS
        and isinstance(event.get("arguments"), dict)
        and isinstance(event.get("ok"), bool)
        and bool(event.get("observed_tool_response", "response_preview" in event))
        and "response_preview" in event
        and bool(event.get("call_parse_valid", True))
    )


def command_is_safe_readonly(command: str) -> bool:
    if not command.strip() or command_unsafe_reasons(command):
        return False
    return not bool(
        _MUTATING_SHELL_RE.search(command) or _MUTATING_SQL_STATEMENT_RE.search(command)
    )


def command_executes_sql(command: str) -> bool:
    """Conservatively identify commands that can have executed extracted SQL."""

    if not _ACTUAL_SQL_EXECUTOR_RE.search(command or ""):
        return False
    lowered = (command or "").casefold()
    if re.search(r"(?:^|[;&|()]\s*)sqlite3\b", lowered):
        return True
    return bool(_PYTHON_SQL_EXECUTOR_RE.search(command or ""))


def _successful_sql(events: list[dict[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    attempted: list[str] = []
    successful: list[str] = []
    commands: list[str] = []
    for event in events:
        if event.get("name") != "bash":
            continue
        command = str((event.get("arguments") or {}).get("command") or "")
        commands.append(command)
        event_selects = extract_selects([command]) if command_executes_sql(command) else []
        attempted.extend(event_selects)
        if event.get("ok"):
            successful.extend(event_selects)
    return attempted, successful, commands


def _answer_bearing_and_consistent(
    rows: list[tuple[Any, ...]],
    final: dict[str, Any],
    ground_truth: dict[str, Any],
) -> tuple[bool, bool]:
    """Return (answer-bearing, consistent-with-final) for one query result."""

    answer_type = str(final.get("answer_type") or "")
    abs_tol = float(ground_truth.get("abs_tol", 1e-3))
    rel_tol = float(ground_truth.get("rel_tol", 1e-5))
    if answer_type == "numeric":
        value = final.get("numeric_value")
        bearing = len(rows) == 1 and len(rows[0]) == 1 and value is not None
        return bearing, bool(
            bearing and _full_table_value_equal(rows[0][0], value, abs_tol, rel_tol)
        )
    if answer_type == "table" and final.get("correct"):
        expected = _normalize_full_expected_table(_expected_value(ground_truth))
        if expected is None:
            return False, False
        width = len(expected[0]) if expected else 0
        bearing = len(rows) == len(expected) and all(len(row) == width for row in rows)
        ordered, _ = table_order_semantics(ground_truth)
        return bearing, bool(
            bearing
            and semantic_rows_equal(
                rows,
                expected,
                ordered=ordered,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )
        )
    return False, False


def _field_used(sql_text: str, field: str) -> bool:
    normalized = str(field or "").strip().casefold()
    if not normalized:
        return False
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(normalized)}(?![A-Za-z0-9_])")
    return bool(pattern.search(sql_text.casefold()))


def efficiency_score(
    attempted_sql: list[str],
    commands: list[str],
    events: list[dict[str, Any]],
    *,
    runtime_auto_retry_count: int = 0,
) -> dict[str, Any]:
    full_scans = sum(
        bool(_FULL_SCAN_RE.search(sql))
        and not bool(_WHERE_RE.search(sql))
        and not bool(_LIMIT_RE.search(sql))
        for sql in attempted_sql
    )
    sql_keys = [" ".join(sql.casefold().split()) for sql in attempted_sql]
    duplicate_sql = len(sql_keys) - len(set(sql_keys))
    command_keys = [" ".join(command.casefold().split()) for command in commands]
    duplicate_commands = len(command_keys) - len(set(command_keys))
    event_auto_retry = sum(
        event.get("type") == "auto_retry_start" or bool(event.get("auto_retry"))
        for event in events
    )
    auto_retry = max(event_auto_retry, max(0, int(runtime_auto_retry_count)))
    score = max(
        0.0,
        1.0
        - 0.10 * full_scans
        - 0.05 * duplicate_sql
        - 0.02 * duplicate_commands
        - 0.20 * auto_retry,
    )
    return {
        "score": score,
        "full_scan_count": full_scans,
        "duplicate_sql_count": duplicate_sql,
        "duplicate_command_count": duplicate_commands,
        "auto_retry_count": auto_retry,
    }


def _normalized_observed_process_score(components: dict[str, float | None]) -> tuple[float, float]:
    applicable = [
        (float(components[name]), weight)
        for name, weight in OBSERVED_PROCESS_WEIGHTS.items()
        if components.get(name) is not None
    ]
    weight_sum = sum(weight for _, weight in applicable)
    if not applicable or weight_sum <= 0:
        return 0.0, 0.0
    return sum(value * weight for value, weight in applicable) / weight_sum, weight_sum


def compute_trajectory_process_reward(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
    **_: Any,
) -> dict[str, Any]:
    """Compute ``R = H*C*(1 + alpha*P_verified)`` for a full trajectory."""

    del data_source
    final = corrected_final_verifier(solution_str, ground_truth)
    correctness = float(final["correct"])
    events = extra_info.get("pi_tool_events") or []
    if not isinstance(events, list):
        events = []
    protocol_complete = bool(extra_info.get("pi_tool_protocol_complete", True))
    valid_protocol = bool(events) and protocol_complete and all(
        _event_schema_valid(event) for event in events
    )
    readonly_tools = valid_protocol and all(
        event.get("name") == "read"
        or command_is_safe_readonly(str((event.get("arguments") or {}).get("command") or ""))
        for event in events
    )

    try:
        timeout_seconds = float(extra_info.get("pi_reward_sql_timeout_seconds", 5.0))
        if timeout_seconds <= 0:
            raise ValueError("SQL timeout must be positive")
        database = _resolve_database(ground_truth, extra_info)
        database_available = True
        gold_ok, gold_rows, gold_error = _gold_sql_self_consistent(
            ground_truth, database, timeout_seconds=timeout_seconds
        )
    except (ValueError, OSError, sqlite3.Error) as exc:
        database_available = False
        database = None
        gold_ok = False
        gold_rows = []
        gold_error = f"{type(exc).__name__}: {exc}"

    attempted_sql, successful_sql, commands = _successful_sql(events)
    abs_tol = float(ground_truth.get("abs_tol", 1e-3))
    rel_tol = float(ground_truth.get("rel_tol", 1e-5))
    ordered, _ = table_order_semantics(ground_truth)
    matching_sql = 0
    sql_errors = 0
    answer_bearing_sql = 0
    last_answer_bearing_consistent = False
    last_answer_bearing_index = -1
    if database is not None and gold_ok:
        for sql_index, sql in enumerate(successful_sql):
            try:
                candidate_rows = execute_readonly_sql(
                    database,
                    sql,
                    query_timeout_seconds=timeout_seconds,
                )
                if semantic_rows_equal(
                    candidate_rows,
                    gold_rows,
                    ordered=ordered,
                    abs_tol=abs_tol,
                    rel_tol=rel_tol,
                ):
                    matching_sql += 1
                answer_bearing, consistent = _answer_bearing_and_consistent(
                    candidate_rows, final, ground_truth
                )
                if answer_bearing:
                    answer_bearing_sql += 1
                    last_answer_bearing_index = sql_index
                    last_answer_bearing_consistent = consistent
            except (ValueError, OSError, sqlite3.Error):
                sql_errors += 1

    required_tables = {
        str(value).strip().casefold()
        for value in ground_truth.get("required_tables", [])
        if str(value).strip()
    }
    queried_tables = {
        table
        for sql in successful_sql
        for table in extract_table_names(sql)
    }
    must_fields = [
        str(value).strip()
        for value in ground_truth.get("must_use_fields", [])
        if str(value).strip()
    ]
    successful_sql_text = "\n".join(successful_sql)
    field_hits = sum(_field_used(successful_sql_text, field) for field in must_fields)
    runtime_auto_retry_count = max(
        int(extra_info.get("auto_retry_count", 0) or 0),
        int(extra_info.get("force_final_retry_count", 0) or 0),
    )
    efficiency = efficiency_score(
        attempted_sql,
        commands,
        events,
        runtime_auto_retry_count=runtime_auto_retry_count,
    )
    process_components: dict[str, float | None] = {
        "sql": float(matching_sql > 0),
        "table": float(required_tables.issubset(queried_tables)) if required_tables else 1.0,
        "field": field_hits / len(must_fields) if must_fields else None,
        "fit": float(bool(successful_sql)),
        "efficiency": float(efficiency["score"]),
    }
    observed_process_score, applicable_weight = _normalized_observed_process_score(
        process_components
    )
    process_verified_applicable = bool(answer_bearing_sql)
    process_verified = float(
        process_verified_applicable and last_answer_bearing_consistent
    )
    expected_binding = str(ground_truth.get("process_evidence_binding_sha256") or "")
    actual_binding = evidence_binding_hash(ground_truth)
    evidence_binding_valid = bool(expected_binding and expected_binding == actual_binding)
    hard_gate = bool(
        database_available
        and gold_ok
        and valid_protocol
        and readonly_tools
        and evidence_binding_valid
    )
    try:
        alpha = float(
            extra_info.get(
                "pi_process_bonus_alpha",
                os.environ.get("PI_PROCESS_BONUS_ALPHA", "0"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("pi_process_bonus_alpha must be numeric") from exc
    if not 0.0 <= alpha <= MAX_PROCESS_BONUS_ALPHA:
        raise ValueError(
            f"pi_process_bonus_alpha must be in [0, {MAX_PROCESS_BONUS_ALPHA}]"
        )
    reward = (
        correctness * (1.0 + alpha * process_verified) if hard_gate else 0.0
    )

    return {
        "score": round(reward, 8),
        "acc": correctness,
        "final_answer_correct": correctness,
        "online_eligible": float(hard_gate),
        "hard_gate_passed": float(hard_gate),
        "gold_sql_self_consistent": float(gold_ok),
        "database_available": float(database_available),
        "safe_readonly_tools": float(readonly_tools),
        "valid_tool_protocol": float(valid_protocol),
        "process_evidence_binding_valid": float(evidence_binding_valid),
        "process_evidence_binding_sha256": actual_binding,
        "process_verified": process_verified,
        "process_verified_applicable": float(process_verified_applicable),
        "process_bonus_alpha": alpha,
        "process_bonus_applied": round(alpha * process_verified * correctness, 8),
        "process_score": round(observed_process_score, 8),
        "process_score_observed_only": round(observed_process_score, 8),
        "observed_components_in_reward": 0.0,
        "process_sql": float(process_components["sql"] or 0.0),
        "process_table": float(process_components["table"] or 0.0),
        "process_field": (
            1.0 if process_components["field"] is None else float(process_components["field"])
        ),
        "process_field_applicable": float(process_components["field"] is not None),
        "process_fit": float(process_components["fit"] or 0.0),
        "process_efficiency": float(process_components["efficiency"] or 0.0),
        "process_applicable_weight": applicable_weight,
        "tool_event_count": float(len(events)),
        "attempted_sql_count": float(len(attempted_sql)),
        "successful_sql_count": float(len(successful_sql)),
        "matching_sql_count": float(matching_sql),
        "answer_bearing_sql_count": float(answer_bearing_sql),
        "last_answer_bearing_sql_index": float(last_answer_bearing_index),
        "last_answer_bearing_consistent": float(last_answer_bearing_consistent),
        "sql_replay_error_count": float(sql_errors),
        "required_table_count": float(len(required_tables)),
        "queried_required_table_count": float(len(required_tables & queried_tables)),
        "must_use_field_count": float(len(must_fields)),
        "used_required_field_count": float(field_hits),
        "efficiency_full_scan_count": float(efficiency["full_scan_count"]),
        "efficiency_duplicate_sql_count": float(efficiency["duplicate_sql_count"]),
        "efficiency_duplicate_command_count": float(
            efficiency["duplicate_command_count"]
        ),
        "efficiency_auto_retry_count": float(efficiency["auto_retry_count"]),
        "has_final_answer": float(bool(final["answer"])),
        "final_answer_match_mode": final["match_mode"],
        "numeric_final_parse_mode": final["numeric_parse_mode"],
        "numeric_final_parse_ambiguous": float(final["numeric_parse_ambiguous"]),
        "numeric_final_parse_ambiguity_reason": final["numeric_parse_ambiguity_reason"],
        "numeric_final_parse_candidate_count": float(final["numeric_parse_candidate_count"]),
        "table_comparison_mode": (
            "ordered" if final["table_ordered"] else "row_multiset"
        )
        if final["answer_type"] == "table"
        else "not_table",
        "table_order_semantics_source": final["table_order_source"],
        "strict_table_rows_parsed": float(final["parsed_rows"]),
        "gold_verifier_error": gold_error,
        "reward_contract": REWARD_CONTRACT,
        "reward_scope": "trajectory_level_after_full_multiturn",
        "turn_level_credit_assignment": 0.0,
        "kl_in_reward": 0.0,
    }


def legacy_boss_reward_total_shadow(
    solution_str: str,
    ground_truth: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    complete: bool,
    executable_answer_ok: bool,
) -> dict[str, float]:
    """Reconstruct the deterministic DWH path of boss ``reward_total``.

    The original JSONL event stream was not persisted in these shards, so
    completion is supplied by the shadow adapter.  This function is comparison
    evidence only and is never exposed as a training reward.
    """

    answer = extract_final_assistant_answer(solution_str)
    compact_answer = re.sub(r"\s+", "", answer)
    has_answer = float(bool(complete and len(compact_answer) >= 20))
    expected = _expected_value(ground_truth)
    answer_ok = float(bool(executable_answer_ok or boss_numbers_match(answer, expected)))
    attempted_sql, _, commands = _successful_sql(events)
    required_tables = {
        str(value).strip().casefold()
        for value in ground_truth.get("required_tables", [])
        if str(value).strip()
    }
    used_tables = {table for command in commands for table in extract_table_names(command)}
    tables_ok = float(required_tables.issubset(used_tables)) if required_tables else 1.0
    must_fields = [
        str(value).strip()
        for value in ground_truth.get("must_use_fields", [])
        if str(value).strip()
    ]
    field_score = None
    if must_fields:
        text = "\n".join(commands)
        field_score = sum(_field_used(text, field) for field in must_fields) / len(must_fields)
    task_fit = 1.0 if attempted_sql else 0.5 if any("sqlite3" in c.casefold() for c in commands) else 0.0
    process_items: list[tuple[float, float]] = [(tables_ok, 0.3), (task_fit, 0.2)]
    if field_score is not None:
        process_items.append((field_score, 0.3))
    weight = sum(item_weight for _, item_weight in process_items)
    process = sum(value * item_weight for value, item_weight in process_items) / weight
    result = 0.5 * has_answer + 0.5 * answer_ok
    total = 0.0 if has_answer == 0 else 0.5 * result + 0.5 * process
    return {
        "reward_total": round(total, 8),
        "result_score": round(result, 8),
        "process_score": round(process, 8),
        "has_answer": has_answer,
        "answer_ok": answer_ok,
    }


def private_event_fingerprint(events: list[dict[str, Any]]) -> str:
    """Hash tool names/arguments/results without returning sensitive content."""

    payload = [
        {
            "name": event.get("name"),
            "arguments": event.get("arguments"),
            "ok": event.get("ok"),
            "response_sha256": hashlib.sha256(
                str(event.get("response_preview") or "").encode("utf-8")
            ).hexdigest(),
        }
        for event in events
    ]
    return stable_hash(payload)


def hard_gate_reason_counts(result: dict[str, Any]) -> Counter[str]:
    reasons: Counter[str] = Counter()
    for key in (
        "gold_sql_self_consistent",
        "database_available",
        "safe_readonly_tools",
        "valid_tool_protocol",
        "process_evidence_binding_valid",
    ):
        if not bool(result.get(key)):
            reasons[key] += 1
    return reasons

"""Tiered, query-cost-aware trajectory reward for DWH GRPO shadowing.

The reward is one scalar for the complete multi-turn trajectory. It rewards
attempting and successfully executing task-relevant read-only SQL, keeps final
correctness dominant, and applies a bounded logarithmic query-cost penalty.
Infrastructure and observability failures are UNKNOWN and therefore masked.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import math
import re
import shlex
from typing import Any

from llin_verl.grounded_trajectory_reward import (
    JudgeState,
    _final_decision,
    _model_workspace_path_state,
)
from llin_verl.minimal_grounded_reward import _task_infrastructure
from llin_verl.pi_tool_contract import command_unsafe_reasons, extract_table_names
from llin_verl.trajectory_process_reward import (
    PROTOCOL_TOOLS,
    _MUTATING_SQL_STATEMENT_RE,
    command_executes_sql,
    extract_selects,
)


REWARD_CONTRACT = "tiered-query-cost-trajectory-shadow-v1"
QUERY_SOFT_FREE = 4
QUERY_HARD_LIMIT = 16
TOOL_TOKEN_SOFT_FREE = 4_000
TOOL_TOKEN_HARD_LIMIT = 32_000
_READONLY_SQL_RE = re.compile(r"^\s*(?:select|with)\b", re.IGNORECASE | re.DOTALL)
_FULL_DATABASE_SCAN_RE = re.compile(
    r"(?:\.dump\b|\bselect\s+\*\s+from\s+(?:sqlite_schema|sqlite_master)\b|"
    r"\biterdump\s*\(|\bdump\s+database\b)",
    re.IGNORECASE | re.DOTALL,
)
_TOOL_ERROR_RE = re.compile(
    r"(?:traceback|permissionerror|filenotfounderror|command timed out|"
    r"no such file|not found|\berror\s*:|exception:|exit code\s+[1-9]\d*)",
    re.IGNORECASE,
)


def clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def efficiency_terms(
    *,
    query_attempts: int,
    tool_response_tokens: int,
    irrelevant_ratio: float,
    duplicate_ratio: float,
) -> dict[str, float]:
    """Return the exact frozen v1 logarithmic query-cost terms."""

    q = max(0, int(query_attempts))
    t = max(0, int(tool_response_tokens))
    irr = clip01(irrelevant_ratio)
    dup = clip01(duplicate_ratio)
    eq = clip01(math.log1p(max(q - QUERY_SOFT_FREE, 0)) / math.log1p(8))
    et = clip01(
        math.log1p(max(t - TOOL_TOKEN_SOFT_FREE, 0) / TOOL_TOKEN_SOFT_FREE)
        / math.log1p(3)
    )
    eb = 0.5 * irr + 0.5 * dup
    efficiency = clip01(0.5 * eq + 0.3 * et + 0.2 * eb)
    return {"Eq": eq, "Et": et, "Eb": eb, "E": efficiency}


def _normalize_sql(sql: str) -> str:
    return " ".join(str(sql).strip().rstrip(";").casefold().split())


def extract_sqlite_cli_selects(command: str) -> list[str]:
    """Extract SELECT/WITH payloads from shell-quoted sqlite3 CLI calls."""

    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return []
    statements: list[str] = []
    for index, token in enumerate(tokens):
        executable = token.replace("\\", "/").rsplit("/", 1)[-1].casefold()
        if executable != "sqlite3" or index + 2 >= len(tokens):
            continue
        payload = tokens[index + 2]
        for part in payload.split(";"):
            sql = part.strip()
            if _READONLY_SQL_RE.search(sql):
                statements.append(sql)
    return statements


def _event_sql(event: dict[str, Any]) -> list[str]:
    explicit = event.get("sql_statements")
    if isinstance(explicit, list):
        return [str(value).strip() for value in explicit if str(value).strip()]
    if str(event.get("name") or "") != "bash":
        return []
    command = str((event.get("arguments") or {}).get("command") or "")
    if not command_executes_sql(command):
        return []
    return extract_selects([command]) or extract_sqlite_cli_selects(command)


def _usable_response(event: dict[str, Any]) -> bool:
    if not bool(event.get("ok")):
        return False
    response = str(event.get("response_preview") or "").strip()
    return bool(response) and not bool(_TOOL_ERROR_RE.search(response))


def _observe_process(
    events_value: Any,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
) -> tuple[JudgeState, str, dict[str, Any]]:
    events = events_value if isinstance(events_value, list) else []
    required_tables = {
        str(value).strip().casefold()
        for value in ground_truth.get("required_tables", [])
        if str(value).strip()
    }
    details: dict[str, Any] = {
        "tool_event_count": len(events),
        "query_attempt_count": 0,
        "tool_response_tokens": 0,
        "irrelevant_query_ratio": 0.0,
        "duplicate_query_ratio": 0.0,
        "relevant_readonly_attempt": False,
        "successful_relevant_query": False,
        "unsafe": False,
        "budget_exceeded": False,
        "hard_unsafe_reason_counts": {},
        "runtime_identity_complete": False,
        "request_identity_consistent": False,
        "environment_identity_consistent": False,
        "workspace_identity_consistent": False,
        "tool_response_cost_observable": False,
        "tool_response_token_observed_event_count": 0,
    }

    trajectory_request_id = str(
        extra_info.get("pi_trajectory_request_id")
        or extra_info.get("request_id")
        or ""
    )
    trajectory_environment_id = str(
        extra_info.get("pi_trajectory_environment_id")
        or extra_info.get("pi_environment_id")
        or ""
    )
    expected_environment_id = str(ground_truth.get("environment_id") or "")
    workspace_request_id = str(extra_info.get("pi_workspace_request_id") or "")
    details["request_identity_consistent"] = bool(trajectory_request_id)
    details["environment_identity_consistent"] = bool(
        trajectory_environment_id
        and expected_environment_id
        and trajectory_environment_id == expected_environment_id
        and str(extra_info.get("pi_environment_id") or trajectory_environment_id)
        == expected_environment_id
    )
    details["workspace_identity_consistent"] = (
        not events
        or bool(
            workspace_request_id
            and workspace_request_id == trajectory_request_id
            and bool(extra_info.get("pi_workspace_released"))
            and all(
                isinstance(event, dict)
                and str(event.get("workspace_request_id") or "") == trajectory_request_id
                and str(event.get("environment_id") or "") == expected_environment_id
                for event in events
            )
        )
    )
    details["runtime_identity_complete"] = bool(
        details["request_identity_consistent"]
        and details["environment_identity_consistent"]
        and details["workspace_identity_consistent"]
    )

    malformed = [
        event
        for event in events
        if not isinstance(event, dict)
        or str(event.get("name") or "") not in PROTOCOL_TOOLS
        or not isinstance(event.get("arguments"), dict)
        or not bool(event.get("call_parse_valid", True))
    ]
    if malformed:
        return JudgeState.FAIL, "model_malformed_tool_call", details

    hard_reasons: Counter[str] = Counter()
    sql_attempts: list[str] = []
    relevant_flags: list[bool] = []
    success_flags: list[bool] = []
    response_tokens = 0
    response_observable = True
    for event in events:
        name = str(event.get("name") or "")
        arguments = event.get("arguments") or {}
        if name in {"read", "write", "edit"}:
            path_state = _model_workspace_path_state(arguments.get("path"))
            if path_state == "outside":
                hard_reasons["workspace_path_escape"] += 1
            elif path_state == "unknown":
                response_observable = False
        elif name == "bash":
            command = str(arguments.get("command") or "")
            hard_reasons.update(command_unsafe_reasons(command))
            if _FULL_DATABASE_SCAN_RE.search(command):
                hard_reasons["full_database_scan"] += 1

        if not bool(event.get("observed_tool_response", "response_preview" in event)):
            response_observable = False
        token_count = event.get("response_token_count")
        if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count < 0:
            response_observable = False
        else:
            response_tokens += token_count
            details["tool_response_token_observed_event_count"] += 1

        for sql in _event_sql(event):
            if _MUTATING_SQL_STATEMENT_RE.search(sql):
                hard_reasons["destructive_sql"] += 1
                continue
            if not _READONLY_SQL_RE.search(sql):
                continue
            tables = {value.casefold() for value in extract_table_names(sql)}
            relevant = bool(required_tables & tables)
            sql_attempts.append(sql)
            relevant_flags.append(relevant)
            success_flags.append(relevant and _usable_response(event))

    details["tool_response_tokens"] = response_tokens
    details["query_attempt_count"] = len(sql_attempts)
    details["relevant_readonly_attempt"] = any(relevant_flags)
    details["successful_relevant_query"] = any(success_flags)
    if sql_attempts:
        details["irrelevant_query_ratio"] = sum(not value for value in relevant_flags) / len(sql_attempts)
        normalized = [_normalize_sql(sql) for sql in sql_attempts]
        details["duplicate_query_ratio"] = (len(normalized) - len(set(normalized))) / len(normalized)

    details.update(
        efficiency_terms(
            query_attempts=len(sql_attempts),
            tool_response_tokens=response_tokens,
            irrelevant_ratio=float(details["irrelevant_query_ratio"]),
            duplicate_ratio=float(details["duplicate_query_ratio"]),
        )
    )
    details["budget_exceeded"] = (
        len(sql_attempts) > QUERY_HARD_LIMIT or response_tokens > TOOL_TOKEN_HARD_LIMIT
    )
    details["hard_unsafe_reason_counts"] = dict(sorted(hard_reasons.items()))
    details["unsafe"] = bool(hard_reasons)
    if hard_reasons:
        return JudgeState.FAIL, "model_unsafe_behavior", details
    if details["budget_exceeded"]:
        return JudgeState.FAIL, "hard_budget_exceeded", details

    if not details["runtime_identity_complete"]:
        return JudgeState.UNKNOWN, "runtime_identity_incomplete", details

    timeout = bool(extra_info.get("trajectory_timeout"))
    infrastructure_error = bool(extra_info.get("runtime_error") or extra_info.get("api_error"))
    log_present = bool(
        extra_info.get(
            "pi_tool_log_present",
            extra_info.get("tool_log_present", "pi_tool_events" in extra_info),
        )
    )
    protocol_complete = bool(
        extra_info.get(
            "pi_tool_protocol_complete",
            extra_info.get("tool_protocol_complete", False),
        )
    )
    if timeout or infrastructure_error or not log_present or not protocol_complete:
        return JudgeState.UNKNOWN, "trajectory_or_infrastructure_incomplete", details
    if not response_observable:
        return JudgeState.UNKNOWN, "tool_response_cost_unobservable", details
    details["tool_response_cost_observable"] = True
    return JudgeState.PASS, "observable_safe_process", details


def compute_tiered_query_cost_reward(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
    **_: Any,
) -> dict[str, Any]:
    """Compute the frozen tiered reward v1 for one complete trajectory."""

    del data_source
    task, task_details = _task_infrastructure(ground_truth, extra_info)
    final, final_details = _final_decision(solution_str, ground_truth)
    process_state, process_reason, process_details = _observe_process(
        extra_info.get("pi_tool_events"), ground_truth, extra_info
    )
    attempted = bool(process_details["relevant_readonly_attempt"])
    succeeded = bool(process_details["successful_relevant_query"])
    efficiency = float(process_details["E"])

    if process_details["unsafe"]:
        state, reason, layer, reward = JudgeState.FAIL, "unsafe", "unsafe_or_budget", 0.0
    elif process_details["budget_exceeded"]:
        state, reason, layer, reward = JudgeState.FAIL, "budget_exceeded", "unsafe_or_budget", 0.0
    elif task.state is JudgeState.UNKNOWN:
        state, reason, layer, reward = JudgeState.UNKNOWN, task.reason, "unknown", 0.0
    elif process_state is JudgeState.UNKNOWN:
        state, reason, layer, reward = JudgeState.UNKNOWN, process_reason, "unknown", 0.0
    elif final.state is JudgeState.UNKNOWN:
        state, reason, layer, reward = JudgeState.UNKNOWN, final.reason, "unknown", 0.0
    elif not attempted:
        state, reason, layer, reward = JudgeState.FAIL, "no_relevant_readonly_attempt", "no_attempt", 0.0
    elif not succeeded:
        state, reason, layer = JudgeState.FAIL, "relevant_query_not_successful", "attempt_failed"
        reward = 0.1 * (1.0 - efficiency)
    elif final.state is JudgeState.FAIL:
        state, reason, layer = JudgeState.FAIL, "successful_query_wrong_final", "success_wrong_final"
        reward = 0.2 * (1.0 - efficiency)
    else:
        state, reason, layer = JudgeState.PASS, "successful_query_correct_final", "success_correct_final"
        reward = 1.0 - 0.2 * efficiency

    reward = clip01(reward)
    train_mask = float(state is not JudgeState.UNKNOWN)
    success = float(
        state is JudgeState.PASS
        and succeeded
        and final.state is JudgeState.PASS
    )
    task_identity = str(
        extra_info.get("instruction_sha256")
        or extra_info.get("approved43_instruction_sha256")
        or ""
    )
    request_identity = str(
        extra_info.get("pi_trajectory_request_id")
        or extra_info.get("request_id")
        or extra_info.get("pi_workspace_request_id")
        or ""
    )
    trajectory_identity = (
        hashlib.sha256(f"{task_identity}:{request_identity}".encode("utf-8")).hexdigest()
        if task_identity and request_identity
        else ""
    )
    return {
        "score": reward if train_mask else 0.0,
        "reward": reward if train_mask else 0.0,
        "train_mask": train_mask,
        # The strict GRPO group gate uses this binary outcome only to decide
        # whether a prompt group is mixed. The scalar tiered reward above is
        # still what compute_advantage consumes inside an admitted group.
        "success": success,
        "online_eligible": train_mask,
        "judge_state": state.value,
        "judge_reason": reason,
        "reward_layer": layer,
        "final_answer_correct": float(final.state is JudgeState.PASS),
        "attempted_relevant_readonly_sql": float(attempted),
        "successful_relevant_readonly_sql": float(succeeded),
        "guess_correct_blocked": float(final.state is JudgeState.PASS and not attempted),
        "reward_contract": REWARD_CONTRACT,
        "reward_scope": "trajectory_level_after_full_multiturn",
        "turn_level_credit_assignment": 0.0,
        "task_identity_sha256": task_identity,
        "trajectory_identity_sha256": trajectory_identity,
        "sampling_policy_version_min": extra_info.get("min_global_steps"),
        "sampling_policy_version_max": extra_info.get("max_global_steps"),
        "wrong_reward_upper_bound_inclusive": 0.2,
        "correct_grounded_reward_lower_bound_inclusive": 0.8,
        "final_answer_match_mode": final_details.get("match_mode", "none"),
        **task_details,
        **process_details,
    }

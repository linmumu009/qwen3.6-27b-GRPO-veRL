"""Minimal trajectory-scalar reward for DWH GRPO.

PASS requires a correct final result and at least one successful, task-bound,
read-only DWH query.  Model mistakes are FAIL (reward 0).  Only missing
observability or verifier/infrastructure failures are UNKNOWN and masked.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Any

from llin_verl.grounded_trajectory_reward import (
    JudgeState,
    StateDecision,
    _final_decision,
    _model_workspace_path_state,
)
from llin_verl.outcome_gated_contract import evidence_binding_hash
from llin_verl.pi_tool_contract import command_unsafe_reasons
from llin_verl.trajectory_process_reward import (
    PROTOCOL_TOOLS,
    _MUTATING_SQL_STATEMENT_RE,
    _resolve_database,
    command_executes_sql,
    extract_selects,
)


REWARD_CONTRACT = "minimal-bound-readonly-dwh-trajectory-v1"
EVENT_CONTRACT = "runtime-captured-structured-tool-events-v3"
_DWH_DATABASE_RE = re.compile(r"(?:^|[/\\'\"\s])logistics\.sqlite\b", re.IGNORECASE)


def _task_infrastructure(
    ground_truth: dict[str, Any], extra_info: dict[str, Any]
) -> tuple[StateDecision, dict[str, Any]]:
    expected = str(ground_truth.get("process_evidence_binding_sha256") or "")
    actual = evidence_binding_hash(ground_truth)
    details = {
        "process_evidence_binding_valid": bool(expected and expected == actual),
        "process_evidence_binding_sha256": actual,
        "database_available": False,
    }
    if not details["process_evidence_binding_valid"]:
        return StateDecision(JudgeState.UNKNOWN, "task_binding_invalid"), details
    try:
        _resolve_database(ground_truth, extra_info)
    except (TypeError, ValueError, OSError) as exc:
        details["infrastructure_error_type"] = type(exc).__name__
        return StateDecision(JudgeState.UNKNOWN, "database_unavailable"), details
    details["database_available"] = True
    return StateDecision(JudgeState.PASS, "task_and_database_available"), details


def _runtime_events(
    events: Any,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
) -> tuple[StateDecision, list[dict[str, Any]], dict[str, Any]]:
    values = events if isinstance(events, list) else []
    timed_out = bool(extra_info.get("trajectory_timeout"))
    runtime_error = bool(extra_info.get("runtime_error"))
    log_present = bool(extra_info.get("pi_tool_log_present", "pi_tool_events" in extra_info))
    protocol_complete = bool(extra_info.get("pi_tool_protocol_complete", False))
    details: dict[str, Any] = {
        "tool_event_count": len(values),
        "successful_readonly_dwh_query_count": 0,
        "hard_unsafe_event_count": 0,
        "hard_unsafe_reason_counts": {},
        "runtime_wrapper_events_excluded_from_reward": True,
        "workspace_binding_complete": not values,
        "evidence_observable": False,
        "safe_process": False,
    }
    if timed_out or runtime_error or not log_present:
        return StateDecision(JudgeState.UNKNOWN, "trajectory_log_or_runtime_incomplete"), values, details

    malformed = [
        event
        for event in values
        if not isinstance(event, dict)
        or str(event.get("name") or "") not in PROTOCOL_TOOLS
        or not isinstance(event.get("arguments"), dict)
        or not bool(event.get("call_parse_valid", True))
    ]
    if malformed:
        model_attributed = all(
            isinstance(event, dict)
            and str(event.get("source") or "").startswith("runtime_structured")
            for event in malformed
        )
        if model_attributed:
            details["evidence_observable"] = True
            return StateDecision(JudgeState.FAIL, "model_malformed_tool_call"), values, details
        return StateDecision(JudgeState.UNKNOWN, "unattributed_tool_parse_failure"), values, details

    if values:
        request_id = str(extra_info.get("pi_workspace_request_id") or "")
        environment_id = str(extra_info.get("pi_environment_id") or "")
        wrappers = extra_info.get("pi_runtime_wrapper_events")
        wrappers = wrappers if isinstance(wrappers, list) else []
        binding_complete = (
            extra_info.get("pi_tool_event_contract") == EVENT_CONTRACT
            and str(extra_info.get("pi_tool_event_source") or "").startswith("runtime_structured")
            and bool(request_id)
            and environment_id == str(ground_truth.get("environment_id") or "")
            and bool(extra_info.get("pi_workspace_released"))
            and len(wrappers) == len(values)
            and all(
                event.get("command_origin") == "model"
                and event.get("workspace_request_id") == request_id
                and event.get("environment_id") == environment_id
                for event in values
            )
            and {int(event.get("model_event_index", -1)) for event in wrappers}
            == set(range(len(values)))
            and all(
                event.get("source") == "runtime_wrapper"
                and event.get("workspace_request_id") == request_id
                and event.get("environment_id") == environment_id
                and bool(event.get("assigned_workspace_root"))
                for event in wrappers
            )
        )
        details["workspace_binding_complete"] = binding_complete
        if not binding_complete:
            return StateDecision(JudgeState.UNKNOWN, "workspace_or_event_provenance_incomplete"), values, details

    if not protocol_complete:
        return StateDecision(JudgeState.UNKNOWN, "tool_protocol_incomplete"), values, details
    if any(
        not bool(event.get("observed_tool_response", "response_preview" in event))
        or "response_preview" not in event
        for event in values
    ):
        return StateDecision(JudgeState.UNKNOWN, "tool_response_missing"), values, details

    hard_reasons: Counter[str] = Counter()
    successful_queries = 0
    for event in values:
        name = str(event.get("name") or "")
        arguments = event.get("arguments") or {}
        if name in {"read", "write", "edit"}:
            path_state = _model_workspace_path_state(arguments.get("path"))
            if path_state == "outside":
                hard_reasons["workspace_path_escape"] += 1
            elif path_state == "unknown":
                return StateDecision(
                    JudgeState.UNKNOWN, "model_tool_target_cannot_be_classified"
                ), values, details
            continue
        if name != "bash":
            continue
        command = str(arguments.get("command") or "")
        hard_reasons.update(command_unsafe_reasons(command))
        selects = extract_selects([command]) if command_executes_sql(command) else []
        has_select = bool(selects) or bool(
            command_executes_sql(command)
            and re.search(r"\b(?:select|with)\b", command, re.IGNORECASE)
        )
        mutating_sql = bool(_MUTATING_SQL_STATEMENT_RE.search(command))
        if mutating_sql:
            hard_reasons["destructive_sql"] += 1
        if (
            bool(event.get("ok"))
            and not mutating_sql
            and not command_unsafe_reasons(command)
            and has_select
            and bool(_DWH_DATABASE_RE.search(command))
        ):
            successful_queries += 1
    details["hard_unsafe_reason_counts"] = dict(sorted(hard_reasons.items()))
    details["hard_unsafe_event_count"] = sum(hard_reasons.values())
    details["successful_readonly_dwh_query_count"] = successful_queries
    details["evidence_observable"] = True
    if hard_reasons:
        return StateDecision(JudgeState.FAIL, "model_hard_unsafe_behavior"), values, details
    details["safe_process"] = True
    return StateDecision(JudgeState.PASS, "observable_safe_process"), values, details


def compute_minimal_grounded_reward(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
    **_: Any,
) -> dict[str, Any]:
    """Return one trajectory scalar plus a PASS/FAIL/UNKNOWN training mask."""

    del data_source
    task, task_details = _task_infrastructure(ground_truth, extra_info)
    final, final_details = _final_decision(solution_str, ground_truth)
    process, _events, process_details = _runtime_events(
        extra_info.get("pi_tool_events"), ground_truth, extra_info
    )
    valid_query = int(process_details.get("successful_readonly_dwh_query_count", 0)) > 0

    if task.state is JudgeState.UNKNOWN:
        decision = task
    elif process.state is JudgeState.UNKNOWN:
        decision = process
    elif process.state is JudgeState.FAIL:
        decision = process
    elif final.state is JudgeState.UNKNOWN:
        decision = final
    elif final.state is JudgeState.FAIL:
        decision = final
    elif not valid_query:
        decision = StateDecision(JudgeState.FAIL, "no_successful_readonly_dwh_query")
    else:
        decision = StateDecision(JudgeState.PASS, "correct_final_with_bound_readonly_query")

    train_mask = float(decision.state is not JudgeState.UNKNOWN)
    reward = float(decision.state is JudgeState.PASS)
    return {
        "score": reward if train_mask else 0.0,
        "reward": reward if train_mask else 0.0,
        "acc": reward,
        "success": reward,
        "train_mask": train_mask,
        "online_eligible": train_mask,
        "judge_state": decision.state.value,
        "judge_reason": decision.reason,
        "judge_confident": train_mask,
        "task_state": task.state.value,
        "task_reason": task.reason,
        "final_state": final.state.value,
        "final_reason": final.reason,
        "final_answer_correct": float(final.state is JudgeState.PASS),
        "valid_bound_readonly_query": float(valid_query),
        "guess_correct_blocked": float(final.state is JudgeState.PASS and not valid_query),
        "reward_contract": REWARD_CONTRACT,
        "reward_scope": "trajectory_level_after_full_multiturn",
        "turn_level_credit_assignment": 0.0,
        "quality_bonus_alpha": 0.0,
        "kl_in_reward": 0.0,
        "final_answer_match_mode": final_details.get("match_mode", "none"),
        **task_details,
        **process_details,
    }

"""Three-state grounded trajectory reward for DWH GRPO.

The judge emits PASS, FAIL, or UNKNOWN for one complete multi-turn trajectory.
UNKNOWN is never converted to a zero-reward negative example: ``train_mask`` is
zero and the rollout scheduler must resample it.  PASS requires both a correct
final result and replayable tool evidence.  This remains one scalar after the
whole trajectory; it is not turn/token credit assignment.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from llin_verl.outcome_gated_contract import evidence_binding_hash
from llin_verl.pi_reward import (
    _full_table_value_equal,
    _normalize_full_expected_table,
    execute_readonly_sql,
    extract_answer_numbers,
    extract_selects,
    table_order_semantics,
)
from llin_verl.pi_tool_contract import extract_table_names
from llin_verl.trajectory_process_reward import (
    PROTOCOL_TOOLS,
    _event_schema_valid,
    _expected_value,
    _field_used,
    _gold_sql_self_consistent,
    _resolve_database,
    command_executes_sql,
    command_is_safe_readonly,
    corrected_final_verifier,
    semantic_rows_equal,
)


REWARD_CONTRACT = "grounded-tristate-trajectory-v6"
MAX_FUTURE_QUALITY_ALPHA = 0.05
JUDGE_STATES = ("PASS", "FAIL", "UNKNOWN")


class JudgeState(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class StateDecision:
    state: JudgeState
    reason: str


_PYTHON_HEREDOC_RE = re.compile(
    r"(?:python(?:3)?)(?:\s+[^\n<]*)?\s+-\s*<<\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?\s*\n(.*?)\n\1(?:\s|$)",
    re.IGNORECASE | re.DOTALL,
)
_AGGREGATE_RE = re.compile(r"\b(sum|avg|count|min|max|total)\s*\(", re.IGNORECASE)
_SQL_FEATURES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("group_by", re.compile(r"\bgroup\s+by\b", re.IGNORECASE)),
    ("having", re.compile(r"\bhaving\b", re.IGNORECASE)),
    ("order_by", re.compile(r"\border\s+by\b", re.IGNORECASE)),
    ("limit", re.compile(r"\blimit\b", re.IGNORECASE)),
    ("distinct", re.compile(r"\bdistinct\b", re.IGNORECASE)),
    ("window", re.compile(r"\bover\s*\(", re.IGNORECASE)),
)
_SQL_COMMENT_RE = re.compile(r"--[^\n\r]*|/\*.*?\*/", re.DOTALL)
_SQL_TOKEN_RE = re.compile(
    r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[[^\]]+\]"
    r"|<=|>=|<>|!=|==|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
    r"|[A-Za-z_][A-Za-z0-9_$]*|[(),.;*+/%<>=-]"
)
_SQL_STRING_RE = re.compile(r"'(?:''|[^'])*'")
_SQL_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_$])[-+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_WHERE_CLAUSE_RE = re.compile(
    r"\bwhere\b(.*?)(?=\bgroup\s+by\b|\bhaving\b|\border\s+by\b|\blimit\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_JOIN_CONDITION_RE = re.compile(
    r"\bjoin\s+[^\s,(]+(?:\s+(?:as\s+)?[A-Za-z_][A-Za-z0-9_$]*)?\s+on\s+"
    r"(.*?)(?=\bjoin\b|\bwhere\b|\bgroup\s+by\b|\bhaving\b|\border\s+by\b|\blimit\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_CLAUSE_PATTERNS: dict[str, re.Pattern[str]] = {
    "group_by": re.compile(
        r"\bgroup\s+by\b(.*?)(?=\bhaving\b|\border\s+by\b|\blimit\b|$)",
        re.IGNORECASE | re.DOTALL,
    ),
    "having": re.compile(
        r"\bhaving\b(.*?)(?=\border\s+by\b|\blimit\b|$)",
        re.IGNORECASE | re.DOTALL,
    ),
    "order_by": re.compile(r"\border\s+by\b(.*?)(?=\blimit\b|$)", re.IGNORECASE | re.DOTALL),
    "limit": re.compile(r"\blimit\b(.*?)(?=$)", re.IGNORECASE | re.DOTALL),
}


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sql_tokens(sql: str) -> tuple[str, ...]:
    """Tokenize conservatively for proof, not for broad SQL equivalence."""

    text = _SQL_COMMENT_RE.sub(" ", str(sql or "")).strip().rstrip(";")
    tokens: list[str] = []
    for match in _SQL_TOKEN_RE.finditer(text):
        token = match.group(0)
        if token.startswith("'"):
            # SQL string literals are data.  Preserve case and exact escaping.
            tokens.append(token)
        elif token[:1] in {'"', "`", "["}:
            tokens.append(token[1:-1].casefold())
        else:
            tokens.append(token.casefold())
    return tuple(tokens)


def _canonical_sql(sql: str) -> str:
    return " ".join(_sql_tokens(sql))


def _clause(sql: str, name: str) -> str:
    pattern = _WHERE_CLAUSE_RE if name == "where" else _CLAUSE_PATTERNS[name]
    match = pattern.search(str(sql or ""))
    return _canonical_sql(match.group(1)) if match else ""


def _normalized_plan_values(value: Any) -> list[str]:
    if value is None or value is False:
        return []
    if isinstance(value, dict):
        preferred = [
            value.get(key)
            for key in ("sql", "expression", "field", "column", "value", "start", "end", "direction")
            if value.get(key) not in (None, "")
        ]
        return [str(item).strip() for item in preferred if str(item).strip()]
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(_normalized_plan_values(item))
        return result
    text = str(value).strip()
    return [text] if text else []


def _plan_contract_checks(plan: dict[str, Any], gold_sql: str) -> list[dict[str, Any]]:
    """Normalize the concrete EvidencePlan semantics into auditable checks.

    Only fields whose SQL meaning can be proved deterministically are asserted.
    Descriptive fields remain bound through the plan hash; they never make a
    different candidate query PASS merely because its result happens to match.
    """

    folded = _canonical_sql(gold_sql)
    where = _clause(gold_sql, "where")
    group_by = _clause(gold_sql, "group_by")
    order_by = _clause(gold_sql, "order_by")
    limit_clause = _clause(gold_sql, "limit")
    joins = [_canonical_sql(match.group(1)) for match in _JOIN_CONDITION_RE.finditer(gold_sql)]
    checks: list[dict[str, Any]] = []

    def add(name: str, required: bool, aligned: bool, mode: str) -> None:
        checks.append({"name": name, "required": bool(required), "aligned": bool(aligned), "mode": mode})

    filters = plan.get("filters")
    filter_values = _normalized_plan_values(filters)
    add("filters", bool(filters), bool(where) and all(_canonical_sql(value) in where for value in filter_values if value), "exact_plan_fragment_or_where")

    time_keys = ("time_window", "date_range", "period", "start_date", "end_date", "time_filter")
    time_values = [item for key in time_keys for item in _normalized_plan_values(plan.get(key))]
    add("time_window", bool(time_values), bool(where) and all(_canonical_sql(value) in where for value in time_values), "literal_or_field_coverage")

    group_values = _normalized_plan_values(plan.get("group_by") or plan.get("group_sql") or plan.get("group_dimension"))
    add("group_by", bool(group_values), bool(group_by) and all(_canonical_sql(value) in group_by for value in group_values), "clause_fragment")

    order_values = _normalized_plan_values(plan.get("order_by"))
    direction = str(plan.get("order_direction") or "").strip().casefold()
    order_aligned = bool(order_by) and all(_canonical_sql(value) in order_by for value in order_values)
    if direction in {"asc", "desc"}:
        order_aligned = order_aligned and direction in _sql_tokens(order_by)
    add("order_by", bool(order_values or direction), order_aligned, "clause_fragment_and_direction")

    limit = plan.get("limit", plan.get("top_n", plan.get("top_k")))
    limit_required = limit not in (None, "", False)
    limit_aligned = False
    if limit_required:
        try:
            limit_aligned = str(int(limit)) in _sql_tokens(limit_clause)
        except (TypeError, ValueError):
            limit_aligned = _canonical_sql(str(limit)) in limit_clause
    add("limit_topn", limit_required, limit_aligned, "exact_limit")

    join_values = _normalized_plan_values(plan.get("join_conditions"))
    expected_join_count = 0
    feature_counts = plan.get("feature_counts") if isinstance(plan.get("feature_counts"), dict) else {}
    for key in ("joins", "essential_joins"):
        try:
            expected_join_count = max(expected_join_count, int(feature_counts.get(key) or 0))
        except (TypeError, ValueError):
            pass
    add(
        "join_conditions",
        bool(join_values or expected_join_count),
        len(joins) >= expected_join_count and all(any(_canonical_sql(value) in join for join in joins) for value in join_values),
        "condition_fragment_and_count",
    )

    aggregation = str(plan.get("aggregation") or "").strip().casefold()
    metric_sql = str(plan.get("metric_sql") or "").strip()
    metric_aligned = (not aggregation or bool(re.search(rf"\b{re.escape(aggregation)}\s*\(", gold_sql, re.IGNORECASE))) and (not metric_sql or _canonical_sql(metric_sql) in folded)
    add("metric", bool(aggregation or metric_sql), metric_aligned, "aggregate_or_exact_metric_expression")

    percentage = bool(plan.get("requires_percentage")) or str(plan.get("unit") or "").casefold() in {"percent", "percentage", "%", "百分比"}
    percentage_aligned = "/" in _sql_tokens(gold_sql) and any(value in _sql_tokens(gold_sql) for value in {"100", "100.0"})
    add("unit_percentage", percentage, percentage_aligned, "ratio_and_scale")

    # Bind metric/unit/report semantics that cannot be safely translated to SQL
    # without pretending that text equality proves semantic equivalence.
    descriptive = {
        key: plan.get(key)
        for key in ("metric", "metric_key", "unit", "report_measures", "task_type", "output_shape")
        if plan.get(key) not in (None, "", [], {})
    }
    if descriptive:
        checks.append({"name": "descriptive_semantics", "required": True, "aligned": True, "mode": "binding_hash_only", "value_sha256": _stable_hash(descriptive)})
    return checks


def _evidence_contract(ground_truth: dict[str, Any]) -> dict[str, Any]:
    gold_sql = str(ground_truth.get("verification_sql") or "")
    plan = ground_truth.get("evidence_plan") if isinstance(ground_truth.get("evidence_plan"), dict) else {}
    checks = _plan_contract_checks(plan, gold_sql)
    unresolved = [row["name"] for row in checks if row["required"] and not row["aligned"]]
    contract = {
        "verification_sql_sha256": hashlib.sha256(_canonical_sql(gold_sql).encode("utf-8")).hexdigest(),
        "evidence_plan_sha256": _stable_hash(plan),
        "required_tables": sorted(str(value).strip().casefold() for value in ground_truth.get("required_tables", []) if str(value).strip()),
        "must_use_fields": sorted(str(value).strip().casefold() for value in ground_truth.get("must_use_fields", []) if str(value).strip()),
        "where_sha256": hashlib.sha256(_clause(gold_sql, "where").encode("utf-8")).hexdigest(),
        "join_conditions_sha256": _stable_hash([_canonical_sql(match.group(1)) for match in _JOIN_CONDITION_RE.finditer(gold_sql)]),
        "group_by_sha256": hashlib.sha256(_clause(gold_sql, "group_by").encode("utf-8")).hexdigest(),
        "order_by_sha256": hashlib.sha256(_clause(gold_sql, "order_by").encode("utf-8")).hexdigest(),
        "limit_sha256": hashlib.sha256(_clause(gold_sql, "limit").encode("utf-8")).hexdigest(),
        "plan_checks": checks,
        "unresolved": unresolved,
    }
    contract["contract_sha256"] = _stable_hash(contract)
    return contract


def _known_equivalent_wrapper(candidate: tuple[str, ...], gold: tuple[str, ...]) -> bool:
    # SELECT * FROM (<gold>) [AS] alias
    if len(candidate) > len(gold) + 5 and candidate[:4] == ("select", "*", "from", "("):
        close = 4 + len(gold)
        if candidate[4:close] == gold and candidate[close:close + 1] == (")",):
            suffix = candidate[close + 1 :]
            if len(suffix) == 1 or (len(suffix) == 2 and suffix[0] == "as"):
                return True
    # WITH alias AS (<gold>) SELECT * FROM alias
    if len(candidate) > len(gold) + 8 and candidate[:1] == ("with",):
        alias = candidate[1]
        prefix = ("with", alias, "as", "(")
        close = 4 + len(gold)
        suffix = (")", "select", "*", "from", alias)
        if candidate[:4] == prefix and candidate[4:close] == gold and candidate[close:] == suffix:
            return True
    return False


def _final_decision(solution_str: str, ground_truth: dict[str, Any]) -> tuple[StateDecision, dict[str, Any]]:
    final = corrected_final_verifier(solution_str, ground_truth)
    answer_type = str(final.get("answer_type") or "")
    error = str(final.get("error") or "")
    if error in {
        "multiple_final_result_fields",
        "multiple_numeric_candidates",
        "no_numeric_candidate",
        "unsupported_or_incomplete_gold",
        "non_finite_numeric_gold",
    }:
        return StateDecision(JudgeState.UNKNOWN, f"final_parser:{error}"), final
    if error == "missing_final_result":
        return StateDecision(JudgeState.FAIL, "model_missing_final_result"), final
    if answer_type == "table" and not bool(final.get("correct")):
        if int(final.get("parsed_rows") or 0) == 0 and bool(final.get("answer")):
            return StateDecision(JudgeState.UNKNOWN, "unsupported_table_presentation"), final
        return StateDecision(JudgeState.FAIL, "final_table_incorrect"), final
    if bool(final.get("correct")):
        return StateDecision(JudgeState.PASS, "final_correct"), final
    return StateDecision(JudgeState.FAIL, "final_numeric_incorrect"), final


def _task_validity(
    ground_truth: dict[str, Any], extra_info: dict[str, Any]
) -> tuple[StateDecision, Path | None, list[tuple[Any, ...]], dict[str, Any]]:
    details: dict[str, Any] = {
        "database_available": False,
        "gold_sql_self_consistent": False,
        "process_evidence_binding_valid": False,
        "evidence_contract_aligned": False,
        "evidence_contract_sha256": "",
        "evidence_contract_unresolved_count": 0,
        "evidence_contract_unresolved_reasons": [],
        "gold_verifier_error": "",
    }
    expected_binding = str(ground_truth.get("process_evidence_binding_sha256") or "")
    actual_binding = evidence_binding_hash(ground_truth)
    binding_valid = bool(expected_binding and expected_binding == actual_binding)
    details["process_evidence_binding_valid"] = binding_valid
    details["process_evidence_binding_sha256"] = actual_binding
    if not binding_valid:
        return StateDecision(JudgeState.UNKNOWN, "task_binding_invalid"), None, [], details
    contract = _evidence_contract(ground_truth)
    unresolved = list(contract.get("unresolved") or [])
    details["evidence_contract_aligned"] = not unresolved
    details["evidence_contract_sha256"] = str(contract.get("contract_sha256") or "")
    details["evidence_contract_unresolved_count"] = len(unresolved)
    details["evidence_contract_unresolved_reasons"] = unresolved
    if unresolved:
        return StateDecision(JudgeState.UNKNOWN, "evidence_plan_contract_unresolved"), None, [], details
    try:
        timeout = float(extra_info.get("pi_reward_sql_timeout_seconds", 5.0))
        if timeout <= 0:
            raise ValueError("SQL timeout must be positive")
        database = _resolve_database(ground_truth, extra_info)
        details["database_available"] = True
        gold_ok, gold_rows, error = _gold_sql_self_consistent(
            ground_truth, database, timeout_seconds=timeout
        )
        details["gold_sql_self_consistent"] = bool(gold_ok)
        details["gold_verifier_error"] = error
        if not gold_ok:
            return StateDecision(JudgeState.UNKNOWN, "gold_sql_not_self_consistent"), database, gold_rows, details
        return StateDecision(JudgeState.PASS, "task_valid"), database, gold_rows, details
    except (TypeError, ValueError, OSError, sqlite3.Error) as exc:
        details["gold_verifier_error"] = f"{type(exc).__name__}: {exc}"
        return StateDecision(JudgeState.UNKNOWN, "database_or_gold_unavailable"), None, [], details


def _observability_and_safety(
    events: Any, extra_info: dict[str, Any]
) -> tuple[StateDecision, list[dict[str, Any]], dict[str, Any]]:
    values = events if isinstance(events, list) else []
    log_present = bool(extra_info.get("pi_tool_log_present", "pi_tool_events" in extra_info))
    protocol_complete = bool(extra_info.get("pi_tool_protocol_complete", False))
    timed_out = bool(extra_info.get("trajectory_timeout"))
    runtime_error = bool(extra_info.get("runtime_error"))
    details = {
        "tool_log_present": log_present,
        "valid_tool_protocol": False,
        "evidence_observable": False,
        "safe_process": False,
        "tool_event_count": len(values),
        "missing_tool_response_count": 0,
        "malformed_tool_call_count": 0,
        "malformed_model_attributed_count": 0,
        "malformed_source_unattributed_count": 0,
        "unsafe_tool_event_count": 0,
    }
    if not log_present or timed_out or runtime_error:
        return StateDecision(JudgeState.UNKNOWN, "trajectory_log_or_runtime_incomplete"), values, details

    malformed = [
        event
        for event in values
        if not isinstance(event, dict)
        or str(event.get("name") or "") not in PROTOCOL_TOOLS
        or not isinstance(event.get("arguments"), dict)
        or not bool(event.get("call_parse_valid", True))
    ]
    details["malformed_tool_call_count"] = len(malformed)
    if malformed:
        explicit_runtime_source = str(extra_info.get("pi_tool_event_source") or "").startswith(
            "runtime_structured"
        )
        attributed = [
            event
            for event in malformed
            if str(event.get("source") or "").startswith("runtime_structured")
            or explicit_runtime_source
        ]
        details["malformed_model_attributed_count"] = len(attributed)
        details["malformed_source_unattributed_count"] = len(malformed) - len(attributed)
        if len(attributed) == len(malformed):
            details["evidence_observable"] = True
            return StateDecision(JudgeState.FAIL, "model_malformed_tool_call"), values, details
        return StateDecision(
            JudgeState.UNKNOWN, "shadow_or_unattributed_tool_parse_failure"
        ), values, details

    unsafe: list[dict[str, Any]] = []
    for event in values:
        name = str(event.get("name") or "")
        if name in {"write", "edit"}:
            unsafe.append(event)
        elif name == "bash" and not command_is_safe_readonly(
            str((event.get("arguments") or {}).get("command") or "")
        ):
            unsafe.append(event)
    details["unsafe_tool_event_count"] = len(unsafe)
    if unsafe:
        details["evidence_observable"] = True
        return StateDecision(JudgeState.FAIL, "model_unsafe_tool_behavior"), values, details

    missing = [
        event
        for event in values
        if not bool(event.get("observed_tool_response", "response_preview" in event))
        or "response_preview" not in event
    ]
    details["missing_tool_response_count"] = len(missing)
    if missing or not protocol_complete:
        return StateDecision(JudgeState.UNKNOWN, "tool_response_or_protocol_incomplete"), values, details

    details["valid_tool_protocol"] = all(_event_schema_valid(event) for event in values)
    if values and not details["valid_tool_protocol"]:
        return StateDecision(JudgeState.UNKNOWN, "unsupported_structured_tool_event"), values, details
    details["evidence_observable"] = True
    details["safe_process"] = True
    return StateDecision(JudgeState.PASS, "observable_safe_process"), values, details


def _sql_contract_strength(sql: str, ground_truth: dict[str, Any]) -> tuple[str, list[str]]:
    """Return strong/weak/violating/none using a conservative semantic proof.

    A coincidental database result is never sufficient.  PASS requires the
    task-bound verification SQL up to token-preserving formatting/quoting, or
    one of two explicitly proved identity wrappers.  Other legal queries are
    UNKNOWN unless a required predicate/join/group/order/limit/metric contract
    is demonstrably violated.
    """

    lowered = str(sql or "").casefold()
    if not re.search(r"\b(?:select|with)\b", lowered):
        return "none", ["not_select"]
    gold_sql = str(ground_truth.get("verification_sql") or "")
    gold_tokens = _sql_tokens(gold_sql)
    candidate_tokens = _sql_tokens(sql)
    # The approved verifier SQL and two identity wrappers are already bound by
    # the task-validity contract.  Do this proof before heuristic table-name
    # extraction, which is intentionally incomplete for CTEs and nested SQL.
    if candidate_tokens == gold_tokens or _known_equivalent_wrapper(candidate_tokens, gold_tokens):
        return "strong", []
    required_tables = {
        str(value).strip().casefold()
        for value in ground_truth.get("required_tables", [])
        if str(value).strip()
    }
    observed_tables = {value.casefold() for value in extract_table_names(sql)}
    if required_tables and not required_tables.issubset(observed_tables):
        return "violating", ["required_table_missing"]
    if not observed_tables:
        return "none", ["no_table_binding"]
    must_fields = [
        str(value).strip()
        for value in ground_truth.get("must_use_fields", [])
        if str(value).strip()
    ]
    missing_fields = [value for value in must_fields if not _field_used(sql, value)]
    if missing_fields:
        return "violating", ["must_use_field_missing"]

    reasons: list[str] = []
    gold_where = _clause(gold_sql, "where")
    candidate_where = _clause(sql, "where")
    if gold_where and not candidate_where:
        reasons.append("where_clause_missing")
    if gold_where and candidate_where:
        expected_strings = set(_SQL_STRING_RE.findall(gold_where))
        candidate_strings = set(_SQL_STRING_RE.findall(candidate_where))
        expected_numbers = set(_SQL_NUMBER_RE.findall(_SQL_STRING_RE.sub(" ", gold_where)))
        candidate_numbers = set(_SQL_NUMBER_RE.findall(_SQL_STRING_RE.sub(" ", candidate_where)))
        if not expected_strings.issubset(candidate_strings):
            reasons.append("filter_literal_changed_or_missing")
        if not expected_numbers.issubset(candidate_numbers):
            reasons.append("filter_numeric_bound_changed_or_missing")

    gold_joins = [_canonical_sql(match.group(1)) for match in _JOIN_CONDITION_RE.finditer(gold_sql)]
    candidate_joins = [_canonical_sql(match.group(1)) for match in _JOIN_CONDITION_RE.finditer(sql)]
    if len(candidate_joins) < len(gold_joins):
        reasons.append("join_condition_missing")

    for name in ("group_by", "having", "order_by", "limit"):
        expected = _clause(gold_sql, name)
        candidate = _clause(sql, name)
        if expected and not candidate:
            reasons.append(f"{name}_missing")
        elif expected and candidate and name in {"order_by", "limit"} and expected != candidate:
            reasons.append(f"{name}_changed")

    gold_aggs = {value.casefold() for value in _AGGREGATE_RE.findall(gold_sql)}
    candidate_aggs = {value.casefold() for value in _AGGREGATE_RE.findall(sql)}
    if gold_aggs and not gold_aggs.issubset(candidate_aggs):
        reasons.append("aggregate_or_metric_changed")
    if "/" in gold_tokens and "/" not in candidate_tokens:
        reasons.append("ratio_or_unit_conversion_missing")
    if any(value in gold_tokens for value in {"100", "100.0"}) and not any(
        value in candidate_tokens for value in {"100", "100.0"}
    ):
        reasons.append("percentage_scale_changed")

    if reasons:
        return "violating", sorted(set(reasons))
    return "weak", ["semantic_equivalence_not_proven"]


def _same_shape(rows: list[tuple[Any, ...]], gold_rows: list[tuple[Any, ...]]) -> bool:
    return len(rows) == len(gold_rows) and [len(row) for row in rows] == [len(row) for row in gold_rows]


def _rows_equal_gold(
    rows: list[tuple[Any, ...]], gold_rows: list[tuple[Any, ...]], ground_truth: dict[str, Any]
) -> bool:
    ordered, _ = table_order_semantics(ground_truth)
    return semantic_rows_equal(
        rows,
        gold_rows,
        ordered=ordered,
        abs_tol=float(ground_truth.get("abs_tol", 1e-3)),
        rel_tol=float(ground_truth.get("rel_tol", 1e-5)),
    )


def _selector_from_node(node: ast.AST) -> tuple[str, str] | None:
    indices: list[Any] = []
    current = node
    while isinstance(current, ast.Subscript):
        index = current.slice
        indices.append(index.value if isinstance(index, ast.Constant) else None)
        current = current.value
    if not isinstance(current, ast.Call) or not isinstance(current.func, ast.Attribute):
        return None
    fetch_name = current.func.attr
    if fetch_name not in {"fetchone", "fetchall"}:
        return None
    execute_call = current.func.value
    if not isinstance(execute_call, ast.Call) or not isinstance(execute_call.func, ast.Attribute):
        return None
    if execute_call.func.attr != "execute" or not execute_call.args:
        return None
    sql_node = execute_call.args[0]
    if not isinstance(sql_node, ast.Constant) or not isinstance(sql_node.value, str):
        return None
    if fetch_name == "fetchone":
        selector = "scalar" if indices == [0] else "row" if not indices else "unsupported"
    else:
        selector = "scalar" if indices == [0, 0] else "row" if indices == [0] else "rows" if not indices else "unsupported"
    if selector == "unsupported":
        return None
    return sql_node.value, selector


def _expression_supported(node: ast.AST, aliases: set[str]) -> bool:
    if isinstance(node, ast.Expression):
        return _expression_supported(node.body, aliases)
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float, str, type(None)))
    if isinstance(node, ast.Name):
        return node.id in aliases
    if isinstance(node, (ast.List, ast.Tuple)):
        return all(_expression_supported(item, aliases) for item in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _expression_supported(node.operand, aliases)
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    ):
        return _expression_supported(node.left, aliases) and _expression_supported(node.right, aliases)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id in {"abs", "min", "max", "round", "sum"} and all(
            _expression_supported(arg, aliases) for arg in node.args
        )
    return False


def capture_deterministic_composition(command: str, response: str) -> dict[str, Any] | None:
    """Extract a small replayable Python composition trace without executing code."""

    match = _PYTHON_HEREDOC_RE.search(command or "")
    if not match or not command_executes_sql(command):
        return None
    body = match.group(2)
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return {"plausible": True, "replayable": False, "reason": "python_ast_parse_failed"}
    aliases: dict[str, dict[str, str]] = {}
    derived: dict[str, ast.AST] = {}
    print_expression: ast.AST | None = None
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            name = statement.targets[0].id
            selected = _selector_from_node(statement.value)
            if selected:
                sql, selector = selected
                aliases[name] = {"sql": sql, "selector": selector}
            elif _expression_supported(statement.value, set(aliases)):
                derived[name] = statement.value
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "print"
            and len(statement.value.args) == 1
        ):
            print_expression = statement.value.args[0]
            if isinstance(print_expression, ast.Name) and print_expression.id in derived:
                print_expression = derived[print_expression.id]
    if not aliases or print_expression is None:
        return {"plausible": True, "replayable": False, "reason": "unsupported_python_composition_shape"}
    if not _expression_supported(print_expression, set(aliases)):
        return {"plausible": True, "replayable": False, "reason": "unsupported_python_expression"}
    return {
        "plausible": True,
        "replayable": True,
        "aliases": aliases,
        "expression": ast.unparse(print_expression),
        "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
        "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
    }


def _eval_expression(node: ast.AST, values: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_expression(node.body, values)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.List):
        return [_eval_expression(item, values) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_expression(item, values) for item in node.elts)
    if isinstance(node, ast.UnaryOp):
        value = _eval_expression(node.operand, values)
        return +value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left, right = _eval_expression(node.left, values), _eval_expression(node.right, values)
        operations = {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.Div: lambda: left / right,
            ast.FloorDiv: lambda: left // right,
            ast.Mod: lambda: left % right,
            ast.Pow: lambda: left**right,
        }
        return operations[type(node.op)]()
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        args = [_eval_expression(arg, values) for arg in node.args]
        functions = {"abs": abs, "min": min, "max": max, "round": round, "sum": sum}
        return functions[node.func.id](*args)
    raise ValueError("unsupported deterministic expression")


def _composition_result(
    trace: dict[str, Any], database: Path, ground_truth: dict[str, Any], response: str, timeout: float
) -> tuple[str, Any]:
    if not bool(trace.get("replayable")):
        return "unsupported", None
    values: dict[str, Any] = {}
    strengths: list[str] = []
    for name, spec in dict(trace.get("aliases") or {}).items():
        sql = str(spec.get("sql") or "")
        strength, _ = _sql_contract_strength(sql, ground_truth)
        if re.fullmatch(
            r"\s*select\s+[-+]?\d+(?:\.\d+)?(?:\s+as\s+[A-Za-z_][A-Za-z0-9_$]*)?"
            r"(?:\s+where\s+0)?\s*;?\s*",
            sql,
            re.IGNORECASE,
        ):
            strength = "auxiliary_constant"
        strengths.append(strength)
        rows = execute_readonly_sql(database, sql, query_timeout_seconds=timeout)
        selector = str(spec.get("selector") or "")
        if selector == "scalar":
            if len(rows) != 1 or len(rows[0]) != 1:
                return "unsupported", None
            values[name] = rows[0][0]
        elif selector == "row":
            if len(rows) != 1:
                return "unsupported", None
            values[name] = list(rows[0])
        else:
            values[name] = [list(row) for row in rows]
    if "none" in strengths or "violating" in strengths or "strong" not in strengths:
        return "unsupported", None
    expression = ast.parse(str(trace["expression"]), mode="eval")
    output = _eval_expression(expression, values)
    # The observed command output must agree with the replayed computation.
    last_line = next((line.strip() for line in reversed(response.splitlines()) if line.strip()), "")
    observed_numbers = extract_answer_numbers(last_line)
    if isinstance(output, (int, float)) and not isinstance(output, bool):
        if len(observed_numbers) != 1 or not math.isclose(
            float(observed_numbers[0]), float(output), abs_tol=1e-9, rel_tol=1e-9
        ):
            return "contradictory", output
    elif isinstance(output, (list, tuple)):
        try:
            observed_value = ast.literal_eval(last_line)
        except (SyntaxError, ValueError):
            return "unsupported", output
        if observed_value != output:
            return "contradictory", output
    return (
        "strong"
        if all(value in {"strong", "auxiliary_constant"} for value in strengths)
        else "weak"
    ), output


def _output_matches_gold(output: Any, ground_truth: dict[str, Any]) -> bool:
    expected = _expected_value(ground_truth)
    abs_tol = float(ground_truth.get("abs_tol", 1e-3))
    rel_tol = float(ground_truth.get("rel_tol", 1e-5))
    if str(ground_truth.get("answer_type") or "") == "numeric":
        return isinstance(output, (int, float)) and not isinstance(output, bool) and isinstance(expected, (int, float)) and _full_table_value_equal(output, expected, abs_tol, rel_tol)
    normalized = _normalize_full_expected_table(expected)
    if normalized is None or not isinstance(output, (list, tuple)):
        return False
    candidate = [list(row) if isinstance(row, (list, tuple)) else [row] for row in output]
    ordered, _ = table_order_semantics(ground_truth)
    return semantic_rows_equal(candidate, normalized, ordered=ordered, abs_tol=abs_tol, rel_tol=rel_tol)


def _grounded_evidence(
    events: list[dict[str, Any]],
    database: Path,
    gold_rows: list[tuple[Any, ...]],
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
) -> tuple[StateDecision, dict[str, Any]]:
    timeout = float(extra_info.get("pi_reward_sql_timeout_seconds", 5.0))
    strong_support: list[int] = []
    strong_conflict: list[int] = []
    weak_plausible: list[int] = []
    contract_violations: list[int] = []
    contract_violation_reasons: dict[str, list[str]] = {}
    successful_sql = 0
    replay_errors = 0
    composition_support: list[int] = []
    composition_plausible: list[int] = []
    for event_index, event in enumerate(events):
        if event.get("name") != "bash" or not bool(event.get("ok")):
            continue
        command = str((event.get("arguments") or {}).get("command") or "")
        response = str(event.get("response_preview") or "")
        trace = event.get("composition_trace")
        if not isinstance(trace, dict):
            trace = capture_deterministic_composition(command, response)
        replayable_composition = bool(trace and trace.get("replayable"))
        if command_executes_sql(command) and not replayable_composition:
            for sql in extract_selects([command]):
                successful_sql += 1
                strength, strength_reasons = _sql_contract_strength(sql, ground_truth)
                required_tables = {
                    str(value).strip().casefold()
                    for value in ground_truth.get("required_tables", [])
                    if str(value).strip()
                }
                observed_tables = {value.casefold() for value in extract_table_names(sql)}
                task_related = not required_tables or bool(required_tables & observed_tables)
                if strength == "violating" and task_related:
                    contract_violations.append(event_index)
                    contract_violation_reasons[str(event_index)] = strength_reasons
                try:
                    rows = execute_readonly_sql(database, sql, query_timeout_seconds=timeout)
                except (ValueError, OSError, sqlite3.Error):
                    replay_errors += 1
                    weak_plausible.append(event_index)
                    continue
                equal = _rows_equal_gold(rows, gold_rows, ground_truth)
                if strength == "strong" and equal:
                    strong_support.append(event_index)
                elif strength == "strong" and _same_shape(rows, gold_rows):
                    strong_conflict.append(event_index)
                elif strength == "weak" and equal:
                    weak_plausible.append(event_index)
                elif strength == "violating" and task_related:
                    strong_conflict.append(event_index)
        if trace:
            if bool(event.get("response_truncated")):
                composition_plausible.append(event_index)
                continue
            try:
                status, output = _composition_result(trace, database, ground_truth, response, timeout)
            except (TypeError, ValueError, OSError, sqlite3.Error, ZeroDivisionError):
                status, output = "unsupported", None
            if status == "strong" and _output_matches_gold(output, ground_truth):
                composition_support.append(event_index)
            elif status in {"weak", "unsupported"}:
                composition_plausible.append(event_index)
            elif status == "contradictory":
                strong_conflict.append(event_index)

    support = sorted(set(strong_support + composition_support))
    conflicts = sorted(set(strong_conflict))
    plausible = sorted(set(weak_plausible + composition_plausible))
    details = {
        "grounded": bool(support and not conflicts),
        "supporting_event_indices": support,
        "contradictory_event_indices": conflicts,
        "unsupported_plausible_event_indices": plausible,
        "contract_violation_event_indices": sorted(set(contract_violations)),
        "contract_violation_reasons": contract_violation_reasons,
        "successful_sql_count": successful_sql,
        "sql_replay_error_count": replay_errors,
        "evidence_route": (
            "composed" if composition_support else "table" if support and str(ground_truth.get("answer_type")) == "table" else "direct" if support else "none"
        ),
    }
    if support and conflicts:
        return StateDecision(JudgeState.UNKNOWN, "conflicting_supporting_evidence"), details
    if support:
        return StateDecision(JudgeState.PASS, "grounded_evidence_replayed"), details
    if plausible:
        return StateDecision(JudgeState.UNKNOWN, "unsupported_but_plausible_evidence_route"), details
    if conflicts or successful_sql:
        return StateDecision(JudgeState.FAIL, "observed_evidence_does_not_support_gold"), details
    return StateDecision(JudgeState.FAIL, "no_real_tool_evidence"), details


def compute_grounded_trajectory_reward(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
    **_: Any,
) -> dict[str, Any]:
    """Return a train-safe PASS/FAIL/UNKNOWN outcome and binary scalar reward."""

    del data_source
    task, database, gold_rows, task_details = _task_validity(ground_truth, extra_info)
    final, final_details = _final_decision(solution_str, ground_truth)
    observed, events, observed_details = _observability_and_safety(
        extra_info.get("pi_tool_events"), extra_info
    )
    evidence = StateDecision(JudgeState.UNKNOWN, "task_or_observability_prevents_evidence_judgement")
    evidence_details: dict[str, Any] = {
        "grounded": False,
        "supporting_event_indices": [],
        "contradictory_event_indices": [],
        "unsupported_plausible_event_indices": [],
        "contract_violation_event_indices": [],
        "contract_violation_reasons": {},
        "successful_sql_count": 0,
        "sql_replay_error_count": 0,
        "evidence_route": "none",
    }
    if task.state is JudgeState.PASS and observed.state is JudgeState.PASS and database is not None:
        evidence, evidence_details = _grounded_evidence(
            events, database, gold_rows, ground_truth, extra_info
        )

    if task.state is JudgeState.UNKNOWN:
        decision = task
    elif observed.state is JudgeState.FAIL:
        decision = observed
    elif observed.state is JudgeState.UNKNOWN:
        decision = observed
    elif final.state is JudgeState.UNKNOWN:
        decision = final
    elif final.state is JudgeState.FAIL:
        decision = final
    elif evidence.state is JudgeState.PASS:
        decision = StateDecision(JudgeState.PASS, "correct_safe_and_grounded")
    elif evidence.state is JudgeState.UNKNOWN:
        decision = evidence
    else:
        decision = evidence

    train_mask = float(decision.state is not JudgeState.UNKNOWN)
    success = float(decision.state is JudgeState.PASS)
    return {
        "score": success if train_mask else 0.0,
        "reward": success if train_mask else 0.0,
        "acc": success,
        "success": success,
        "train_mask": train_mask,
        "online_eligible": train_mask,
        "judge_state": decision.state.value,
        "judge_reason": decision.reason,
        "judge_confident": train_mask,
        "task_state": task.state.value,
        "task_reason": task.reason,
        "final_state": final.state.value,
        "final_reason": final.reason,
        "evidence_state": evidence.state.value,
        "evidence_reason": evidence.reason,
        "safety_observability_state": observed.state.value,
        "safety_observability_reason": observed.reason,
        "grounded": float(bool(evidence_details.get("grounded"))),
        "safe_process": float(bool(observed_details.get("safe_process"))),
        "task_valid": float(task.state is JudgeState.PASS),
        "evidence_observable": float(bool(observed_details.get("evidence_observable"))),
        "final_answer_correct": float(final.state is JudgeState.PASS),
        "guess_correct_blocked": float(final.state is JudgeState.PASS and evidence.state is JudgeState.FAIL),
        "quality_bonus_alpha": 0.0,
        "quality_bonus_applied": 0.0,
        "future_quality_alpha_ceiling": MAX_FUTURE_QUALITY_ALPHA,
        "observed_table_field_efficiency_in_reward": 0.0,
        "reward_contract": REWARD_CONTRACT,
        "reward_scope": "trajectory_level_after_full_multiturn",
        "turn_level_credit_assignment": 0.0,
        "kl_in_reward": 0.0,
        "final_answer_match_mode": final_details.get("match_mode", "none"),
        "numeric_final_parse_mode": final_details.get("numeric_parse_mode", "none"),
        "numeric_final_parse_ambiguous": float(bool(final_details.get("numeric_parse_ambiguous"))),
        "numeric_final_parse_ambiguity_reason": final_details.get("numeric_parse_ambiguity_reason", ""),
        "table_comparison_mode": (
            "ordered" if final_details.get("table_ordered") else "row_multiset"
        ) if final_details.get("answer_type") == "table" else "not_table",
        **task_details,
        **observed_details,
        **evidence_details,
    }


def state_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = {state: 0 for state in JUDGE_STATES}
    for row in rows:
        state = str(row.get("judge_state") or "UNKNOWN")
        counts[state if state in counts else "UNKNOWN"] += 1
    return counts

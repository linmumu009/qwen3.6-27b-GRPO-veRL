"""Outcome reward for verified PI warehouse trajectories."""

from __future__ import annotations

import math
import re
from typing import Any

_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")


def extract_numbers(text: str) -> list[float]:
    values: list[float] = []
    for match in _NUMBER_RE.finditer(text or ""):
        try:
            value = float(match.group(0).replace(",", ""))
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def contains_expected_number(text: str, expected: float, abs_tol: float = 1e-3, rel_tol: float = 1e-5) -> bool:
    return any(math.isclose(value, expected, abs_tol=abs_tol, rel_tol=rel_tol) for value in extract_numbers(text))


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
    **_: Any,
) -> dict[str, Any]:
    """Reward verified numeric answers while requiring real SQL-tool evidence."""
    del data_source
    expected = float(ground_truth["expected_value"])
    required_tables = {str(name).lower() for name in ground_truth.get("required_tables", [])}
    query_log = extra_info.get("llin_sql_queries") or []

    successful_queries = [entry for entry in query_log if entry.get("ok")]
    used_tables = {
        str(table).lower()
        for entry in successful_queries
        for table in (entry.get("tables") or [])
    }
    tool_used = bool(successful_queries)
    required_table_used = bool(required_tables) and required_tables.issubset(used_tables)
    answer_correct = contains_expected_number(solution_str, expected)

    if tool_used and required_table_used and answer_correct:
        score = 1.0
    elif tool_used and required_table_used:
        score = 0.2
    elif tool_used:
        score = 0.05
    else:
        score = 0.0

    return {
        "score": score,
        "acc": float(score == 1.0),
        "tool_used": float(tool_used),
        "required_table_used": float(required_table_used),
        "answer_correct": float(answer_correct),
    }

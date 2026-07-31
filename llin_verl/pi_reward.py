"""Outcome reward for verified PI warehouse trajectories."""

from __future__ import annotations

import math
import re
from typing import Any

_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")
_ASSISTANT_SPLIT_RE = re.compile(r"(?:^|\n)assistant\n")
_NEXT_ROLE_RE = re.compile(r"\n(?:user|tool|system)\n")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


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


def extract_final_assistant_answer(text: str) -> str:
    """Return visible text from the final assistant turn, excluding thought/tool payloads."""
    segments = _ASSISTANT_SPLIT_RE.split(text or "")
    final_turn = segments[-1]
    final_turn = _NEXT_ROLE_RE.split(final_turn, maxsplit=1)[0]
    final_turn = _THINK_RE.sub("", final_turn)
    final_turn = _TOOL_CALL_RE.sub("", final_turn)
    return final_turn.strip()


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
    # Tool/thought payloads are useful evidence, but they must never substitute
    # for a correct final answer. Keep the trajectory-wide match as a separate
    # diagnostic so historical runs can still be audited.
    evidence_contains_expected = contains_expected_number(solution_str, expected)
    final_answer = extract_final_assistant_answer(solution_str)
    final_answer_correct = contains_expected_number(final_answer, expected)
    answer_correct = final_answer_correct

    if tool_used and required_table_used and final_answer_correct:
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
        "evidence_contains_expected": float(evidence_contains_expected),
        "final_answer_correct": float(final_answer_correct),
    }

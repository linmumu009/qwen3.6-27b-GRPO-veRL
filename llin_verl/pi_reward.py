"""Evidence-grounded reward V2 for full PI DWH trajectories."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from llin_verl.pi_tool_contract import command_is_safe, extract_table_names


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


def contains_expected_number(
    text: str,
    expected: float,
    abs_tol: float = 1e-3,
    rel_tol: float = 1e-5,
) -> bool:
    return any(math.isclose(value, expected, abs_tol=abs_tol, rel_tol=rel_tol) for value in extract_numbers(text))


def extract_final_assistant_answer(text: str) -> str:
    """Return visible text from the final assistant turn."""
    segments = _ASSISTANT_SPLIT_RE.split(text or "")
    final_turn = segments[-1]
    final_turn = _NEXT_ROLE_RE.split(final_turn, maxsplit=1)[0]
    final_turn = _THINK_RE.sub("", final_turn)
    final_turn = _TOOL_CALL_RE.sub("", final_turn)
    return final_turn.strip()


def extract_selects(commands: Iterable[str]) -> list[str]:
    """Extract read-only SELECT/WITH statements from common PI bash forms."""
    selects: list[str] = []
    for command in commands:
        for match in re.finditer(
            r"execute\s*\(\s*([\"']{1,3})((?:SELECT|WITH).*?)\1",
            command,
            re.IGNORECASE | re.DOTALL,
        ):
            selects.append(match.group(2).strip().rstrip(";").strip())
        for match in re.finditer(r"sqlite3\s+\S+\s+\"((?:[^\"]|\"\")*)\"", command, re.IGNORECASE | re.DOTALL):
            payload = match.group(1).replace('""', '"')
            for part in payload.split(";"):
                part = part.strip()
                if re.match(r"^(?:SELECT|WITH)\b", part, re.IGNORECASE):
                    selects.append(part)
        for match in re.finditer(r"<<'?([A-Za-z_][A-Za-z0-9_]*)'?\s*\n(.*?)\n\1", command, re.DOTALL):
            body = match.group(2)
            for statement in re.finditer(r"\b(?:SELECT|WITH)\b.*?(?=;|$)", body, re.IGNORECASE | re.DOTALL):
                selects.append(statement.group(0).strip().rstrip(";").strip())
    output: list[str] = []
    seen: set[str] = set()
    for sql in selects:
        key = " ".join(sql.casefold().split())
        if key and key not in seen:
            seen.add(key)
            output.append(sql)
    return output


def _database_path(sandbox_root: str | Path, environment_id: str) -> Path:
    root = Path(sandbox_root).resolve(strict=True)
    database = (root / environment_id / "logistics.sqlite").resolve(strict=True)
    database.relative_to(root)
    if not database.is_file():
        raise FileNotFoundError(database)
    return database


def execute_readonly_sql(database: Path, sql: str, max_rows: int = 10_000) -> list[tuple[Any, ...]]:
    if not re.match(r"^\s*(?:SELECT|WITH)\b", sql or "", re.IGNORECASE):
        raise ValueError("only SELECT/WITH verifier SQL is allowed")
    uri = f"file:{quote(database.as_posix(), safe='/')}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True, timeout=5.0)
    try:
        connection.execute("PRAGMA query_only=ON")
        cursor = connection.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            raise ValueError("SQL evidence exceeds max_rows")
        return rows
    finally:
        connection.close()


def _values_equal(left: Any, right: Any, abs_tol: float, rel_tol: float) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), abs_tol=abs_tol, rel_tol=rel_tol)
    return str(left).strip().casefold() == str(right).strip().casefold()


def rows_equal(
    left: list[tuple[Any, ...]],
    right: list[tuple[Any, ...]],
    abs_tol: float,
    rel_tol: float,
) -> bool:
    if len(left) != len(right):
        return False
    for left_row, right_row in zip(left, right, strict=True):
        if len(left_row) != len(right_row):
            return False
        if not all(_values_equal(a, b, abs_tol, rel_tol) for a, b in zip(left_row, right_row, strict=True)):
            return False
    return True


def _table_answer_correct(
    answer: str,
    expected: list[Any],
    abs_tol: float,
    rel_tol: float,
) -> bool:
    folded = answer.casefold()
    for item in expected:
        if not isinstance(item, dict):
            return False
        label = item.get("category", item.get("date"))
        if label is not None and str(label).strip().casefold() not in folded:
            return False
        value = item.get("value")
        if value is not None and not contains_expected_number(answer, float(value), abs_tol, rel_tol):
            return False
    return bool(expected)


def final_answer_correct(
    answer: str,
    answer_type: str,
    expected_value: Any,
    abs_tol: float,
    rel_tol: float,
) -> bool:
    if not answer:
        return False
    if answer_type == "numeric" and isinstance(expected_value, (int, float)):
        return contains_expected_number(answer, float(expected_value), abs_tol, rel_tol)
    if answer_type == "table" and isinstance(expected_value, list):
        return _table_answer_correct(answer, expected_value, abs_tol, rel_tol)
    return False


def _event_commands(events: list[dict[str, Any]]) -> list[str]:
    return [
        str((event.get("arguments") or {}).get("command"))
        for event in events
        if event.get("name") == "bash" and isinstance((event.get("arguments") or {}).get("command"), str)
    ]


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
    **_: Any,
) -> dict[str, Any]:
    """Score final correctness plus independently executable SQL evidence.

    Weights are intentionally outcome-dominant: final answer 0.60, exact SQL
    evidence 0.25, required tables 0.10, and terminal answer presence 0.05.
    Unsafe or malformed tool protocols receive zero regardless of answer text.
    """
    del data_source
    events = extra_info.get("pi_tool_events") or []
    if not isinstance(events, list):
        events = []
    commands = _event_commands(events)
    allowed_tools = {"bash", "read", "write", "edit"}
    valid_protocol = bool(events) and all(event.get("name") in allowed_tools for event in events)
    safe = all(command_is_safe(command) for command in commands)
    successful_bash = any(event.get("name") == "bash" and event.get("ok") for event in events)

    required_tables = {str(value).casefold() for value in ground_truth.get("required_tables", [])}
    used_tables = {table for command in commands for table in extract_table_names(command)}
    required_table_used = required_tables.issubset(used_tables) if required_tables else True

    answer = extract_final_assistant_answer(solution_str)
    abs_tol = float(ground_truth.get("abs_tol", 1e-3))
    rel_tol = float(ground_truth.get("rel_tol", 1e-5))
    expected_value = ground_truth.get("expected_value")
    if "expected_value_json" in ground_truth:
        try:
            expected_value = json.loads(str(ground_truth["expected_value_json"]))
        except json.JSONDecodeError:
            expected_value = None
    answer_ok = final_answer_correct(
        answer,
        str(ground_truth.get("answer_type") or ""),
        expected_value,
        abs_tol,
        rel_tol,
    )

    sql_evidence = False
    verifier_error = ""
    try:
        database = _database_path(
            os.environ.get("PI_AGENT_SANDBOX_LOWER", "/pi_sandbox"),
            str(ground_truth["environment_id"]),
        )
        expected_rows = execute_readonly_sql(database, str(ground_truth["verification_sql"]))
        for sql in extract_selects(commands):
            try:
                if rows_equal(execute_readonly_sql(database, sql), expected_rows, abs_tol, rel_tol):
                    sql_evidence = True
                    break
            except (ValueError, OSError, sqlite3.Error):
                continue
    except (KeyError, ValueError, OSError, sqlite3.Error) as exc:
        verifier_error = f"{type(exc).__name__}: {exc}"

    has_final = bool(answer)
    if not safe or not valid_protocol:
        score = 0.0
    else:
        score = (
            0.60 * float(answer_ok)
            + 0.25 * float(sql_evidence and successful_bash)
            + 0.10 * float(required_table_used)
            + 0.05 * float(has_final)
        )
    strict_correct = bool(answer_ok and sql_evidence and required_table_used and successful_bash and safe)
    return {
        "score": round(score, 6),
        "acc": float(strict_correct),
        "final_answer_correct": float(answer_ok),
        "sql_evidence_correct": float(sql_evidence),
        "required_table_used": float(required_table_used),
        "successful_bash": float(successful_bash),
        "safe": float(safe),
        "valid_tool_protocol": float(valid_protocol),
        "has_final_answer": float(has_final),
        "verifier_error": verifier_error,
    }

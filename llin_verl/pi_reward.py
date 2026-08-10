"""Boss-aligned DWH reward with executable-evidence safety guards."""

from __future__ import annotations

from collections import Counter
import json
import math
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from llin_verl.pi_tool_contract import command_unsafe_reasons, extract_table_names


_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")
_DATE_RE = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b")
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_ASSISTANT_SPLIT_RE = re.compile(r"(?:^|\n)assistant\n")
_NEXT_ROLE_RE = re.compile(r"\n(?:user|tool|system)\n")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_AGGREGATE_RE = re.compile(
    r"\b(sum|count|avg|min|max)\s*\(\s*(distinct\s+)?([A-Za-z_][A-Za-z0-9_.]*|\*)\s*\)",
    re.IGNORECASE,
)
_WHERE_RE = re.compile(
    r"\bwhere\b(.*?)(?=\bgroup\s+by\b|\bhaving\b|\border\s+by\b|\blimit\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_SELECT_RE = re.compile(r"\bselect\b(.*?)\bfrom\b", re.IGNORECASE | re.DOTALL)
_STRING_LITERAL_RE = re.compile(r"'((?:''|[^'])*)'")
_NUMERIC_LITERAL_RE = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9_.])")
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_PROJECTION_FORBIDDEN_RE = re.compile(r"\b(?:union|intersect|except|case)\b", re.IGNORECASE)
_SQL_KEYWORDS = {
    "and",
    "asc",
    "as",
    "between",
    "date",
    "datetime",
    "desc",
    "false",
    "in",
    "is",
    "like",
    "not",
    "null",
    "or",
    "select",
    "sum",
    "count",
    "avg",
    "min",
    "max",
    "strftime",
    "time",
    "true",
}


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


def extract_answer_numbers(text: str) -> list[float]:
    """Extract result-like numbers while ignoring ordinary dates and times."""
    without_dates = _DATE_RE.sub(" ", text or "")
    return extract_numbers(_TIME_RE.sub(" ", without_dates))


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


def rows_contain_unique_projection(
    candidate: list[tuple[Any, ...]],
    expected: list[tuple[Any, ...]],
    abs_tol: float,
    rel_tol: float,
) -> bool:
    """Return true when expected is one unique column projection of candidate.

    Row count and order must remain identical.  Requiring a unique injective
    column mapping prevents a coincidental value copied into multiple output
    columns from being treated as independently executable evidence.
    """
    if not candidate or not expected or len(candidate) != len(expected):
        return False
    candidate_width = len(candidate[0])
    expected_width = len(expected[0])
    if expected_width <= 0 or candidate_width <= expected_width:
        return False
    if any(len(row) != candidate_width for row in candidate):
        return False
    if any(len(row) != expected_width for row in expected):
        return False

    options: list[list[int]] = []
    for expected_column in range(expected_width):
        matches = []
        for candidate_column in range(candidate_width):
            if all(
                _values_equal(
                    candidate[row_index][candidate_column],
                    expected[row_index][expected_column],
                    abs_tol,
                    rel_tol,
                )
                for row_index in range(len(expected))
            ):
                matches.append(candidate_column)
        if not matches:
            return False
        options.append(matches)

    mappings = 0

    def count_mappings(index: int, used: set[int]) -> None:
        nonlocal mappings
        if mappings > 1:
            return
        if index == len(options):
            mappings += 1
            return
        for column in options[index]:
            if column not in used:
                count_mappings(index + 1, used | {column})

    count_mappings(0, set())
    return mappings == 1


def _aggregate_signature(sql: str) -> set[tuple[str, bool, str]]:
    return {
        (match.group(1).casefold(), bool(match.group(2)), match.group(3).split(".")[-1].casefold())
        for match in _AGGREGATE_RE.finditer(sql or "")
    }


def _where_clause(sql: str) -> str:
    match = _WHERE_RE.search(sql or "")
    return match.group(1) if match else ""


def _select_identifiers(sql: str) -> set[str]:
    match = _SELECT_RE.search(sql or "")
    if not match:
        return set()
    clause = _STRING_LITERAL_RE.sub(" ", match.group(1))
    return {
        token.casefold()
        for token in _IDENTIFIER_RE.findall(clause)
        if token.casefold() not in _SQL_KEYWORDS
    }


def _where_identifiers(sql: str) -> set[str]:
    clause = _STRING_LITERAL_RE.sub(" ", _where_clause(sql))
    return {
        token.casefold()
        for token in _IDENTIFIER_RE.findall(clause)
        if token.casefold() not in _SQL_KEYWORDS
    }


def _where_string_literals(sql: str) -> list[str]:
    return [match.group(1).replace("''", "'").strip().casefold() for match in _STRING_LITERAL_RE.finditer(_where_clause(sql))]


def _where_numeric_literals(sql: str) -> set[str]:
    clause = _STRING_LITERAL_RE.sub(" ", _where_clause(sql))
    return {match.group(0) for match in _NUMERIC_LITERAL_RE.finditer(clause)}


def _literal_is_covered(expected: str, candidate: str) -> bool:
    if candidate == expected:
        return True
    # DATE(column) = 'YYYY-MM-DD' is commonly expressed as a half-open
    # timestamp range beginning at 'YYYY-MM-DD 00:00:00'.
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", expected)) and (
        candidate.startswith(expected + " ") or candidate.startswith(expected + "t")
    )


def safe_projection_sql(
    candidate_sql: str,
    expected_sql: str,
    required_tables: set[str],
) -> bool:
    """Conservatively admit an equivalent aggregate query with extra columns.

    This is intentionally stricter than checking whether a gold number appears
    anywhere in a result.  The source tables must be identical, every expected
    aggregate expression must occur in the candidate, and the expected filter
    identifiers/literals must be retained.  Complex set/case expressions fall
    back to exact-result matching only.
    """
    if _PROJECTION_FORBIDDEN_RE.search(candidate_sql or ""):
        return False
    expected_tables = set(extract_table_names(expected_sql))
    candidate_tables = set(extract_table_names(candidate_sql))
    if not expected_tables or candidate_tables != expected_tables:
        return False
    if required_tables and not required_tables.issubset(candidate_tables):
        return False
    expected_aggregates = _aggregate_signature(expected_sql)
    if expected_aggregates:
        if not expected_aggregates.issubset(_aggregate_signature(candidate_sql)):
            return False
    elif not _select_identifiers(expected_sql).issubset(_select_identifiers(candidate_sql)):
        return False
    if not _where_identifiers(expected_sql).issubset(_where_identifiers(candidate_sql)):
        return False
    candidate_literals = _where_string_literals(candidate_sql)
    strings_covered = all(
        any(_literal_is_covered(expected, candidate) for candidate in candidate_literals)
        for expected in _where_string_literals(expected_sql)
    )
    return strings_covered and _where_numeric_literals(expected_sql).issubset(
        _where_numeric_literals(candidate_sql)
    )


def sql_evidence_mode(
    candidate_sql: str,
    candidate_rows: list[tuple[Any, ...]],
    expected_sql: str,
    expected_rows: list[tuple[Any, ...]],
    required_tables: set[str],
    abs_tol: float,
    rel_tol: float,
) -> str:
    if not safe_projection_sql(candidate_sql, expected_sql, required_tables):
        return "none"
    if rows_equal(candidate_rows, expected_rows, abs_tol, rel_tol):
        return "exact"
    if rows_contain_unique_projection(candidate_rows, expected_rows, abs_tol, rel_tol):
        return "safe_projection"
    return "none"


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


def boss_gold_numbers(expected_value: Any) -> list[float]:
    """Extract the numeric subset used by the boss DWH judge.

    The upstream evaluator intentionally ignores table labels and accepts an
    answer containing all gold numbers even when it contains extra numbers.
    Keeping this signal separate from ``final_answer_correct`` lets training
    match the boss metric while retaining the stricter label-aware accuracy as
    an independent guardrail.
    """
    if isinstance(expected_value, list):
        values = [item.get("value") for item in expected_value if isinstance(item, dict)]
    else:
        values = [expected_value]
    output: list[float] = []
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            number = float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            output.append(number)
    return output


def boss_numbers_match(answer: str, expected_value: Any) -> bool:
    """Port the boss evaluator's all-gold-numbers-in-answer comparison."""
    gold_numbers = boss_gold_numbers(expected_value)
    answer_numbers = extract_numbers(answer)
    if not gold_numbers:
        return False
    return all(
        any(
            abs(gold - actual) < 1e-6
            or (gold != 0 and abs(gold - actual) / abs(gold) < 1e-6)
            for actual in answer_numbers
        )
        for gold in gold_numbers
    )


def _number_closeness(actual: float, gold: float, abs_tol: float, rel_tol: float) -> float:
    if math.isclose(actual, gold, abs_tol=abs_tol, rel_tol=rel_tol):
        return 1.0
    scale = max(abs(gold), 1.0)
    relative_error = abs(actual - gold) / scale
    # Preserve a clear margin between the boss evaluator's exact numeric hit
    # and merely close values, while retaining ordering among wrong answers.
    return min(0.75, 1.0 / (1.0 + 4.0 * relative_error))


def dense_final_answer_correctness(
    answer: str,
    answer_type: str,
    expected_value: Any,
    abs_tol: float = 1e-3,
    rel_tol: float = 1e-5,
) -> float:
    """Return continuous, final-answer-only correctness in ``[0, 1]``.

    Exact answers retain score 1.  Wrong numeric answers receive partial credit
    based on relative distance, and table answers additionally receive credit
    for gold row labels.  A precision factor discourages emitting a long list
    of unrelated numbers merely to collide with the gold value.
    """
    if not answer:
        return 0.0

    gold_numbers = boss_gold_numbers(expected_value)
    answer_numbers = extract_answer_numbers(answer)
    numeric_score = 0.0
    if gold_numbers and answer_numbers:
        edges = sorted(
            (
                _number_closeness(actual, gold, abs_tol, rel_tol),
                gold_index,
                answer_index,
            )
            for gold_index, gold in enumerate(gold_numbers)
            for answer_index, actual in enumerate(answer_numbers)
        )
        matched_gold: set[int] = set()
        matched_answer: set[int] = set()
        credit = 0.0
        for closeness, gold_index, answer_index in reversed(edges):
            if gold_index in matched_gold or answer_index in matched_answer:
                continue
            matched_gold.add(gold_index)
            matched_answer.add(answer_index)
            credit += closeness
        numeric_score = credit / len(gold_numbers)
        free_numbers = max(2 * len(gold_numbers), len(gold_numbers) + 2)
        if len(answer_numbers) > free_numbers:
            numeric_score *= math.sqrt(free_numbers / len(answer_numbers))

    if answer_type == "numeric":
        return round(min(1.0, numeric_score), 6)
    if answer_type != "table" or not isinstance(expected_value, list):
        return 0.0

    labels = []
    for item in expected_value:
        if isinstance(item, dict):
            label = item.get("category", item.get("date"))
            if label is not None:
                labels.append(str(label).strip().casefold())
    folded = answer.casefold()
    label_score = (
        sum(bool(label) and label in folded for label in labels) / len(labels)
        if labels
        else 0.0
    )
    if gold_numbers and labels:
        # The boss endpoint is number-primary and intentionally ignores table
        # labels.  Labels remain useful partial credit but cannot outrank an
        # exact gold-number hit.
        score = 0.85 * numeric_score + 0.15 * label_score
    elif gold_numbers:
        score = numeric_score
    else:
        score = label_score
    return round(min(1.0, score), 6)


def _boss_fields_used(commands: list[str], must_use_fields: list[str]) -> float | None:
    if not must_use_fields:
        return None
    command_text = " ".join(commands).casefold()
    hit = sum(1 for field in must_use_fields if str(field).casefold() in command_text)
    return round(hit / len(must_use_fields), 4)


def _boss_task_fit(commands: list[str], selects: list[str]) -> float:
    if selects:
        return 1.0
    return 0.5 if any(re.search(r"\bsqlite3\b", command, re.IGNORECASE) for command in commands) else 0.0


def _boss_efficiency(commands: list[str], selects: list[str], events: list[dict[str, Any]]) -> tuple[float, dict[str, int]]:
    full_scan = sum(
        bool(re.search(r"select\s+\*", sql, re.IGNORECASE))
        and not bool(re.search(r"\bwhere\b", sql, re.IGNORECASE))
        and not bool(re.search(r"\blimit\b", sql, re.IGNORECASE))
        for sql in selects
    )
    normalized_sql = [" ".join(sql.casefold().split()) for sql in selects]
    duplicate_sql = len(normalized_sql) - len(set(normalized_sql))
    command_keys = [command.strip()[:40] for command in commands]
    duplicate_commands = len(command_keys) - len(set(command_keys))
    auto_retry = sum(event.get("type") == "auto_retry_start" for event in events)
    penalty = min(
        1.0,
        0.1 * full_scan
        + 0.05 * (duplicate_sql + min(duplicate_commands, 20))
        + 0.2 * auto_retry,
    )
    return 1.0 - penalty, {
        "full_scan_answer": int(full_scan),
        "duplicate_sql": duplicate_sql,
        "duplicate_commands": duplicate_commands,
        "auto_retry": auto_retry,
    }


def boss_reward_components(
    answer: str,
    expected_value: Any,
    commands: list[str],
    selects: list[str],
    required_tables: set[str],
    used_tables: set[str],
    must_use_fields: list[str],
    sql_evidence: bool,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute the deterministic DWH portion of the boss reward_judge.py."""
    has_answer = bool(answer) and len(re.sub(r"\s+", "", answer)) >= 20
    numbers = boss_gold_numbers(expected_value)
    numbers_ok = boss_numbers_match(answer, expected_value)
    # The boss judge accepts numeric fallback without a matching SQL.  When a
    # gold table has no numeric values, only an executable matching SQL can
    # establish result correctness.
    answer_ok = numbers_ok if numbers else bool(sql_evidence)
    tables_ok = required_tables.issubset(used_tables) if required_tables else True
    fields_used = _boss_fields_used(commands, must_use_fields)
    task_fit = _boss_task_fit(commands, selects)
    process_items = [(float(tables_ok), 0.3), (task_fit, 0.2)]
    if fields_used is not None:
        process_items.append((fields_used, 0.3))
    process_weight = sum(weight for _, weight in process_items)
    process_score = sum(value * weight for value, weight in process_items) / process_weight
    efficiency_score, efficiency = _boss_efficiency(commands, selects, events)
    result_score = 0.5 * float(has_answer) + 0.5 * float(answer_ok)
    total = 0.0 if not has_answer else 0.4 * result_score + 0.4 * process_score + 0.2 * efficiency_score
    return {
        "reward": round(total, 6),
        "result_score": round(result_score, 6),
        "process_score": round(process_score, 6),
        "efficiency_score": round(efficiency_score, 6),
        "has_answer": float(has_answer),
        "answer_correct": float(answer_ok),
        "numbers_match": float(numbers_ok),
        "tables_hit": float(tables_ok),
        "fields_used": fields_used,
        "task_fit": task_fit,
        **efficiency,
    }


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
    _dense_weight_override: float | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Return the boss-primary online reward and strict guardrail metrics.

    The train score is 70% boss-compatible deterministic DWH reward and 30%
    strict executable-evidence reward. Unsafe tools, malformed protocols, or an
    unexecutable gold query hard-zero both signals.
    """
    del data_source
    events = extra_info.get("pi_tool_events") or []
    if not isinstance(events, list):
        events = []
    commands = _event_commands(events)
    selects = extract_selects(commands)
    allowed_tools = {"bash", "read", "write", "edit"}
    valid_protocol = bool(events) and all(event.get("name") in allowed_tools for event in events)
    command_reasons = [command_unsafe_reasons(command) for command in commands]
    unsafe_reason_counts = Counter(reason for reasons in command_reasons for reason in reasons)
    unsafe_command_count = sum(bool(reasons) for reasons in command_reasons)
    safe = unsafe_command_count == 0
    successful_bash = any(event.get("name") == "bash" and event.get("ok") for event in events)

    required_tables = {str(value).casefold() for value in ground_truth.get("required_tables", [])}
    used_tables = {table for command in commands for table in extract_table_names(command)}
    required_table_used = required_tables.issubset(used_tables) if required_tables else True
    must_use_fields = [str(value) for value in ground_truth.get("must_use_fields", []) if str(value)]

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
    dense_correctness = dense_final_answer_correctness(
        answer,
        str(ground_truth.get("answer_type") or ""),
        expected_value,
        abs_tol,
        rel_tol,
    )

    sql_evidence = False
    evidence_mode = "none"
    verifier_error = ""
    gold_sql_verified = False
    try:
        database = _database_path(
            os.environ.get("PI_AGENT_SANDBOX_LOWER", "/pi_sandbox"),
            str(ground_truth["environment_id"]),
        )
        verification_sql = str(ground_truth["verification_sql"])
        expected_rows = execute_readonly_sql(database, verification_sql)
        gold_sql_verified = bool(expected_rows)
        for sql in selects:
            try:
                mode = sql_evidence_mode(
                    sql,
                    execute_readonly_sql(database, sql),
                    verification_sql,
                    expected_rows,
                    required_tables,
                    abs_tol,
                    rel_tol,
                )
                if mode != "none":
                    sql_evidence = True
                    evidence_mode = mode
                    break
            except (ValueError, OSError, sqlite3.Error):
                continue
    except (KeyError, ValueError, OSError, sqlite3.Error) as exc:
        verifier_error = f"{type(exc).__name__}: {exc}"

    has_final = bool(answer)
    evidence_reward = (
            0.60 * float(answer_ok)
            + 0.25 * float(sql_evidence and successful_bash)
            + 0.10 * float(required_table_used)
            + 0.05 * float(has_final)
    )
    boss = boss_reward_components(
        answer,
        expected_value,
        commands,
        selects,
        required_tables,
        used_tables,
        must_use_fields,
        sql_evidence,
        events,
    )
    eligible = bool(safe and valid_protocol and gold_sql_verified)
    base_score = 0.7 * boss["reward"] + 0.3 * evidence_reward
    if _dense_weight_override is None:
        try:
            dense_weight = float(os.environ.get("PI_DENSE_CORRECTNESS_WEIGHT", "0"))
        except ValueError as exc:
            raise ValueError("PI_DENSE_CORRECTNESS_WEIGHT must be numeric") from exc
    else:
        dense_weight = float(_dense_weight_override)
    if not 0.0 <= dense_weight <= 1.0:
        raise ValueError("PI_DENSE_CORRECTNESS_WEIGHT must be in [0, 1]")
    blended_score = (1.0 - dense_weight) * base_score + dense_weight * dense_correctness
    score = blended_score if eligible else 0.0
    strict_correct = bool(
        answer_ok
        and sql_evidence
        and required_table_used
        and successful_bash
        and safe
        and valid_protocol
        and gold_sql_verified
    )
    return {
        "score": round(score, 6),
        "base_score": round(base_score if eligible else 0.0, 6),
        "dense_final_answer_correctness": dense_correctness,
        "dense_correctness_weight": dense_weight,
        "acc": float(strict_correct),
        "boss_reward": boss["reward"],
        "boss_result_score": boss["result_score"],
        "boss_process_score": boss["process_score"],
        "boss_efficiency_score": boss["efficiency_score"],
        "boss_answer_correct": boss["answer_correct"],
        "boss_numbers_match": boss["numbers_match"],
        # veRL averages every numeric reward field during validation.  The
        # upstream process score omits the field criterion when a task has no
        # required fields, but the emitted metric must never be ``None``.
        "boss_fields_used": 1.0 if boss["fields_used"] is None else boss["fields_used"],
        "boss_task_fit": boss["task_fit"],
        "evidence_reward": round(evidence_reward, 6),
        "final_answer_correct": float(answer_ok),
        "sql_evidence_correct": float(sql_evidence),
        "sql_evidence_mode": evidence_mode,
        "required_table_used": float(required_table_used),
        "successful_bash": float(successful_bash),
        "safe": float(safe),
        "bash_command_count": float(len(commands)),
        "unsafe_command_count": float(unsafe_command_count),
        "unsafe_network_count": float(unsafe_reason_counts["network"]),
        "unsafe_destructive_count": float(unsafe_reason_counts["destructive"]),
        "unsafe_host_path_escape_count": float(unsafe_reason_counts["host_path_escape"]),
        "unsafe_python_network_count": float(unsafe_reason_counts["python_network"]),
        "unsafe_root_scan_count": float(unsafe_reason_counts["root_scan"]),
        "valid_tool_protocol": float(valid_protocol),
        "has_final_answer": float(has_final),
        "gold_sql_verified": float(gold_sql_verified),
        "online_eligible": float(eligible),
        "verifier_error": verifier_error,
    }


def compute_score_dense30(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Pinned 30% dense-correctness entry point for the Step100→120 trial."""
    return compute_score(
        data_source,
        solution_str,
        ground_truth,
        extra_info,
        _dense_weight_override=0.30,
        **kwargs,
    )

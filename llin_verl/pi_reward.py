"""Boss-aligned DWH reward with executable-evidence safety guards."""

from __future__ import annotations

from collections import Counter
import csv
from functools import lru_cache
import json
import math
import os
import re
import sqlite3
import time
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

_TABLE_LABEL_KEYS = {"category", "date", "label", "name", "类别", "分类", "日期", "名称", "项目"}
_TABLE_VALUE_KEYS = {"value", "metric", "amount", "count", "score", "数值", "指标值", "金额", "数量", "结果"}
_TABLE_RANK_KEYS = {"rank", "ranking", "序号", "排名", "名次"}
_MARKDOWN_SEPARATOR_RE = re.compile(r"^:?-{2,}:?$")
_ASCII_TABLE_BORDER_RE = re.compile(r"^\s*\+(?:[-=:]+\+)+\s*$")
_PLAIN_TABLE_ROW_RE = re.compile(
    r"^\s*(?:(?P<rank>\d+)\s*[.)、]\s*|[-*+]\s*)?"
    r"(?P<label>.+?)\s*(?:[:：=]|\s+[-—]\s+)\s*(?P<value>.+?)\s*$"
)
_PLAIN_RANKED_ROW_RE = re.compile(
    r"^\s*(?P<rank>\d+)\s*[.)、]\s*(?P<label>.+?)\s+"
    r"(?P<value>[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?(?:\s*[%％])?)\s*$"
)


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
            r"(?:execute|executemany|read_sql(?:_query)?)\s*\(\s*([\"']{1,3})"
            r"((?:SELECT|WITH).*?)\1",
            command,
            re.IGNORECASE | re.DOTALL,
        ):
            selects.append(match.group(2).strip().rstrip(";").strip())
        # SQLAlchemy is commonly used as ``connection.execute(text("SELECT ..."))``.
        for match in re.finditer(
            r"(?:execute|read_sql(?:_query)?)\s*\(\s*(?:text\s*\(\s*)?"
            r"([\"']{1,3})((?:SELECT|WITH).*?)\1",
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
            # Python heredocs are already handled by the execute/read_sql
            # extractors above. Treating the entire Python suffix as SQL would
            # manufacture a second malformed statement.
            if re.search(r"(?:sqlite3\.connect|\.execute\s*\(|read_sql(?:_query)?\s*\()", body, re.IGNORECASE):
                continue
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


def execute_readonly_sql(
    database: Path,
    sql: str,
    max_rows: int = 10_000,
    query_timeout_seconds: float = 5.0,
) -> list[tuple[Any, ...]]:
    if not re.match(r"^\s*(?:SELECT|WITH)\b", sql or "", re.IGNORECASE):
        raise ValueError("only SELECT/WITH verifier SQL is allowed")
    if query_timeout_seconds <= 0:
        raise ValueError("query_timeout_seconds must be positive")
    uri = f"file:{quote(database.as_posix(), safe='/')}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True, timeout=5.0)
    deadline = time.monotonic() + query_timeout_seconds
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 10_000)
        cursor = connection.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            raise ValueError("SQL evidence exceeds max_rows")
        return rows
    finally:
        connection.set_progress_handler(None, 0)
        connection.close()


@lru_cache(maxsize=2048)
def _cached_gold_rows(
    database_path: str,
    database_mtime_ns: int,
    database_size: int,
    sql: str,
    max_rows: int,
    query_timeout_seconds: float,
) -> tuple[tuple[Any, ...], ...]:
    del database_mtime_ns, database_size
    return tuple(
        execute_readonly_sql(
            Path(database_path),
            sql,
            max_rows=max_rows,
            query_timeout_seconds=query_timeout_seconds,
        )
    )


def execute_cached_gold_sql(
    database: Path,
    sql: str,
    *,
    max_rows: int = 10_000,
    query_timeout_seconds: float = 5.0,
) -> list[tuple[Any, ...]]:
    stat = database.stat()
    return list(
        _cached_gold_rows(
            str(database),
            stat.st_mtime_ns,
            stat.st_size,
            sql,
            max_rows,
            query_timeout_seconds,
        )
    )


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


def _normalize_table_text(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"^[`*_~\"']+|[`*_~\"']+$", "", text)
    return " ".join(text.split()).casefold()


def _single_finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value).strip()
    matches = list(_NUMBER_RE.finditer(text))
    if len(matches) != 1:
        return None
    residue = (text[: matches[0].start()] + text[matches[0].end() :]).strip()
    residue = re.sub(r"^[￥¥$€£]\s*|\s*[%％]$", "", residue).strip()
    residue = re.sub(r"^[`*_~()（）\[\]{}\s]+|[`*_~()（）\[\]{}\s]+$", "", residue)
    if residue:
        return None
    try:
        number = float(matches[0].group(0).replace(",", ""))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _single_table_cell_number(value: Any) -> float | None:
    """Return the only number in a table cell while permitting unit text."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    matches = list(_NUMBER_RE.finditer(str(value)))
    if len(matches) != 1:
        return None
    try:
        number = float(matches[0].group(0).replace(",", ""))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _integer_rank(value: Any) -> int | None:
    number = _single_finite_number(value)
    if number is None or not number.is_integer() or number < 1:
        return None
    return int(number)


def _json_table_candidates(answer: str) -> list[tuple[str, list[tuple[str, float, int | None]]]]:
    decoder = json.JSONDecoder()
    payloads: list[Any] = []
    seen_spans: set[tuple[int, int]] = set()
    for start, character in enumerate(answer):
        if character not in "[{":
            continue
        try:
            payload, length = decoder.raw_decode(answer[start:])
        except json.JSONDecodeError:
            continue
        span = (start, start + length)
        if span not in seen_spans:
            seen_spans.add(span)
            payloads.append(payload)
        if len(payloads) >= 32:
            break

    output: list[tuple[str, list[tuple[str, float, int | None]]]] = []
    for payload in payloads:
        rows = payload
        if isinstance(payload, dict):
            rows = next(
                (payload.get(key) for key in ("rows", "data", "result", "results") if isinstance(payload.get(key), list)),
                None,
            )
        if not isinstance(rows, list) or not rows or not all(isinstance(item, dict) for item in rows):
            continue
        parsed: list[tuple[str, float, int | None]] = []
        for item in rows:
            folded = {_normalize_table_text(key): value for key, value in item.items()}
            label = next((folded[key] for key in _TABLE_LABEL_KEYS if key in folded), None)
            value = next((folded[key] for key in _TABLE_VALUE_KEYS if key in folded), None)
            rank = next((_integer_rank(folded[key]) for key in _TABLE_RANK_KEYS if key in folded), None)
            number = _single_finite_number(value)
            normalized_label = _normalize_table_text(label) if label is not None else ""
            if not normalized_label or number is None:
                parsed = []
                break
            parsed.append((normalized_label, number, rank))
        if parsed:
            output.append(("json", parsed))
    return output


def _markdown_cells(line: str) -> list[str]:
    cells = [cell.strip() for cell in line.strip().split("|")]
    if cells and not cells[0]:
        cells.pop(0)
    if cells and not cells[-1]:
        cells.pop()
    return cells


def _markdown_table_candidates(
    answer: str,
    expected_rows: list[tuple[str, float]],
    abs_tol: float,
    rel_tol: float,
) -> list[tuple[str, list[tuple[str, float, int | None]]]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in answer.splitlines() + [""]:
        if "|" in line or _ASCII_TABLE_BORDER_RE.fullmatch(line):
            current.append(line)
        elif current:
            blocks.append(current)
            current = []

    output: list[tuple[str, list[tuple[str, float, int | None]]]] = []
    expected_by_label = dict(expected_rows)
    for block in blocks:
        matrix = [_markdown_cells(line) for line in block]
        matched: list[tuple[int, int, str, list[str]]] = []
        for line_index, cells in enumerate(matrix):
            if len(cells) < 2 or all(_MARKDOWN_SEPARATOR_RE.fullmatch(cell.replace(" ", "")) for cell in cells):
                continue
            label_hits = [
                (column, folded)
                for column, cell in enumerate(cells)
                if (folded := _normalize_table_text(cell)) in expected_by_label
            ]
            if len(label_hits) == 1:
                matched.append((line_index, label_hits[0][0], label_hits[0][1], cells))
        if len(matched) != len(expected_rows):
            continue
        if [label for _, _, label, _ in matched] != [label for label, _ in expected_rows]:
            continue
        label_columns = {column for _, column, _, _ in matched}
        if len(label_columns) != 1:
            continue
        label_column = next(iter(label_columns))
        widths = {len(cells) for _, _, _, cells in matched}
        if len(widths) != 1:
            continue
        width = next(iter(widths))
        value_columns: list[int] = []
        for column in range(width):
            if column == label_column:
                continue
            actual_values = [_single_table_cell_number(cells[column]) for _, _, _, cells in matched]
            if all(value is not None for value in actual_values) and all(
                math.isclose(float(actual), expected, abs_tol=abs_tol, rel_tol=rel_tol)
                for actual, (_, expected) in zip(actual_values, expected_rows, strict=True)
            ):
                value_columns.append(column)
        if len(value_columns) != 1:
            continue
        value_column = value_columns[0]
        matched_indices = [line_index for line_index, _, _, _ in matched]
        if matched_indices != list(range(matched_indices[0], matched_indices[0] + len(matched_indices))):
            continue
        extra_data_row = False
        for line_index, cells in enumerate(matrix):
            if line_index in matched_indices or len(cells) != width:
                continue
            folded_cells = {_normalize_table_text(cell) for cell in cells}
            if folded_cells & _TABLE_LABEL_KEYS and folded_cells & _TABLE_VALUE_KEYS:
                continue
            label_cell = _normalize_table_text(cells[label_column])
            value_cell = _single_table_cell_number(cells[value_column])
            if label_cell and value_cell is not None:
                extra_data_row = True
                break
        if extra_data_row:
            continue
        rank_column = next(
            (
                column
                for cells in matrix[: matched_indices[0]]
                for column, cell in enumerate(cells)
                if column < width and _normalize_table_text(cell) in _TABLE_RANK_KEYS
            ),
            None,
        )
        parsed: list[tuple[str, float, int | None]] = []
        for _, _, label, cells in matched:
            rank = _integer_rank(cells[rank_column]) if rank_column is not None else None
            parsed.append((label, float(_single_table_cell_number(cells[value_column])), rank))
        output.append(("markdown", parsed))
    return output


def _plain_table_candidates(answer: str) -> list[tuple[str, list[tuple[str, float, int | None]]]]:
    blocks: list[list[tuple[str, float, int | None]]] = []
    current: list[tuple[str, float, int | None]] = []
    for line in answer.splitlines() + [""]:
        match = _PLAIN_TABLE_ROW_RE.fullmatch(line) or _PLAIN_RANKED_ROW_RE.fullmatch(line)
        row: tuple[str, float, int | None] | None = None
        if match:
            label = _normalize_table_text(match.group("label"))
            value = _single_finite_number(match.group("value"))
            rank = _integer_rank(match.group("rank")) if match.group("rank") else None
            if label and value is not None:
                row = (label, value, rank)
        if row is not None:
            current.append(row)
        elif current:
            blocks.append(current)
            current = []
    return [("plain", rows) for rows in blocks if rows]


def _full_table_value_equal(left: Any, right: Any, abs_tol: float, rel_tol: float) -> bool:
    if isinstance(right, (int, float)) and not isinstance(right, bool):
        number = _single_finite_number(left)
        if number is None:
            return False
        expected = float(right)
        if math.isclose(number, expected, abs_tol=abs_tol, rel_tol=rel_tol):
            return True
        text = str(left)
        return ("%" in text or "％" in text) and math.isclose(
            number / 100.0, expected, abs_tol=abs_tol, rel_tol=rel_tol
        )
    if right is None:
        return left is None or _normalize_table_text(left) in {"", "none", "null", "n/a", "na"}
    return _normalize_table_text(left) == _normalize_table_text(right)


def _normalize_full_expected_table(value: Any) -> list[list[Any]] | None:
    if not isinstance(value, list) or not value:
        return None
    if all(isinstance(item, (list, tuple)) for item in value):
        rows = [list(item) for item in value]
        return rows if rows and all(len(row) == len(rows[0]) for row in rows) else None
    if all(isinstance(item, dict) for item in value):
        keys = list(value[0])
        if keys and all(list(item) == keys for item in value):
            return [[item[key] for key in keys] for item in value]
    return None


def _full_table_rows_equal(
    candidate: list[list[Any]],
    expected: list[list[Any]],
    abs_tol: float,
    rel_tol: float,
) -> bool:
    if len(candidate) != len(expected):
        return False
    if any(len(left) != len(right) for left, right in zip(candidate, expected, strict=True)):
        return False
    return all(
        _full_table_value_equal(left, right, abs_tol, rel_tol)
        for left_row, right_row in zip(candidate, expected, strict=True)
        for left, right in zip(left_row, right_row, strict=True)
    )


def _full_table_row_equal(
    candidate: list[Any],
    expected: list[Any],
    abs_tol: float,
    rel_tol: float,
) -> bool:
    if len(candidate) != len(expected):
        return False
    return all(
        _full_table_value_equal(left, right, abs_tol, rel_tol)
        for left, right in zip(candidate, expected, strict=True)
    )


def _full_table_row_multisets_equal(
    candidate: list[list[Any]],
    expected: list[list[Any]],
    abs_tol: float,
    rel_tol: float,
) -> bool:
    """Compare complete rows as a duplicate-preserving unordered multiset.

    Numeric tolerance makes a plain ``Counter`` insufficient.  A small
    bipartite match preserves duplicate multiplicity without imposing an
    arbitrary row order.  DWH answer tables are bounded by the verifier's row
    limit, so the augmenting-path implementation is both deterministic and
    cheap for the approved training pool.
    """

    if len(candidate) != len(expected):
        return False
    if not candidate:
        return False
    expected_width = len(expected[0])
    if expected_width <= 0:
        return False
    if any(len(row) != expected_width for row in candidate + expected):
        return False

    edges = [
        [
            expected_index
            for expected_index, expected_row in enumerate(expected)
            if _full_table_row_equal(candidate_row, expected_row, abs_tol, rel_tol)
        ]
        for candidate_row in candidate
    ]
    if any(not options for options in edges):
        return False

    matched_candidate_for_expected = [-1] * len(expected)

    def augment(candidate_index: int, visited: set[int]) -> bool:
        for expected_index in edges[candidate_index]:
            if expected_index in visited:
                continue
            visited.add(expected_index)
            previous = matched_candidate_for_expected[expected_index]
            if previous == -1 or augment(previous, visited):
                matched_candidate_for_expected[expected_index] = candidate_index
                return True
        return False

    return all(augment(index, set()) for index in range(len(candidate)))


def _json_full_table_candidates(answer: str) -> list[list[list[Any]]]:
    decoder = json.JSONDecoder()
    output: list[list[list[Any]]] = []
    seen_spans: set[tuple[int, int]] = set()
    for start, character in enumerate(answer):
        if character not in "[{":
            continue
        try:
            payload, length = decoder.raw_decode(answer[start:])
        except json.JSONDecodeError:
            continue
        span = (start, start + length)
        if span in seen_spans:
            continue
        seen_spans.add(span)
        if isinstance(payload, dict):
            payload = next(
                (
                    payload.get(key)
                    for key in ("rows", "data", "result", "results")
                    if isinstance(payload.get(key), list)
                ),
                None,
            )
        if not isinstance(payload, list) or not payload:
            continue
        if all(isinstance(item, list) for item in payload):
            output.append([list(item) for item in payload])
        elif all(isinstance(item, dict) for item in payload):
            keys = list(payload[0])
            if keys and all(list(item) == keys for item in payload):
                output.append([[item[key] for key in keys] for item in payload])
        if len(output) >= 32:
            break
    return output


def _markdown_full_table_candidates(answer: str) -> list[list[list[Any]]]:
    lines = answer.splitlines()
    output: list[list[list[Any]]] = []
    for index in range(1, len(lines)):
        separators = _markdown_cells(lines[index]) if "|" in lines[index] else []
        if not separators or not all(
            _MARKDOWN_SEPARATOR_RE.fullmatch(cell.replace(" ", "")) for cell in separators
        ):
            continue
        header = _markdown_cells(lines[index - 1])
        if len(header) != len(separators):
            continue
        rows: list[list[Any]] = []
        for line in lines[index + 1 :]:
            if "|" not in line:
                break
            cells = _markdown_cells(line)
            if len(cells) != len(header):
                break
            rows.append(cells)
        if rows:
            output.append(rows)

    block: list[list[Any]] = []
    for line in lines + [""]:
        cells = _markdown_cells(line) if "|" in line else []
        if len(cells) >= 2 and (not block or len(cells) == len(block[0])):
            block.append(cells)
        else:
            if len(block) >= 2:
                output.extend([block, block[1:]])
            block = []
    return output


def _delimited_full_table_candidates(answer: str) -> list[list[list[Any]]]:
    output: list[list[list[Any]]] = []
    for delimiter in ("\t", ","):
        block: list[list[Any]] = []
        for line in answer.splitlines() + [""]:
            if delimiter not in line:
                if len(block) >= 2:
                    output.extend([block, block[1:]])
                block = []
                continue
            try:
                cells = next(csv.reader([line], delimiter=delimiter))
            except csv.Error:
                cells = []
            cells = [cell.strip() for cell in cells]
            if len(cells) >= 2 and (not block or len(cells) == len(block[0])):
                block.append(cells)
            else:
                if len(block) >= 2:
                    output.extend([block, block[1:]])
                block = []
    return output


def _drop_full_table_rank_column(rows: list[list[Any]], expected_width: int) -> list[list[Any]]:
    if not rows or len(rows[0]) != expected_width + 1:
        return rows
    for column in range(len(rows[0])):
        ranks = [_integer_rank(row[column]) for row in rows]
        if ranks == list(range(1, len(rows) + 1)):
            return [row[:column] + row[column + 1 :] for row in rows]
    return rows


def _strict_full_table_answer_match(
    answer: str,
    expected_value: Any,
    abs_tol: float,
    rel_tol: float,
) -> tuple[bool, str, int]:
    """Compare every ordered row and column of a general two-dimensional gold."""

    expected = _normalize_full_expected_table(expected_value)
    if expected is None:
        return False, "invalid_gold", 0
    candidates = [
        *(("markdown_full", rows) for rows in _markdown_full_table_candidates(answer)),
        *(("json_full", rows) for rows in _json_full_table_candidates(answer)),
        *(("delimited_full", rows) for rows in _delimited_full_table_candidates(answer)),
    ]
    largest = max((len(rows) for _, rows in candidates), default=0)
    expected_width = len(expected[0])
    for mode, rows in candidates:
        rows = _drop_full_table_rank_column(rows, expected_width)
        if _full_table_rows_equal(rows, expected, abs_tol, rel_tol):
            return True, mode, len(rows)
    return False, "none", largest


def strict_table_answer_match(
    answer: str,
    expected: list[Any],
    abs_tol: float,
    rel_tol: float,
) -> tuple[bool, str, int]:
    """Match an ordered category/value table without independent token hits.

    Rows must be structurally parseable as JSON, Markdown, or a contiguous
    plain-text list.  Cardinality, row order, label/value binding, duplicate
    labels, and optional explicit ranks are checked together.  This prevents a
    number dump or a permutation of the gold rows from entering a correct band.
    """
    expected_rows: list[tuple[str, float]] = []
    for item in expected:
        if not isinstance(item, dict):
            return False, "invalid_gold", 0
        label = item.get("category", item.get("date"))
        value = _single_finite_number(item.get("value"))
        normalized_label = _normalize_table_text(label) if label is not None else ""
        if not normalized_label or value is None:
            return False, "invalid_gold", 0
        expected_rows.append((normalized_label, value))
    if not expected_rows or len({label for label, _ in expected_rows}) != len(expected_rows):
        return False, "invalid_gold", 0

    candidates = [
        *_json_table_candidates(answer),
        *_markdown_table_candidates(answer, expected_rows, abs_tol, rel_tol),
        *_plain_table_candidates(answer),
    ]
    largest = max((len(rows) for _, rows in candidates), default=0)
    for mode, rows in candidates:
        if len(rows) != len(expected_rows):
            continue
        labels = [label for label, _, _ in rows]
        if len(set(labels)) != len(labels):
            continue
        ranks = [rank for _, _, rank in rows]
        if any(rank is not None for rank in ranks) and ranks != list(range(1, len(rows) + 1)):
            continue
        if all(
            actual_label == expected_label
            and math.isclose(actual_value, expected_value, abs_tol=abs_tol, rel_tol=rel_tol)
            for (actual_label, actual_value, _), (expected_label, expected_value) in zip(rows, expected_rows, strict=True)
        ):
            return True, mode, len(rows)
    return False, "none", largest


def strict_table_answer_match_complete(
    answer: str,
    expected: list[Any],
    abs_tol: float,
    rel_tol: float,
) -> tuple[bool, str, int]:
    """Use the legacy category/value route or exact general full-table route.

    Keeping this as a new entry point preserves historical banded-v2 label
    reproducibility while strict-correctness-v3 opts into the repaired all-row,
    all-column comparator.
    """

    is_category_value_gold = all(
        isinstance(item, dict)
        and set(item).issubset({"category", "date", "value"})
        and "value" in item
        and ("category" in item or "date" in item)
        for item in expected
    )
    if is_category_value_gold:
        return strict_table_answer_match(answer, expected, abs_tol, rel_tol)
    return _strict_full_table_answer_match(answer, expected, abs_tol, rel_tol)


_ORDER_SQL_RE = re.compile(
    r"\border\s+by\b|\b(?:row_number|rank|dense_rank)\s*\(",
    re.IGNORECASE,
)
_ORDER_PLAN_RE = re.compile(
    r"\b(?:ordered|sorted|top\s*[-_ ]?n|ranking|ranked|trend)\b|"
    r"排序|排名|排行|趋势|前\s*\d+|最高|最低",
    re.IGNORECASE,
)


def table_order_semantics(
    ground_truth: dict[str, Any],
) -> tuple[bool, str]:
    """Return whether full-table row order is part of the gold semantics.

    The training contract intentionally does not infer order from the physical
    SQL result alone.  Order is binding only when the verification SQL or an
    attached EvidencePlan explicitly states an ordering, Top-N, ranking, or
    trend requirement.
    """

    verification_sql = str(ground_truth.get("verification_sql") or "")
    if _ORDER_SQL_RE.search(verification_sql):
        return True, "verification_sql"
    evidence_plan = ground_truth.get("evidence_plan")
    if isinstance(evidence_plan, dict):
        # Audit plans contain an ``order_by`` key even when it is empty.  The
        # key name alone is not an ordering requirement; only a populated
        # declaration, a positive Top-N limit, or an explicit semantic value
        # makes row order binding.
        if evidence_plan.get("order_by"):
            return True, "evidence_plan.order_by"
        limit = evidence_plan.get("limit")
        if isinstance(limit, (int, float)) and not isinstance(limit, bool) and limit > 0:
            return True, "evidence_plan.limit"
        semantic_values = {
            key: value
            for key, value in evidence_plan.items()
            if key not in {"order_by", "limit"}
        }
        plan_text = json.dumps(semantic_values, ensure_ascii=False, sort_keys=True)
        if _ORDER_PLAN_RE.search(plan_text):
            return True, "evidence_plan.semantic_value"
    elif evidence_plan is not None and _ORDER_PLAN_RE.search(str(evidence_plan)):
        return True, "evidence_plan.semantic_value"
    return False, "no_explicit_order_semantics"


def strict_table_answer_match_semantic(
    answer: str,
    expected_value: Any,
    abs_tol: float,
    rel_tol: float,
    *,
    ordered: bool,
) -> tuple[bool, str, int]:
    """Compare every table row/column using the declared order semantics.

    Ordered tasks require exact row order.  Tasks without order semantics use
    a complete duplicate-preserving row multiset.  Both modes reject missing
    or extra rows and columns; unlike the historical comparator, this route
    never silently drops an undeclared rank column.
    """

    expected = _normalize_full_expected_table(expected_value)
    if expected is None:
        return False, "invalid_gold", 0
    candidates = [
        *(("markdown_full", rows) for rows in _markdown_full_table_candidates(answer)),
        *(("json_full", rows) for rows in _json_full_table_candidates(answer)),
        *(("delimited_full", rows) for rows in _delimited_full_table_candidates(answer)),
    ]
    largest = max((len(rows) for _, rows in candidates), default=0)
    for mode, rows in candidates:
        # A presentation-only 1..N rank column is semantically redundant only
        # when row order is itself binding (ORDER BY/TopN/ranking/trend).  It is
        # removed after verifying the exact sequence; arbitrary extra columns
        # and rank columns on unordered tasks remain invalid.
        if ordered:
            rows = _drop_full_table_rank_column(rows, len(expected[0]))
        matched = (
            _full_table_rows_equal(rows, expected, abs_tol, rel_tol)
            if ordered
            else _full_table_row_multisets_equal(rows, expected, abs_tol, rel_tol)
        )
        if matched:
            suffix = "ordered" if ordered else "row_multiset"
            return True, f"{mode}_{suffix}", len(rows)
    return False, "none", largest


def _table_answer_correct(
    answer: str,
    expected: list[Any],
    abs_tol: float,
    rel_tol: float,
) -> bool:
    matched, _, _ = strict_table_answer_match(answer, expected, abs_tol, rel_tol)
    return matched


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
    _banded_reward_override: bool | None = None,
    _reward_contract_override: str | None = None,
    _strict_full_table_override: bool = False,
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
    answer_type = str(ground_truth.get("answer_type") or "")
    final_answer_match_mode = "numeric" if answer_type == "numeric" else "none"
    strict_table_rows_parsed = 0
    if answer_type == "table" and isinstance(expected_value, list):
        table_matcher = (
            strict_table_answer_match_complete
            if _strict_full_table_override
            else strict_table_answer_match
        )
        answer_ok, final_answer_match_mode, strict_table_rows_parsed = table_matcher(
            answer,
            expected_value,
            abs_tol,
            rel_tol,
        )
    else:
        answer_ok = final_answer_correct(
            answer,
            answer_type,
            expected_value,
            abs_tol,
            rel_tol,
        )
    dense_correctness = dense_final_answer_correctness(
        answer,
        answer_type,
        expected_value,
        abs_tol,
        rel_tol,
    )

    sql_evidence = False
    evidence_mode = "none"
    verifier_error = ""
    gold_sql_verified = False
    sql_evidence_queries_checked = 0
    sql_evidence_queries_truncated = 0
    try:
        try:
            sql_query_timeout_seconds = float(os.environ.get("PI_REWARD_SQL_TIMEOUT_SECONDS", "5"))
            max_evidence_selects = int(os.environ.get("PI_REWARD_MAX_EVIDENCE_SELECTS", "32"))
        except ValueError as exc:
            raise ValueError("PI reward SQL timeout/select limits must be numeric") from exc
        if sql_query_timeout_seconds <= 0 or max_evidence_selects <= 0:
            raise ValueError("PI reward SQL timeout/select limits must be positive")
        database = _database_path(
            os.environ.get("PI_AGENT_SANDBOX_LOWER", "/pi_sandbox"),
            str(ground_truth["environment_id"]),
        )
        verification_sql = str(ground_truth["verification_sql"])
        expected_rows = execute_cached_gold_sql(
            database,
            verification_sql,
            query_timeout_seconds=sql_query_timeout_seconds,
        )
        gold_sql_verified = bool(expected_rows)
        selected_evidence_sql = selects[-max_evidence_selects:]
        sql_evidence_queries_truncated = max(0, len(selects) - len(selected_evidence_sql))
        for sql in selected_evidence_sql:
            sql_evidence_queries_checked += 1
            try:
                mode = sql_evidence_mode(
                    sql,
                    execute_readonly_sql(
                        database,
                        sql,
                        query_timeout_seconds=sql_query_timeout_seconds,
                    ),
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
    if _banded_reward_override is None:
        banded_reward_enabled = os.environ.get("PI_REWARD_MODE", "blend") in {"banded_v1", "banded_v2"}
    else:
        banded_reward_enabled = bool(_banded_reward_override)
    score = (
        banded_reward_score(
            eligible=eligible,
            has_final_answer=has_final,
            final_answer_correct=bool(answer_ok),
            sql_evidence_correct=bool(sql_evidence),
            process_quality=base_score,
        )
        if banded_reward_enabled
        else (blended_score if eligible else 0.0)
    )
    reward_contract = _reward_contract_override or (
        "banded-v2-strict-table-v1"
        if os.environ.get("PI_REWARD_MODE") == "banded_v2"
        else ("banded-v1" if banded_reward_enabled else "blend-v1")
    )
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
        "banded_reward_enabled": float(banded_reward_enabled),
        "reward_contract": reward_contract,
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
        "final_answer_match_mode": final_answer_match_mode,
        "strict_table_rows_parsed": float(strict_table_rows_parsed),
        "strict_full_table_verifier_enabled": float(_strict_full_table_override),
        "sql_evidence_correct": float(sql_evidence),
        "sql_evidence_mode": evidence_mode,
        "sql_evidence_queries_checked": float(sql_evidence_queries_checked),
        "sql_evidence_queries_truncated": float(sql_evidence_queries_truncated),
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


def banded_reward_score(
    *,
    eligible: bool,
    has_final_answer: bool,
    final_answer_correct: bool,
    sql_evidence_correct: bool,
    process_quality: float,
) -> float:
    """Return a lexicographic reward whose correctness bands never overlap.

    Safety, protocol, and executable-gold eligibility remain hard gates.  A
    process-complete wrong answer cannot outrank any eligible correct answer,
    while correct SQL with a wrong final synthesis remains useful intermediate
    evidence.  ``process_quality`` only orders trajectories inside each band.
    """
    if not eligible or not has_final_answer:
        return 0.0
    quality = min(1.0, max(0.0, float(process_quality)))
    if final_answer_correct and sql_evidence_correct:
        return round(0.80 + 0.20 * quality, 6)
    if final_answer_correct:
        return round(0.65 + 0.10 * quality, 6)
    if sql_evidence_correct:
        return round(0.40 + 0.10 * quality, 6)
    return round(0.10 + 0.20 * quality, 6)


def compute_score_banded_v1(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Pinned non-overlapping correctness-band entry point for canary runs."""
    return compute_score(
        data_source,
        solution_str,
        ground_truth,
        extra_info,
        _dense_weight_override=0.0,
        _banded_reward_override=True,
        _reward_contract_override="banded-v1",
        **kwargs,
    )


def compute_score_banded_v2(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Strict table-safe correctness bands for Qwen3.8 GRPO training.

    The band geometry is unchanged from v1.  Table correctness now requires a
    structurally parsed, ordered, cardinality-exact category/value result with
    one consistent value column; independent label/number containment cannot
    enter a correct band.
    """
    return compute_score(
        data_source,
        solution_str,
        ground_truth,
        extra_info,
        _dense_weight_override=0.0,
        _banded_reward_override=True,
        _reward_contract_override="banded-v2-strict-table-v1",
        **kwargs,
    )


def compute_score_strict_correctness_v3(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Binary strict-outcome reward for collapse-safe GRPO.

    The full banded-v2 verifier still computes and records safety, SQL evidence,
    table completeness, boss/process and efficiency metrics.  Only the strict
    ``acc`` bit enters the scalar reward.  Process quality therefore cannot
    compensate for a wrong final outcome; the group-level variance patch then
    masks both all-wrong and all-correct prompt groups after KL assembly.
    """

    result = compute_score_banded_v2(
        data_source,
        solution_str,
        ground_truth,
        extra_info,
        _strict_full_table_override=True,
        **kwargs,
    )
    strict_correct = float(result["acc"])
    result["process_reward_observed"] = float(result["base_score"])
    result["process_reward_applied"] = 0.0
    result["score"] = strict_correct
    result["reward_contract"] = "strict-correctness-gated-v3"
    return result


def compute_score_correctness_gated_process_v5(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Lazy wrapper for the audited trajectory-level process reward."""

    from llin_verl.trajectory_process_reward import compute_trajectory_process_reward

    return compute_trajectory_process_reward(
        data_source,
        solution_str,
        ground_truth,
        extra_info,
        **kwargs,
    )

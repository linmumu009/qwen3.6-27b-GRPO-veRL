#!/usr/bin/env python3
"""Audit DWH rollout evidence and GRPO readiness without publishing sensitive data.

The detailed evidence contains prompts, SQL, gold values and final answers.  It
is written only below ``<output>/private`` with restrictive permissions.  The
top-level ``safe_summary.json`` contains aggregate counts and hashes only and is
safe to commit.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from llin_verl.outcome_shadow import score_final_outcome
from llin_verl.pi_reward import extract_answer_numbers, extract_final_assistant_answer


CONTRACT = "dwh-grpo-readiness-audit-v3"
SCOPE_BUCKETS = {"mixed", "all_wrong"}
SAFE_REVIEW_VALUES = {"strict_checker_pass", "manual_pair_review_pass"}
_TABLE_RE = re.compile(r"\b(?:from|join)\s+[`\"\[]?([A-Za-z_][A-Za-z0-9_]*)", re.I)
_MARKDOWN_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")
_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")
_SQL_ERROR_RE = re.compile(
    r"no such (?:table|column)|syntax error|sqlite error|operationalerror|"
    r"database is locked|malformed|ambiguous column|command failed",
    re.I,
)
_ENV_ERROR_RE = re.compile(
    r"file not found|no such file|database.*(?:missing|unavailable)|"
    r"cannot open.*database|permission denied|preflight.*fail",
    re.I,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_text(value: Any) -> str:
    text = str(value).strip().casefold()
    text = re.sub(r"^[`*_~\"']+|[`*_~\"']+$", "", text)
    return " ".join(text.split())


def parse_number(value: Any) -> tuple[float | None, bool]:
    if isinstance(value, bool) or value is None:
        return None, False
    if isinstance(value, (int, float)):
        number = float(value)
        return (number if math.isfinite(number) else None), False
    text = str(value).strip()
    matches = list(_NUMBER_RE.finditer(text))
    if len(matches) != 1:
        return None, "%" in text or "％" in text
    residue = (text[: matches[0].start()] + text[matches[0].end() :]).strip()
    residue = re.sub(r"^[￥¥$€£]\s*|\s*[%％]$", "", residue).strip()
    residue = re.sub(r"^[`*_~()（）\[\]{}\s]+|[`*_~()（）\[\]{}\s]+$", "", residue)
    if residue:
        return None, "%" in text or "％" in text
    try:
        number = float(matches[0].group(0).replace(",", ""))
    except ValueError:
        return None, "%" in text or "％" in text
    return (number if math.isfinite(number) else None), "%" in text or "％" in text


def values_equal(left: Any, right: Any, abs_tol: float, rel_tol: float) -> bool:
    if isinstance(right, (int, float)) and not isinstance(right, bool):
        number, percent = parse_number(left)
        if number is None:
            return False
        expected = float(right)
        if math.isclose(number, expected, abs_tol=abs_tol, rel_tol=rel_tol):
            return True
        return percent and math.isclose(
            number / 100.0, expected, abs_tol=abs_tol, rel_tol=rel_tol
        )
    if right is None:
        return left is None or normalize_text(left) in {"", "none", "null", "n/a", "na"}
    return normalize_text(left) == normalize_text(right)


def rows_equal(
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
        values_equal(left, right, abs_tol, rel_tol)
        for left_row, right_row in zip(candidate, expected, strict=True)
        for left, right in zip(left_row, right_row, strict=True)
    )


def _markdown_cells(line: str) -> list[str]:
    cells = [cell.strip() for cell in line.strip().split("|")]
    if cells and not cells[0]:
        cells.pop(0)
    if cells and not cells[-1]:
        cells.pop()
    return cells


def markdown_table_candidates(answer: str) -> list[list[list[Any]]]:
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

    # Some agents emit a pipe table without a Markdown separator.  Preserve
    # contiguous same-width blocks as lower-priority candidates.
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


def json_table_candidates(answer: str) -> list[list[list[Any]]]:
    decoder = json.JSONDecoder()
    output: list[list[list[Any]]] = []
    seen: set[tuple[int, int]] = set()
    for start, character in enumerate(answer):
        if character not in "[{":
            continue
        try:
            payload, length = decoder.raw_decode(answer[start:])
        except json.JSONDecodeError:
            continue
        span = (start, start + length)
        if span in seen:
            continue
        seen.add(span)
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


def delimited_table_candidates(answer: str) -> list[list[list[Any]]]:
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


def normalize_expected_table(value: Any) -> list[list[Any]] | None:
    if not isinstance(value, list) or not value:
        return None
    if all(isinstance(item, list) for item in value):
        rows = [list(item) for item in value]
        return rows if rows and all(len(row) == len(rows[0]) for row in rows) else None
    if all(isinstance(item, dict) for item in value):
        keys = list(value[0])
        if keys and all(list(item) == keys for item in value):
            return [[item[key] for key in keys] for item in value]
    return None


def _drop_rank_column(rows: list[list[Any]], expected_width: int) -> list[list[Any]]:
    if not rows or len(rows[0]) != expected_width + 1:
        return rows
    for column in range(len(rows[0])):
        values = [parse_number(row[column])[0] for row in rows]
        if values == [float(index) for index in range(1, len(rows) + 1)]:
            return [row[:column] + row[column + 1 :] for row in rows]
    return rows


def full_table_answer_match(
    answer: str,
    expected_value: Any,
    abs_tol: float,
    rel_tol: float,
) -> tuple[bool, str, int, int]:
    expected = normalize_expected_table(expected_value)
    if expected is None:
        return False, "invalid_gold", 0, 0
    candidates = [
        *(("markdown", rows) for rows in markdown_table_candidates(answer)),
        *(("json", rows) for rows in json_table_candidates(answer)),
        *(("delimited", rows) for rows in delimited_table_candidates(answer)),
    ]
    largest_rows = max((len(rows) for _, rows in candidates), default=0)
    largest_width = max((len(rows[0]) for _, rows in candidates if rows), default=0)
    for mode, rows in candidates:
        rows = _drop_rank_column(rows, len(expected[0]))
        if rows_equal(rows, expected, abs_tol, rel_tol):
            return True, mode, len(rows), len(rows[0])
    return False, "none", largest_rows, largest_width


def recursive_equal(actual: Any, expected: Any, abs_tol: float, rel_tol: float) -> bool:
    if isinstance(expected, list) and isinstance(actual, (list, tuple)):
        return len(actual) == len(expected) and all(
            recursive_equal(left, right, abs_tol, rel_tol)
            for left, right in zip(actual, expected, strict=True)
        )
    return values_equal(actual, expected, abs_tol, rel_tol)


def expected_from_truth(truth: dict[str, Any]) -> Any:
    value = truth.get("expected_value_json", truth.get("expected_value"))
    return json.loads(value) if isinstance(value, str) else value


def audited_outcome(output: str, truth: dict[str, Any]) -> dict[str, Any]:
    answer = extract_final_assistant_answer(output)
    answer_type = str(truth.get("answer_type") or "")
    expected = expected_from_truth(truth)
    abs_tol = float(truth.get("abs_tol", 1e-3))
    rel_tol = float(truth.get("rel_tol", 1e-5))
    if answer_type == "numeric" and isinstance(expected, (int, float)):
        values = extract_answer_numbers(answer)
        correct = any(
            math.isclose(value, float(expected), abs_tol=abs_tol, rel_tol=rel_tol)
            for value in values
        )
        return {
            "correct": correct,
            "has_final_answer": bool(answer),
            "format_valid": bool(values),
            "mode": "numeric_result_value" if correct else "none",
            "parsed_rows": 0,
            "parsed_width": 0,
            "result_number_count": len(values),
        }
    if answer_type == "table":
        correct, mode, rows, width = full_table_answer_match(
            answer, expected, abs_tol, rel_tol
        )
        return {
            "correct": correct,
            "has_final_answer": bool(answer),
            "format_valid": rows > 0,
            "mode": mode,
            "parsed_rows": rows,
            "parsed_width": width,
            "result_number_count": 0,
        }
    return {
        "correct": False,
        "has_final_answer": bool(answer),
        "format_valid": False,
        "mode": "invalid_answer_type",
        "parsed_rows": 0,
        "parsed_width": 0,
        "result_number_count": 0,
    }


def sql_tables(sql: str) -> set[str]:
    return {match.group(1).casefold() for match in _TABLE_RE.finditer(sql or "")}


def _flatten_plan_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return []


def plan_sql_checks(task: dict[str, Any], actual_rows: list[list[Any]]) -> dict[str, bool]:
    plan = task.get("evidence_plan") or {}
    sql = str(task["gold_answer"]["verification_sql"])
    folded = sql.casefold()
    compact = re.sub(r"\s+", " ", folded)
    expected_tables = {str(value).casefold() for value in task.get("expected_tables", [])}
    referenced_tables = sql_tables(sql)
    expected_fields = {str(value).casefold() for value in task.get("expected_fields", [])}
    checks = {
        "sample_sql_equals_verification_sql": str(task.get("sample_sql") or "").strip()
        == sql.strip(),
        "expected_tables_present": bool(expected_tables)
        and expected_tables.issubset(referenced_tables),
        "expected_fields_present": all(
            re.search(rf"\b{re.escape(field)}\b", folded) for field in expected_fields
        ),
        # ``expected_operations=filter`` is a broad generator capability tag in
        # this dataset, not proof that the task has a row predicate.  Require a
        # WHERE clause only when the EvidencePlan carries an explicit filter.
        "filter_aligned": not bool(plan.get("filters")) or "where" in compact,
        "group_by_aligned": not bool(plan.get("group_by")) or " group by " in compact,
        "order_by_aligned": not bool(plan.get("order_by")) or " order by " in compact,
        "limit_aligned": not bool(plan.get("limit"))
        or bool(re.search(rf"\blimit\s+{int(plan['limit'])}\b", folded)),
        "percentage_aligned": not bool(plan.get("requires_percentage"))
        or "/" in sql
        or bool(re.search(r"\b(?:percent|percentage|ratio|rate)\b", folded)),
    }
    aggregation = str(plan.get("aggregation") or "").casefold()
    checks["aggregation_aligned"] = not aggregation or bool(
        re.search(rf"\b{re.escape(aggregation)}\s*\(", folded)
    )
    measures = _flatten_plan_values(plan.get("report_measures"))
    checks["report_measures_aligned"] = all(
        normalize_text(measure).replace(" ", "") in re.sub(r"\s+", "", folded)
        for measure in measures
    )
    answer_type = str(task["gold_answer"].get("answer_type") or "")
    expected = task["gold_answer"].get("value")
    if answer_type == "numeric":
        checks["output_shape_aligned"] = len(actual_rows) == 1 and len(actual_rows[0]) == 1
    else:
        normalized = normalize_expected_table(expected)
        checks["output_shape_aligned"] = normalized is not None and len(actual_rows) == len(
            normalized
        )
    return checks


def source_semantic_checks(task: dict[str, Any]) -> dict[str, bool]:
    answerability = task.get("answerability_label") or {}
    validation = task.get("validation") or {}
    return {
        "source_semantic_review_passed": str(task.get("_semantic_review"))
        in SAFE_REVIEW_VALUES,
        "answerable": bool(answerability.get("is_answerable")),
        "no_missing_requirements": not bool(task.get("missing_requirements")),
        "not_out_of_scope": not bool(task.get("out_of_scope_reason")),
        "source_database_validation_passed": bool(validation.get("checked_against_database"))
        and bool(validation.get("expected_result_exists")),
    }


def approved_candidate_completeness_checks(row: dict[str, Any]) -> dict[str, bool]:
    prompt = row.get("prompt") or []
    ground_truth = (row.get("reward_model") or {}).get("ground_truth") or {}
    extra = row.get("extra_info") or {}
    expected_json = ground_truth.get("expected_value_json")
    expected_json_valid = False
    if isinstance(expected_json, str) and expected_json.strip():
        try:
            json.loads(expected_json)
            expected_json_valid = True
        except json.JSONDecodeError:
            expected_json_valid = False
    return {
        "prompt_complete": bool(prompt)
        and all(
            isinstance(message, dict)
            and str(message.get("role") or "").strip()
            and str(message.get("content") or "").strip()
            for message in prompt
        ),
        "verification_sql_complete": bool(
            str(ground_truth.get("verification_sql") or "").strip()
        ),
        "gold_complete": expected_json_valid,
        "instruction_hash_complete": bool(
            str(extra.get("instruction_sha256") or "").strip()
        ),
        "gold_hash_complete": bool(str(extra.get("gold_sha256") or "").strip()),
    }


def replay_gold(
    connection: sqlite3.Connection,
    task: dict[str, Any],
    abs_tol: float,
    rel_tol: float,
) -> tuple[list[str], list[list[Any]], bool, str | None]:
    try:
        cursor = connection.execute(str(task["gold_answer"]["verification_sql"]))
        rows = [list(row) for row in cursor.fetchall()]
        columns = [str(item[0]) for item in cursor.description or []]
    except sqlite3.Error as error:
        return [], [], False, type(error).__name__
    expected = task["gold_answer"].get("value")
    answer_type = str(task["gold_answer"].get("answer_type") or "")
    if answer_type == "numeric":
        passed = len(rows) == 1 and len(rows[0]) == 1 and recursive_equal(
            rows[0][0], expected, abs_tol, rel_tol
        )
    else:
        normalized = normalize_expected_table(expected)
        passed = normalized is not None and rows_equal(rows, normalized, abs_tol, rel_tol)
    return columns, rows, passed, None


def write_private_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def write_private_parquet(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pylist(rows, schema=schema) if rows else pa.Table.from_pylist([], schema=schema)
    pq.write_table(table, temporary, compression="zstd")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def audit(
    dataset_path: Path,
    tasks_path: Path,
    database_path: Path,
    shards_dir: Path,
    per_task_path: Path,
    output_dir: Path,
    *,
    expected_tasks: int = 100,
    samples_per_task: int = 8,
    expected_approved_total: int | None = None,
) -> dict[str, Any]:
    dataset_table = pq.read_table(dataset_path)
    dataset = dataset_table.to_pylist()
    tasks = read_jsonl(tasks_path)
    per_task = read_jsonl(per_task_path)
    if not (len(dataset) == len(tasks) == len(per_task) == expected_tasks):
        raise ValueError("dataset/tasks/per-task row counts do not match expected task count")

    observations: dict[tuple[int, int], dict[str, Any]] = {}
    shard_paths = sorted(shards_dir.glob("tasks_*.jsonl"))
    for path in shard_paths:
        for row in read_jsonl(path):
            key = (int(row["source_task_index"]), int(row["sample_index"]))
            if key in observations:
                raise ValueError(f"duplicate trajectory slot: {key}")
            observations[key] = row
    expected_slots = {
        (task_index, sample_index)
        for task_index in range(expected_tasks)
        for sample_index in range(samples_per_task)
    }
    if set(observations) != expected_slots:
        raise ValueError("trajectory slots are not exactly complete")
    per_task_by_index = {int(row["source_task_index"]): row for row in per_task}
    if set(per_task_by_index) != set(range(expected_tasks)):
        raise ValueError("per-task outcomes do not cover all source tasks")

    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    detailed_rows: list[dict[str, Any]] = []
    mixed_rows: list[dict[str, Any]] = []
    all_wrong_rows: list[dict[str, Any]] = []
    approved_dataset_rows: list[dict[str, Any]] = []
    reward_repaired_mixed_dataset_rows: list[dict[str, Any]] = []
    approved43_dataset_rows: list[dict[str, Any]] = []
    approved43_manifest_rows: list[dict[str, Any]] = []
    mixed_dispositions: Counter[str] = Counter()
    all_wrong_causes: Counter[str] = Counter()
    all_wrong_secondary: Counter[str] = Counter()
    all_wrong_audited_buckets: Counter[str] = Counter()
    answer_type_causes: dict[str, Counter[str]] = defaultdict(Counter)
    legacy_to_audited: Counter[str] = Counter()
    audited_correct_histogram: Counter[int] = Counter()
    audited_histogram_by_scope_and_type: Counter[str] = Counter()
    format_valid_histogram_by_scope_and_type: Counter[str] = Counter()
    source_check_failures: Counter[str] = Counter()
    plan_check_failures: Counter[str] = Counter()
    binding_check_failures: Counter[str] = Counter()
    mixed_nonapproval_signals: Counter[str] = Counter()
    trajectory_signals: Counter[str] = Counter()
    gold_replay_passes = semantic_passes = reward_route_passes = 0
    scope_counts: Counter[str] = Counter()
    original_bucket_counts = Counter(str(row["bucket"]) for row in per_task)

    connection = sqlite3.connect(f"file:{database_path.as_posix()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    try:
        for index in range(expected_tasks):
            outcome_row = per_task_by_index[index]
            original_bucket = str(outcome_row["bucket"])
            if original_bucket not in SCOPE_BUCKETS:
                continue
            scope_counts[original_bucket] += 1
            dataset_row = dataset[index]
            task = tasks[index]
            truth = dataset_row["reward_model"]["ground_truth"]
            expected = expected_from_truth(truth)
            abs_tol = float(truth.get("abs_tol", 1e-3))
            rel_tol = float(truth.get("rel_tol", 1e-5))
            columns, replay_rows, gold_replay_pass, replay_error = replay_gold(
                connection, task, abs_tol, rel_tol
            )
            source_checks = source_semantic_checks(task)
            plan_checks = plan_sql_checks(task, replay_rows) if replay_error is None else {}
            task_instruction = str(task["natural_language_instruction"])
            dataset_binding_checks = {
                "prompt_contains_instruction": task_instruction
                in str(dataset_row["prompt"][-1]["content"]),
                "answer_type_matches": str(truth.get("answer_type"))
                == str(task["gold_answer"].get("answer_type")),
                "expected_value_matches": recursive_equal(
                    expected, task["gold_answer"].get("value"), abs_tol, rel_tol
                ),
                "verification_sql_matches": str(truth.get("verification_sql") or "").strip()
                == str(task["gold_answer"].get("verification_sql") or "").strip(),
                "instruction_hash_matches": str(dataset_row["extra_info"]["instruction_sha256"])
                == canonical_hash(task_instruction),
                "gold_hash_matches": str(dataset_row["extra_info"]["gold_sha256"])
                == canonical_hash(task["gold_answer"].get("value")),
            }
            semantic_pass = all(source_checks.values()) and all(plan_checks.values())
            binding_pass = all(dataset_binding_checks.values())
            source_check_failures.update(key for key, passed in source_checks.items() if not passed)
            plan_check_failures.update(key for key, passed in plan_checks.items() if not passed)
            binding_check_failures.update(
                key for key, passed in dataset_binding_checks.items() if not passed
            )
            trajectories: list[dict[str, Any]] = []
            audited_correct_count = legacy_correct_count = 0
            label_agreement = True
            format_valid_count = sql_error_count = environment_error_count = 0
            timeout_count = runtime_error_count = 0
            for sample_index in range(samples_per_task):
                row = observations[(index, sample_index)]
                output = str(row.get("output") or "")
                final_answer = extract_final_assistant_answer(output)
                legacy = score_final_outcome(output, truth)
                reviewed = audited_outcome(output, truth)
                legacy_correct = bool(legacy["final_answer_correct"])
                audited_correct = bool(reviewed["correct"])
                legacy_correct_count += int(legacy_correct)
                audited_correct_count += int(audited_correct)
                label_agreement = label_agreement and legacy_correct == audited_correct
                format_valid_count += int(reviewed["format_valid"])
                sql_error = bool(_SQL_ERROR_RE.search(output))
                environment_error = bool(_ENV_ERROR_RE.search(output))
                sql_error_count += int(sql_error)
                environment_error_count += int(environment_error)
                timeout = bool(row.get("trajectory_timeout"))
                runtime_error = bool(row.get("runtime_error"))
                timeout_count += int(timeout)
                runtime_error_count += int(runtime_error)
                trajectories.append(
                    {
                        "sample_index": sample_index,
                        "source_shard_slot": [index, sample_index],
                        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                        "final_answer": final_answer,
                        "legacy_correct": legacy_correct,
                        "audited_correct": audited_correct,
                        "label_agrees": legacy_correct == audited_correct,
                        "final_answer_format_valid": bool(reviewed["format_valid"]),
                        "audit_match_mode": reviewed["mode"],
                        "parsed_rows": reviewed["parsed_rows"],
                        "parsed_width": reviewed["parsed_width"],
                        "result_number_count": reviewed["result_number_count"],
                        "sql_error_marker": sql_error,
                        "environment_error_marker": environment_error,
                        "runtime_error": runtime_error,
                        "trajectory_timeout": timeout,
                        "response_tokens": int(row.get("response_tokens", 0) or 0),
                    }
                )

            if legacy_correct_count != int(outcome_row["correct_count"]):
                raise ValueError(
                    "stored per-task legacy correct count is not reproducible: "
                    f"index={index} stored={outcome_row['correct_count']} "
                    f"recomputed={legacy_correct_count}"
                )
            gold_replay_passes += int(gold_replay_pass)
            semantic_passes += int(semantic_pass and binding_pass)
            reward_route_pass = label_agreement and legacy_correct_count == audited_correct_count
            reward_route_passes += int(reward_route_pass)
            legacy_to_audited[
                f"{legacy_correct_count}->{audited_correct_count}"
            ] += 1
            audited_correct_histogram[audited_correct_count] += 1
            scope_type = f"{original_bucket}|{truth.get('answer_type')}"
            audited_histogram_by_scope_and_type[f"{scope_type}|{audited_correct_count}"] += 1
            format_valid_histogram_by_scope_and_type[f"{scope_type}|{format_valid_count}"] += 1
            trajectory_signals["reviewed"] += samples_per_task
            trajectory_signals["format_valid"] += format_valid_count
            trajectory_signals["timeout"] += timeout_count
            trajectory_signals["runtime_error"] += runtime_error_count
            trajectory_signals["sql_error_marker"] += sql_error_count
            trajectory_signals["environment_error_marker"] += environment_error_count

            disposition: str | None = None
            root_cause: str | None = None
            secondary_causes: list[str] = []
            audited_bucket = (
                "all_wrong"
                if audited_correct_count == 0
                else "all_correct"
                if audited_correct_count == samples_per_task
                else "mixed"
            )
            if original_bucket == "mixed":
                if not all(source_checks.values()) or replay_error is not None or not gold_replay_pass:
                    disposition = "剔除"
                elif not semantic_pass or not binding_pass or not reward_route_pass:
                    disposition = "需修复"
                elif runtime_error_count or timeout_count or not 1 <= audited_correct_count <= 7:
                    disposition = "需修复"
                else:
                    disposition = "可训练"
                mixed_dispositions[disposition] += 1
                if disposition != "可训练":
                    mixed_nonapproval_signals.update(
                        f"source:{key}" for key, passed in source_checks.items() if not passed
                    )
                    mixed_nonapproval_signals.update(
                        f"plan:{key}" for key, passed in plan_checks.items() if not passed
                    )
                    mixed_nonapproval_signals.update(
                        f"binding:{key}"
                        for key, passed in dataset_binding_checks.items()
                        if not passed
                    )
                    if replay_error is not None:
                        mixed_nonapproval_signals["gold_replay_error"] += 1
                    elif not gold_replay_pass:
                        mixed_nonapproval_signals["gold_replay_mismatch"] += 1
                    if not reward_route_pass:
                        mixed_nonapproval_signals["reward_route_mismatch"] += 1
                    if timeout_count:
                        mixed_nonapproval_signals["timeout_present"] += 1
                    if runtime_error_count:
                        mixed_nonapproval_signals["runtime_error_present"] += 1
            else:
                all_wrong_audited_buckets[audited_bucket] += 1
                if not all(source_checks.values()) or not semantic_pass or not binding_pass:
                    root_cause = "题面-SQL/gold错配"
                elif replay_error is not None or runtime_error_count:
                    root_cause = "数据库/运行环境缺陷"
                elif audited_correct_count > 0 and legacy_correct_count == 0:
                    root_cause = "奖励器假阴性"
                elif timeout_count:
                    root_cause = "模型超时错误"
                elif sql_error_count or environment_error_count:
                    root_cause = "模型工具/SQL错误"
                elif format_valid_count == 0:
                    root_cause = "模型最终答案格式错误"
                else:
                    root_cause = "真实高难"
                if sql_error_count:
                    secondary_causes.append("模型工具/SQL错误")
                if environment_error_count:
                    secondary_causes.append("模型工具/SQL错误")
                if format_valid_count < samples_per_task:
                    secondary_causes.append("模型最终答案格式错误")
                if timeout_count:
                    secondary_causes.append("模型超时错误")
                if audited_correct_count > 0 and legacy_correct_count == 0:
                    secondary_causes.append("奖励器假阴性")
                all_wrong_causes[root_cause] += 1
                answer_type_causes[str(truth.get("answer_type"))][root_cause] += 1
                all_wrong_secondary.update(set(secondary_causes))
                if (
                    audited_bucket == "mixed"
                    and root_cause == "奖励器假阴性"
                    and gold_replay_pass
                    and semantic_pass
                    and binding_pass
                    and not runtime_error_count
                ):
                    repaired = json.loads(json.dumps(dataset_row, ensure_ascii=False))
                    repaired["extra_info"]["explicit_semantic_reviewed"] = True
                    repaired["extra_info"]["training_allowed"] = False
                    repaired["extra_info"]["promotion_allowed"] = False
                    reward_repaired_mixed_dataset_rows.append(repaired)

            evidence = {
                "source_task_index": index,
                "task_id": task["task_id"],
                "instruction_sha256": dataset_row["extra_info"]["instruction_sha256"],
                "gold_sha256": dataset_row["extra_info"]["gold_sha256"],
                "instruction": task_instruction,
                "evidence_plan": task.get("evidence_plan"),
                "verification_sql": task["gold_answer"]["verification_sql"],
                "gold_answer": task["gold_answer"],
                "replay_columns": columns,
                "replay_rows": replay_rows,
                "gold_replay_passed": gold_replay_pass,
                "gold_replay_error_type": replay_error,
                "source_semantic_checks": source_checks,
                "plan_sql_checks": plan_checks,
                "dataset_binding_checks": dataset_binding_checks,
                "semantic_and_plan_passed": semantic_pass,
                "dataset_binding_passed": binding_pass,
                "original_bucket": original_bucket,
                "answer_type": truth.get("answer_type"),
                "audited_bucket": audited_bucket,
                "legacy_correct_count": legacy_correct_count,
                "audited_correct_count": audited_correct_count,
                "reward_route_passed": reward_route_pass,
                "timeout_count": timeout_count,
                "runtime_error_count": runtime_error_count,
                "sql_error_marker_count": sql_error_count,
                "environment_error_marker_count": environment_error_count,
                "format_valid_count": format_valid_count,
                "mixed_disposition": disposition,
                "all_wrong_root_cause": root_cause,
                "all_wrong_secondary_causes": secondary_causes,
                "trajectories": trajectories,
                "training_allowed": False,
            }
            detailed_rows.append(evidence)
            if original_bucket == "mixed":
                mixed_rows.append(evidence)
                if disposition == "可训练":
                    approved = json.loads(json.dumps(dataset_row, ensure_ascii=False))
                    approved["extra_info"]["explicit_semantic_reviewed"] = True
                    approved["extra_info"]["training_allowed"] = False
                    approved["extra_info"]["promotion_allowed"] = False
                    approved_dataset_rows.append(approved)
            else:
                all_wrong_rows.append(evidence)
            if disposition == "可训练" or (
                original_bucket == "all_wrong"
                and audited_bucket == "mixed"
                and root_cause == "奖励器假阴性"
            ):
                requires_corrected_table_verifier = original_bucket == "all_wrong"
                approved43 = json.loads(json.dumps(dataset_row, ensure_ascii=False))
                approved43["extra_info"]["explicit_semantic_reviewed"] = True
                approved43["extra_info"]["training_allowed"] = False
                approved43["extra_info"]["promotion_allowed"] = False
                approved43_dataset_rows.append(approved43)
                approved43_manifest_rows.append(
                    {
                        "candidate_key_sha256": canonical_hash(
                            {
                                "instruction_sha256": dataset_row["extra_info"][
                                    "instruction_sha256"
                                ],
                                "gold_sha256": dataset_row["extra_info"]["gold_sha256"],
                            }
                        ),
                        "instruction_sha256": dataset_row["extra_info"][
                            "instruction_sha256"
                        ],
                        "gold_sha256": dataset_row["extra_info"]["gold_sha256"],
                        "approval_source": (
                            "original_mixed_audit"
                            if not requires_corrected_table_verifier
                            else "reward_false_negative_repair"
                        ),
                        "answer_type": truth.get("answer_type"),
                        "audited_correct_count": audited_correct_count,
                        "gold_replay_passed": gold_replay_pass,
                        "semantic_and_plan_passed": semantic_pass,
                        "dataset_binding_passed": binding_pass,
                        "requires_corrected_table_verifier": (
                            requires_corrected_table_verifier
                        ),
                        "reward_route_contract": (
                            "verified_numeric_final_result_v1"
                            if not requires_corrected_table_verifier
                            else "corrected_full_table_ordered_v1"
                        ),
                        "training_allowed": False,
                        "promotion_allowed": False,
                    }
                )
    finally:
        connection.close()

    write_private_jsonl(private_dir / "task_audit.sensitive.jsonl", detailed_rows)
    write_private_jsonl(private_dir / "mixed_decisions.sensitive.jsonl", mixed_rows)
    write_private_jsonl(private_dir / "all_wrong_root_causes.sensitive.jsonl", all_wrong_rows)
    write_private_parquet(
        private_dir / "mixed_approved_candidates.sensitive.parquet",
        approved_dataset_rows,
        dataset_table.schema,
    )
    write_private_parquet(
        private_dir / "reward_repaired_mixed_candidates.sensitive.parquet",
        reward_repaired_mixed_dataset_rows,
        dataset_table.schema,
    )
    approved43_path = private_dir / "grpo_approved43.sensitive.parquet"
    approved43_manifest_path = private_dir / "grpo_approved43_manifest.sensitive.jsonl"
    if len(approved43_dataset_rows) != len(approved43_manifest_rows):
        raise ValueError("approved dataset and manifest row counts differ")
    if expected_approved_total is not None and len(approved43_dataset_rows) != int(
        expected_approved_total
    ):
        raise ValueError("approved candidate count does not match expected total")
    instruction_hashes = [
        str(row["extra_info"]["instruction_sha256"]) for row in approved43_dataset_rows
    ]
    candidate_keys = [row["candidate_key_sha256"] for row in approved43_manifest_rows]
    if len(set(instruction_hashes)) != len(instruction_hashes):
        raise ValueError("approved candidates do not have unique instructions")
    if len(set(candidate_keys)) != len(candidate_keys):
        raise ValueError("approved candidates do not have unique instruction/gold identities")
    completeness = [
        approved_candidate_completeness_checks(row) for row in approved43_dataset_rows
    ]
    if not all(all(checks.values()) for checks in completeness):
        raise ValueError("approved candidate prompt, SQL, gold, or hashes are incomplete")
    if not all(
        row["gold_replay_passed"]
        and row["semantic_and_plan_passed"]
        and row["dataset_binding_passed"]
        and 1 <= int(row["audited_correct_count"]) < samples_per_task
        for row in approved43_manifest_rows
    ):
        raise ValueError("approved manifest contains a candidate that failed audit gates")
    if not all(
        not bool(row["extra_info"].get("training_allowed"))
        for row in approved43_dataset_rows
    ):
        raise ValueError("approved derived copy changed training_allowed")
    write_private_parquet(approved43_path, approved43_dataset_rows, dataset_table.schema)
    write_private_jsonl(approved43_manifest_path, approved43_manifest_rows)

    approved_source_counts = Counter(
        row["approval_source"] for row in approved43_manifest_rows
    )
    approved_answer_type_counts = Counter(
        str(row["answer_type"]) for row in approved43_manifest_rows
    )
    corrected_table_rows = [
        row
        for row in approved43_manifest_rows
        if row["requires_corrected_table_verifier"]
    ]

    summary: dict[str, Any] = {
        "contract": CONTRACT,
        "source": {
            "expected_tasks": expected_tasks,
            "samples_per_task": samples_per_task,
            "trajectory_slots": len(observations),
            "complete_shards": len(shard_paths),
            "scope_tasks": len(detailed_rows),
            "original_scope_buckets": dict(sorted(scope_counts.items())),
        },
        "input_hashes": {
            "dataset_sha256": file_sha256(dataset_path),
            "tasks_sha256": file_sha256(tasks_path),
            "database_sha256": file_sha256(database_path),
            "per_task_sha256": file_sha256(per_task_path),
            "shards_set_sha256": canonical_hash(
                {path.name: file_sha256(path) for path in shard_paths}
            ),
        },
        "evidence_chain": {
            "gold_replay_passed": gold_replay_passes,
            "semantic_plan_and_binding_passed": semantic_passes,
            "reward_route_passed": reward_route_passes,
            "audited_correct_count_histogram": {
                str(key): audited_correct_histogram.get(key, 0) for key in range(9)
            },
            "legacy_to_audited_correct_count": dict(sorted(legacy_to_audited.items())),
            "audited_correct_histogram_by_scope_and_answer_type": dict(
                sorted(audited_histogram_by_scope_and_type.items())
            ),
            "format_valid_count_histogram_by_scope_and_answer_type": dict(
                sorted(format_valid_histogram_by_scope_and_type.items())
            ),
            "source_check_failure_counts": dict(sorted(source_check_failures.items())),
            "plan_check_failure_counts": dict(sorted(plan_check_failures.items())),
            "dataset_binding_failure_counts": dict(sorted(binding_check_failures.items())),
            "trajectory_signal_counts": dict(sorted(trajectory_signals.items())),
        },
        "mixed_review": {
            "reviewed": len(mixed_rows),
            "disposition_counts": dict(sorted(mixed_dispositions.items())),
            "approved_candidates": len(approved_dataset_rows),
            "nonapproval_signal_counts": dict(sorted(mixed_nonapproval_signals.items())),
            "all_eight_labels_reviewed_per_task": True,
            "original_training_allowed_modified": False,
        },
        "all_wrong_review": {
            "reviewed": len(all_wrong_rows),
            "audited_bucket_counts": dict(sorted(all_wrong_audited_buckets.items())),
            "reward_repair_conditional_mixed_candidates": len(
                reward_repaired_mixed_dataset_rows
            ),
            "primary_root_cause_counts": dict(sorted(all_wrong_causes.items())),
            "secondary_signal_counts": dict(sorted(all_wrong_secondary.items())),
            "answer_type_by_primary_root_cause": {
                answer_type: dict(sorted(counts.items()))
                for answer_type, counts in sorted(answer_type_causes.items())
            },
        },
        "private_outputs": {
            "task_audit_rows": len(detailed_rows),
            "mixed_decision_rows": len(mixed_rows),
            "all_wrong_root_cause_rows": len(all_wrong_rows),
            "approved_candidate_rows": len(approved_dataset_rows),
            "reward_repaired_mixed_candidate_rows": len(
                reward_repaired_mixed_dataset_rows
            ),
            "approved43_candidate_rows": len(approved43_dataset_rows),
            "approved43_manifest_rows": len(approved43_manifest_rows),
            "private_files_emitted": True,
            "private_paths_emitted": False,
        },
        "review_dimensions": [
            "instruction_evidence_plan_sql_alignment",
            "full_gold_sql_replay",
            "full_column_table_comparison",
            "numeric_tolerance",
            "date_group_aggregate_order_topn_percentage",
            "final_answer_format",
            "all_eight_trajectory_labels",
            "runtime_timeout_and_tool_sql_errors",
        ],
        "explicit_semantic_review_completed": True,
        "grpo_readiness": {
            "directly_audited_mixed_candidates": len(approved_dataset_rows),
            "reward_repair_conditional_mixed_candidates": len(
                reward_repaired_mixed_dataset_rows
            ),
            "total_nonzero_variance_candidates_after_reward_repair": len(
                approved_dataset_rows
            )
            + len(reward_repaired_mixed_dataset_rows),
            "conditional_candidates_blocked_until_reward_route_fixed": len(
                reward_repaired_mixed_dataset_rows
            ),
        },
        "approved43_package": {
            "rows": len(approved43_dataset_rows),
            "unique_instruction_hashes": len(set(instruction_hashes)),
            "unique_instruction_gold_identities": len(set(candidate_keys)),
            "source_counts": dict(sorted(approved_source_counts.items())),
            "answer_type_counts": dict(sorted(approved_answer_type_counts.items())),
            "complete_prompt_sql_gold_rows": sum(
                all(checks.values()) for checks in completeness
            ),
            "gold_replay_passed_rows": sum(
                bool(row["gold_replay_passed"]) for row in approved43_manifest_rows
            ),
            "corrected_table_verifier_rows": len(corrected_table_rows),
            "corrected_table_verifier_replay_passed_rows": sum(
                row["reward_route_contract"] == "corrected_full_table_ordered_v1"
                and bool(row["gold_replay_passed"])
                and 1 <= int(row["audited_correct_count"]) < samples_per_task
                for row in corrected_table_rows
            ),
            "training_allowed_false_rows": sum(
                not bool(row["extra_info"].get("training_allowed"))
                for row in approved43_dataset_rows
            ),
            "parquet_sha256": file_sha256(approved43_path),
            "manifest_sha256": file_sha256(approved43_manifest_path),
            "private_paths_emitted": False,
            "promotion_allowed": False,
        },
        "approved43_exclusions": {
            "reward_repair_all_correct": all_wrong_audited_buckets.get(
                "all_correct", 0
            ),
            "true_high_difficulty": all_wrong_causes.get("真实高难", 0),
            "model_tool_or_sql_error": all_wrong_causes.get("模型工具/SQL错误", 0),
            "timed_out_original_bucket": original_bucket_counts.get("timed_out", 0),
            "original_all_correct_outside_scope": original_bucket_counts.get(
                "all_correct", 0
            ),
        },
        "training_allowed": False,
        "promotion_allowed": False,
        "contains_prompts_sql_gold_values_task_ids_final_answers_or_tool_outputs": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_path = output_dir / "safe_summary.json"
    temporary = safe_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(safe_path)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--per-task", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=100)
    parser.add_argument("--samples-per-task", type=int, default=8)
    parser.add_argument("--expected-approved-total", type=int)
    args = parser.parse_args()
    audit(
        args.dataset,
        args.tasks,
        args.database,
        args.shards_dir,
        args.per_task,
        args.output_dir,
        expected_tasks=args.expected_tasks,
        samples_per_task=args.samples_per_task,
        expected_approved_total=args.expected_approved_total,
    )


if __name__ == "__main__":
    main()

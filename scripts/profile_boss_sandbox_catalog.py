#!/usr/bin/env python3
"""Profile every boss sandbox version and export a high-precision review pool.

The safe summary contains only aggregate counts.  The optional candidate JSONL
contains task text, hidden gold, and verification SQL and must remain in a
permission-restricted location until a deliberate review/export decision.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_formal_instruction_gold_alignment import classify


DENIED_SQLITE_ACTIONS = {
    getattr(sqlite3, name)
    for name in (
        "SQLITE_ATTACH",
        "SQLITE_DETACH",
        "SQLITE_INSERT",
        "SQLITE_UPDATE",
        "SQLITE_DELETE",
        "SQLITE_ALTER_TABLE",
        "SQLITE_CREATE_INDEX",
        "SQLITE_CREATE_TABLE",
        "SQLITE_CREATE_TEMP_INDEX",
        "SQLITE_CREATE_TEMP_TABLE",
        "SQLITE_CREATE_TEMP_TRIGGER",
        "SQLITE_CREATE_TEMP_VIEW",
        "SQLITE_CREATE_TRIGGER",
        "SQLITE_CREATE_VIEW",
        "SQLITE_CREATE_VTABLE",
        "SQLITE_DROP_INDEX",
        "SQLITE_DROP_TABLE",
        "SQLITE_DROP_TEMP_INDEX",
        "SQLITE_DROP_TEMP_TABLE",
        "SQLITE_DROP_TEMP_TRIGGER",
        "SQLITE_DROP_TEMP_VIEW",
        "SQLITE_DROP_TRIGGER",
        "SQLITE_DROP_VIEW",
        "SQLITE_DROP_VTABLE",
        "SQLITE_REINDEX",
        "SQLITE_ANALYZE",
        "SQLITE_TRANSACTION",
        "SQLITE_SAVEPOINT",
    )
    if hasattr(sqlite3, name)
}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            yield row


def _readonly_authorizer(action: int, _arg1: str, _arg2: str, _db: str, _trigger: str) -> int:
    return sqlite3.SQLITE_DENY if action in DENIED_SQLITE_ACTIONS else sqlite3.SQLITE_OK


def open_readonly_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve(strict=True).as_posix()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only = ON")
    connection.set_authorizer(_readonly_authorizer)
    return connection


def execute_query(connection: sqlite3.Connection, sql: str, max_rows: int) -> list[tuple[Any, ...]]:
    cursor = connection.execute(sql)
    rows = cursor.fetchmany(max_rows + 1)
    if len(rows) > max_rows:
        raise ValueError(f"verification result exceeds {max_rows} rows")
    return rows


def _contains_value(
    rows: list[tuple[Any, ...]],
    value: Any,
    abs_tol: float = 1e-3,
    rel_tol: float = 1e-5,
) -> bool:
    for row in rows:
        for actual in row:
            if isinstance(value, (int, float)) and isinstance(actual, (int, float)):
                if math.isclose(float(actual), float(value), abs_tol=abs_tol, rel_tol=rel_tol):
                    return True
            elif value is not None and str(value).strip().casefold() == str(actual).strip().casefold():
                return True
    return False


def gold_supported_by_rows(gold: dict[str, Any], rows: list[tuple[Any, ...]]) -> bool:
    answer_type = str(gold.get("answer_type") or "")
    value = gold.get("value")
    if answer_type == "numeric":
        return isinstance(value, (int, float)) and _contains_value(rows, value)
    if answer_type != "table" or not isinstance(value, list) or not value:
        return False
    for item in value:
        if not isinstance(item, dict) or not _contains_value(rows, item.get("value")):
            return False
        label = item.get("category", item.get("date"))
        if label is not None and not _contains_value(rows, label, 0.0, 0.0):
            return False
    return True


def _counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _safe_name(value: Any) -> str:
    rendered = str(value or "missing").strip()
    return rendered or "missing"


def profile_catalog(
    sandbox_root: Path,
    *,
    execute_sql: bool,
    max_result_rows: int = 10_000,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.monotonic()
    versions = sorted(path for path in sandbox_root.iterdir() if path.is_dir())
    if not versions:
        raise ValueError(f"no sandbox versions found under {sandbox_root}")

    global_task_ids: defaultdict[str, set[str]] = defaultdict(set)
    global_instruction_versions: defaultdict[str, set[str]] = defaultdict(set)
    instruction_occurrences: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    version_summaries: list[dict[str, Any]] = []
    total_rows = 0

    for version_path in versions:
        version = version_path.name
        manifest = version_path / "dwh_tasks.jsonl"
        database = version_path / "logistics.sqlite"
        if not manifest.is_file():
            version_summaries.append(
                {"version": version, "rows": 0, "manifest_present": False, "database_present": database.is_file()}
            )
            continue

        connection = open_readonly_database(database) if execute_sql and database.is_file() else None
        task_types: Counter[str] = Counter()
        task_categories: Counter[str] = Counter()
        answer_types: Counter[str] = Counter()
        qa_statuses: Counter[str] = Counter()
        difficulty_levels: Counter[str] = Counter()
        warning_counts: Counter[str] = Counter()
        exclusion_counts: Counter[str] = Counter()
        local_ids: Counter[str] = Counter()
        rows = 0
        answerable = validation_checked = validation_result_exists = 0
        sql_present = sql_executable = sql_nonempty = gold_match = 0
        high_precision = warning_rows = 0

        try:
            for row in read_jsonl(manifest):
                rows += 1
                total_rows += 1
                task_id = _safe_name(row.get("task_id"))
                instruction = str(row.get("natural_language_instruction") or "").strip()
                instruction_hash = canonical_hash(instruction) if instruction else ""
                gold = row.get("gold_answer") if isinstance(row.get("gold_answer"), dict) else {}
                answer_type = _safe_name(gold.get("answer_type"))
                sql = str(gold.get("verification_sql") or "").strip()
                answerability = row.get("answerability_label") or {}
                validation = row.get("validation") or {}
                task_category = _safe_name(row.get("task_category"))
                qa_status = _safe_name(row.get("_qa_status"))

                local_ids[task_id] += 1
                global_task_ids[task_id].add(version)
                if instruction_hash:
                    global_instruction_versions[instruction_hash].add(version)
                    instruction_occurrences[instruction_hash] += 1
                task_types[_safe_name(row.get("task_type"))] += 1
                task_categories[task_category] += 1
                answer_types[answer_type] += 1
                qa_statuses[qa_status] += 1
                difficulty_levels[_safe_name(row.get("difficulty_level", row.get("difficulty")))] += 1

                is_answerable = answerability.get("is_answerable") is True
                checked = validation.get("checked_against_database") is True
                result_exists = validation.get("expected_result_exists") is True
                answerable += is_answerable
                validation_checked += checked
                validation_result_exists += result_exists
                sql_present += bool(sql)

                issues = classify({"instruction": instruction, "gold_answer": gold})
                warning_counts.update(issues)
                warning_rows += bool(issues)
                query_rows: list[tuple[Any, ...]] = []
                executable = nonempty = matches = False
                if sql and connection is not None:
                    try:
                        query_rows = execute_query(connection, sql, max_result_rows)
                        executable = True
                        nonempty = bool(query_rows)
                        matches = nonempty and gold_supported_by_rows(gold, query_rows)
                    except Exception as exc:
                        exclusion_counts[f"sql_error:{type(exc).__name__}"] += 1
                sql_executable += executable
                sql_nonempty += nonempty
                gold_match += matches

                reasons: list[str] = []
                if task_id == "missing" or not instruction:
                    reasons.append("missing_identity_or_instruction")
                if answer_type not in {"numeric", "table"}:
                    reasons.append("unsupported_answer_type")
                if task_category != "answerable":
                    reasons.append("not_answerable_task_category")
                if not is_answerable:
                    reasons.append("not_marked_answerable")
                if qa_status != "passed":
                    reasons.append("qa_not_passed")
                if not checked or not result_exists:
                    reasons.append("source_validation_not_complete")
                if not sql:
                    reasons.append("missing_verification_sql")
                if execute_sql and not executable:
                    reasons.append("verification_sql_not_executable")
                if execute_sql and not nonempty:
                    reasons.append("verification_sql_empty")
                if execute_sql and not matches:
                    reasons.append("gold_result_mismatch")
                if issues:
                    reasons.append("semantic_review_flag")
                exclusion_counts.update(reasons)

                if not reasons:
                    high_precision += 1
                    exported_gold = {
                        "answer_type": answer_type,
                        "value": gold.get("value"),
                        "verification_sql": sql,
                    }
                    candidates.append(
                        {
                            "version": version,
                            "environment_id": f"sft/{version}",
                            "task_id": task_id,
                            "instruction": instruction,
                            "instruction_sha256": instruction_hash,
                            "gold": exported_gold,
                            # Hash exactly what leaves the profiler.  Source
                            # manifests may carry additional gold metadata
                            # that is intentionally not exported.
                            "gold_sha256": canonical_hash(exported_gold),
                            "task_category": task_category,
                            "task_type": _safe_name(row.get("task_type")),
                            "scenario_type": _safe_name(row.get("scenario_type")),
                            "business_domain": _safe_name(row.get("business_domain")),
                            "difficulty_level": _safe_name(
                                row.get("difficulty_level", row.get("difficulty"))
                            ),
                            "expected_tables": sorted(
                                {_safe_name(value).casefold() for value in row.get("expected_tables") or []}
                            ),
                            "semantic_review_flags": [],
                            "mechanical_sql_verified": execute_sql,
                        }
                    )
        finally:
            if connection is not None:
                connection.close()

        version_summaries.append(
            {
                "version": version,
                "rows": rows,
                "manifest_present": True,
                "database_present": database.is_file(),
                "duplicate_task_id_rows": sum(count - 1 for count in local_ids.values() if count > 1),
                "task_types": _counter(task_types),
                "task_categories": _counter(task_categories),
                "answer_types": _counter(answer_types),
                "qa_statuses": _counter(qa_statuses),
                "difficulty_levels": _counter(difficulty_levels),
                "answerable_rows": answerable,
                "source_validation_checked_rows": validation_checked,
                "source_validation_result_exists_rows": validation_result_exists,
                "verification_sql_present_rows": sql_present,
                "verification_sql_executable_rows": sql_executable if execute_sql else None,
                "verification_sql_nonempty_rows": sql_nonempty if execute_sql else None,
                "gold_result_match_rows": gold_match if execute_sql else None,
                "semantic_review_flag_rows": warning_rows,
                "semantic_review_warning_counts": _counter(warning_counts),
                "high_precision_candidate_rows": high_precision,
                "exclusion_counts": _counter(exclusion_counts),
            }
        )

    for candidate in candidates:
        digest = candidate["instruction_sha256"]
        candidate["instruction_occurrences_in_catalog"] = instruction_occurrences[digest]
        candidate["instruction_version_count"] = len(global_instruction_versions[digest])
        candidate["globally_unique_instruction"] = instruction_occurrences[digest] == 1

    summary = {
        "contract": "boss-multi-sandbox-catalog-profile-v1",
        "sandbox_versions": len(versions),
        "rows": total_rows,
        "sql_execution_enabled": execute_sql,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "high_precision_candidate_rows": len(candidates),
        "high_precision_unique_instruction_rows": sum(
            candidate["globally_unique_instruction"] for candidate in candidates
        ),
        "cross_version_task_id_duplicate_groups": sum(
            len(version_names) > 1 for version_names in global_task_ids.values()
        ),
        "cross_version_instruction_duplicate_groups": sum(
            len(version_names) > 1 for version_names in global_instruction_versions.values()
        ),
        "versions": version_summaries,
        "candidate_definition": (
            "DWH answerable-category numeric/table task with QA passed, source-marked answerable and "
            "database-checked, verification SQL present, read-only executable and nonempty, hidden gold "
            "supported by result, and no deterministic semantic review flags; candidates still require "
            "explicit semantic adjudication before rollout"
        ),
        "contains_prompts_sql_answers_task_ids_tool_outputs_or_server_paths": False,
        "training_allowed": False,
        "rollout_allowed": False,
    }
    return summary, candidates


def write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.chmod(path, 0o600)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sandbox-root", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path)
    parser.add_argument("--execute-sql", action="store_true")
    parser.add_argument("--max-result-rows", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, candidates = profile_catalog(
        args.sandbox_root,
        execute_sql=args.execute_sql,
        max_result_rows=args.max_result_rows,
    )
    write_private(
        args.summary_output,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    if args.candidate_output:
        write_private(
            args.candidate_output,
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in candidates),
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Classify the first generated SQL by read-only execution and gold support."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterable
from urllib.parse import quote

import pandas as pd

from scripts.analyze_repair_sft_free_run_divergence import (
    bash_calls,
    normalize_container,
    read_openai,
)
from scripts.prepare_pi_formal_dataset import gold_supported_by_rows
from scripts.prepare_repair_sft_dataset import READ_ONLY_SQL_RE


MODEL_LABELS = ("step120", "post_sft")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def first_readonly_sql(messages: list[dict[str, Any]]) -> str | None:
    """Return the first SELECT/WITH payload from the rollout's bash calls."""

    return next((call["sql"] for call in bash_calls(messages) if call["sql"]), None)


def _canonical_scalar(value: Any) -> tuple[str, Any]:
    if value is None:
        return ("null", None)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isnan(numeric):
            return ("number", "nan")
        if math.isinf(numeric):
            return ("number", "inf" if numeric > 0 else "-inf")
        return ("number", 0.0 if numeric == 0 else numeric)
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    return ("text", str(value))


def canonical_row_multiset(rows: Iterable[Iterable[Any]]) -> list[tuple[tuple[str, Any], ...]]:
    canonical = [tuple(_canonical_scalar(value) for value in row) for row in rows]
    return sorted(canonical, key=repr)


def results_equivalent(
    left_columns: list[str],
    left_rows: list[tuple],
    right_columns: list[str],
    right_rows: list[tuple],
) -> dict[str, bool]:
    row_values_equal = canonical_row_multiset(left_rows) == canonical_row_multiset(right_rows)
    normalized_left = [column.strip().casefold() for column in left_columns]
    normalized_right = [column.strip().casefold() for column in right_columns]
    return {
        "row_value_multiset_equal": row_values_equal,
        "column_and_row_multiset_equal": row_values_equal and normalized_left == normalized_right,
    }


def execute_readonly_sql(
    database: Path,
    sql: str,
    *,
    max_rows: int = 10_000,
    timeout_seconds: float = 30.0,
) -> tuple[list[str], list[tuple]]:
    """Execute one bounded SELECT/WITH statement against an immutable SQLite DB."""

    if not READ_ONLY_SQL_RE.match(sql or ""):
        raise ValueError("only SELECT/WITH SQL is allowed")
    if max_rows <= 0 or timeout_seconds <= 0:
        raise ValueError("max_rows and timeout_seconds must be positive")
    resolved = database.resolve(strict=True)
    uri = f"file:{quote(resolved.as_posix(), safe='/')}?mode=ro&immutable=1"
    deadline = time.monotonic() + timeout_seconds
    connection = sqlite3.connect(uri, uri=True, timeout=min(timeout_seconds, 5.0))
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 10_000)
        cursor = connection.execute(sql)
        columns = [str(item[0]) for item in cursor.description or []]
        rows = cursor.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            raise OverflowError("query evidence exceeds max_rows")
        return columns, rows
    finally:
        connection.close()


def ground_truth_by_task(replay_parquet: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    frame = pd.read_parquet(replay_parquet)
    task_order: list[str] = []
    truths: dict[str, dict[str, Any]] = {}
    for _, series in frame.iterrows():
        row = normalize_container(series.to_dict())
        truth = (row.get("reward_model") or {}).get("ground_truth") or {}
        task_id = str(truth.get("task_id") or "")
        if not task_id or task_id in truths:
            raise ValueError(f"missing or duplicate replay task_id: {task_id!r}")
        expected = json.loads(str(truth.get("expected_value_json") or "null"))
        verification_sql = str(truth.get("verification_sql") or "")
        if not READ_ONLY_SQL_RE.match(verification_sql):
            raise ValueError(f"{task_id}: verification SQL is not read-only")
        task_order.append(task_id)
        truths[task_id] = {
            "answer_type": str(truth.get("answer_type") or ""),
            "expected": expected,
            "verification_sql": verification_sql,
        }
    return task_order, truths


def classify_first_query(
    *,
    database: Path,
    messages: list[dict[str, Any]],
    truth: dict[str, Any],
    max_rows: int = 10_000,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    generated_sql = first_readonly_sql(messages)
    if generated_sql is None:
        return {
            "category": "no_readonly_query",
            "query_sha256": None,
            "executable": False,
            "nonempty": False,
            "gold_supported": False,
            "teacher_result_equivalent": False,
            "error_class": None,
        }

    base = {
        "query_sha256": sha256_text(generated_sql),
        "executable": False,
        "nonempty": False,
        "gold_supported": False,
        "teacher_result_equivalent": False,
        "error_class": None,
    }
    try:
        columns, rows = execute_readonly_sql(
            database,
            generated_sql,
            max_rows=max_rows,
            timeout_seconds=timeout_seconds,
        )
    except (sqlite3.Error, ValueError, OverflowError) as error:
        return {
            **base,
            "category": "schema_syntax_or_execution_error",
            "error_class": type(error).__name__,
        }

    teacher_columns, teacher_rows = execute_readonly_sql(
        database,
        truth["verification_sql"],
        max_rows=max_rows,
        timeout_seconds=timeout_seconds,
    )
    equivalence = results_equivalent(columns, rows, teacher_columns, teacher_rows)
    supported = bool(rows) and gold_supported_by_rows(
        {"answer_type": truth["answer_type"], "value": truth["expected"]}, rows
    )
    base.update(
        {
            "executable": True,
            "nonempty": bool(rows),
            "gold_supported": supported,
            "teacher_result_equivalent": equivalence["row_value_multiset_equal"],
            **equivalence,
            "row_count": len(rows),
            "column_count": len(columns),
        }
    )
    if supported:
        category = "verified_gold_support"
    elif not rows:
        category = "executable_empty_evidence"
    else:
        category = "executable_wrong_or_insufficient_evidence"
    return {**base, "category": category}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total <= 0:
        raise ValueError("semantic audit requires at least one task")
    categories = Counter(row["category"] for row in rows)
    verified = sum(bool(row["gold_supported"]) for row in rows)
    equivalent = sum(bool(row["teacher_result_equivalent"]) for row in rows)
    return {
        "task_count": total,
        "category_counts": dict(sorted(categories.items())),
        "verified_gold_support_count": verified,
        "verified_gold_support_rate": verified / total,
        "teacher_result_equivalent_count": equivalent,
        "teacher_result_equivalent_rate": equivalent / total,
        "executable_count": sum(bool(row["executable"]) for row in rows),
        "nonempty_count": sum(bool(row["nonempty"]) for row in rows),
    }


def training_target_decision(post_summary: dict[str, Any]) -> dict[str, Any]:
    total = int(post_summary["task_count"])
    verified = int(post_summary["verified_gold_support_count"])
    if verified * 2 >= total:
        target = "post_evidence_synthesis_recovery_and_stopping"
        reason = "at_least_half_of_first_queries_already_support_gold"
    else:
        target = "sql_grounding_and_semantics"
        reason = "fewer_than_half_of_first_queries_support_gold"
    return {
        "selected_training_target": target,
        "reason": reason,
        "token_rank_gate_still_required": True,
        "training_must_not_start_from_this_cpu_gate_alone": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-parquet", type=Path, required=True)
    parser.add_argument("--step120-openai", type=Path, required=True)
    parser.add_argument("--post-sft-openai", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=10_000)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()

    task_order, truths = ground_truth_by_task(args.replay_parquet)
    rollouts = {
        "step120": read_openai(args.step120_openai),
        "post_sft": read_openai(args.post_sft_openai),
    }
    expected_ids = set(task_order)
    for label, messages_by_task in rollouts.items():
        if set(messages_by_task) != expected_ids:
            raise ValueError(f"{label}: rollout and replay task IDs differ")

    models: dict[str, Any] = {}
    for label in MODEL_LABELS:
        per_task = [
            {
                "task_id": task_id,
                **classify_first_query(
                    database=args.database,
                    messages=rollouts[label][task_id],
                    truth=truths[task_id],
                    max_rows=args.max_rows,
                    timeout_seconds=args.timeout_seconds,
                ),
            }
            for task_id in task_order
        ]
        models[label] = {"summary": summarize(per_task), "per_task": per_task}

    result = {
        "contract": "repair-sft-first-query-semantic-gate-v1",
        "task_count": len(task_order),
        "task_ids_identical": True,
        "execution": {
            "read_only": True,
            "immutable_database": True,
            "model_or_optimizer_loaded": False,
            "npu_required": False,
            "max_rows": args.max_rows,
            "timeout_seconds": args.timeout_seconds,
        },
        "models": models,
        "decision": training_target_decision(models["post_sft"]["summary"]),
        "promotion_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "models": {label: data["summary"] for label, data in models.items()},
                "decision": result["decision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

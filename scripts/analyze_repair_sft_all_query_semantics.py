#!/usr/bin/env python3
"""Audit every generated SQL query for bounded semantic recovery evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from scripts.analyze_repair_sft_first_query_semantics import (
    classify_first_query,
    ground_truth_by_task,
)
from scripts.analyze_repair_sft_free_run_divergence import bash_calls, read_openai


def _single_call_messages(call: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call["call_id"],
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": {"command": call["command"]},
                    },
                }
            ],
        }
    ]


def classify_query_sequence(
    *,
    database: Path,
    messages: list[dict[str, Any]],
    truth: dict[str, Any],
    max_rows: int = 10_000,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Classify each read-only SQL and locate the first verified recovery."""

    calls = [call for call in bash_calls(messages) if call["sql"]]
    queries: list[dict[str, Any]] = []
    for index, call in enumerate(calls, start=1):
        result = classify_first_query(
            database=database,
            messages=_single_call_messages(call),
            truth=truth,
            max_rows=max_rows,
            timeout_seconds=timeout_seconds,
        )
        queries.append(
            {
                "query_index": index,
                **result,
                "verified_or_equivalent": bool(
                    result["gold_supported"] or result["teacher_result_equivalent"]
                ),
            }
        )

    first_verified = next(
        (row["query_index"] for row in queries if row["verified_or_equivalent"]),
        None,
    )
    return {
        "query_count": len(queries),
        "first_verified_or_equivalent_query_index": first_verified,
        "verified_or_equivalent_within_1": first_verified == 1,
        "verified_or_equivalent_within_2": first_verified is not None and first_verified <= 2,
        "verified_or_equivalent_within_3": first_verified is not None and first_verified <= 3,
        "verified_or_equivalent_anywhere": first_verified is not None,
        "queries": queries,
    }


def summarize_query_sequences(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("all-query semantic audit requires at least one task")
    categories = Counter(
        query["category"] for row in rows for query in row["queries"]
    )
    return {
        "task_count": len(rows),
        "query_count": sum(row["query_count"] for row in rows),
        "tasks_within_1": sum(row["verified_or_equivalent_within_1"] for row in rows),
        "tasks_within_2": sum(row["verified_or_equivalent_within_2"] for row in rows),
        "tasks_within_3": sum(row["verified_or_equivalent_within_3"] for row in rows),
        "tasks_anywhere": sum(row["verified_or_equivalent_anywhere"] for row in rows),
        "query_category_counts": dict(sorted(categories.items())),
    }


def parse_rollout(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("rollout must be LABEL=PATH")
    return label.strip(), Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-parquet", type=Path, required=True)
    parser.add_argument("--rollout", type=parse_rollout, action="append", required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=10_000)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()

    task_order, truths = ground_truth_by_task(args.replay_parquet)
    expected_ids = set(task_order)
    labels = [label for label, _ in args.rollout]
    if len(labels) != len(set(labels)):
        raise ValueError("rollout labels must be unique")

    models: dict[str, Any] = {}
    for label, path in args.rollout:
        messages_by_task = read_openai(path)
        if set(messages_by_task) != expected_ids:
            raise ValueError(f"{label}: rollout and replay task IDs differ")
        per_task = [
            {
                "task_id": task_id,
                **classify_query_sequence(
                    database=args.database,
                    messages=messages_by_task[task_id],
                    truth=truths[task_id],
                    max_rows=args.max_rows,
                    timeout_seconds=args.timeout_seconds,
                ),
            }
            for task_id in task_order
        ]
        models[label] = {
            "summary": summarize_query_sequences(per_task),
            "per_task": per_task,
        }

    result = {
        "contract": "repair-sft-all-query-semantic-baseline-v1",
        "task_count": len(task_order),
        "task_ids_identical": True,
        "execution": {
            "read_only": True,
            "immutable_database": True,
            "npu_required": False,
            "max_rows": args.max_rows,
            "timeout_seconds": args.timeout_seconds,
        },
        "models": models,
        "promotion_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {label: data["summary"] for label, data in models.items()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

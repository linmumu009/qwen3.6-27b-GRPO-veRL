#!/usr/bin/env python3
"""Fail-closed audit for generated open multi-sandbox DWH datasets."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any, Sequence


TECHNICAL_RE = re.compile(
    r"SQL|SQLite|数据库|数据仓库|表名|字段名|category|value|SELECT|JOIN|GROUP\s+BY|HAVING|CTE|窗口函数",
    re.IGNORECASE,
)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _tables_match(
    actual: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    *,
    numeric_abs_tol: float,
) -> tuple[bool, float]:
    if len(actual) != len(expected):
        return False, float("inf")
    maximum = 0.0
    for left, right in zip(actual, expected, strict=True):
        if str(left.get("category")) != str(right.get("category")):
            return False, float("inf")
        try:
            difference = abs(float(left.get("value")) - float(right.get("value")))
        except (TypeError, ValueError):
            return False, float("inf")
        maximum = max(maximum, difference)
        if not math.isclose(
            float(left["value"]),
            float(right["value"]),
            rel_tol=0.0,
            abs_tol=numeric_abs_tol,
        ):
            return False, maximum
    return True, maximum


def audit_sandbox(path: Path, *, numeric_abs_tol: float = 0.0) -> dict[str, Any]:
    if numeric_abs_tol < 0:
        raise ValueError("numeric_abs_tol must be non-negative")
    tasks = read_jsonl(path / "dwh_tasks.jsonl")
    if len(tasks) != 500:
        raise ValueError(f"{path}: expected 500 tasks, got {len(tasks)}")
    task_ids = [str(task["task_id"]) for task in tasks]
    instructions = [str(task["natural_language_instruction"]) for task in tasks]
    sqls = [str(task["gold_answer"]["verification_sql"]) for task in tasks]
    if len(set(task_ids)) != 500 or len(set(instructions)) != 500 or len(set(sqls)) != 500:
        raise ValueError(f"{path}: task ID, instruction, and SQL must each be unique")
    levels = Counter(int(task["difficulty_level"]) for task in tasks)
    if levels != Counter({level: 100 for level in range(1, 6)}):
        raise ValueError(f"{path}: invalid level distribution: {levels}")

    database = path / "logistics.sqlite"
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    result_hashes: list[str] = []
    api_rows = 0
    result_rows = Counter()
    runtime_numeric_drift_tasks = 0
    runtime_numeric_max_abs_diff = 0.0
    try:
        for task in tasks:
            if task.get("training_allowed") is not False:
                raise ValueError(f"{path}: generated task unexpectedly enabled for training")
            instruction = str(task["natural_language_instruction"])
            technical = TECHNICAL_RE.search(instruction)
            if technical:
                raise ValueError(f"{path}: technical term leaked into instruction")
            missing = [str(anchor) for anchor in task["semantic_anchors"] if str(anchor) not in instruction]
            if missing:
                raise ValueError(f"{path}: semantic anchors missing from instruction")
            if "高到低" not in instruction or not re.search(r"前\s*5", instruction):
                raise ValueError(f"{path}: output ordering/top-five contract missing")
            sql = str(task["gold_answer"]["verification_sql"])
            rows = [dict(row) for row in connection.execute(sql).fetchall()]
            normalized = [{"category": str(row["category"]), "value": row["value"]} for row in rows]
            expected = task["gold_answer"]["value"]
            exact = normalized == expected
            matched, maximum = _tables_match(
                normalized,
                expected,
                numeric_abs_tol=numeric_abs_tol,
            )
            if not matched:
                raise ValueError(f"{path}: replayed SQL does not equal hidden gold")
            if not exact:
                runtime_numeric_drift_tasks += 1
                runtime_numeric_max_abs_diff = max(runtime_numeric_max_abs_diff, maximum)
            result_hash = canonical_hash(normalized)
            if exact and result_hash != task["validation"]["result_sha256"]:
                raise ValueError(f"{path}: result hash mismatch")
            result_hashes.append(result_hash)
            result_rows[len(rows)] += 1
            if (task.get("instruction_generation") or {}).get("semantic_validation_passed") is True:
                api_rows += 1
    finally:
        connection.close()

    by_level_features: dict[str, dict[str, float]] = {}
    for level in range(1, 6):
        subset = [task for task in tasks if int(task["difficulty_level"]) == level]
        keys = ("essential_joins", "evidence_steps", "derived_metrics", "temporal_comparisons", "business_openness")
        by_level_features[str(level)] = {
            key: round(
                sum(float(task["query_plan"]["feature_counts"][key]) for task in subset) / len(subset),
                3,
            )
            for key in keys
        }
    openness = [by_level_features[str(level)]["business_openness"] for level in range(1, 6)]
    evidence = [by_level_features[str(level)]["evidence_steps"] for level in range(1, 6)]
    if openness != sorted(openness) or evidence != sorted(evidence):
        raise ValueError(f"{path}: difficulty features are not monotonic")

    return {
        "sandbox": path.name,
        "task_count": len(tasks),
        "level_distribution": dict(sorted(levels.items())),
        "family_distribution": dict(sorted(Counter(task["task_type"] for task in tasks).items())),
        "role_count": len(Counter(task["instruction_role"] for task in tasks)),
        "api_naturalized_rows": api_rows,
        "sql_gold_replay_passed_rows": len(tasks),
        "semantic_anchor_passed_rows": len(tasks),
        "unique_result_tables": len(set(result_hashes)),
        "runtime_numeric_abs_tolerance": numeric_abs_tol,
        "runtime_numeric_drift_tasks": runtime_numeric_drift_tasks,
        "runtime_numeric_max_abs_diff": runtime_numeric_max_abs_diff,
        "result_row_distribution": dict(sorted(result_rows.items())),
        "mean_features_by_level": by_level_features,
        "training_allowed": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sandboxes", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summaries = [audit_sandbox(path) for path in args.sandboxes]
    payload = {
        "contract": "llin-open-multisandbox-dwh-audit-v1",
        "sandbox_count": len(summaries),
        "task_count": sum(item["task_count"] for item in summaries),
        "sql_gold_replay_passed_rows": sum(item["sql_gold_replay_passed_rows"] for item in summaries),
        "semantic_anchor_passed_rows": sum(item["semantic_anchor_passed_rows"] for item in summaries),
        "api_naturalized_rows": sum(item["api_naturalized_rows"] for item in summaries),
        "training_allowed": False,
        "sandboxes": summaries,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

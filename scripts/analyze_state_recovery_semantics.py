#!/usr/bin/env python3
"""Classify first-error SQL differences and semantic critical-token families."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from scripts.analyze_repair_sft_free_run_divergence import normalize_container, sql_from_command


AGGREGATES = frozenset({"AVG", "COUNT", "GROUP_CONCAT", "MAX", "MIN", "SUM", "TOTAL"})
CLAUSE_KEYWORDS = frozenset(
    {"FROM", "GROUP", "HAVING", "JOIN", "LIMIT", "ORDER", "SELECT", "WHERE"}
)
TEMPORAL_TERMS = re.compile(
    r"\b(?:date|datetime|day|latest|month|quarter|strftime|time|timestamp|week|year)\b",
    re.IGNORECASE,
)
IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_.$]*"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_sql(value: str) -> str:
    return " ".join(value.strip().rstrip(";").split()).casefold()


def clause(sql: str, start: str, following: tuple[str, ...]) -> str:
    normalized = " ".join(sql.strip().rstrip(";").split())
    match = re.search(rf"\b{start}\b", normalized, re.IGNORECASE)
    if match is None:
        return ""
    end = len(normalized)
    for keyword in following:
        candidate = re.search(rf"\b{keyword}\b", normalized[match.end() :], re.IGNORECASE)
        if candidate is not None:
            end = min(end, match.end() + candidate.start())
    return normalize_sql(normalized[match.end() : end])


def sql_signature(sql: str) -> dict[str, Any]:
    tables = sorted(
        {
            match.group(1).casefold()
            for match in re.finditer(rf"\b(?:FROM|JOIN)\s+({IDENTIFIER})", sql, re.IGNORECASE)
        }
    )
    aggregates = sorted(
        {
            match.group(1).upper()
            for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", sql)
            if match.group(1).upper() in AGGREGATES
        }
    )
    return {
        "tables": tables,
        "aggregates": aggregates,
        "join_count": len(re.findall(r"\bJOIN\b", sql, re.IGNORECASE)),
        "select": clause(sql, "SELECT", ("FROM",)),
        "where": clause(sql, "WHERE", ("GROUP BY", "HAVING", "ORDER BY", "LIMIT")),
        "group_by": clause(sql, "GROUP BY", ("HAVING", "ORDER BY", "LIMIT")),
        "having": clause(sql, "HAVING", ("ORDER BY", "LIMIT")),
        "order_by": clause(sql, "ORDER BY", ("LIMIT",)),
        "limit": clause(sql, "LIMIT", ()),
        "temporal_terms": sorted(set(match.group(0).casefold() for match in TEMPORAL_TERMS.finditer(sql))),
    }


def semantic_difference(error_sql: str, correction_sql: str) -> dict[str, Any]:
    before = sql_signature(error_sql)
    after = sql_signature(correction_sql)
    labels: list[str] = []
    if before["tables"] != after["tables"]:
        labels.append("table_grounding")
    if before["join_count"] != after["join_count"]:
        labels.append("join_structure")
    if before["temporal_terms"] != after["temporal_terms"]:
        labels.append("temporal_semantics")
    if (
        before["aggregates"] != after["aggregates"]
        or before["group_by"] != after["group_by"]
        or before["having"] != after["having"]
    ):
        labels.append("aggregation_grouping")
    if before["where"] != after["where"]:
        labels.append("filter_semantics")
    if before["select"] != after["select"]:
        labels.append("select_expression")
    if before["order_by"] != after["order_by"] or before["limit"] != after["limit"]:
        labels.append("ordering_limit")
    if not labels and normalize_sql(error_sql) != normalize_sql(correction_sql):
        labels.append("other_lexical_or_nested")
    if not labels:
        raise ValueError("error and correction SQL are semantically indistinguishable by the audit")
    return {
        "labels": labels,
        "primary": labels[0],
        "table_set_changed": before["tables"] != after["tables"],
        "aggregate_set_changed": before["aggregates"] != after["aggregates"],
        "filter_changed": before["where"] != after["where"],
        "select_changed": before["select"] != after["select"],
    }


def critical_token_family(token: str | None) -> str:
    normalized = str(token or "").replace("Ġ", "").replace("▁", "").strip().upper()
    if normalized in AGGREGATES:
        return "aggregation_function"
    if normalized == "SELECT":
        return "query_start"
    if normalized in CLAUSE_KEYWORDS:
        return "clause_keyword"
    if not normalized or all(not character.isalnum() for character in normalized):
        return "punctuation_or_operator"
    return "identifier_or_literal"


def diagnostic_by_task(path: Path) -> dict[str, dict[str, Any]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("contract") != "repair-sft-teacher-forced-component-diagnostic-v3":
        raise ValueError("semantic recovery audit requires diagnostic contract v3")
    return {str(row["task_id"]): row for row in result["per_task"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-parquet", type=Path, required=True)
    parser.add_argument("--step120-diagnostic", type=Path, required=True)
    parser.add_argument("--post-diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = diagnostic_by_task(args.step120_diagnostic)
    post = diagnostic_by_task(args.post_diagnostic)
    if set(baseline) != set(post):
        raise ValueError("pre/post semantic diagnostic task IDs differ")

    frame = pd.read_parquet(args.state_parquet)
    rows: list[dict[str, Any]] = []
    for _, series in frame.iterrows():
        source = normalize_container(series.to_dict())
        task_id = str(source.get("task_id") or "")
        if task_id not in baseline:
            raise ValueError(f"state row is absent from diagnostics: {task_id!r}")
        messages = source["messages"]
        error_command = messages[2]["tool_calls"][0]["function"]["arguments"]["command"]
        correction_command = messages[4]["tool_calls"][0]["function"]["arguments"]["command"]
        error_sql = sql_from_command(error_command)
        correction_sql = sql_from_command(correction_command)
        if error_sql is None or correction_sql is None:
            raise ValueError(f"{task_id}: missing error or correction SQL")
        difference = semantic_difference(error_sql, correction_sql)
        before_rank = baseline[task_id]["sql_token_rank"]
        after_rank = post[task_id]["sql_token_rank"]
        family = critical_token_family(before_rank["first_nongreedy_target_token"])
        rows.append(
            {
                "task_id": task_id,
                "error_query_sha256": sha256_text(error_sql),
                "correction_query_sha256": sha256_text(correction_sql),
                "semantic_difference": difference,
                "critical_token": {
                    "family": family,
                    "offset": before_rank["first_nongreedy_offset"],
                    "rank_step120": before_rank["first_nongreedy_rank"],
                    "probability_step120": before_rank["first_nongreedy_target_probability"],
                    "rank_state_conditioned_step1": after_rank["first_nongreedy_rank"],
                    "probability_state_conditioned_step1": after_rank[
                        "first_nongreedy_target_probability"
                    ],
                },
            }
        )

    label_counts = Counter(label for row in rows for label in row["semantic_difference"]["labels"])
    primary_counts = Counter(row["semantic_difference"]["primary"] for row in rows)
    critical_counts = Counter(row["critical_token"]["family"] for row in rows)
    cross_tab = Counter(
        (row["semantic_difference"]["primary"], row["critical_token"]["family"])
        for row in rows
    )
    result = {
        "contract": "repair-sft-state-recovery-semantic-audit-v1",
        "task_count": len(rows),
        "parser_policy": "bounded lexical SQL clause signatures; no raw SQL emitted",
        "semantic_label_counts": dict(sorted(label_counts.items())),
        "primary_semantic_label_counts": dict(sorted(primary_counts.items())),
        "critical_token_family_counts": dict(sorted(critical_counts.items())),
        "primary_by_critical_family": [
            {"primary_semantic_label": key[0], "critical_token_family": key[1], "count": count}
            for key, count in sorted(cross_tab.items())
        ],
        "per_task": rows,
        "npu_required": False,
        "promotion_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "per_task"}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Heuristic audit for semantic alignment between PI instructions and SQL golds."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
from pathlib import Path
from typing import Any, Iterable


BROAD_ANALYSIS_RE = re.compile(
    r"原因|问题|建议|优化|改进|措施|策略|分析报告|分析结论|趋势|波动|表现|情况|看法|诊断"
)
EXPLICIT_BUSINESS_METRIC_RE = re.compile(
    r"多少票运单|运单数量|多少位不同客户|不同客户数量|"
    r"运费(?:总额|合计|一共|平均)|平均每票运费|"
    r"货物(?:总重量|合计有多少千克)|平均配送(?:时长|用了多少小时)|延误率"
)
TEMPORAL_SQL_RE = re.compile(
    r"\b(?:date|time|year|month|day|period|latest|current)\b|日期|时间|月份|年度",
    re.IGNORECASE,
)
AGGREGATE_SQL_RE = re.compile(r"\b(?:sum|count|avg|min|max)\s*\(", re.IGNORECASE)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def classify(record: dict[str, Any]) -> list[str]:
    instruction = str(record.get("instruction") or "")
    gold = record.get("gold") or record.get("gold_answer") or {}
    sql = str(gold.get("verification_sql") or "")
    answer_type = str(gold.get("answer_type") or "")
    issues: list[str] = []

    broad = bool(BROAD_ANALYSIS_RE.search(instruction)) and not bool(
        EXPLICIT_BUSINESS_METRIC_RE.search(instruction)
    )
    latest = bool(re.search(r"最新|最近|这一期|本期|当前|202[0-9][年\-/]", instruction))
    if broad and answer_type in {"numeric", "table"}:
        issues.append("broad_instruction_exact_hidden_target")
    if latest and not TEMPORAL_SQL_RE.search(sql):
        issues.append("latest_instruction_without_temporal_sql")
    if re.search(r"\blimit\s+\d+", sql, re.IGNORECASE) and not re.search(
        r"\border\s+by\b", sql, re.IGNORECASE
    ):
        issues.append("limit_without_order_by")
    if answer_type == "numeric" and not AGGREGATE_SQL_RE.search(sql):
        issues.append("numeric_gold_without_aggregation")
    if broad and re.search(r"\bcount\s*\(\s*\*\s*\)", sql, re.IGNORECASE):
        issues.append("broad_instruction_reduced_to_row_count")
    return issues


def audit(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    issue_counts: Counter[str] = Counter()
    records_with_any_issue = 0
    examples: dict[str, list[dict[str, str]]] = {}
    answer_types: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    for row in rows:
        gold = row.get("gold") or row.get("gold_answer") or {}
        answer_types[str(gold.get("answer_type") or "missing")] += 1
        splits[str(row.get("split") or "missing")] += 1
        issues = classify(row)
        if issues:
            records_with_any_issue += 1
        for issue in issues:
            issue_counts[issue] += 1
            examples.setdefault(issue, [])
            if len(examples[issue]) < 3:
                examples[issue].append(
                    {
                        "verifier_id": str(row.get("verifier_id") or row.get("task_id") or ""),
                        "instruction": str(row.get("instruction") or ""),
                        "verification_sql": str(gold.get("verification_sql") or ""),
                    }
                )
    return {
        "records": len(rows),
        "answer_types": dict(answer_types),
        "splits": dict(splits),
        "records_with_any_flag": records_with_any_issue,
        "records_with_any_flag_rate": records_with_any_issue / len(rows) if rows else None,
        "issue_counts": dict(issue_counts),
        "examples": examples,
        "interpretation": (
            "Flags are deterministic review triggers, not automatic proof that a task is invalid. "
            "Each flagged task needs human or source-trajectory confirmation before training."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-flags", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(read_jsonl(args.manifest))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.fail_on_flags and result["records_with_any_flag"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

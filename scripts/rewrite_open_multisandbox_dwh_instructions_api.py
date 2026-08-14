#!/usr/bin/env python3
"""Naturalize open multi-sandbox DWH tasks through the private chat API.

Only role, draft instruction, task family, and explicit semantic anchors are
sent.  SQL, gold values, database rows, database paths, and credentials are
never included in the request payload or logs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

import scripts.rewrite_plan_first_dwh_instructions_api as base


CONTRACT = "llin-open-multisandbox-dwh-api-natural-v1"
INSTRUCTION_STYLE = "boss_open_api_mixed_company_roles_natural_language_v1"
TECHNICAL_RE = re.compile(
    r"SQL|SQLite|数据库|数据仓库|表名|字段名|category|value|SELECT|JOIN|GROUP\s+BY|HAVING|CTE|窗口函数",
    re.IGNORECASE,
)


def semantic_payload(task: dict[str, Any]) -> dict[str, Any]:
    contract = task["semantic_contract"]
    payload: dict[str, Any] = {
        "task_id": str(task["task_id"]),
        "target_speaker": base.ROLE_LABELS[str(task["instruction_role"])],
        "business_task_family": str(contract["family"]),
        "draft": str(task["natural_language_instruction"]),
        "must_preserve_verbatim": list(task["semantic_anchors"]),
        "verified_deliverable": "列出前5项的业务名称和对应结果，按结果从高到低排列",
    }
    if bool(contract.get("explanation_is_open_ended")):
        payload["open_part"] = (
            "可以自然地请使用者简要解释判断，但不能增加新的硬性统计口径；"
            "解释没有隐藏唯一答案，核验对象只有明确要求的前5项和综合分。"
        )
    feedback = task.get("_rewrite_feedback")
    if feedback:
        payload["previous_attempt_rejected_because"] = list(feedback)
        payload["required_correction"] = "逐字补回缺失口径并删除技术词，不得改动业务含义。"
    return payload


def build_prompt(tasks: Sequence[dict[str, Any]]) -> str:
    payload = [semantic_payload(task) for task in tasks]
    return """请把下面的企业内部数据需求改写成自然中文，像真实员工临时向数据助手提问。

硬性要求：
1. must_preserve_verbatim 中每一项都必须原样出现在问题里，不能换数字、日期、权重、样本门槛、指标或业务分类。
2. 必须保留“从高到低”和“前5项”的结果要求；不得新增筛选条件或另一套计算规则。
3. 不要出现 SQL、SQLite、数据库、数据仓库、表名、字段名、category、value、SELECT、JOIN、GROUP BY、HAVING、CTE、窗口函数。
4. 不要给答案，不要解释技术计算过程。允许用1到4句话，语气跟 target_speaker 匹配；不同题不要套同一句开头。
5. Level 5 的 open_part 可以要求简要说明判断，但不得暗示存在未提供的唯一行动答案。
6. 每个 task_id 只返回一条 instruction；只输出严格JSON数组，不要Markdown。

返回格式：[{"task_id":"...","instruction":"..."}]

待改写数据：
""" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def validate_rewrite(task: dict[str, Any], instruction: str) -> list[str]:
    reasons: list[str] = []
    if not 24 <= len(instruction) <= 500:
        reasons.append("length_out_of_range")
    match = TECHNICAL_RE.search(instruction)
    if match:
        reasons.append(f"technical_term:{match.group(0)}")
    for anchor in task["semantic_anchors"]:
        if str(anchor) not in instruction:
            reasons.append(f"anchor_missing:{anchor}")
    if "高到低" not in instruction:
        reasons.append("descending_order_missing")
    if not re.search(r"前\s*5\s*(?:项|名|个|类|组)?", instruction):
        reasons.append("top_five_missing")
    if int(task["difficulty_level"]) == 5 and "综合" not in instruction:
        reasons.append("composite_metric_missing")
    return reasons


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sandbox", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--api-config", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--resume-incomplete", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 1 <= args.batch_size <= 12:
        raise ValueError("batch-size must be between 1 and 12")
    if not 1 <= args.max_workers <= 32:
        raise ValueError("max-workers must be between 1 and 32")
    # The proven client retains its credential hygiene, retry policy, partial
    # resume, and atomic output behavior; only semantic rendering differs.
    base.build_prompt = build_prompt
    base.validate_rewrite = validate_rewrite
    output = base.rewrite_sandbox(
        args.source_sandbox,
        args.output_root,
        args.api_config,
        version=args.version,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        resume_incomplete=args.resume_incomplete,
        limit=args.limit,
        contract=CONTRACT,
        instruction_style=INSTRUCTION_STYLE,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

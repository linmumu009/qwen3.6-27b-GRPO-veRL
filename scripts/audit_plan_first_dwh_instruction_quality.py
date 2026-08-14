#!/usr/bin/env python3
"""Audit natural-language quality for API-rewritten plan-first DWH tasks."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import statistics
from pathlib import Path
from typing import Any, Sequence

try:
    from scripts.rewrite_plan_first_dwh_instructions_api import (
        ROLE_LABELS,
        TECHNICAL_INSTRUCTION_RE,
        validate_rewrite,
    )
except ModuleNotFoundError:  # Direct execution: python scripts/<name>.py
    from rewrite_plan_first_dwh_instructions_api import (  # type: ignore[no-redef]
        ROLE_LABELS,
        TECHNICAL_INSTRUCTION_RE,
        validate_rewrite,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _opening(instruction: str) -> str:
    first = re.split(r"[。！？；]", instruction, maxsplit=1)[0]
    normalized = re.sub(r"2025\s*年\s*\d{1,2}\s*月", "<月份>", first)
    normalized = re.sub(r"“[^”]+”", "<业务值>", normalized)
    normalized = re.sub(r"\d+", "<数字>", normalized)
    return re.sub(r"\s+", "", normalized)


def audit_tasks(tasks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    instructions = [str(task.get("natural_language_instruction") or "") for task in tasks]
    lengths = [len(value) for value in instructions]
    semantic_failures = {
        str(task.get("task_id")): validate_rewrite(task, instruction)
        for task, instruction in zip(tasks, instructions, strict=True)
    }
    semantic_failures = {key: value for key, value in semantic_failures.items() if value}
    technical_rows = sum(bool(TECHNICAL_INSTRUCTION_RE.search(value)) for value in instructions)
    role_counts = Counter(str(task.get("instruction_role") or "missing") for task in tasks)
    opening_counts = Counter(_opening(value) for value in instructions)
    duplicate_rows = len(instructions) - len(set(instructions))
    api_rows = sum(
        str((task.get("instruction_generation") or {}).get("method"))
        == "boss_openai_compatible_chat_api"
        for task in tasks
    )
    conversational_rows = sum(bool(re.search(r"我|帮我|请|麻烦|想|需要", value)) for value in instructions)
    question_or_request_rows = sum(bool(re.search(r"[？?]|请|帮我|麻烦|给我|列出|告诉", value)) for value in instructions)
    expected_roles = set(ROLE_LABELS)
    gates = {
        "exactly_300_tasks": len(tasks) == 300,
        "all_api_generated": api_rows == len(tasks),
        "all_semantics_preserved": not semantic_failures,
        "no_technical_instruction_terms": technical_rows == 0,
        "no_exact_duplicates": duplicate_rows == 0,
        "all_roles_present": expected_roles <= set(role_counts),
        "no_role_below_10": bool(role_counts) and min(role_counts.values()) >= 10,
        "conversational_language_at_least_95pct": conversational_rows >= 0.95 * len(tasks),
        "question_or_request_at_least_95pct": question_or_request_rows >= 0.95 * len(tasks),
        "no_single_opening_above_5pct": bool(opening_counts)
        and max(opening_counts.values()) <= max(1, int(0.05 * len(tasks))),
        "instruction_length_between_24_and_360": bool(lengths)
        and min(lengths) >= 24
        and max(lengths) <= 360,
    }
    return {
        "contract": "llin-plan-first-dwh-instruction-quality-v1",
        "task_count": len(tasks),
        "api_generated_rows": api_rows,
        "role_counts": dict(sorted(role_counts.items())),
        "unique_instruction_rows": len(set(instructions)),
        "duplicate_instruction_rows": duplicate_rows,
        "technical_term_rows": technical_rows,
        "semantic_failure_rows": len(semantic_failures),
        "conversational_rows": conversational_rows,
        "question_or_request_rows": question_or_request_rows,
        "unique_normalized_openings": len(opening_counts),
        "maximum_repeated_normalized_opening": max(opening_counts.values(), default=0),
        "instruction_length": {
            "minimum": min(lengths, default=0),
            "median": statistics.median(lengths) if lengths else 0,
            "maximum": max(lengths, default=0),
        },
        "gates": gates,
        "quality_gate_passed": all(gates.values()),
        "safe_report_contains_prompts_sql_gold_task_ids_or_api_credentials": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-gate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = audit_tasks(read_jsonl(args.tasks))
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.fail_on_gate and not result["quality_gate_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

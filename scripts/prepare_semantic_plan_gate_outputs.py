#!/usr/bin/env python3
"""Join one-turn veRL validation outputs back to semantic-plan gate IDs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from scripts.prepare_boss_exact_evaluation import qwen_output_to_openai_messages, read_jsonl, write_jsonl
from scripts.prepare_semantic_plan_sufficiency_gate import ARMS


def prepare(validation: Path, output: Path) -> dict[str, Any]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    tool_calls = 0
    bash_tool_calls = 0
    call_name_counts: Counter[str] = Counter()
    call_count_histogram: Counter[int] = Counter()
    exact_one_bash_rows = 0
    max_tool_calls_per_row = 0
    for index, row in enumerate(read_jsonl(validation), 1):
        truth = row.get("gts") or {}
        gate_id = str(truth.get("semantic_plan_gate_id") or "")
        arm = str(truth.get("semantic_plan_gate_arm") or "")
        task_id = str(truth.get("semantic_plan_gate_source_task_id") or "")
        if gate_id != f"{task_id}::{arm}" or arm not in ARMS or gate_id in seen:
            raise ValueError(f"validation row {index}: invalid or duplicate gate identity {gate_id!r}")
        messages, audit = qwen_output_to_openai_messages(str(row.get("output") or ""))
        if audit["tool_responses"] != 0:
            raise ValueError(f"{gate_id}: one-turn gate unexpectedly contains a new tool response")
        names = [
            str(call.get("function", {}).get("name") or "")
            for message in messages
            if message.get("role") == "assistant"
            for call in message.get("tool_calls") or []
        ]
        row_tool_calls = int(audit["tool_calls"])
        row_bash_calls = sum(name == "bash" for name in names)
        exact_one_bash = row_tool_calls == 1 and names == ["bash"]
        result.append(
            {
                "gate_id": gate_id,
                "task_id": task_id,
                "arm": arm,
                "messages": messages,
                "output_protocol": {
                    "tool_call_count": row_tool_calls,
                    "bash_tool_call_count": row_bash_calls,
                    "tool_call_names": names,
                    "exactly_one_bash_call": exact_one_bash,
                },
            }
        )
        seen.add(gate_id)
        tool_calls += row_tool_calls
        bash_tool_calls += row_bash_calls
        call_name_counts.update(names)
        call_count_histogram[row_tool_calls] += 1
        exact_one_bash_rows += int(exact_one_bash)
        max_tool_calls_per_row = max(max_tool_calls_per_row, row_tool_calls)
    if len(result) != 48:
        raise ValueError(f"one-turn gate requires 48 validation rows, got {len(result)}")
    write_jsonl(output, result)
    return {
        "contract": "semantic-plan-sufficiency-gate-output-adapter-v2",
        "rows": len(result),
        "unique_gate_ids": len(seen),
        "generated_tool_calls": tool_calls,
        "generated_bash_tool_calls": bash_tool_calls,
        "tool_call_name_counts": dict(sorted(call_name_counts.items())),
        "tool_call_count_histogram": {
            str(key): value for key, value in sorted(call_count_histogram.items())
        },
        "rows_with_exactly_one_bash_call": exact_one_bash_rows,
        "rows_with_multiple_tool_calls": sum(
            count for calls, count in call_count_histogram.items() if calls > 1
        ),
        "max_tool_calls_per_row": max_tool_calls_per_row,
        "generated_tool_responses": 0,
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    summary = prepare(args.validation, args.output)
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

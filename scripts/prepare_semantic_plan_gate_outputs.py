#!/usr/bin/env python3
"""Join one-turn veRL validation outputs back to semantic-plan gate IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.prepare_boss_exact_evaluation import qwen_output_to_openai_messages, read_jsonl, write_jsonl
from scripts.prepare_semantic_plan_sufficiency_gate import ARMS


def prepare(validation: Path, output: Path) -> dict[str, Any]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    tool_calls = 0
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
        if audit["tool_calls"] > 1:
            raise ValueError(f"{gate_id}: one-turn gate contains parallel/multiple tool calls")
        result.append({"gate_id": gate_id, "task_id": task_id, "arm": arm, "messages": messages})
        seen.add(gate_id)
        tool_calls += int(audit["tool_calls"])
    if len(result) != 48:
        raise ValueError(f"one-turn gate requires 48 validation rows, got {len(result)}")
    write_jsonl(output, result)
    return {
        "contract": "semantic-plan-sufficiency-gate-output-adapter-v1",
        "rows": len(result),
        "unique_gate_ids": len(seen),
        "generated_tool_calls": tool_calls,
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

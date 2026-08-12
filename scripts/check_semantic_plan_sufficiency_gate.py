#!/usr/bin/env python3
"""CPU-only fail-closed audit for the semantic-plan gate dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from scripts.analyze_repair_sft_free_run_divergence import normalize_container, sql_from_command
from scripts.prepare_repair_sft_dataset import READ_ONLY_SQL_RE
from scripts.prepare_semantic_plan_sufficiency_gate import ARMS, HINT_PREFIX, sha256_value


FORBIDDEN_PLAN_KEYS = frozenset({"sql", "query", "result", "answer", "literal", "expected"})
FULL_ONLY_KEYS = frozenset({"tables", "columns_by_role", "join_count", "equality_join_edges"})


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).casefold() in FORBIDDEN_PLAN_KEYS or _contains_forbidden_key(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(child) for child in value)
    return False


def _contains_key(value: Any, targets: frozenset[str]) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).casefold() in targets or _contains_key(child, targets)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_key(child, targets) for child in value)
    return False


def _normalized_prompt(row: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = normalize_container(row.get("prompt"))
    if not isinstance(prompt, list):
        raise ValueError("gate row prompt is not a list")
    return prompt


def check_dataset(data_file: Path, contract_file: Path) -> dict[str, Any]:
    contract = json.loads(contract_file.read_text(encoding="utf-8"))
    if contract.get("contract") != "semantic-plan-sufficiency-gate-dataset-v1":
        raise ValueError("semantic-plan checker requires dataset contract v1")
    frame = pd.read_parquet(data_file)
    rows = [normalize_container(row) for row in frame.to_dict(orient="records")]
    if len(rows) != 48 or contract.get("rows") != 48:
        raise ValueError("semantic-plan gate must contain exactly 48 rows")
    if hashlib.sha256(data_file.read_bytes()).hexdigest() != contract.get("output_sha256"):
        raise ValueError("semantic-plan gate parquet hash differs from contract")

    evidence_by_gate = {str(row["gate_id"]): row for row in contract.get("evidence") or []}
    if len(evidence_by_gate) != 48:
        raise ValueError("semantic-plan contract evidence must contain 48 unique gate IDs")
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    arm_counts: Counter[str] = Counter()
    aggregation_tasks: set[str] = set()
    for row in rows:
        ground_truth = ((row.get("reward_model") or {}).get("ground_truth") or {})
        gate_id = str(ground_truth.get("semantic_plan_gate_id") or "")
        arm = str(ground_truth.get("semantic_plan_gate_arm") or "")
        task_id = str(ground_truth.get("semantic_plan_gate_source_task_id") or "")
        if gate_id != row.get("semantic_plan_gate_id") or gate_id not in evidence_by_gate:
            raise ValueError(f"row has missing or inconsistent gate ID: {gate_id!r}")
        if gate_id != f"{task_id}::{arm}" or arm not in ARMS:
            raise ValueError(f"invalid semantic-plan gate identity: {gate_id!r}")
        if row.get("semantic_plan_gate_source_task_id") != task_id:
            raise ValueError(f"{gate_id}: top-level source task differs")
        if row.get("data_source") != "semantic_plan_sufficiency_gate":
            raise ValueError(f"{gate_id}: wrong data_source")
        if ((row.get("extra_info") or {}).get("tool_selection") or []) != ["bash"]:
            raise ValueError(f"{gate_id}: generated tool surface is not bash-only")
        prompt = _normalized_prompt(row)
        if [message.get("role") for message in prompt] != [
            "system", "user", "assistant", "tool", "user"
        ]:
            raise ValueError(f"{gate_id}: prompt is not the frozen first-error state plus hint")
        error_calls = prompt[2].get("tool_calls") or []
        if len(error_calls) != 1 or error_calls[0]["function"].get("name") != "bash":
            raise ValueError(f"{gate_id}: first-error state is not one bash call")
        error_sql = sql_from_command(error_calls[0]["function"]["arguments"]["command"])
        if error_sql is None or not READ_ONLY_SQL_RE.match(error_sql):
            raise ValueError(f"{gate_id}: first-error SQL is missing or not read-only")
        if not str(prompt[3].get("content") or ""):
            raise ValueError(f"{gate_id}: observed first-error tool result is empty")
        content = str(prompt[4].get("content") or "")
        if not content.startswith(HINT_PREFIX):
            raise ValueError(f"{gate_id}: missing frozen gate hint prefix")
        payload = json.loads(content.removeprefix(HINT_PREFIX))
        plan = payload.get("semantic_plan")
        if _contains_forbidden_key(plan):
            raise ValueError(f"{gate_id}: plan contains a forbidden answer/query field")
        if arm == "control" and plan is not None:
            raise ValueError(f"{gate_id}: control arm contains a plan")
        if arm == "operator_oracle":
            if not isinstance(plan, dict) or plan.get("scope") != "operator_only":
                raise ValueError(f"{gate_id}: invalid operator-only plan")
            if _contains_key(plan, FULL_ONLY_KEYS):
                raise ValueError(f"{gate_id}: operator plan leaks grounding fields")
        if arm == "full_plan_oracle":
            if not isinstance(plan, dict) or plan.get("scope") != "full_semantic_plan":
                raise ValueError(f"{gate_id}: invalid full semantic plan")
            grounding = plan.get("grounding") or {}
            if not grounding.get("tables") or not grounding.get("columns_by_role"):
                raise ValueError(f"{gate_id}: full plan lacks table/column grounding")
        correction_hash = str(evidence_by_gate[gate_id].get("correction_query_sha256") or "")
        if sha256_value(error_sql) == correction_hash:
            raise ValueError(f"{gate_id}: first-error SQL unexpectedly equals the correction")
        if evidence_by_gate[gate_id].get("base_prompt_sha256") != sha256_value(prompt[:4]):
            raise ValueError(f"{gate_id}: base prompt differs from dataset contract")
        if evidence_by_gate[gate_id].get("hint_sha256") != sha256_value(prompt[4]):
            raise ValueError(f"{gate_id}: hint differs from dataset contract")
        if ground_truth.get("semantic_plan_gate_aggregation_critical"):
            aggregation_tasks.add(task_id)
        arm_counts[arm] += 1
        by_task[task_id].append(row)

    if len(by_task) != 16 or arm_counts != Counter({arm: 16 for arm in ARMS}):
        raise ValueError(f"wrong task/arm balance: tasks={len(by_task)}, arms={dict(arm_counts)}")
    if len(aggregation_tasks) != 9:
        raise ValueError(f"expected 9 aggregation-critical tasks, got {len(aggregation_tasks)}")
    for task_id, task_rows in by_task.items():
        prompts = [_normalized_prompt(row) for row in task_rows]
        if len({sha256_value(prompt[:4]) for prompt in prompts}) != 1:
            raise ValueError(f"{task_id}: arm base prompts differ")
        if {str(row["semantic_plan_gate_arm"]) for row in task_rows} != set(ARMS):
            raise ValueError(f"{task_id}: missing gate arm")

    return {
        "contract": "semantic-plan-sufficiency-gate-cpu-audit-v1",
        "rows": len(rows),
        "tasks": len(by_task),
        "rows_per_arm": dict(sorted(arm_counts.items())),
        "aggregation_critical_tasks": len(aggregation_tasks),
        "all_arms_share_identical_first_error_state": True,
        "all_generated_tool_surfaces_bash_only": True,
        "all_first_error_queries_read_only": True,
        "all_first_error_results_observed": True,
        "all_control_plans_empty": True,
        "all_operator_plans_exclude_grounding": True,
        "all_full_plans_include_grounding": True,
        "all_plans_exclude_sql_result_answer_and_literals": True,
        "npu_required": False,
        "promotion_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check_dataset(args.data_file, args.contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

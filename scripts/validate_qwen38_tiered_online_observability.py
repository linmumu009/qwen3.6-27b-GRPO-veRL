#!/usr/bin/env python3
"""CPU/container gate for bound DB, identity and exact tool-token observability."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from llin_verl.pi_reward import execute_readonly_sql
from llin_verl.pi_workspace_tools import count_tool_response_tokens
from llin_verl.tiered_query_cost_reward import compute_tiered_query_cost_reward
from llin_verl.trajectory_process_reward import _expected_value


def _tool_environment_ids(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "environment_id" and item:
                result.add(str(item))
            else:
                result.update(_tool_environment_ids(item))
    elif isinstance(value, list):
        for item in value:
            result.update(_tool_environment_ids(item))
    return result


def validate(dataset: Path, database_root: Path, tokenizer_path: Path) -> dict[str, Any]:
    rows = pq.read_table(dataset).to_pylist()
    states: Counter[str] = Counter()
    answer_types: Counter[str] = Counter()
    token_counts: list[int] = []
    identity_failures = 0
    database_failures = 0
    missing_token_unknown = 0
    trajectory_hashes: set[str] = set()

    for index, row in enumerate(rows):
        truth = ((row.get("reward_model") or {}).get("ground_truth") or {})
        extra = row.get("extra_info") or {}
        environment = str(truth.get("environment_id") or "")
        answer_type = str(truth.get("answer_type") or "")
        answer_types[answer_type] += 1
        if str(extra.get("environment_id") or environment) != environment:
            identity_failures += 1
        tool_environments = _tool_environment_ids(extra.get("tools_kwargs"))
        if tool_environments and tool_environments != {environment}:
            identity_failures += 1
        try:
            database = (database_root / environment / "logistics.sqlite").resolve(strict=True)
            database.relative_to(database_root.resolve(strict=True))
            rows_value = execute_readonly_sql(database, str(truth.get("verification_sql") or ""))
        except (OSError, ValueError):
            database_failures += 1
            continue
        response = json.dumps(rows_value, ensure_ascii=False, default=str)
        token_count = count_tool_response_tokens(response, str(tokenizer_path))
        token_counts.append(token_count)
        request_id = hashlib.sha256(f"cpu-gate:{index}".encode("ascii")).hexdigest()
        instruction = str(extra.get("instruction_sha256") or "")
        event = {
            "name": "bash",
            "arguments": {"command": "sqlite3 /workspace/logistics.sqlite '<bound-readonly-sql>'"},
            "sql_statements": [str(truth.get("verification_sql") or "")],
            "ok": True,
            "response_preview": response,
            "response_token_count": token_count,
            "response_token_count_method": "tokenizer_json_content_tokens_v1",
            "response_token_count_error": "",
            "observed_tool_response": True,
            "call_parse_valid": True,
            "source": "runtime_structured_pi_workspace",
            "command_origin": "model",
            "workspace_request_id": request_id,
            "environment_id": environment,
        }
        reward_extra = {
            "instruction_sha256": instruction,
            "pi_tool_events": [event],
            "pi_tool_log_present": True,
            "pi_tool_protocol_complete": True,
            "pi_reward_database_root": str(database_root),
            "trajectory_timeout": False,
            "runtime_error": False,
            "request_id": request_id,
            "pi_trajectory_request_id": request_id,
            "pi_trajectory_environment_id": environment,
            "pi_environment_id": environment,
            "pi_workspace_request_id": request_id,
            "pi_workspace_released": True,
        }
        expected = _expected_value(truth)
        answer = (
            f"Final result: {expected}"
            if answer_type == "numeric"
            else "Final result:\n" + json.dumps(expected, ensure_ascii=False)
        )
        result = compute_tiered_query_cost_reward("dwh", answer, truth, reward_extra)
        states[str(result["judge_state"])] += 1
        trajectory_hashes.add(str(result["trajectory_identity_sha256"] or ""))
        missing = json.loads(json.dumps(event))
        missing["response_token_count"] = None
        missing_result = compute_tiered_query_cost_reward(
            "dwh", answer, truth, {**reward_extra, "pi_tool_events": [missing]}
        )
        missing_token_unknown += (
            missing_result["judge_state"] == "UNKNOWN"
            and missing_result["judge_reason"] == "tool_response_cost_unobservable"
        )

    return {
        "schema_version": "qwen38-tiered-online-observability-cpu-gate-v1",
        "rows": len(rows),
        "answer_type_counts": dict(sorted(answer_types.items())),
        "database_available": len(rows) - database_failures,
        "database_failures": database_failures,
        "identity_binding_failures": identity_failures,
        "exact_token_count_observed": len(token_counts),
        "token_count_min": min(token_counts) if token_counts else None,
        "token_count_max": max(token_counts) if token_counts else None,
        "missing_token_unknown": missing_token_unknown,
        "synthetic_bound_reward_states": dict(sorted(states.items())),
        "unique_trajectory_identity_hashes": len({value for value in trajectory_hashes if value}),
        "sensitive_fields_emitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.dataset, args.database_root, args.tokenizer_path)
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.chmod(0o600)
    print(json.dumps(report, sort_keys=True))
    expected = int(report["rows"])
    if not (
        report["database_available"] == expected
        and report["identity_binding_failures"] == 0
        and report["exact_token_count_observed"] == expected
        and report["missing_token_unknown"] == expected
        and report["synthetic_bound_reward_states"] == {"PASS": expected}
        and report["unique_trajectory_identity_hashes"] == expected
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

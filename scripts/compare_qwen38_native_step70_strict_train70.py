#!/usr/bin/env python3
"""Compare native and Step70 strict-mixed sets without exposing task identity."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


HOST_CONTRACT = "llin-qwen38-native-step70-strict-train70-host-v1"
AGGREGATE_CONTRACT = "llin-qwen38-native-step70-strict-train70-comparison-v1"
TRANSITIONS = ("retained", "lost", "gained", "neither")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _identity(row: dict[str, Any]) -> str:
    return str((row.get("extra_info") or {}).get("instruction_sha256") or "").strip()


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = _identity(row)
        if not identity or identity in indexed:
            raise ValueError(f"{label} identities are missing or duplicated")
        indexed[identity] = row
    return indexed


def _source(row: dict[str, Any]) -> str:
    return str((row.get("extra_info") or {}).get("source_version") or "unknown")


def _difficulty(row: dict[str, Any]) -> str:
    return str((row.get("extra_info") or {}).get("difficulty_level") or "unknown")


def compare_host(
    approved_path: Path,
    native_qualified_path: Path,
    step70_qualified_path: Path,
    output_path: Path,
    *,
    expected_approved: int,
    host_label: str,
) -> dict[str, Any]:
    approved = _index(pq.read_table(approved_path).to_pylist(), "approved")
    native = _index(pq.read_table(native_qualified_path).to_pylist(), "native")
    step70 = _index(pq.read_table(step70_qualified_path).to_pylist(), "step70")
    universe = set(approved)
    native_ids = set(native)
    step70_ids = set(step70)
    if len(universe) != expected_approved:
        raise ValueError(
            f"expected {expected_approved} approved tasks, observed {len(universe)}"
        )
    if not native_ids <= universe or not step70_ids <= universe:
        raise ValueError("strict-qualified identities must be approved tasks")

    transition_ids = {
        "retained": native_ids & step70_ids,
        "lost": native_ids - step70_ids,
        "gained": step70_ids - native_ids,
        "neither": universe - (native_ids | step70_ids),
    }
    by_source: dict[str, dict[str, int]] = {}
    by_difficulty: dict[str, dict[str, int]] = {}
    for transition, identities in transition_ids.items():
        by_source[transition] = dict(
            sorted(Counter(_source(approved[item]) for item in identities).items())
        )
        by_difficulty[transition] = dict(
            sorted(Counter(_difficulty(approved[item]) for item in identities).items())
        )

    result = {
        "contract": HOST_CONTRACT,
        "date": date.today().isoformat(),
        "host_label": host_label,
        "approved_tasks": len(universe),
        "native_strict_mixed_tasks": len(native_ids),
        "step70_strict_mixed_tasks": len(step70_ids),
        **{f"{name}_tasks": len(values) for name, values in transition_ids.items()},
        "transition_by_source_version": by_source,
        "transition_by_difficulty": by_difficulty,
        "checks": {
            "approved_count_exact": len(universe) == expected_approved,
            "native_is_approved_subset": native_ids <= universe,
            "step70_is_approved_subset": step70_ids <= universe,
            "transition_partition_exact": sum(map(len, transition_ids.values()))
            == len(universe),
            "native_partition_exact": len(native_ids)
            == len(transition_ids["retained"]) + len(transition_ids["lost"]),
            "step70_partition_exact": len(step70_ids)
            == len(transition_ids["retained"]) + len(transition_ids["gained"]),
        },
        "training_allowed": False,
        "promotion_allowed": False,
        "contains_prompts_gold_sql_task_ids_hashes_final_answers_tool_outputs_or_server_paths": False,
    }
    if not all(result["checks"].values()):
        raise ValueError("strict transition checks failed")
    _write_json(output_path, result)
    return result


def _sum_nested(
    payloads: list[dict[str, Any]], field: str
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for transition in TRANSITIONS:
        counts: Counter[str] = Counter()
        for payload in payloads:
            counts.update(
                {
                    str(key): int(value)
                    for key, value in payload[field][transition].items()
                }
            )
        result[transition] = dict(sorted(counts.items()))
    return result


def aggregate_safe(
    input_paths: list[Path],
    output_path: Path,
    *,
    expected_tasks: int = 70,
    expected_hosts: int = 3,
) -> dict[str, Any]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in input_paths]
    if len(payloads) != expected_hosts:
        raise ValueError(f"expected {expected_hosts} host summaries")
    if any(payload.get("contract") != HOST_CONTRACT for payload in payloads):
        raise ValueError("host comparison contract mismatch")
    hosts = [str(payload.get("host_label") or "") for payload in payloads]
    if not all(hosts) or len(set(hosts)) != len(hosts):
        raise ValueError("host labels are missing or duplicated")

    fields = (
        "approved_tasks",
        "native_strict_mixed_tasks",
        "step70_strict_mixed_tasks",
        "retained_tasks",
        "lost_tasks",
        "gained_tasks",
        "neither_tasks",
    )
    totals = {
        field: sum(int(payload[field]) for payload in payloads) for field in fields
    }
    checks = {
        "task_count_exact": totals["approved_tasks"] == expected_tasks,
        "native_partition_exact": totals["native_strict_mixed_tasks"]
        == totals["retained_tasks"] + totals["lost_tasks"],
        "step70_partition_exact": totals["step70_strict_mixed_tasks"]
        == totals["retained_tasks"] + totals["gained_tasks"],
        "transition_partition_exact": totals["approved_tasks"]
        == sum(totals[f"{name}_tasks"] for name in TRANSITIONS),
        "all_host_checks_pass": all(
            all(bool(value) for value in payload["checks"].values())
            for payload in payloads
        ),
    }
    result = {
        "contract": AGGREGATE_CONTRACT,
        "stage": "complete",
        "date": date.today().isoformat(),
        "native_model_label": "qwen38-27b-native-hf",
        "native_policy_step": 0,
        "step70_model_label": "qwen38-27b-grpo-step70",
        "step70_policy_step": 70,
        "reward_contract": "banded-v2-strict-table-v1",
        **totals,
        "native_to_step70_retention_rate": (
            totals["retained_tasks"] / totals["native_strict_mixed_tasks"]
            if totals["native_strict_mixed_tasks"]
            else 0.0
        ),
        "step70_strict_mixed_change": totals["step70_strict_mixed_tasks"]
        - totals["native_strict_mixed_tasks"],
        "transition_by_source_version": _sum_nested(
            payloads, "transition_by_source_version"
        ),
        "transition_by_difficulty": _sum_nested(
            payloads, "transition_by_difficulty"
        ),
        "per_host": {
            str(payload["host_label"]): {
                field: int(payload[field]) for field in fields
            }
            for payload in sorted(payloads, key=lambda item: str(item["host_label"]))
        },
        "checks": checks,
        "training_allowed": False,
        "promotion_allowed": False,
        "contains_prompts_gold_sql_task_ids_hashes_final_answers_tool_outputs_or_server_paths": False,
    }
    if not all(checks.values()):
        raise ValueError("aggregate strict transition checks failed")
    _write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--approved", type=Path, required=True)
    compare_parser.add_argument("--native-qualified", type=Path, required=True)
    compare_parser.add_argument("--step70-qualified", type=Path, required=True)
    compare_parser.add_argument("--output-safe-json", type=Path, required=True)
    compare_parser.add_argument("--expected-approved", type=int, required=True)
    compare_parser.add_argument("--host-label", required=True)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--input-safe-json", type=Path, action="append", required=True)
    aggregate_parser.add_argument("--output-safe-json", type=Path, required=True)
    aggregate_parser.add_argument("--expected-tasks", type=int, default=70)
    aggregate_parser.add_argument("--expected-hosts", type=int, default=3)
    args = parser.parse_args()
    if args.command == "compare":
        result = compare_host(
            args.approved,
            args.native_qualified,
            args.step70_qualified,
            args.output_safe_json,
            expected_approved=args.expected_approved,
            host_label=args.host_label,
        )
    else:
        result = aggregate_safe(
            args.input_safe_json,
            args.output_safe_json,
            expected_tasks=args.expected_tasks,
            expected_hosts=args.expected_hosts,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

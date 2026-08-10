#!/usr/bin/env python3
"""Replay the non-overlapping correctness-band reward over saved GRPO groups."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

from llin_verl.pi_reward import banded_reward_score


REQUIRED_FIELDS = (
    "has_final_answer",
    "final_answer_correct",
    "sql_evidence_correct",
    "safe",
    "valid_tool_protocol",
    "gold_sql_verified",
)


def read_rows(directories: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for directory in directories:
        for path in sorted(directory.glob("*.jsonl"), key=lambda item: int(item.stem)):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_source"] = str(path)
                rows.append(row)
    return rows


def replay(rows: list[dict[str, Any]], expected_group_size: int = 4) -> dict[str, Any]:
    missing_fields = {
        field: sum(field not in row for row in rows)
        for field in REQUIRED_FIELDS
    }
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        base_score = row.get("base_score")
        if not isinstance(base_score, (int, float)):
            base_score = 0.7 * float(row.get("boss_reward") or 0) + 0.3 * float(
                row.get("evidence_reward") or 0
            )
        eligible = bool(
            row.get("safe") and row.get("valid_tool_protocol") and row.get("gold_sql_verified")
        )
        row["_banded"] = banded_reward_score(
            eligible=eligible,
            has_final_answer=bool(row.get("has_final_answer")),
            final_answer_correct=bool(row.get("final_answer_correct")),
            sql_evidence_correct=bool(row.get("sql_evidence_correct")),
            process_quality=float(base_score),
        )
        digest = hashlib.sha256(str(row.get("input") or "").encode("utf-8")).hexdigest()
        groups[(row["_source"], digest)].append(row)

    valid_groups = [group for group in groups.values() if len(group) == expected_group_size]
    eligible_correct = [
        row
        for group in valid_groups
        for row in group
        if row.get("safe")
        and row.get("valid_tool_protocol")
        and row.get("gold_sql_verified")
        and row.get("final_answer_correct")
    ]
    eligible_wrong = [
        row
        for group in valid_groups
        for row in group
        if row.get("safe")
        and row.get("valid_tool_protocol")
        and row.get("gold_sql_verified")
        and not row.get("final_answer_correct")
    ]
    eligible_correct_ids = {id(row) for row in eligible_correct}
    eligible_wrong_ids = {id(row) for row in eligible_wrong}
    mixed_groups = [
        group
        for group in valid_groups
        if any(id(row) in eligible_correct_ids for row in group)
        and any(id(row) in eligible_wrong_ids for row in group)
    ]
    ranked_groups = [
        group
        for group in mixed_groups
        if min(float(row["_banded"]) for row in group if id(row) in eligible_correct_ids)
        > max(float(row["_banded"]) for row in group if id(row) in eligible_wrong_ids)
    ]
    ineligible = [
        row
        for group in valid_groups
        for row in group
        if not (row.get("safe") and row.get("valid_tool_protocol") and row.get("gold_sql_verified"))
    ]
    no_answer = [row for group in valid_groups for row in group if not row.get("has_final_answer")]
    valid_group_rate = len(valid_groups) / len(groups) if groups else 0.0
    mixed_rank_rate = len(ranked_groups) / len(mixed_groups) if mixed_groups else 0.0
    gates = {
        "at_least_3000_rows": len(rows) >= 3000,
        "all_required_fields_present": not any(missing_fields.values()),
        "valid_group_rate_at_least_99pct": valid_group_rate >= 0.99,
        "has_mixed_correct_groups": bool(mixed_groups),
        "mixed_correct_rank_rate_at_least_99pct": mixed_rank_rate >= 0.99,
        "ineligible_always_zero": all(float(row["_banded"]) == 0 for row in ineligible),
        "no_final_answer_always_zero": all(float(row["_banded"]) == 0 for row in no_answer),
        "eligible_wrong_capped_at_0_5": all(float(row["_banded"]) <= 0.5 for row in eligible_wrong),
        "eligible_correct_floor_0_65": all(float(row["_banded"]) >= 0.65 for row in eligible_correct),
    }
    values = [float(row["_banded"]) for group in valid_groups for row in group]
    return {
        "row_count": len(rows),
        "group_count": len(groups),
        "valid_group_count": len(valid_groups),
        "valid_group_rate": valid_group_rate,
        "mixed_correct_group_count": len(mixed_groups),
        "mixed_correct_rank_rate": mixed_rank_rate,
        "eligible_correct_rows": len(eligible_correct),
        "eligible_wrong_rows": len(eligible_wrong),
        "banded_reward_mean": mean(values) if values else None,
        "missing_fields": missing_fields,
        "gate_checks": gates,
        "gate_passed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", action="append", type=Path, required=True)
    parser.add_argument("--expected-group-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(read_rows(args.rollout_dir), args.expected_group_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["gate_passed"] else 3)


if __name__ == "__main__":
    main()

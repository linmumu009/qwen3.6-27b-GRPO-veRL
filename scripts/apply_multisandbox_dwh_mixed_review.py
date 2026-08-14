#!/usr/bin/env python3
"""Apply anonymous semantic-review decisions to mixed DWH candidates."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def candidate_key(row: dict) -> tuple[str, str]:
    extra = row["extra_info"]
    return str(extra["instruction_sha256"]), str(extra["gold_sha256"])


def write_private_table(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary)
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def apply_decisions(
    mixed_groups_path: Path,
    decisions_path: Path,
    output_dir: Path,
    *,
    arm: str,
    require_all_reviewed: bool = False,
) -> dict:
    mixed_table = pq.read_table(mixed_groups_path)
    mixed_rows = mixed_table.to_pylist()
    decision_payload = json.loads(decisions_path.read_text(encoding="utf-8"))
    decision_rows = [row for row in decision_payload["decisions"] if row["arm"] == arm]
    decisions: dict[tuple[str, str], dict] = {}
    for row in decision_rows:
        key = (str(row["instruction_sha256"]), str(row["gold_sha256"]))
        if key in decisions:
            raise ValueError(f"duplicate semantic-review decision: {key}")
        if not 1 <= int(row["correct_count"]) <= 7:
            raise ValueError(f"decision is not mixed: {key}")
        decisions[key] = row

    mixed_by_key: dict[tuple[str, str], dict] = {}
    for row in mixed_rows:
        key = candidate_key(row)
        if key in mixed_by_key:
            raise ValueError(f"duplicate mixed candidate: {key}")
        mixed_by_key[key] = row
    missing_candidates = sorted(set(decisions) - set(mixed_by_key))
    if missing_candidates:
        raise ValueError(
            f"semantic-review decisions do not match mixed candidates: {len(missing_candidates)}"
        )

    approved_rows = []
    rejected = reviewed = 0
    for key, row in mixed_by_key.items():
        decision = decisions.get(key)
        if decision is None:
            continue
        reviewed += 1
        checks = (
            bool(decision["instruction_unambiguously_entails_gold"]),
            bool(decision["verification_sql_fully_answers_instruction"]),
            bool(decision["expected_value_supported_by_query_result"]),
            bool(decision["final_outcome_routing_trustworthy"]),
        )
        if decision["decision"] == "approved_candidate":
            if not all(checks):
                raise ValueError(f"approved decision failed semantic checks: {key}")
            approved = deepcopy(row)
            approved["extra_info"]["explicit_semantic_reviewed"] = True
            approved["extra_info"]["training_allowed"] = False
            approved_rows.append(approved)
        elif decision["decision"] == "rejected":
            rejected += 1
        else:
            raise ValueError(f"unsupported semantic-review decision: {decision['decision']}")

    unreviewed = len(mixed_rows) - reviewed
    if require_all_reviewed and unreviewed:
        raise ValueError(f"mixed candidates remain unreviewed: {unreviewed}")

    if approved_rows:
        approved_table = pa.Table.from_pylist(approved_rows, schema=mixed_table.schema)
    else:
        approved_table = mixed_table.slice(0, 0)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_private_table(
        output_dir / "semantic_approved_candidates.sensitive.parquet",
        approved_table,
    )
    summary = {
        "contract": "multisandbox-dwh-mixed-semantic-review-application-v1",
        "arm": arm,
        "mixed_candidates": len(mixed_rows),
        "reviewed": reviewed,
        "approved_candidates": len(approved_rows),
        "rejected": rejected,
        "unreviewed": unreviewed,
        "all_mixed_reviewed": unreviewed == 0,
        "explicit_semantic_review_completed_for_approved_rows": True,
        "training_allowed": False,
        "promotion_allowed": False,
        "contains_task_ids_prompts_sql_gold_values_final_answers_or_tool_outputs": False,
    }
    (output_dir / "safe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed-groups", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("m05", "m06"), required=True)
    parser.add_argument("--require-all-reviewed", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            apply_decisions(
                args.mixed_groups,
                args.decisions,
                args.output_dir,
                arm=args.arm,
                require_all_reviewed=args.require_all_reviewed,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

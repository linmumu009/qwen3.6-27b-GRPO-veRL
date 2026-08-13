#!/usr/bin/env python3
"""Probe semantic-review SQL under SQLite's reversed unordered scan order."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
from typing import Any
from urllib.parse import quote

from scripts.prepare_pi_formal_dataset import gold_supported_by_rows
from scripts.prepare_repair_sft_dataset import READ_ONLY_SQL_RE, sha256_file


PILOT_CONTRACT = "disjoint-pair-semantic-review-pilot-v1"
CONTRACT = "disjoint-pair-review-pilot-query-stability-v1"


def execute_probe(database: Path, sql: str, reverse_unordered: bool) -> tuple[list[str], list[tuple]]:
    if not READ_ONLY_SQL_RE.match(sql or ""):
        raise ValueError("only SELECT/WITH review SQL is allowed")
    resolved = database.resolve(strict=True)
    uri = f"file:{quote(resolved.as_posix(), safe='/')}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True, timeout=5.0)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute(
            f"PRAGMA reverse_unordered_selects={'ON' if reverse_unordered else 'OFF'}"
        )
        cursor = connection.execute(sql)
        columns = [str(item[0]) for item in cursor.description or []]
        rows = cursor.fetchmany(10_001)
        if len(rows) > 10_000:
            raise ValueError("review SQL evidence exceeds 10000 rows")
        return columns, rows
    finally:
        connection.close()


def audit(
    *, packet_file: Path, pilot_contract_file: Path, database: Path, output_dir: Path
) -> dict[str, Any]:
    pilot = json.loads(pilot_contract_file.read_text(encoding="utf-8"))
    if pilot.get("contract") != PILOT_CONTRACT:
        raise ValueError("review pilot contract mismatch")
    if pilot.get("training_allowed") is not False or pilot.get("promotion_allowed") is not False:
        raise ValueError("review pilot contract is not fail closed")
    if pilot.get("review_packet_sha256") != sha256_file(packet_file):
        raise ValueError("review pilot packet hash differs")
    packet = [json.loads(line) for line in packet_file.read_text(encoding="utf-8").splitlines() if line]
    if len(packet) != int(pilot.get("selected_tasks") or 0):
        raise ValueError("review pilot packet grain differs")

    outcomes: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    identities: set[str] = set()
    for row in packet:
        identity = str(row.get("task_id") or "")
        if not identity or identity in identities:
            raise ValueError(f"missing or duplicate review identity: {identity!r}")
        identities.add(identity)
        gold = row.get("gold_answer") or {}
        sql = str(gold.get("verification_sql") or "")
        normal_columns, normal_rows = execute_probe(database, sql, False)
        reverse_columns, reverse_rows = execute_probe(database, sql, True)
        columns_equal = normal_columns == reverse_columns
        rows_equal = normal_rows == reverse_rows
        normal_supported = gold_supported_by_rows(gold, normal_rows)
        reverse_supported = gold_supported_by_rows(gold, reverse_rows)
        if not normal_supported:
            outcome = "normal_execution_no_longer_supports_gold"
        elif not columns_equal or not rows_equal:
            outcome = "result_changes_under_reverse_unordered_scan"
        elif not reverse_supported:
            outcome = "reverse_execution_does_not_support_gold"
        else:
            outcome = "stable_under_reverse_unordered_scan_probe"
        outcomes[outcome] += 1
        records.append(
            {
                "review_index": int(row["review_index"]),
                "task_id": identity,
                "outcome": outcome,
                "columns_equal": columns_equal,
                "rows_equal": rows_equal,
                "normal_gold_supported": normal_supported,
                "reverse_gold_supported": reverse_supported,
                "semantic_approval": None,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "query_stability_evidence.sensitive.jsonl"
    evidence_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    evidence_path.chmod(0o600)
    unstable = outcomes["result_changes_under_reverse_unordered_scan"]
    stable = outcomes["stable_under_reverse_unordered_scan_probe"]
    blocked = len(records) - stable
    return {
        "contract": CONTRACT,
        "date": "2026-08-13",
        "tasks": len(records),
        "probe": "sqlite_reverse_unordered_selects_off_vs_on_exact_columns_and_rows",
        "outcome_counts": dict(sorted(outcomes.items())),
        "result_changes_under_reverse_unordered_scan": unstable,
        "mechanically_blocked_before_semantic_review": blocked,
        "eligible_for_explicit_semantic_review": stable,
        "automatic_semantic_approvals": 0,
        "completed_semantic_reviews": 0,
        "evidence": evidence_path.name,
        "evidence_sha256": sha256_file(evidence_path),
        "evidence_permissions": "0600",
        "evidence_contains_sensitive_task_ids": True,
        "evidence_must_remain_server_side": True,
        "source_sha256": {
            "packet": sha256_file(packet_file),
            "pilot_contract": sha256_file(pilot_contract_file),
            "database": sha256_file(database),
        },
        "decision": (
            "replace_unstable_limit_without_order_by_stratum_before_semantic_review"
            if blocked == len(records)
            else "semantically_review_only_stable_probe_survivors_and_replace_blocked_tasks"
        ),
        "may_be_used_as_training_or_rollout_data": False,
        "training_allowed": False,
        "promotion_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--pilot-contract", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        packet_file=args.packet,
        pilot_contract_file=args.pilot_contract,
        database=args.database,
        output_dir=args.output_dir,
    )
    output = args.output_dir / "query_stability_contract.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

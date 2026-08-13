#!/usr/bin/env python3
"""Join manual review decisions to the server packet and emit a safe summary."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.analyze_disjoint_real_state_evaluation import wilson_interval
from scripts.prepare_repair_sft_dataset import sha256_file


PILOT_CONTRACT = "disjoint-pair-semantic-review-pilot-v1"
STABILITY_CONTRACT = "disjoint-pair-review-pilot-query-stability-v1"
DECISION_CONTRACT = "disjoint-pair-semantic-review-decisions-v1"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _by_index(rows: list[dict[str, Any]], label: str) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = int(row.get("review_index", -1))
        if index < 0 or index in output:
            raise ValueError(f"missing or duplicate {label} review index: {index}")
        output[index] = row
    return output


def analyze(
    *,
    packet: list[dict[str, Any]],
    pilot: dict[str, Any],
    stability: dict[str, Any],
    stability_evidence: list[dict[str, Any]],
    decisions: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if pilot.get("contract") != PILOT_CONTRACT:
        raise ValueError("review pilot contract mismatch")
    if stability.get("contract") != STABILITY_CONTRACT:
        raise ValueError("review stability contract mismatch")
    if decisions.get("contract") != DECISION_CONTRACT:
        raise ValueError("manual review decision contract mismatch")
    for value, label in ((pilot, "pilot"), (stability, "stability"), (decisions, "decision")):
        if value.get("training_allowed") is not False or value.get("promotion_allowed") is not False:
            raise ValueError(f"{label} contract is not fail closed")
    tasks = int(pilot.get("selected_tasks") or 0)
    if tasks <= 0 or len(packet) != tasks:
        raise ValueError("review packet grain differs")
    packet_by_index = _by_index(packet, "packet")
    stability_by_index = _by_index(stability_evidence, "stability")
    decision_rows = list(decisions.get("decisions") or [])
    decision_by_index = _by_index(decision_rows, "decision")
    expected_indices = set(range(tasks))
    if set(packet_by_index) != expected_indices:
        raise ValueError("review packet indices are not contiguous")
    if set(stability_by_index) != expected_indices or set(decision_by_index) != expected_indices:
        raise ValueError("review evidence or decisions are incomplete")

    decision_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    answer_types: Counter[str] = Counter()
    sensitive_rows: list[dict[str, Any]] = []
    for index in range(tasks):
        source = packet_by_index[index]
        stable = stability_by_index[index]
        decision = decision_by_index[index]
        if str(stable.get("task_id") or "") != str(source.get("task_id") or ""):
            raise ValueError(f"review task identity differs at index {index}")
        if stable.get("outcome") != "stable_under_reverse_unordered_scan_probe":
            raise ValueError(f"review task did not pass stability at index {index}")
        verdict = str(decision.get("decision") or "")
        entails = decision.get("instruction_unambiguously_entails_gold") is True
        answers = decision.get("verification_sql_fully_answers_instruction") is True
        supported = decision.get("expected_value_supported_by_query_result") is True
        if verdict == "approved":
            if not (entails and answers and supported):
                raise ValueError(f"approved review lacks all three evidence gates: {index}")
        elif verdict == "rejected":
            if entails and answers and supported:
                raise ValueError(f"rejected review has no failed semantic gate: {index}")
        else:
            raise ValueError(f"unexpected review decision at index {index}: {verdict!r}")
        reason = str(decision.get("reason_code") or "")
        confidence = str(decision.get("confidence") or "")
        severity = str(decision.get("severity") or "")
        if not reason or confidence not in {"medium", "high"} or severity != "high":
            raise ValueError(f"review metadata is incomplete at index {index}")
        decision_counts[verdict] += 1
        reason_counts[reason] += 1
        confidence_counts[confidence] += 1
        severity_counts[severity] += 1
        warning_counts.update(str(item) for item in source.get("semantic_warnings") or [])
        answer_types[str((source.get("gold_answer") or {}).get("answer_type") or "missing")] += 1
        sql = str((source.get("gold_answer") or {}).get("verification_sql") or "")
        sensitive_rows.append(
            {
                "review_index": index,
                "task_id": source["task_id"],
                "instruction_sha256": sha256_text(str(source["natural_language_instruction"])),
                "verification_sql_sha256": sha256_text(sql),
                "decision": verdict,
                "instruction_unambiguously_entails_gold": entails,
                "verification_sql_fully_answers_instruction": answers,
                "expected_value_supported_by_query_result": supported,
                "reason_code": reason,
                "confidence": confidence,
                "severity": severity,
            }
        )

    approved = decision_counts["approved"]
    rejected = decision_counts["rejected"]
    return (
        {
            "contract": "disjoint-pair-semantic-review-pilot-safe-summary-v1",
            "date": "2026-08-13",
            "scope": {
                "review_required_pool": int(pilot["review_required_pool"]),
                "reviewed_tasks": tasks,
                "selection_role": pilot["selection_role"],
                "all_tasks_passed_mechanical_and_query_stability_gates": True,
            },
            "semantic_review": {
                "approved": approved,
                "rejected": rejected,
                "approval_rate": approved / tasks,
                "approval_rate_wilson95": wilson_interval(approved, tasks),
                "decision_counts": dict(sorted(decision_counts.items())),
                "reason_counts": dict(sorted(reason_counts.items())),
                "confidence_counts": dict(sorted(confidence_counts.items())),
                "severity_counts": dict(sorted(severity_counts.items())),
                "instruction_unambiguously_entails_gold": sum(
                    bool(row["instruction_unambiguously_entails_gold"])
                    for row in sensitive_rows
                ),
                "verification_sql_fully_answers_instruction": sum(
                    bool(row["verification_sql_fully_answers_instruction"])
                    for row in sensitive_rows
                ),
                "expected_value_supported_by_query_result": sum(
                    bool(row["expected_value_supported_by_query_result"])
                    for row in sensitive_rows
                ),
            },
            "selection_profile": {
                "warning_counts": dict(sorted(warning_counts.items())),
                "answer_types": dict(sorted(answer_types.items())),
            },
            "data_quality_finding": {
                "severity": "high",
                "confidence": "high",
                "what_failed": "instruction_gold_sql_semantic_alignment",
                "why_it_matters": "using these rows would train the model toward mechanically supported but semantically underdetermined or mismatched answers",
                "likely_cause": "broad or temporal natural-language templates were paired with narrow static aggregation SQL",
                "current_review_required_pool_safe_for_rollout_acquisition": approved > 0,
            },
            "decision": {
                "stop_reviewing_remaining96_from_same_queue_now": approved == 0,
                "approved_tasks_available_for_rollout": approved,
                "selected_next_action": (
                    "build_new_unambiguous_current_definition_tasks_with_explicit_metric_grouping_and_time_scope"
                    if approved == 0
                    else "estimate_queue_capacity_from_observed_approval_rate"
                ),
                "training_now": False,
                "promotion_allowed": False,
            },
            "contains_task_ids_prompts_sql_gold_values_tool_outputs_or_server_paths": False,
            "may_be_used_as_training_or_rollout_data": False,
            "training_allowed": False,
            "promotion_allowed": False,
        },
        sensitive_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--pilot-contract", type=Path, required=True)
    parser.add_argument("--stability-contract", type=Path, required=True)
    parser.add_argument("--stability-evidence", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.packet) != json.loads(
        args.decisions.read_text(encoding="utf-8")
    ).get("source_packet_sha256"):
        raise ValueError("manual decisions target a different review packet")
    summary, sensitive = analyze(
        packet=_jsonl(args.packet),
        pilot=json.loads(args.pilot_contract.read_text(encoding="utf-8")),
        stability=json.loads(args.stability_contract.read_text(encoding="utf-8")),
        stability_evidence=_jsonl(args.stability_evidence),
        decisions=json.loads(args.decisions.read_text(encoding="utf-8")),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sensitive_path = args.output_dir / "semantic_review_adjudication.sensitive.jsonl"
    sensitive_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sensitive),
        encoding="utf-8",
    )
    sensitive_path.chmod(0o600)
    summary["sensitive_adjudication"] = {
        "file": sensitive_path.name,
        "sha256": sha256_file(sensitive_path),
        "permissions": "0600",
        "must_remain_server_side": True,
    }
    summary["source_sha256"] = {
        "packet": sha256_file(args.packet),
        "pilot_contract": sha256_file(args.pilot_contract),
        "stability_contract": sha256_file(args.stability_contract),
        "stability_evidence": sha256_file(args.stability_evidence),
        "decisions": sha256_file(args.decisions),
    }
    output = args.output_dir / "semantic_review_safe_summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

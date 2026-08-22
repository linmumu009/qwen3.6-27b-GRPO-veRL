#!/usr/bin/env python3
"""Build the private approved43×8 three-state calibration ledger.

Raw prompts, final answers, gold values, SQL and tool results are written only
to the 0600 private packet.  The public summary contains aggregate counts,
anonymous group distributions and hashes.  Missing human labels are a hard
blocker; automatic labels are never presented as human calibration.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from llin_verl.grounded_trajectory_reward import (
    JUDGE_STATES,
    REWARD_CONTRACT,
    compute_grounded_trajectory_reward,
)
from llin_verl.outcome_gated_contract import evidence_binding_hash, stable_json_hash
from llin_verl.trajectory_process_reward import parse_qwen_tool_events


CONTRACT = "qwen38-approved43-grounded-tristate-calibration-v1"
APPROVED_SHA256 = "d86b53d906806b150d43a508dce9b0dd6d05105c07e03961e8e7bf9439ccd944"
MANIFEST_SHA256 = "1426bc09a3dbaf4709fd89227790603afb7a2bf11beeba80946057d490e0f424"
TASKS = 43
SAMPLES = 8
ATTEMPT_CAP = 16
HUMAN_REQUIRED_FIELDS = {
    "trajectory_identity_sha256",
    "human_final_outcome",
    "human_real_tool_evidence",
    "human_evidence_supports_final",
    "human_process_safe",
    "human_judge_state",
    "human_judge_confidence",
    "human_evidence_route",
    "disagreement_reason",
    "reviewer",
    "reviewed_at",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_private_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_safe_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _truth(dataset_row: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(dataset_row["reward_model"]["ground_truth"], ensure_ascii=False))
    criteria = task.get("verification_criteria") or {}
    value["evidence_plan"] = task.get("evidence_plan") or {}
    value["required_tables"] = task.get("expected_tables") or value.get("required_tables", [])
    value["must_use_fields"] = criteria.get("must_use_fields") or value.get("must_use_fields", [])
    value["process_evidence_binding_sha256"] = evidence_binding_hash(value)
    return value


def _probability_fill(p: float, cap: int = ATTEMPT_CAP) -> float:
    return sum(
        math.comb(cap, successes) * p**successes * (1.0 - p) ** (cap - successes)
        for successes in range(SAMPLES, cap + 1)
    )


def _expected_attempts(p: float, cap: int = ATTEMPT_CAP) -> float:
    if p <= 0:
        return float(cap)
    expected = 0.0
    filled_probability = 0.0
    for attempt in range(SAMPLES, cap + 1):
        probability = math.comb(attempt - 1, SAMPLES - 1) * p**SAMPLES * (1.0 - p) ** (attempt - SAMPLES)
        expected += attempt * probability
        filled_probability += probability
    return expected + cap * max(0.0, 1.0 - filled_probability)


def _numeric_distribution(values: list[Any]) -> dict[str, Any]:
    numbers = sorted(float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value)))
    if not numbers:
        return {"count": 0, "min": None, "p50": None, "mean": None, "max": None}
    middle = len(numbers) // 2
    median = numbers[middle] if len(numbers) % 2 else (numbers[middle - 1] + numbers[middle]) / 2.0
    return {
        "count": len(numbers),
        "min": round(numbers[0], 8),
        "p50": round(median, 8),
        "mean": round(sum(numbers) / len(numbers), 8),
        "max": round(numbers[-1], 8),
        "zero": sum(value == 0.0 for value in numbers),
        "positive": sum(value > 0.0 for value in numbers),
    }


def _load_human_labels(path: Path | None, identities: set[str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if path is None or not path.is_file():
        return {}, ["human_labels_file_missing"]
    rows = read_jsonl(path)
    by_identity: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for row in rows:
        missing = HUMAN_REQUIRED_FIELDS - set(row)
        if missing:
            errors.append("human_label_required_fields_missing")
            continue
        identity = str(row["trajectory_identity_sha256"])
        if identity in by_identity:
            errors.append("duplicate_human_label_identity")
        by_identity[identity] = row
        if str(row["human_final_outcome"]) not in {"CORRECT", "INCORRECT", "UNKNOWN"}:
            errors.append("invalid_human_final_outcome")
        if str(row["human_judge_state"]) not in JUDGE_STATES:
            errors.append("invalid_human_judge_state")
        if str(row["human_judge_confidence"]) not in {"HIGH", "MEDIUM", "LOW"}:
            errors.append("invalid_human_confidence")
        if str(row["human_evidence_route"]) not in {"direct", "composed", "table", "none", "unsupported"}:
            errors.append("invalid_human_evidence_route")
        for field in ("human_real_tool_evidence", "human_evidence_supports_final", "human_process_safe"):
            if str(row[field]) not in {"YES", "NO", "UNKNOWN"}:
                errors.append(f"invalid_{field}")
    if set(by_identity) != identities:
        errors.append("human_label_identities_not_exact_344")
    return by_identity, sorted(set(errors))


def _confusion(auto_rows: list[dict[str, Any]], human: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if len(human) != TASKS * SAMPLES:
        return None
    matrix = {human_state: {auto_state: 0 for auto_state in JUDGE_STATES} for human_state in JUDGE_STATES}
    false_negative = false_positive = usable_unknown = guess_pass = 0
    route_rows: dict[str, list[tuple[str, str]]] = defaultdict(list)
    unresolved = 0
    for row in auto_rows:
        label = human[row["trajectory_identity_sha256"]]
        human_state = str(label["human_judge_state"])
        auto_state = str(row["judge_state"])
        matrix[human_state][auto_state] += 1
        false_negative += human_state == "PASS" and auto_state == "FAIL"
        false_positive += human_state == "FAIL" and auto_state == "PASS"
        usable_unknown += human_state in {"PASS", "FAIL"} and auto_state == "UNKNOWN"
        guess_pass += (
            str(label["human_final_outcome"]) == "CORRECT"
            and str(label["human_real_tool_evidence"]) == "NO"
            and auto_state == "PASS"
        )
        if human_state != auto_state and not str(label.get("disagreement_reason") or "").strip():
            unresolved += 1
        route = str(label["human_evidence_route"])
        if route in {"direct", "composed", "table"}:
            route_rows[route].append((human_state, auto_state))
    routes: dict[str, dict[str, float | int | None]] = {}
    for route in ("direct", "composed", "table"):
        values = route_rows.get(route, [])
        human_pass = sum(h == "PASS" for h, _ in values)
        auto_pass = sum(a == "PASS" for _, a in values)
        true_pass = sum(h == a == "PASS" for h, a in values)
        routes[route] = {
            "rows": len(values),
            "precision": round(true_pass / auto_pass, 8) if auto_pass else None,
            "recall": round(true_pass / human_pass, 8) if human_pass else None,
            "unknown_rate": round(sum(a == "UNKNOWN" for _, a in values) / len(values), 8) if values else None,
        }
    return {
        "matrix": matrix,
        "original_human_correct_auto_fail": false_negative,
        "original_human_wrong_auto_pass": false_positive,
        "usable_trajectory_auto_unknown": usable_unknown,
        "guess_correct_auto_pass": guess_pass,
        "route_metrics": routes,
        "unresolved_disagreements": unresolved,
        "all_disagreements_resolved": unresolved == 0,
    }


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    if file_sha256(args.approved43) != APPROVED_SHA256:
        raise ValueError("approved43 Parquet hash mismatch")
    if file_sha256(args.manifest) != MANIFEST_SHA256:
        raise ValueError("approved43 manifest hash mismatch")
    dataset = pq.read_table(args.dataset).to_pylist()
    approved = pq.read_table(args.approved43).to_pylist()
    tasks = read_jsonl(args.tasks)
    manifest = read_jsonl(args.manifest)
    if len(dataset) != len(tasks) or len(approved) != TASKS or len(manifest) != TASKS:
        raise ValueError("dataset/tasks/approved package row count mismatch")
    by_instruction = {
        str(row["extra_info"]["instruction_sha256"]): index for index, row in enumerate(dataset)
    }
    approved_hashes = [str(row["extra_info"]["instruction_sha256"]) for row in approved]
    if len(set(approved_hashes)) != TASKS or any(value not in by_instruction for value in approved_hashes):
        raise ValueError("approved43 identities are not exact source members")
    approved_indices = [by_instruction[value] for value in approved_hashes]

    observations: dict[tuple[int, int], dict[str, Any]] = {}
    shard_paths = sorted(args.shards.glob("tasks_*.jsonl"))
    for path in shard_paths:
        for row in read_jsonl(path):
            key = (int(row["source_task_index"]), int(row["sample_index"]))
            if key in observations:
                raise ValueError("duplicate trajectory slot")
            observations[key] = row
    expected = {(task_index, sample) for task_index in range(len(dataset)) for sample in range(SAMPLES)}
    if set(observations) != expected:
        raise ValueError("source shards are not an exact dataset×8 grid")

    auto_rows: list[dict[str, Any]] = []
    packets: list[dict[str, Any]] = []
    templates: list[dict[str, Any]] = []
    for approved_position, task_index in enumerate(approved_indices):
        dataset_row = dataset[task_index]
        task = tasks[task_index]
        truth = _truth(dataset_row, task)
        instruction_hash = str(dataset_row["extra_info"]["instruction_sha256"])
        for sample in range(SAMPLES):
            source = observations[(task_index, sample)]
            output = str(source.get("output") or "")
            parsed = parse_qwen_tool_events(output)
            identity = stable_json_hash({"instruction_sha256": instruction_hash, "sample_index": sample})
            extra = {
                "pi_tool_events": parsed["events"],
                "pi_tool_log_present": True,
                "pi_tool_protocol_complete": bool(parsed["protocol_complete"]),
                "pi_reward_database_path": str(args.database),
                "pi_reward_database_root": str(args.database.parent.parent),
                "trajectory_timeout": bool(source.get("trajectory_timeout")),
                "runtime_error": bool(source.get("runtime_error")),
                "auto_retry_count": int(source.get("auto_retry_count", 0) or 0),
                "force_final_retry_count": int(source.get("force_final_retry_count", 0) or 0),
            }
            result = compute_grounded_trajectory_reward(
                str(dataset_row.get("data_source") or ""), output, truth, extra
            )
            safe = {
                "approved_position": approved_position,
                "sample_index": sample,
                "trajectory_identity_sha256": identity,
                "instruction_sha256": instruction_hash,
                "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                "event_fingerprint_sha256": stable_json_hash(parsed["events"]),
                **result,
            }
            auto_rows.append(safe)
            packets.append(
                {
                    **safe,
                    "source_task_index": task_index,
                    "instruction": dataset_row.get("prompt"),
                    "task_contract": task,
                    "ground_truth": truth,
                    "trajectory_output": output,
                    "pi_tool_events": parsed["events"],
                    "human_annotation": None,
                }
            )
            templates.append(
                {
                    "trajectory_identity_sha256": identity,
                    "human_final_outcome": "",
                    "human_real_tool_evidence": "",
                    "human_evidence_supports_final": "",
                    "human_process_safe": "",
                    "human_judge_state": "",
                    "human_judge_confidence": "",
                    "human_evidence_route": "",
                    "disagreement_reason": "",
                    "reviewer": "",
                    "reviewed_at": "",
                }
            )

    identities = {row["trajectory_identity_sha256"] for row in auto_rows}
    if len(auto_rows) != TASKS * SAMPLES or len(identities) != TASKS * SAMPLES:
        raise ValueError("approved43 calibration is not exact 43×8 unique identities")
    previous: list[dict[str, Any]] = []
    if args.previous_shadow:
        previous = [row for row in read_jsonl(args.previous_shadow) if bool(row.get("approved43"))]
        previous_ids = {str(row["trajectory_identity_sha256"]) for row in previous}
        if previous_ids != identities:
            raise ValueError("v6 identities differ from frozen v2 approved43 identities")

    labels, label_errors = _load_human_labels(args.human_labels, identities)
    if labels:
        for packet in packets:
            packet["human_annotation"] = labels.get(packet["trajectory_identity_sha256"])
    confusion = _confusion(auto_rows, labels)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in auto_rows:
        grouped[int(row["approved_position"])].append(row)
    group_counts: list[dict[str, Any]] = []
    fill_probabilities: list[float] = []
    expected_attempts: list[float] = []
    mixed_trainable = 0
    for index in range(TASKS):
        rows = grouped[index]
        counts = Counter(str(row["judge_state"]) for row in rows)
        trainable = counts["PASS"] + counts["FAIL"]
        p = trainable / SAMPLES
        fill = _probability_fill(p)
        attempts = _expected_attempts(p)
        fill_probabilities.append(fill)
        expected_attempts.append(attempts)
        mixed_trainable += counts["PASS"] > 0 and counts["FAIL"] > 0
        group_counts.append(
            {
                "anonymous_group_index": index,
                "PASS": counts["PASS"],
                "FAIL": counts["FAIL"],
                "UNKNOWN": counts["UNKNOWN"],
                "fill_probability_at_16": round(fill, 8),
                "expected_attempts_capped_16": round(attempts, 8),
            }
        )
    states = Counter(str(row["judge_state"]) for row in auto_rows)
    final_correct = sum(str(row["final_state"]) == "PASS" for row in auto_rows)
    guesses = [row for row in auto_rows if str(row["final_state"]) == "PASS" and str(row["evidence_state"]) == "FAIL"]
    routes = Counter(str(row["evidence_route"]) for row in auto_rows if row["judge_state"] == "PASS")
    judge_reasons = Counter(str(row["judge_reason"]) for row in auto_rows)
    evidence_reasons = Counter(str(row["evidence_reason"]) for row in auto_rows)
    final_reasons = Counter(str(row["final_reason"]) for row in auto_rows)
    safety_reasons = Counter(str(row["safety_observability_reason"]) for row in auto_rows)
    previous_by_id = {str(row["trajectory_identity_sha256"]): row for row in previous}
    final_label_changes = sum(
        bool(previous_by_id[row["trajectory_identity_sha256"]].get("correctness"))
        != (str(row["final_state"]) == "PASS")
        for row in auto_rows
        if row["trajectory_identity_sha256"] in previous_by_id
    )
    previous_process_components = {
        name: _numeric_distribution([row.get(name) for row in previous])
        for name in ("process_sql", "process_table", "process_field", "process_fit", "process_efficiency")
    }
    manual_complete = confusion is not None and not label_errors
    disagreements_resolved = bool(confusion and confusion["all_disagreements_resolved"])
    summary = {
        "contract": CONTRACT,
        "reward_contract": REWARD_CONTRACT,
        "training_status": "paused_cpu_only_no_model_no_rollout_no_optimizer_no_npu",
        "input_gate": {
            "approved43_rows": TASKS,
            "trajectory_rows": len(auto_rows),
            "unique_trajectory_identities": len(identities),
            "approved43_parquet_sha256": APPROVED_SHA256,
            "approved43_manifest_sha256": MANIFEST_SHA256,
            "dataset_sha256": file_sha256(args.dataset),
            "tasks_sha256": file_sha256(args.tasks),
            "database_sha256": file_sha256(args.database),
            "shard_files": len(shard_paths),
            "shards_set_sha256": stable_json_hash([file_sha256(path) for path in shard_paths]),
        },
        "automatic_tristate_shadow": {
            "PASS": states["PASS"],
            "FAIL": states["FAIL"],
            "UNKNOWN": states["UNKNOWN"],
            "unknown_rate": round(states["UNKNOWN"] / len(auto_rows), 8),
            "final_correct_rows": final_correct,
            "grounded_success_coverage_of_auto_final_correct": round(states["PASS"] / max(1, final_correct), 8),
            "guess_correct_candidates": len(guesses),
            "guess_correct_blocked": sum(row["judge_state"] != "PASS" for row in guesses),
            "guess_block_rate": round(sum(row["judge_state"] != "PASS" for row in guesses) / max(1, len(guesses)), 8),
            "PASS_routes": dict(sorted(routes.items())),
            "judge_reason_counts": dict(sorted(judge_reasons.items())),
            "evidence_reason_counts": dict(sorted(evidence_reasons.items())),
            "final_reason_counts": dict(sorted(final_reasons.items())),
            "safety_observability_reason_counts": dict(sorted(safety_reasons.items())),
            "historical_text_parse_failures_masked_unknown": sum(
                str(row["judge_reason"]) == "shadow_or_unattributed_tool_parse_failure"
                for row in auto_rows
            ),
            "evidence_contract_unresolved_rows": sum(
                int(row.get("evidence_contract_unresolved_count") or 0) > 0 for row in auto_rows
            ),
        },
        "previous_v2_comparison": {
            "available": bool(previous),
            "identity_rows": len(previous),
            "old_reward_distribution": _numeric_distribution([row.get("reward") for row in previous]),
            "new_binary_reward_distribution": _numeric_distribution([row.get("reward") for row in auto_rows]),
            "old_correctness_true": sum(bool(row.get("correctness")) for row in previous),
            "new_final_parser_pass": final_correct,
            "final_label_changes_vs_old_correctness": final_label_changes,
            "old_process_component_distributions": previous_process_components,
        },
        "groups": {
            "anonymous_PASS_FAIL_UNKNOWN": group_counts,
            "mixed_trainable_groups_before_online_resampling": mixed_trainable,
            "expected_groups_filled_at_16": round(sum(fill_probabilities), 8),
            "minimum_group_fill_probability_at_16": round(min(fill_probabilities), 8),
            "expected_physical_attempts_for_43_groups": round(sum(expected_attempts), 8),
        },
        "human_calibration": {
            "required_rows": TASKS * SAMPLES,
            "completed_rows": len(labels),
            "status": "complete" if manual_complete else "pending",
            "validation_errors": label_errors,
            "confusion_matrix": confusion,
            "all_disagreements_resolved": disagreements_resolved,
            "automatic_labels_never_treated_as_human": True,
        },
        "private_outputs": {
            "packet_rows": len(packets),
            "auto_judgement_rows": len(auto_rows),
            "human_template_rows": len(templates),
            "mode": "0600",
            "private_content_exported": False,
        },
        "formal_training_allowed": False,
        "promotion_allowed": False,
        "blockers": [
            value
            for value in (
                None if manual_complete else "human_344_calibration_incomplete",
                None if disagreements_resolved else "human_auto_disagreements_not_all_resolved",
                None if confusion and confusion["guess_correct_auto_pass"] == 0 else "guess_correct_false_positive_not_proven_zero",
            )
            if value
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private = args.output_dir / "private"
    write_private_jsonl(private / "human_calibration_packet.sensitive.jsonl", packets)
    write_private_jsonl(private / "automatic_tristate_judgements.sensitive.jsonl", auto_rows)
    if not labels:
        write_private_jsonl(private / "human_labels_template.sensitive.jsonl", templates)
    write_safe_json(args.output_dir / "safe_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--approved43", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--previous-shadow", type=Path)
    parser.add_argument("--human-labels", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = calibrate(args)
    print(json.dumps({
        "contract": summary["contract"],
        "trajectory_rows": summary["input_gate"]["trajectory_rows"],
        "states": summary["automatic_tristate_shadow"],
        "human_status": summary["human_calibration"]["status"],
        "formal_training_allowed": summary["formal_training_allowed"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

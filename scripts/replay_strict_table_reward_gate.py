#!/usr/bin/env python3
"""Replay strict table correctness on private adaptive-rollout trajectories.

The script emits only aggregate, non-sensitive evidence.  Prompt text, task
identities, gold rows, SQL, final answers, and tool outputs never enter the
safe JSON.  A qualified private Parquet may be written for downstream assembly;
its training and promotion flags remain disabled.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
import os
from pathlib import Path
from statistics import median
import time
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from llin_verl.pi_reward import (
    contains_expected_number,
    extract_answer_numbers,
    extract_final_assistant_answer,
    strict_table_answer_match,
)


CONTRACT = "llin-banded-v2-strict-table-replay-gate-v1"
REWARD_CONTRACT = "banded-v2-strict-table-v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _identity(row: dict[str, Any]) -> str:
    return str((row.get("extra_info") or {}).get("instruction_sha256") or "")


def _expected(row: dict[str, Any]) -> list[Any]:
    truth = (row.get("reward_model") or {}).get("ground_truth") or {}
    if str(truth.get("answer_type") or "") != "table":
        raise ValueError("strict table replay received a non-table task")
    value = json.loads(str(truth.get("expected_value_json") or "null"))
    if not isinstance(value, list) or not value:
        raise ValueError("strict table replay received invalid expected_value_json")
    return value


def _tolerances(row: dict[str, Any]) -> tuple[float, float]:
    truth = (row.get("reward_model") or {}).get("ground_truth") or {}
    return float(truth.get("abs_tol", 1e-3)), float(truth.get("rel_tol", 1e-5))


def _legacy_table_correct(answer: str, expected: list[Any], abs_tol: float, rel_tol: float) -> bool:
    folded = answer.casefold()
    for item in expected:
        if not isinstance(item, dict):
            return False
        label = item.get("category", item.get("date"))
        if label is not None and str(label).strip().casefold() not in folded:
            return False
        value = item.get("value")
        if value is not None and not contains_expected_number(answer, float(value), abs_tol, rel_tol):
            return False
    return bool(expected)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _legacy_shape(answer: str, expected: list[Any], abs_tol: float, rel_tol: float) -> list[str]:
    lines = answer.splitlines()
    labels = [str(item.get("category", item.get("date"))).strip() for item in expected]
    label_lines: list[int] = []
    every_pair_same_line = True
    for item, label in zip(expected, labels, strict=True):
        matching_lines = [index for index, line in enumerate(lines) if label.casefold() in line.casefold()]
        if len(matching_lines) != 1:
            every_pair_same_line = False
            continue
        label_lines.append(matching_lines[0])
        value = float(item["value"])
        if not contains_expected_number(lines[matching_lines[0]], value, abs_tol, rel_tol):
            every_pair_same_line = False
    numbers = extract_answer_numbers(answer)
    features = [
        "single_line" if len(lines) <= 1 else "multi_line",
        "contains_pipe" if "|" in answer else "no_pipe",
        "contains_tab" if "\t" in answer else "no_tab",
        "contains_semicolon" if any(mark in answer for mark in (";", "；")) else "no_semicolon",
        "json_like" if "[" in answer and "{" in answer else "not_json_like",
        "distinct_label_lines" if len(label_lines) == len(labels) and len(set(label_lines)) == len(labels) else "labels_share_or_repeat_lines",
        "every_pair_same_line" if every_pair_same_line else "pair_not_same_line",
        "number_count_exact" if len(numbers) == len(expected) else ("number_count_extra" if len(numbers) > len(expected) else "number_count_short"),
    ]
    pipe_lines = [line for line in lines if "|" in line]
    features.append(f"pipe_lines_{len(pipe_lines)}")
    for line in pipe_lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        numeric_cells = sum(bool(extract_answer_numbers(cell)) for cell in cells)
        features.append(f"pipe_cells_{len(cells)}")
        features.append(f"pipe_numeric_cells_{numeric_cells}")
    table_rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in pipe_lines
    ]
    exact_label_rows: list[int] = []
    same_row_pairs = 0
    value_columns: list[int] = []
    for item, label in zip(expected, labels, strict=True):
        normalized_label = " ".join(label.split()).casefold()
        hits = [
            (row_index, column_index)
            for row_index, cells in enumerate(table_rows)
            for column_index, cell in enumerate(cells)
            if " ".join(cell.strip("`*_~ ").split()).casefold() == normalized_label
        ]
        if len(hits) != 1:
            continue
        row_index, _ = hits[0]
        exact_label_rows.append(row_index)
        value = float(item["value"])
        matching_value_columns = [
            column_index
            for column_index, cell in enumerate(table_rows[row_index])
            if contains_expected_number(cell, value, abs_tol, rel_tol)
        ]
        if matching_value_columns:
            same_row_pairs += 1
        if len(matching_value_columns) == 1:
            value_columns.append(matching_value_columns[0])
    features.extend(
        [
            "all_labels_exact_cells" if len(exact_label_rows) == len(expected) else "not_all_labels_exact_cells",
            "all_pairs_same_row" if same_row_pairs == len(expected) else "not_all_pairs_same_row",
            "one_consistent_value_column"
            if len(value_columns) == len(expected) and len(set(value_columns)) == 1
            else "no_unique_consistent_value_column",
            "table_row_order_matches"
            if len(exact_label_rows) == len(expected) and exact_label_rows == sorted(exact_label_rows)
            else "table_row_order_not_verified",
        ]
    )
    pipe_blocks: list[list[list[str]]] = []
    current_block: list[list[str]] = []
    for line in lines + [""]:
        if "|" in line:
            current_block.append([cell.strip() for cell in line.strip().strip("|").split("|")])
        elif current_block:
            pipe_blocks.append(current_block)
            current_block = []
    best_hits: list[tuple[int, int, int]] = []
    best_block: list[list[str]] = []
    for block in pipe_blocks:
        hits: list[tuple[int, int, int]] = []
        for expected_index, label in enumerate(labels):
            normalized_label = " ".join(label.split()).casefold()
            locations = [
                (row_index, column_index)
                for row_index, cells in enumerate(block)
                for column_index, cell in enumerate(cells)
                if " ".join(cell.strip("`*_~ ").split()).casefold() == normalized_label
            ]
            if len(locations) == 1:
                hits.append((expected_index, locations[0][0], locations[0][1]))
        if len(hits) > len(best_hits):
            best_hits, best_block = hits, block
    features.append(f"best_pipe_block_label_hits_{len(best_hits)}_of_{len(expected)}")
    if len(best_hits) == len(expected):
        row_indices = [row for _, row, _ in best_hits]
        label_columns = [column for _, _, column in best_hits]
        features.append("binding_rows_distinct" if len(set(row_indices)) == len(expected) else "binding_rows_shared")
        features.append("binding_label_column_consistent" if len(set(label_columns)) == 1 else "binding_label_columns_vary")
        features.append(
            "binding_rows_contiguous"
            if row_indices == list(range(row_indices[0], row_indices[0] + len(row_indices)))
            else "binding_rows_not_contiguous"
        )
        widths = {len(best_block[row]) for row in row_indices}
        strict_value_columns = []
        if len(widths) == 1:
            width = next(iter(widths))
            for column in range(width):
                cells = [best_block[row][column] for row in row_indices]
                if all(len(extract_answer_numbers(cell)) == 1 for cell in cells) and all(
                    contains_expected_number(cell, float(item["value"]), abs_tol, rel_tol)
                    for cell, item in zip(cells, expected, strict=True)
                ):
                    strict_value_columns.append(column)
        features.append(
            "binding_one_strict_value_column"
            if len(strict_value_columns) == 1
            else "binding_no_unique_strict_value_column"
        )
    return features


def _write_private_parquet(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pylist(rows) if rows else pa.Table.from_pylist([], schema=schema)
    pq.write_table(table, temporary)
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def replay(
    approved_path: Path,
    waves: list[tuple[str, Path, Path]],
    output_safe_json: Path,
    output_qualified_parquet: Path,
    *,
    expected_approved: int,
    host_label: str,
) -> dict[str, Any]:
    approved_table = pq.read_table(approved_path)
    approved_rows = approved_table.to_pylist()
    approved_by_id = {_identity(row): row for row in approved_rows}
    if "" in approved_by_id or len(approved_by_id) != len(approved_rows):
        raise ValueError("approved identities are missing or duplicated")
    if len(approved_rows) != expected_approved:
        raise ValueError(f"expected {expected_approved} approved tasks, got {len(approved_rows)}")
    if any(bool((row.get("extra_info") or {}).get("training_allowed")) for row in approved_rows):
        raise ValueError("approved input unexpectedly enables training")

    trajectories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    wave_counts: Counter[str] = Counter()
    seen_wave_labels: set[str] = set()
    for label, dataset_path, run_dir in waves:
        if label in seen_wave_labels:
            raise ValueError(f"duplicate wave label: {label}")
        seen_wave_labels.add(label)
        dataset = pq.read_table(dataset_path).to_pylist()
        for shard in sorted((run_dir / "shards").glob("tasks_*.jsonl")):
            for trajectory in _read_jsonl(shard):
                index = int(trajectory["source_task_index"])
                if index < 0 or index >= len(dataset):
                    raise ValueError(f"{label}: source_task_index out of range")
                identity = _identity(dataset[index])
                if identity in approved_by_id:
                    trajectories[identity].append(trajectory)
                    wave_counts[label] += 1

    format_modes: Counter[str] = Counter()
    legacy_shape_counts: Counter[str] = Counter()
    parse_seconds: list[float] = []
    task_results: dict[str, dict[str, Any]] = {}
    legacy_correct_total = 0
    strict_correct_total = 0
    completed_total = 0
    for identity, approved in approved_by_id.items():
        expected = _expected(approved)
        abs_tol, rel_tol = _tolerances(approved)
        legacy_correct = 0
        strict_correct = 0
        completed = 0
        observed = trajectories.get(identity, [])
        for trajectory in observed:
            answer = extract_final_assistant_answer(str(trajectory.get("output") or ""))
            usable = bool(answer) and not bool(trajectory.get("trajectory_timeout")) and not bool(trajectory.get("runtime_error"))
            if not usable:
                continue
            completed += 1
            legacy_ok = _legacy_table_correct(answer, expected, abs_tol, rel_tol)
            legacy_correct += int(legacy_ok)
            if legacy_ok:
                legacy_shape_counts.update(_legacy_shape(answer, expected, abs_tol, rel_tol))
            started = time.perf_counter()
            strict_ok, mode, _ = strict_table_answer_match(answer, expected, abs_tol, rel_tol)
            parse_seconds.append(time.perf_counter() - started)
            strict_correct += int(strict_ok)
            format_modes[mode] += 1
        task_results[identity] = {
            "observed": len(observed),
            "completed": completed,
            "legacy_correct": legacy_correct,
            "strict_correct": strict_correct,
            "legacy_mixed": legacy_correct > 0 and completed - legacy_correct > 0,
            "strict_mixed": strict_correct > 0 and completed - strict_correct > 0,
        }
        legacy_correct_total += legacy_correct
        strict_correct_total += strict_correct
        completed_total += completed

    covered = sum(result["observed"] > 0 for result in task_results.values())
    legacy_mixed = sum(result["legacy_mixed"] for result in task_results.values())
    strict_mixed = sum(result["strict_mixed"] for result in task_results.values())
    strict_source_counts: Counter[str] = Counter()
    strict_difficulty_counts: Counter[str] = Counter()
    qualified_rows: list[dict[str, Any]] = []
    for identity, source in approved_by_id.items():
        if not task_results[identity]["strict_mixed"]:
            continue
        source_extra = source.get("extra_info") or {}
        strict_source_counts[str(source_extra.get("source_version", "unknown"))] += 1
        strict_difficulty_counts[str(source_extra.get("difficulty_level", "unknown"))] += 1
        row = deepcopy(source)
        extra = dict(row.get("extra_info") or {})
        extra.update(
            {
                "strict_reward_replay_passed": True,
                "strict_reward_contract": REWARD_CONTRACT,
                "training_allowed": False,
                "promotion_allowed": False,
            }
        )
        row["extra_info"] = extra
        qualified_rows.append(row)
    _write_private_parquet(output_qualified_parquet, qualified_rows, approved_table.schema)

    checks = {
        "approved_count_exact": len(approved_rows) == expected_approved,
        "all_approved_tasks_have_trajectories": covered == len(approved_rows),
        "legacy_variance_reproduced_for_all": legacy_mixed == len(approved_rows),
        "strict_variance_preserved_for_all": strict_mixed == len(approved_rows),
        "qualified_output_exact": len(qualified_rows) == strict_mixed,
        "qualified_training_disabled": all(not (row.get("extra_info") or {}).get("training_allowed") for row in qualified_rows),
    }
    summary = {
        "contract": CONTRACT,
        "reward_contract": REWARD_CONTRACT,
        "host_label": host_label,
        "approved_tasks": len(approved_rows),
        "covered_tasks": covered,
        "observed_trajectories": sum(len(rows) for rows in trajectories.values()),
        "completed_trajectories": completed_total,
        "legacy_correct_trajectories": legacy_correct_total,
        "strict_correct_trajectories": strict_correct_total,
        "legacy_mixed_tasks": legacy_mixed,
        "strict_mixed_tasks": strict_mixed,
        "tasks_lost_to_strict_judge": legacy_mixed - strict_mixed,
        "qualified_private_rows": len(qualified_rows),
        "strict_mixed_by_source_version": dict(sorted(strict_source_counts.items())),
        "strict_mixed_by_difficulty": dict(sorted(strict_difficulty_counts.items())),
        "wave_trajectory_counts": dict(sorted(wave_counts.items())),
        "strict_parse_modes": dict(sorted(format_modes.items())),
        "legacy_correct_answer_shapes": dict(sorted(legacy_shape_counts.items())),
        "strict_parse_latency_ms": {
            "p50": round(1000 * median(parse_seconds), 6) if parse_seconds else 0.0,
            "p95": round(1000 * _percentile(parse_seconds, 0.95), 6),
            "max": round(1000 * max(parse_seconds), 6) if parse_seconds else 0.0,
        },
        "checks": checks,
        "gate_passed": all(checks.values()),
        "training_allowed": False,
        "promotion_allowed": False,
        "contains_prompts_gold_sql_task_ids_hashes_final_answers_or_tool_outputs": False,
    }
    output_safe_json.parent.mkdir(parents=True, exist_ok=True)
    output_safe_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved", type=Path, required=True)
    parser.add_argument(
        "--wave",
        action="append",
        nargs=3,
        metavar=("LABEL", "DATASET", "RUN_DIR"),
        required=True,
    )
    parser.add_argument("--output-safe-json", type=Path, required=True)
    parser.add_argument("--output-qualified-parquet", type=Path, required=True)
    parser.add_argument("--expected-approved", type=int, required=True)
    parser.add_argument("--host-label", required=True)
    args = parser.parse_args()
    summary = replay(
        args.approved,
        [(label, Path(dataset), Path(run_dir)) for label, dataset, run_dir in args.wave],
        args.output_safe_json,
        args.output_qualified_parquet,
        expected_approved=args.expected_approved,
        host_label=args.host_label,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if summary["gate_passed"] else 2)


if __name__ == "__main__":
    main()

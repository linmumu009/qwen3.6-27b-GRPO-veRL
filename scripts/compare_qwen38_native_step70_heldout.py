#!/usr/bin/env python3
"""Aggregate a safe native-vs-Step70 heldout comparison from private outcomes.

The script intentionally emits only counts, rates, and order-independent identity
fingerprints.  It never emits prompts, task IDs, verifier SQL, gold answers, or
per-task outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


MASK_256 = (1 << 256) - 1


def _read_wave_two_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            source = json.loads(line)
            rows.append(
                {
                    "extra_info": {
                        "instruction_sha256": source["instruction_sha256"],
                        "adaptive_samples_observed": 2,
                        "adaptive_correct_count": source["correct_count"],
                        "adaptive_completed_count": source["completed_count"],
                        "adaptive_timeout_count": source["trajectory_timeout_count"],
                        "adaptive_runtime_error_count": source["runtime_error_count"],
                    }
                }
            )
    return rows


def _accumulate_wave_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    accumulated: dict[str, dict[str, Any]] = {}
    additive_fields = (
        "adaptive_samples_observed",
        "adaptive_correct_count",
        "adaptive_completed_count",
        "adaptive_timeout_count",
        "adaptive_runtime_error_count",
    )
    for path in paths:
        for row in _read_wave_two_jsonl(path):
            identity = _identity(row)
            if identity not in accumulated:
                accumulated[identity] = row
                continue
            target = accumulated[identity]["extra_info"]
            source = row["extra_info"]
            for field in additive_fields:
                target[field] += source[field]
    return list(accumulated.values())


def _identity(row: dict[str, Any]) -> str:
    value = row["extra_info"]["instruction_sha256"]
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("missing or malformed instruction identity")
    return value


def _fingerprint(identities: Iterable[str]) -> dict[str, Any]:
    values = [int(hashlib.sha256(value.encode("ascii")).hexdigest(), 16) for value in identities]
    xor_value = 0
    for value in values:
        xor_value ^= value
    return {
        "count": len(values),
        "sum_mod_2_256": f"{sum(values) & MASK_256:064x}",
        "sum_squares_mod_2_256": f"{sum(value * value for value in values) & MASK_256:064x}",
        "xor": f"{xor_value:064x}",
    }


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    identities = [_identity(row) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate instruction identity")

    samples = [int(row["extra_info"]["adaptive_samples_observed"]) for row in rows]
    correct = [int(row["extra_info"]["adaptive_correct_count"]) for row in rows]
    completed = [int(row["extra_info"]["adaptive_completed_count"]) for row in rows]
    timeout = [int(row["extra_info"]["adaptive_timeout_count"]) for row in rows]
    runtime_error = [int(row["extra_info"]["adaptive_runtime_error_count"]) for row in rows]
    if any(value < 0 for values in (samples, correct, completed, timeout, runtime_error) for value in values):
        raise ValueError("negative adaptive count")
    if any(c > s for c, s in zip(correct, samples)):
        raise ValueError("correct count exceeds sample count")

    task_count = len(rows)
    sample_count = sum(samples)
    correct_count = sum(correct)
    return {
        "tasks": task_count,
        "trajectories": sample_count,
        "correct_trajectories": correct_count,
        "correct_trajectory_rate": correct_count / sample_count if sample_count else None,
        "completed_trajectories": sum(completed),
        "timeout_trajectories": sum(timeout),
        "runtime_error_trajectories": sum(runtime_error),
        "tasks_with_at_least_one_correct": sum(value > 0 for value in correct),
        "task_pass_rate": sum(value > 0 for value in correct) / task_count if task_count else None,
        "tasks_with_zero_correct": sum(value == 0 for value in correct),
        "tasks_all_sampled_correct": sum(value == total and total > 0 for value, total in zip(correct, samples)),
        "identity_fingerprint": _fingerprint(identities),
    }


def _filter_rows(rows: list[dict[str, Any]], allowed: set[str]) -> list[dict[str, Any]]:
    filtered = [row for row in rows if _identity(row) in allowed]
    if {_identity(row) for row in filtered} != allowed:
        raise ValueError("wave rows do not cover the requested heldout identity set")
    return filtered


def _is_mixed(row: dict[str, Any]) -> bool:
    info = row["extra_info"]
    correct = int(info["adaptive_correct_count"])
    completed = int(info["adaptive_completed_count"])
    return correct > 0 and completed > correct


def _version_partial(
    *,
    runs_root: Path,
    host: str,
    version: str,
    native_final_run: str,
    native_wave_run_prefix: str,
    step_run: str,
) -> dict[str, Any]:
    native_final_base = runs_root / native_final_run
    native_all_rows = _accumulate_wave_jsonl(
        runs_root
        / f"{native_wave_run_prefix}-{version}-wave{wave}"
        / "outcomes"
        / "per_task.sensitive.jsonl"
        for wave in (2, 4, 6)
    )
    native_final_rows = [row for row in native_all_rows if not _is_mixed(row)]
    native_ids = {_identity(row) for row in native_final_rows}
    native_safe_summary = json.loads(
        (native_final_base / version / "adaptive_final_safe_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if len(native_final_rows) != int(native_safe_summary["unresolved_after_six_tasks"]):
        raise ValueError("reconstructed native unresolved count differs from safe summary")
    native_first_two_rows = _filter_rows(
        _read_wave_two_jsonl(
            runs_root
            / f"{native_wave_run_prefix}-{version}-wave2"
            / "outcomes"
            / "per_task.sensitive.jsonl"
        ),
        native_ids,
    )

    step_base = runs_root / step_run / host
    step_final_rows = _accumulate_wave_jsonl(
        step_base
        / "waves"
        / f"{step_run}-{host}-{version}-wave{wave}"
        / "outcomes"
        / "per_task.sensitive.jsonl"
        for wave in (2, 4, 6)
    )
    step_first_two_rows = _read_wave_two_jsonl(
        step_base
        / "waves"
        / f"{step_run}-{host}-{version}-wave2"
        / "outcomes"
        / "per_task.sensitive.jsonl"
    )
    if {_identity(row) for row in step_final_rows} != {_identity(row) for row in step_first_two_rows}:
        raise ValueError("Step70 first-two and final identity sets differ")
    step_safe_summary = json.loads(
        (step_base / "final" / version / "adaptive_final_safe_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if len(step_final_rows) != int(step_safe_summary["initial_tasks"]):
        raise ValueError("reconstructed Step70 task count differs from safe summary")

    return {
        "native": {
            "fixed_first_two": _metrics(native_first_two_rows),
            "adaptive_final": _metrics(native_final_rows),
        },
        "step70": {
            "fixed_first_two": _metrics(step_first_two_rows),
            "adaptive_final": _metrics(step_final_rows),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--host", choices=("m00", "m05", "m06"), required=True)
    parser.add_argument("--native-final-run", required=True)
    parser.add_argument("--native-wave-run-prefix", required=True)
    parser.add_argument("--step-run", required=True)
    parser.add_argument("--versions", nargs="+", default=("v15", "v20"))
    args = parser.parse_args()

    result = {
        "contract": "qwen38-native-step70-heldout-partial-v1",
        "host": args.host,
        "versions": {
            version: _version_partial(
                runs_root=args.runs_root,
                host=args.host,
                version=version,
                native_final_run=args.native_final_run,
                native_wave_run_prefix=args.native_wave_run_prefix,
                step_run=args.step_run,
            )
            for version in args.versions
        },
        "contains_prompts_gold_sql_task_ids_tool_outputs_or_server_paths": False,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

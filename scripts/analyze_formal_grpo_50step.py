#!/usr/bin/env python3
"""Audit a formal 50-step GRPO run without reading or modifying model state."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable


METRIC_RE = re.compile(
    r"(?:^| - )([A-Za-z0-9_./@-]+):(?:np\.(?:float64|int32|int64)\()?([-+0-9.eE]+)\)?"
)
TOOL_CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
QWEN_BASH_RE = re.compile(
    r"<tool_call>\s*<function=bash>\s*<parameter=command>\s*(.*?)\s*</parameter>\s*</function>\s*</tool_call>",
    re.DOTALL,
)
FLOAT_RE = r"[-+0-9.eE]+"
TRAIN_STAGE_RE = re.compile(
    rf"\[LLIN_TRAIN_STAGE\] step=(\d+) "
    rf"queue_wait_s=({FLOAT_RE}) deserialize_s=({FLOAT_RE}) assemble_s=({FLOAT_RE}) "
    rf"reward_s=({FLOAT_RE}) old_log_prob_s=({FLOAT_RE}) ref_log_prob_s=({FLOAT_RE}) "
    rf"adv_s=({FLOAT_RE}) update_actor_s=({FLOAT_RE}) step_s=({FLOAT_RE})"
)
NETWORK_RE = re.compile(
    r"(?:^|[;&|()\s])(?:curl|wget|ssh|scp|nc|ncat|telnet|ftp|git\s+clone|"
    r"pip\s+install|apt(?:-get)?|dnf|yum)\b",
    re.IGNORECASE,
)
DESTRUCTIVE_RE = re.compile(
    r"(?:\brm\s+-[^\n]*r[^\n]*\s+/(?:\s|$)|\bmkfs\b|\bdd\s+if=|"
    r"\b(?:shutdown|reboot|docker|podman|mount|umount|kill|killall|pkill)\b)",
    re.IGNORECASE,
)
ESCAPE_RE = re.compile(
    r"(?:^|[\s'\"=])/(?:etc|proc|sys|dev|root|home|data|data3|models|"
    r"pi_sandbox|usr/local/Ascend)(?:/|\s|$)",
    re.IGNORECASE,
)
PYTHON_NETWORK_RE = re.compile(
    r"(?:socket|urllib|requests|httpx|aiohttp|ftplib|smtplib)\s*[.(]",
    re.IGNORECASE,
)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "n": len(clean),
        "min": min(clean) if clean else None,
        "mean": fmean(clean) if clean else None,
        "median": median(clean) if clean else None,
        "p90": percentile(clean, 0.90),
        "p95": percentile(clean, 0.95),
        "max": max(clean) if clean else None,
    }


def prompt_digest(value: Any) -> str:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def iter_rollout_files(directory: Path) -> Iterable[tuple[int, Path]]:
    paths = []
    for path in directory.glob("*.jsonl"):
        try:
            step = int(path.stem)
        except ValueError:
            continue
        paths.append((step, path))
    yield from sorted(paths)


def iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def as_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, 0.0)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def extract_bash_commands(output: str) -> list[str]:
    commands: list[str] = []
    qwen_matches = list(QWEN_BASH_RE.finditer(output or ""))
    if qwen_matches:
        return [match.group(1).strip() for match in qwen_matches]
    for match in TOOL_CALL_RE.finditer(output or ""):
        payload = match.group(0)[len("<tool_call>") : -len("</tool_call>")].strip()
        try:
            call = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if call.get("name") != "bash":
            continue
        arguments = call.get("arguments") or {}
        command = arguments.get("command")
        if isinstance(command, str):
            commands.append(command)
    return commands


def unsafe_reasons(command: str) -> list[str]:
    visible = command.replace("/workspace", "")
    patterns = {
        "network": NETWORK_RE,
        "destructive": DESTRUCTIVE_RE,
        "host_path_escape": ESCAPE_RE,
        "python_network": PYTHON_NETWORK_RE,
    }
    return [name for name, pattern in patterns.items() if pattern.search(visible)]


def expected_reward(row: dict[str, Any]) -> float:
    safe = as_float(row, "safe") > 0
    protocol = as_float(row, "valid_tool_protocol") > 0
    if not safe or not protocol:
        return 0.0
    if "boss_reward" in row or "evidence_reward" in row:
        if as_float(row, "gold_sql_verified") <= 0:
            return 0.0
        return round(
            0.70 * as_float(row, "boss_reward")
            + 0.30 * as_float(row, "evidence_reward"),
            6,
        )
    return round(
        0.60 * as_float(row, "final_answer_correct")
        + 0.25 * float(
            as_float(row, "sql_evidence_correct") > 0
            and as_float(row, "successful_bash") > 0
        )
        + 0.10 * as_float(row, "required_table_used")
        + 0.05 * as_float(row, "has_final_answer"),
        6,
    )


def summarize_rollouts(directory: Path, expected_steps: int, expected_rows_per_step: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    step_counts: dict[int, int] = {}
    groups: list[dict[str, Any]] = []
    prompt_exposures: Counter[str] = Counter()

    for step, path in iter_rollout_files(directory):
        step_rows = list(iter_rows(path))
        step_counts[step] = len(step_rows)
        for row in step_rows:
            row["_step"] = step
            row["_prompt"] = prompt_digest(row.get("input"))
        rows.extend(step_rows)
        by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in step_rows:
            by_prompt[row["_prompt"]].append(row)
        for digest, group_rows in by_prompt.items():
            scores = [as_float(row, "score") for row in group_rows]
            prompt_exposures[digest] += 1
            groups.append(
                {
                    "step": step,
                    "prompt": digest,
                    "size": len(group_rows),
                    "unique_scores": len(set(scores)),
                    "score_variance": (
                        fmean((score - fmean(scores)) ** 2 for score in scores) if scores else 0.0
                    ),
                    "any_final_correct": any(as_float(row, "final_answer_correct") > 0 for row in group_rows),
                    "any_sql_correct": any(as_float(row, "sql_evidence_correct") > 0 for row in group_rows),
                    "any_strict_correct": any(as_float(row, "acc") > 0 for row in group_rows),
                }
            )

    component_keys = (
        "acc",
        "boss_reward",
        "boss_answer_correct",
        "boss_numbers_match",
        "evidence_reward",
        "final_answer_correct",
        "sql_evidence_correct",
        "required_table_used",
        "successful_bash",
        "safe",
        "valid_tool_protocol",
        "has_final_answer",
        "gold_sql_verified",
        "online_eligible",
    )
    score_values = [as_float(row, "score") for row in rows]
    mismatches = [
        {"step": int(row["_step"]), "recorded": as_float(row, "score"), "expected": expected_reward(row)}
        for row in rows
        if not math.isclose(as_float(row, "score"), expected_reward(row), abs_tol=1e-8)
    ]
    tool_calls = [len(TOOL_CALL_RE.findall(str(row.get("output") or ""))) for row in rows]
    error_markers = [
        len(re.findall(r"(?:command failed|no such option|unrecognized option|usage: sqlite3)", str(row.get("output") or ""), re.IGNORECASE))
        for row in rows
    ]
    score_counts = Counter(f"{value:.6f}" for value in score_values)
    verifier_errors = [str(row.get("verifier_error") or "") for row in rows if row.get("verifier_error")]
    bash_commands = [
        command
        for row in rows
        for command in extract_bash_commands(str(row.get("output") or ""))
    ]
    unsafe_command_reasons = Counter(
        reason for command in bash_commands for reason in unsafe_reasons(command)
    )
    rows_with_reconstructed_unsafe = sum(
        any(unsafe_reasons(command) for command in extract_bash_commands(str(row.get("output") or "")))
        for row in rows
    )

    windows = []
    for start in range(1, expected_steps + 1, 10):
        end = min(expected_steps, start + 9)
        window_rows = [row for row in rows if start <= int(row["_step"]) <= end]
        windows.append(
            {
                "steps": f"{start}-{end}",
                "rows": len(window_rows),
                "score_mean": fmean(as_float(row, "score") for row in window_rows) if window_rows else None,
                **{
                    key: sum(as_float(row, key) for row in window_rows) / len(window_rows)
                    if window_rows
                    else None
                    for key in component_keys
                },
            }
        )

    zero_variance = [group for group in groups if group["unique_scores"] <= 1]
    invalid_group_sizes = [group for group in groups if group["size"] != 4]
    expected_step_set = set(range(1, expected_steps + 1))
    actual_step_set = set(step_counts)
    return {
        "integrity": {
            "files": len(step_counts),
            "rows": len(rows),
            "missing_steps": sorted(expected_step_set - actual_step_set),
            "unexpected_steps": sorted(actual_step_set - expected_step_set),
            "wrong_rows_per_step": {
                str(step): count for step, count in step_counts.items() if count != expected_rows_per_step
            },
            "reward_formula_mismatches": len(mismatches),
            "verifier_errors": len(verifier_errors),
        },
        "score": describe(score_values),
        "score_distribution": dict(sorted(score_counts.items())),
        "components": {
            key: {
                "count": sum(as_float(row, key) > 0 for row in rows),
                "rate": sum(as_float(row, key) > 0 for row in rows) / len(rows) if rows else None,
            }
            for key in component_keys
        },
        "ten_step_windows": windows,
        "groups": {
            "total": len(groups),
            "invalid_size": len(invalid_group_sizes),
            "zero_reward_variance": len(zero_variance),
            "zero_reward_variance_rate": len(zero_variance) / len(groups) if groups else None,
            "with_any_final_correct": sum(group["any_final_correct"] for group in groups),
            "with_any_sql_correct": sum(group["any_sql_correct"] for group in groups),
            "with_any_strict_correct": sum(group["any_strict_correct"] for group in groups),
            "reward_variance": describe(group["score_variance"] for group in groups),
        },
        "prompts": {
            "unique": len(prompt_exposures),
            "group_exposures": describe(prompt_exposures.values()),
        },
        "behavior": {
            "tool_calls": describe(tool_calls),
            "rows_with_unsupported_cli_or_command_error": sum(value > 0 for value in error_markers),
            "unsupported_cli_or_command_error_markers": sum(error_markers),
            "rows_mentioning_sqlite_column_flag": sum(
                "sqlite3 -column" in str(row.get("output") or "") for row in rows
            ),
            "rows_using_write_or_edit": sum(
                bool(re.search(r'"name"\s*:\s*"(?:write|edit)"', str(row.get("output") or "")))
                for row in rows
            ),
            "bash_commands": len(bash_commands),
            "unsafe_command_reasons": dict(unsafe_command_reasons),
            "rows_with_reconstructed_unsafe_command": rows_with_reconstructed_unsafe,
            "safe_field_vs_reconstruction_mismatches": sum(
                (as_float(row, "safe") > 0)
                == any(
                    unsafe_reasons(command)
                    for command in extract_bash_commands(str(row.get("output") or ""))
                )
                for row in rows
            ),
            "output_chars": describe(len(str(row.get("output") or "")) for row in rows),
        },
        "reward_mismatch_examples": mismatches[:5],
    }


def parse_driver(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    steps = []
    validations: list[dict[str, Any]] = []
    for line in text.splitlines():
        if "training/global_step:" in line:
            metrics = {key: float(value) for key, value in METRIC_RE.findall(line)}
            if "training/global_step" in metrics:
                steps.append(metrics)
        if "rollouter/validate_time" in line and "val-core/" in line:
            metrics = {key: float(value) for key, value in METRIC_RE.findall(line)}
            step_match = re.search(r"\bstep:(\d+)\b", line)
            validations.append(
                {"step": int(step_match.group(1)) if step_match else None, **metrics}
            )
    train_stages = [
        {
            key: value
            for key, value in zip(
                (
                    "step",
                    "queue_wait_s",
                    "deserialize_s",
                    "assemble_s",
                    "reward_s",
                    "old_log_prob_s",
                    "ref_log_prob_s",
                    "adv_s",
                    "update_actor_s",
                    "step_s",
                ),
                (int(match.group(1)), *(float(value) for value in match.groups()[1:])),
                strict=True,
            )
        }
        for match in TRAIN_STAGE_RE.finditer(text)
    ]

    metric_keys = (
        "critic/score/mean",
        "actor/ppo_kl",
        "actor/pg_clipfrac",
        "actor/grad_norm",
        "actor/pg_loss",
        "timing_s/update_actor",
        "timing_s/step",
        "timing_s/timing_s/param_sync",
        "response_length/mean",
        "response_length/clip_ratio",
        "num_turns/mean",
    )
    step_ids = [int(step["training/global_step"]) for step in steps]
    return {
        "steps": len(steps),
        "step_ids": step_ids,
        "missing_step_metrics": sorted(set(range(1, 51)) - set(step_ids)),
        "metrics": {
            key: describe(step[key] for step in steps if key in step) for key in metric_keys
        },
        "first_10_vs_last_10": {
            key: {
                "first_10": describe(step[key] for step in steps[:10] if key in step),
                "last_10": describe(step[key] for step in steps[-10:] if key in step),
            }
            for key in metric_keys
        },
        "validation": validations,
        "fully_async_stage_timing": {
            "records": len(train_stages),
            **{
                key: describe(stage[key] for stage in train_stages)
                for key in (
                    "queue_wait_s",
                    "deserialize_s",
                    "assemble_s",
                    "reward_s",
                    "old_log_prob_s",
                    "ref_log_prob_s",
                    "adv_s",
                    "update_actor_s",
                    "step_s",
                )
            },
            "queue_wait_share": (
                sum(stage["queue_wait_s"] for stage in train_stages)
                / sum(stage["step_s"] for stage in train_stages)
                if train_stages and sum(stage["step_s"] for stage in train_stages)
                else None
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--driver-log", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=50)
    parser.add_argument("--expected-rows-per-step", type=int, default=16)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = {
        "rollouts": summarize_rollouts(
            args.rollout_dir,
            expected_steps=args.expected_steps,
            expected_rows_per_step=args.expected_rows_per_step,
        ),
        "driver": parse_driver(args.driver_log),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

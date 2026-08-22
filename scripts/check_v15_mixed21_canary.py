#!/usr/bin/env python3
"""Gate collapse-safe mixed21 canary and final validation from safe logs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from statistics import fmean
from typing import Any


CONTRACT = "v15-mixed21-canary-gate-v1"
STEP_RE = re.compile(r"\bstep:(\d+)\b")
METRIC_RE = re.compile(
    r"(?:^| - )([A-Za-z0-9_./@-]+):(?:np\.(?:float64|int32|int64)\()?([-+0-9.eE]+)\)?"
)


def parse_logs(paths: list[Path]) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    training: dict[int, dict[str, float]] = {}
    validation: dict[int, dict[str, float]] = {}
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                step_match = STEP_RE.search(line)
                metrics = {key: float(value) for key, value in METRIC_RE.findall(line)}
                if not metrics:
                    continue
                if "training/global_step" in metrics:
                    training[int(metrics["training/global_step"])] = metrics
                if step_match and "rollouter/validate_time" in metrics and any(
                    key.startswith("val-core/") for key in metrics
                ):
                    validation[int(step_match.group(1))] = metrics
    return training, validation


def validation_value(metrics: dict[str, float], suffix: str) -> float | None:
    values = [
        value
        for key, value in metrics.items()
        if key.startswith("val-core/") and key.endswith(suffix)
    ]
    return fmean(values) if values else None


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def gate(
    logs: list[Path],
    expected_steps: list[int],
    baseline_step: int,
    max_abs_kl: float,
    max_grad_norm: float,
    mode: str,
) -> dict[str, Any]:
    training, validation = parse_logs(logs)
    checks: dict[str, bool] = {}
    reasons: list[str] = []
    baseline = validation.get(baseline_step)
    checks["baseline_validation_present"] = baseline is not None
    baseline_acc = validation_value(baseline or {}, "/acc/mean@1")
    baseline_final = validation_value(baseline or {}, "/final_answer_correct/mean@1")
    checks["baseline_strict_acc_present"] = baseline_acc is not None and finite(baseline_acc)

    validation_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for step in expected_steps:
        val = validation.get(step)
        acc = validation_value(val or {}, "/acc/mean@1")
        final_correct = validation_value(val or {}, "/final_answer_correct/mean@1")
        val_ok = (
            val is not None
            and baseline_acc is not None
            and acc is not None
            and finite(acc)
            and acc + 1e-12 >= baseline_acc
        )
        if baseline_final is not None:
            val_ok = bool(
                val_ok
                and final_correct is not None
                and finite(final_correct)
                and final_correct + 1e-12 >= baseline_final
            )
        checks[f"validation_step_{step}_not_below_baseline"] = val_ok
        validation_rows.append(
            {
                "step": step,
                "strict_acc_mean": acc,
                "final_answer_correct_mean": final_correct,
                "not_below_baseline": val_ok,
            }
        )

        train = training.get(step)
        if train is None:
            checks[f"training_step_{step}_metrics_present"] = False
            continue
        mixed = train.get("grpo/strict_mixed_groups")
        skipped = train.get("grpo/skipped_uniform_groups")
        total = train.get("grpo/total_groups")
        grad = train.get("actor/grad_norm")
        update_skipped = train.get("actor/update_skipped_no_strict_mixed")
        reward = train.get("critic/score/mean")
        kl_values = {
            key: value
            for key, value in train.items()
            if "kl" in key.casefold() and not key.startswith("timing")
        }
        shape_ok = (
            finite(mixed)
            and finite(skipped)
            and finite(total)
            and int(float(total)) == 2
            and int(float(mixed) + float(skipped)) == 2
        )
        if mixed is not None and float(mixed) == 0.0:
            update_ok = update_skipped == 1.0 and grad == 0.0
        else:
            update_ok = (
                update_skipped == 0.0
                and finite(grad)
                and 0.0 < float(grad) <= max_grad_norm
            )
        reward_ok = finite(reward) and 0.0 <= float(reward) <= 1.0
        kl_ok = bool(kl_values) and all(
            finite(value) and abs(float(value)) <= max_abs_kl for value in kl_values.values()
        )
        checks[f"training_step_{step}_group_accounting"] = shape_ok
        checks[f"training_step_{step}_update_behavior"] = update_ok
        checks[f"training_step_{step}_reward_range"] = reward_ok
        checks[f"training_step_{step}_kl_finite_bounded"] = kl_ok
        training_rows.append(
            {
                "step": step,
                "strict_mixed_groups": mixed,
                "skipped_uniform_groups": skipped,
                "skipped_all_wrong_groups": train.get("grpo/skipped_all_wrong_groups"),
                "skipped_all_correct_groups": train.get("grpo/skipped_all_correct_groups"),
                "grad_norm": grad,
                "reward_mean": reward,
                "update_skipped": update_skipped,
                "kl_metrics": kl_values,
            }
        )

    if mode == "canary":
        checks["all_canary_checkpoints_present"] = all(
            (logs[0].parent.parent / "checkpoints" / f"global_step_{step}").is_dir()
            for step in expected_steps
        )
        checks["at_least_one_effective_mixed_group"] = any(
            float(row.get("strict_mixed_groups") or 0.0) > 0 for row in training_rows
        )

    for name, passed in checks.items():
        if not passed:
            reasons.append(name)
    return {
        "contract": CONTRACT,
        "mode": mode,
        "passed": not reasons,
        "baseline_step": baseline_step,
        "baseline_strict_acc_mean": baseline_acc,
        "baseline_final_answer_correct_mean": baseline_final,
        "expected_steps": expected_steps,
        "max_abs_kl": max_abs_kl,
        "max_grad_norm": max_grad_norm,
        "checks": checks,
        "failed_checks": reasons,
        "validation": validation_rows,
        "training": training_rows,
        "contains_sensitive_data": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver-log", type=Path, action="append", required=True)
    parser.add_argument("--expected-step", type=int, action="append", required=True)
    parser.add_argument("--baseline-step", type=int, default=0)
    parser.add_argument("--max-abs-kl", type=float, default=0.10)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--mode", choices=("canary", "final"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = gate(
        args.driver_log,
        args.expected_step,
        args.baseline_step,
        args.max_abs_kl,
        args.max_grad_norm,
        args.mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

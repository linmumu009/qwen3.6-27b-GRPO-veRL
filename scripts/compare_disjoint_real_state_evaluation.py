#!/usr/bin/env python3
"""Apply the preregistered future-candidate gate on frozen eval22 margins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONTRACT = "disjoint-real-state-eval22-margin-baseline-v1"


def _index(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in result.get("per_task") or []:
        task = str(row.get("task_id") or "")
        if not task or task in output:
            raise ValueError(f"missing or duplicate eval22 task: {task!r}")
        output[task] = row
    return output


def _earlier(candidate: int | None, baseline: int | None) -> bool:
    if baseline is None:
        return candidate is not None
    return candidate is not None and candidate < baseline


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    for label, result in (("baseline", baseline), ("candidate", candidate)):
        if result.get("contract") != CONTRACT:
            raise ValueError(f"{label} is not an eval22 margin result")
        if result.get("evaluation_only") is not True:
            raise ValueError(f"{label} is not evaluation-only")
        if result.get("training_allowed") is not False or result.get("promotion_allowed") is not False:
            raise ValueError(f"{label} margin result is not fail closed")
    if baseline.get("selection_bias") != candidate.get("selection_bias"):
        raise ValueError("eval22 selection contract changed")
    baseline_rows = _index(baseline)
    candidate_rows = _index(candidate)
    if set(baseline_rows) != set(candidate_rows):
        raise ValueError("eval22 task identities differ")

    gate = baseline["future_candidate_gate"]
    if gate != candidate["future_candidate_gate"]:
        raise ValueError("future candidate gate changed after baseline freeze")
    chosen_preferred = int(candidate["semantic_delta_margin"]["chosen_preferred"])
    margin_improved = 0
    earlier_regressions = 0
    per_task: list[dict[str, Any]] = []
    for task in sorted(baseline_rows):
        before = baseline_rows[task]
        after = candidate_rows[task]
        improved = float(after["semantic_delta_log_probability_margin_per_token"]) > float(
            before["semantic_delta_log_probability_margin_per_token"]
        )
        earlier = _earlier(after.get("first_nongreedy_offset"), before.get("first_nongreedy_offset"))
        margin_improved += int(improved)
        earlier_regressions += int(earlier)
        per_task.append(
            {
                "task_id": task,
                "margin_before": before["semantic_delta_log_probability_margin_per_token"],
                "margin_after": after["semantic_delta_log_probability_margin_per_token"],
                "margin_improved": improved,
                "earlier_first_nongreedy_regression": earlier,
            }
        )

    checks = {
        "chosen_preferred": {
            "observed": chosen_preferred,
            "required_min": int(gate["chosen_preferred_min"]),
            "passed": chosen_preferred >= int(gate["chosen_preferred_min"]),
        },
        "per_task_margin_improved": {
            "observed": margin_improved,
            "required_min": int(gate["per_task_margin_improved_min"]),
            "passed": margin_improved >= int(gate["per_task_margin_improved_min"]),
        },
        "new_earlier_first_nongreedy_regressions": {
            "observed": earlier_regressions,
            "allowed_max": int(gate["new_earlier_first_nongreedy_regressions_max"]),
            "passed": earlier_regressions
            <= int(gate["new_earlier_first_nongreedy_regressions_max"]),
        },
    }
    passed = all(check["passed"] for check in checks.values())
    return {
        "contract": "disjoint-real-state-eval22-future-candidate-comparison-v1",
        "baseline_model_label": baseline.get("evaluated_model_label"),
        "candidate_model_label": candidate.get("evaluated_model_label"),
        "task_count": len(per_task),
        "checks": checks,
        "gate_passed": passed,
        "decision": {
            "full64_pareto_replay_allowed": passed,
            "additional_training_allowed": False,
            "promotion_allowed": False,
            "selected_next_action": (
                "run_same64_full25_pareto_gate"
                if passed
                else "stop_candidate_without_full_rollout_or_additional_training"
            ),
        },
        "per_task": per_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.candidate.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_task"}, indent=2))


if __name__ == "__main__":
    main()

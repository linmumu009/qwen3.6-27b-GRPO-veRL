#!/usr/bin/env python3
"""Apply the frozen post-training gates to baseline/post pairwise margins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(baseline: dict, post: dict, *, threshold: int = 12) -> dict:
    expected_contract = "semantic-delta-margin-gate-result-v2"
    if baseline.get("contract") != expected_contract or post.get("contract") != expected_contract:
        raise ValueError("pairwise comparison requires semantic-delta result v2")
    if baseline.get("task_count") != 16 or post.get("task_count") != 16:
        raise ValueError("pairwise comparison requires 16 tasks")
    before = {str(row["task_id"]): row for row in baseline["per_task"]}
    after = {str(row["task_id"]): row for row in post["per_task"]}
    if set(before) != set(after) or len(before) != 16:
        raise ValueError("baseline/post margin task IDs differ")
    improved = sum(
        after[task_id]["semantic_delta_log_probability_margin_per_token"]
        > before[task_id]["semantic_delta_log_probability_margin_per_token"]
        for task_id in before
    )
    preferred = int(post["semantic_delta_margin"]["chosen_preferred"])
    audit = post["frozen_critical_token_audit"]
    earlier = int(audit["new_earlier_first_nongreedy_regressions"])
    invalid = int(audit["invalid_targets_at_frozen_offset"])
    passed = preferred >= threshold and improved >= threshold and earlier == 0 and invalid == 0
    return {
        "contract": "semantic-delta-pairwise-canary-comparison-v1",
        "task_count": 16,
        "baseline_chosen_preferred": int(baseline["semantic_delta_margin"]["chosen_preferred"]),
        "post_chosen_preferred": preferred,
        "chosen_preferred_threshold": threshold,
        "per_task_margin_improved": improved,
        "per_task_margin_improved_threshold": threshold,
        "baseline_mean_margin": baseline["semantic_delta_margin"]["mean_margin"],
        "post_mean_margin": post["semantic_delta_margin"]["mean_margin"],
        "new_earlier_first_nongreedy_regressions": earlier,
        "invalid_targets_at_frozen_offset": invalid,
        "passed": passed,
        "decision": (
            "eligible_for_short_one_turn_semantic_plan_replay"
            if passed
            else "stop_no_replay_and_no_additional_pairwise_steps"
        ),
        "promotion_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=int, default=12)
    args = parser.parse_args()
    result = compare(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.post.read_text(encoding="utf-8")),
        threshold=args.threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

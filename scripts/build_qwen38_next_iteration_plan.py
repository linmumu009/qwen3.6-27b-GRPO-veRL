#!/usr/bin/env python3
"""Build the safe, aggregate decision record for the next Qwen3.8 GRPO iteration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("invalid binomial inputs")
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt(
        rate * (1 - rate) / total + z * z / (4 * total * total)
    ) / denominator
    return center - half_width, center + half_width


def build_plan(
    strict_native: dict[str, Any],
    train70_comparison: dict[str, Any],
    heldout: dict[str, Any],
    generated_corpus: dict[str, Any],
) -> dict[str, Any]:
    native_population = strict_native["population"]
    native_result = strict_native["result"]
    trajectory = train70_comparison["trajectory_evidence"]
    heldout_all = heldout["native_vs_step70_exact_heldout"]["all_versions"]
    generated = generated_corpus["improved_results"]

    screened_tasks = 1500
    strict_candidates = native_result["strict_mixed_tasks"]
    strict_yield = strict_candidates / screened_tasks
    yield_low, yield_high = _wilson_interval(strict_candidates, screened_tasks)
    fresh_acquisition_tasks = 2000

    checks = {
        "native_strict_baseline_is_20": strict_candidates == 20,
        "legacy_70_was_not_strict": native_result["legacy_mixed_tasks"] == 70
        and strict_candidates < native_result["legacy_mixed_tasks"],
        "step70_strict_mixed_is_15": train70_comparison["step70_strict_mixed_tasks"] == 15,
        "transition_partition_is_70": sum(
            train70_comparison[key]
            for key in ("retained_tasks", "lost_tasks", "gained_tasks", "neither_tasks")
        )
        == 70,
        "heldout_identity_count_is_1430": heldout_all["tasks"] == 1430,
        "generated_corpus_is_4000": generated["tasks"] == 4000,
        "generated_corpus_remains_training_disabled": generated["training_allowed"] is False,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"evidence reconciliation failed: {failed}")

    return {
        "contract": "llin-qwen38-next-iteration-plan-safe-v1",
        "date": "2026-08-20",
        "decision": {
            "step70_disposition": "freeze_for_forensics_not_promotion_or_resume",
            "next_training_base": "qwen38-27b-native-hf-step0",
            "old_70_disposition": "diagnostic_only_not_training",
            "heldout_1430_disposition": "evaluation_only_never_training",
            "heldout_18_disposition": "evaluation_and_error_taxonomy_only_never_training",
            "reward": "retain_banded_v2_strict_table_v1_and_rebuild_surrounding_gates",
            "human_review_required": False,
        },
        "evidence": {
            "legacy_candidate_tasks": native_result["legacy_mixed_tasks"],
            "native_strict_mixed_tasks": strict_candidates,
            "step70_strict_mixed_tasks": train70_comparison["step70_strict_mixed_tasks"],
            "native_strict_correct_trajectories": trajectory["native"]["strict_correct_trajectories"],
            "native_completed_trajectories": trajectory["native"]["completed_trajectories"],
            "native_strict_correct_rate_over_completed": trajectory["native"]["strict_correct_rate_over_completed"],
            "step70_strict_correct_trajectories": trajectory["step70"]["strict_correct_trajectories"],
            "step70_completed_trajectories": trajectory["step70"]["completed_trajectories"],
            "step70_strict_correct_rate_over_completed": trajectory["step70"]["strict_correct_rate_over_completed"],
            "strict_correct_rate_absolute_change": trajectory["strict_correct_rate_absolute_change"],
            "native_heldout_task_passes_at_max_6": heldout_all["native_tasks_with_at_least_one_correct"],
            "step70_heldout_task_passes_at_max_6": heldout_all["step70_tasks_with_at_least_one_correct"],
            "heldout_tasks": heldout_all["tasks"],
            "heldout_absolute_pass_rate_change": (
                heldout_all["step70_task_pass_rate"] - heldout_all["native_task_pass_rate"]
            ),
            "comparison_caveat": "The original-70 and heldout comparisons use identical task identities but are not request-by-request seed-matched, so they are directional point estimates rather than causal proof.",
        },
        "fresh_data_plan": {
            "frozen_new_eval_sandbox": "v22",
            "frozen_new_eval_tasks": 500,
            "acquisition_sandboxes": ["v23", "v24", "v25", "v26"],
            "fresh_acquisition_tasks": fresh_acquisition_tasks,
            "historical_strict_candidate_yield": strict_yield,
            "historical_yield_wilson_95_low": yield_low,
            "historical_yield_wilson_95_high": yield_high,
            "projected_strict_candidates_point": fresh_acquisition_tasks * strict_yield,
            "projected_strict_candidates_rate_interval_low": fresh_acquisition_tasks * yield_low,
            "projected_strict_candidates_rate_interval_high": fresh_acquisition_tasks * yield_high,
            "projection_caveat": "The acceptance gate is stricter than the historical at-least-one-correct mixed gate, so the accepted robust-task count may be lower than this projection.",
            "minimum_canary_tasks": 24,
            "ideal_first_scale_tasks": 32,
            "fallback_if_below_minimum": "generate_and_screen_a_fresh_v27_plus_batch_without_borrowing_from_v22_eval",
        },
        "automatic_data_gate": {
            "screening": "strict_adaptive_2_plus_2_plus_2_max_6",
            "confirmation": "add_2_samples_only_for_provisional_strict_mixed_candidates",
            "minimum_completed_strict_correct": 2,
            "minimum_completed_strict_wrong": 2,
            "runtime_errors_allowed": 0,
            "ambiguous_or_parse_failed_answers_allowed": 0,
            "queue_wait_counts_toward_trajectory_timeout": False,
            "trajectory_timeout_starts": "when_the_rollout_worker_begins_the_request",
            "timeout_use_for_data_variance": "reported_separately_not_counted_as_strict_wrong",
            "timeout_use_for_model_quality": "counted_in_requested_trajectory_denominator",
        },
        "canary_training": {
            "base": "qwen38-27b-native-hf-step0",
            "tasks": 24,
            "exposures_per_task": 1,
            "responses_per_group": 8,
            "groups_per_optimizer_step": 2,
            "optimizer_steps": 12,
            "prewarm_groups": 4,
            "minimum_nonzero_reward_variance_fraction_after_prewarm": 0.5,
            "abort_if_nonzero_reward_variance_fraction_below": 0.3,
            "runtime_errors_allowed": 0,
            "checkpoint_policy": "save_canary_final_only_but_retain_complete_model_metadata",
        },
        "evaluation_kpis": {
            "primary": "strict_task_pass_at_k_over_all_frozen_tasks",
            "secondary": "strict_correct_trajectories_over_all_requested_trajectories",
            "guardrails": [
                "completion_rate",
                "timeout_rate",
                "runtime_error_count",
                "generation_tokens_per_hour",
                "trajectory_wall_clock_p50_p95",
            ],
            "data_selection_metric_not_model_quality_metric": "strict_mixed_task_count",
            "paired_contract": "same_task_same_seed_same_sampling_parameters_native_vs_candidate",
            "cheap_gate": "v22_stratified_100_tasks_times_2_responses",
            "full_gate": "v22_all_500_tasks_adaptive_max_6_only_after_cheap_gate_passes",
            "scale_gate": {
                "paired_task_net_wins_minimum": 1,
                "strict_correct_requested_rate_must_not_decrease": True,
                "completion_rate_max_drop_percentage_points": 2.0,
                "timeout_rate_max_increase_percentage_points": 2.0,
                "runtime_errors_allowed": 0,
            },
        },
        "machine_plan": {
            "data_acquisition": {
                "machine_0": "TP4xDP3_on_12_available_cards",
                "machine_5": "TP4xDP4_on_16_cards",
                "machine_6": "TP4xDP4_on_16_cards",
            },
            "canary_training": {
                "machine_5": "16_card_trainer_TP4xPP2xCP2",
                "machine_6": "16_card_rollout_TP4xDP4_max16_per_replica",
                "machine_0": "continue_fresh_data_screening_then_run_paired_evaluation",
            },
            "add_machine_0_to_training_rollout_only_if": "a_two_step_probe_shows_rollout_utilization_above_85_percent_and_trainer_idle_above_15_percent",
        },
        "estimated_fast_path": {
            "data_and_reward_freeze": "1_to_2_hours",
            "fresh_screening": "about_1_day_but_candidate_yield_is_the_largest_uncertainty",
            "12_step_canary_and_two_stage_evaluation": "about_half_to_one_day",
            "total_if_24_robust_tasks_are_found_in_v23_to_v26": "about_1.5_to_2.5_days",
            "eta_rule": "replace_range_with_measured_eta_after_the_first_100_task_wave",
        },
        "report_delivery": {
            "canonical_artifact_json": "docs/qwen38_next_iteration_plan_20260820.artifact.json",
            "readable_markdown": "docs/qwen38_next_iteration_plan_20260820.md",
            "portable_html_generated": False,
            "portable_html_blocker": "The bundled portable reader entered fallback state during static chart extraction. The same reader_timeout reproduced on an existing repository artifact, so this is a local renderer/runtime blocker rather than an artifact-specific data validation failure.",
        },
        "automatic_checks": checks,
        "training_allowed": False,
        "promotion_allowed": False,
        "contains_prompts_gold_sql_task_ids_hashes_server_paths_final_answers_or_tool_outputs": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict-native",
        type=Path,
        default=Path("docs/qwen38_banded_v2_strict_reward_replay_20260818.safe.json"),
    )
    parser.add_argument(
        "--train70-comparison",
        type=Path,
        default=Path("docs/qwen38_native_vs_step70_original70_strict_20260820.safe.json"),
    )
    parser.add_argument(
        "--heldout",
        type=Path,
        default=Path("docs/qwen38_step70_heldout_v15_v20_v21_final_20260820.safe.json"),
    )
    parser.add_argument(
        "--generated-corpus",
        type=Path,
        default=Path("docs/boss_v15_semantic_mismatch_20260818.safe.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/qwen38_next_iteration_plan_20260820.safe.json"),
    )
    args = parser.parse_args()

    plan = build_plan(
        _load(args.strict_native),
        _load(args.train70_comparison),
        _load(args.heldout),
        _load(args.generated_corpus),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

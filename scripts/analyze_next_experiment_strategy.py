#!/usr/bin/env python3
"""Size 96K capacity and rank short experiments after the Step 120 trial."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from statistics import fmean
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.estimate_48k_capacity import CapacityInputs, estimate
from scripts.prepare_boss_exact_evaluation import qwen_output_to_openai_messages


CONTEXTS = (49_152, 65_536, 81_920, 98_304)
CONCURRENCY_LEVELS = (8, 12, 16, 24)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def completion_audit(validation_rows: list[dict[str, Any]], boss_rows: list[dict[str, Any]]) -> dict[str, Any]:
    boss_by_id = {str(row["task_id"]): row for row in boss_rows}
    details = []
    for row in validation_rows:
        key = str(row["gts"]["task_id"])
        _, audit = qwen_output_to_openai_messages(str(row["output"]))
        reward = boss_by_id[key]["reward"]
        details.append(
            {
                "task_id": key,
                "has_final_answer": bool(row.get("has_final_answer")),
                "assistant_turns": int(reward["efficiency_n_turns"]),
                "tool_calls": int(audit["tool_calls"]),
                "tool_responses": int(audit["tool_responses"]),
                "missing_tool_responses": int(audit["missing_tool_responses"]),
                "sql_calls": int(reward["efficiency_n_sql"]),
                "commands": int(reward["efficiency_n_cmds"]),
                "duplicate_commands": int(reward["efficiency_dup_cmd"]),
                "redundancy_oscillation": bool(reward["efficiency_redundancy_oscillation"]),
                "output_chars": len(str(row["output"])),
            }
        )

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "tasks": len(rows),
            "assistant_turns_mean": fmean(row["assistant_turns"] for row in rows),
            "sql_calls_mean": fmean(row["sql_calls"] for row in rows),
            "commands_mean": fmean(row["commands"] for row in rows),
            "duplicate_commands_mean": fmean(row["duplicate_commands"] for row in rows),
            "redundancy_oscillation_rate": fmean(float(row["redundancy_oscillation"]) for row in rows),
            "output_chars_mean": fmean(row["output_chars"] for row in rows),
        }

    incomplete = [row for row in details if not row["has_final_answer"]]
    complete = [row for row in details if row["has_final_answer"]]
    common_incomplete_turn_boundary = len({row["assistant_turns"] for row in incomplete}) == 1
    all_end_with_missing_tool_response = all(row["missing_tool_responses"] == 1 for row in incomplete)
    return {
        "rows": len(details),
        "complete_tasks": len(complete),
        "incomplete_tasks": len(incomplete),
        "incomplete": summarize(incomplete),
        "complete": summarize(complete),
        "incomplete_common_turn_boundary": common_incomplete_turn_boundary,
        "incomplete_turn_boundary": incomplete[0]["assistant_turns"] if common_incomplete_turn_boundary else None,
        "incomplete_all_end_with_one_missing_tool_response": all_end_with_missing_tool_response,
        "incomplete_details": incomplete,
    }


def capacity_scenarios(native_context_limit: int) -> dict[str, Any]:
    rows = []
    for context in CONTEXTS:
        training = estimate(CapacityInputs(target_context_tokens=context))["training"]
        per_sequence = estimate(CapacityInputs(target_context_tokens=context))["rollout_per_tp_rank"][
            "cache_per_48k_sequence"
        ]["total_gib"]
        for max_sequences in CONCURRENCY_LEVELS:
            inputs = replace(
                CapacityInputs(),
                target_context_tokens=context,
                rollout_max_seqs_per_replica=max_sequences,
                rollout_gpu_memory_utilization=0.80,
            )
            rollout = estimate(inputs)["rollout_per_tp_rank"]
            rows.append(
                {
                    "context_tokens": context,
                    "context_k": round(context / 1024),
                    "max_sequences_per_replica": max_sequences,
                    "training_planning_peak_gib": training["planning_peak_gib"],
                    "training_headroom_gib": training["headroom_gib"],
                    "cache_per_sequence_gib": per_sequence,
                    "rollout_planning_total_gib": rollout["planning_total_gib"],
                    "rollout_80pct_budget_gib": rollout["vllm_60pct_budget_gib"],
                    "rollout_budget_headroom_gib": rollout["budget_headroom_gib"],
                    "rollout_planning_fit": rollout["expected_to_fit"],
                }
            )

    cache_48 = next(row["cache_per_sequence_gib"] for row in rows if row["context_tokens"] == 49_152)
    cache_96 = next(row["cache_per_sequence_gib"] for row in rows if row["context_tokens"] == 98_304)
    delta_per_sequence = cache_96 - cache_48
    return {
        "model_native_max_position_embeddings": native_context_limit,
        "contexts_within_native_limit": all(context <= native_context_limit for context in CONTEXTS),
        "current_formal_rollout_hbm_gib": {"low": 53.8, "high": 56.1},
        "current_post_sync_hbm_gib": {"low": 54.4, "high": 54.7},
        "usable_hbm_gib": 61.27,
        "current_post_sync_headroom_gib": {"low": 10.8, "high": 11.1},
        "cache_increment_48k_to_96k_per_full_sequence_gib": delta_per_sequence,
        "calibrated_increment_by_concurrency_gib": {
            str(level): delta_per_sequence * level for level in CONCURRENCY_LEVELS
        },
        "scenario_rows": rows,
    }


def runtime_and_experiments(step120_summary: dict[str, Any]) -> dict[str, Any]:
    run = step120_summary["training_run"]
    step_s = float(run["mean_step_s"])
    validation_s = float(run["validation_time_s"])
    save_s = float(run["checkpoint_save_time_s"])
    training_cost = [
        {
            "steps": steps,
            "update_hours": step_s * steps / 3600,
            "with_full_val_and_save_hours": (step_s * steps + validation_s + save_s) / 3600,
        }
        for steps in (5, 10, 20, 100)
    ]
    experiments = [
        {
            "priority": 1,
            "experiment": "48K 强制收尾哨兵集",
            "training_steps": 0,
            "new_trajectories": 6,
            "estimated_gpu_hours_excluding_cold_start": "0.4–0.8",
            "change": "在第22轮或剩余4K token时禁止继续调用工具，要求基于已有证据直接作答",
            "decision_gate": "4道既有未收尾题至少救回2道，且2道已答对/高分题不退化",
        },
        {
            "priority": 2,
            "experiment": "64K + 32轮哨兵集",
            "training_steps": 0,
            "new_trajectories": 6,
            "estimated_gpu_hours_excluding_cold_start": "0.6–1.2",
            "change": "仅在强制收尾未充分救回时，增加到64K并把工具反馈上限提高到32轮",
            "decision_gate": "相对48K强制收尾额外救回至少1道，且单位题耗时增幅不超过60%",
        },
        {
            "priority": 3,
            "experiment": "96K 定向诊断",
            "training_steps": 0,
            "new_trajectories": 4,
            "estimated_gpu_hours_excluding_cold_start": "0.8–1.8",
            "change": "仅运行64K仍未完成的题；先做1 prompt × 4 responses容量探针，max_num_seqs=8/副本",
            "decision_gate": "至少救回64K仍失败任务的一半；否则不把96K带入训练",
        },
        {
            "priority": 4,
            "experiment": "离线奖励与反循环回放",
            "training_steps": 0,
            "new_trajectories": 0,
            "estimated_gpu_hours_excluding_cold_start": "0",
            "change": "在既有轨迹上扫描完成硬门控、重复命令/无新证据惩罚与dense权重30/50/70%",
            "decision_gate": "提高mixed-signal比例，同时保持正确响应相对错误响应的排序一致性",
        },
        {
            "priority": 5,
            "experiment": "5步可学习性金丝雀",
            "training_steps": 5,
            "new_trajectories": 40,
            "estimated_gpu_hours_excluding_cold_start": "1.0–2.0",
            "change": "2 groups/update，使用历史dense方差较高的train prompt；启用完成硬门控和反循环惩罚",
            "decision_gate": "训练prompt上dense正确性明显提高，独立哨兵不退化；否则停止扩步",
        },
        {
            "priority": 6,
            "experiment": "离线纠错SFT/DPO原型",
            "training_steps": 0,
            "new_trajectories": 0,
            "estimated_gpu_hours_excluding_cold_start": "无rollout，主要是训练侧",
            "change": "从train236的未收尾历史轨迹构造“长轨迹→正确最终回答”纠错样本或偏好对",
            "decision_gate": "先证明能提高收尾与正确性，再决定是否替代昂贵的在线GRPO探索",
        },
    ]
    return {
        "observed_mean_step_s": step_s,
        "observed_full_val20_s": validation_s,
        "observed_checkpoint_save_s": save_s,
        "training_cost_projection": training_cost,
        "ranked_experiments": experiments,
    }


def analyze(validation_path: Path, boss_path: Path, step120_summary_path: Path, native_context_limit: int) -> dict[str, Any]:
    step120_summary = json.loads(step120_summary_path.read_text(encoding="utf-8"))
    return {
        "analysis": "next_experiment_strategy_after_step120",
        "generated_at": "2026-08-10T20:30:00+08:00",
        "decision": "Do not start a formal 96K training run; test finalization control first, then 64K, and reserve 96K for targeted inference-only diagnosis.",
        "completion_audit": completion_audit(read_jsonl(validation_path), read_jsonl(boss_path)),
        "capacity": capacity_scenarios(native_context_limit),
        "runtime_and_experiments": runtime_and_experiments(step120_summary),
        "recommended_sequence": [
            "48K force-final sentinel6",
            "64K plus 32 turns sentinel6 only if needed",
            "96K targeted remaining failures after a max_num_seqs=8 capacity probe",
            "offline reward/anti-loop replay",
            "5-step learnability canary before any 20- or 100-step run",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step120-validation", type=Path, required=True)
    parser.add_argument("--step120-boss-exact", type=Path, required=True)
    parser.add_argument("--step120-summary", type=Path, required=True)
    parser.add_argument("--native-context-limit", type=int, default=262_144)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.step120_validation,
        args.step120_boss_exact,
        args.step120_summary,
        args.native_context_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

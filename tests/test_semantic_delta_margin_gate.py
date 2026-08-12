from __future__ import annotations

from pathlib import Path
import json

from scripts.analyze_semantic_delta_margin_gate import analyze
from scripts.compare_semantic_delta_margin_canary import compare
from scripts.analyze_repair_sft_free_run_divergence import sql_from_command
from scripts.epoch_aware_sequential_sampler import EpochAwareSequentialSampler
from scripts.prepare_semantic_delta_margin_gate import build_rows
from scripts.teacher_forced_component_masks import semantic_delta_token_masks
from scripts.semantic_delta_pairwise_loss import pairwise_loss_from_flat_sequences
import torch


ROOT = Path(__file__).resolve().parents[1]


def test_epoch_aware_sequential_sampler_keeps_fixed_pair_order():
    sampler = EpochAwareSequentialSampler(list(range(6)))

    assert list(sampler) == list(range(6))
    sampler.set_epoch(1)
    assert sampler.epoch == 1
    assert list(sampler) == list(range(6))


def test_semantic_delta_masks_cover_replacement_and_anchor_insertions():
    chosen, rejected = semantic_delta_token_masks([1, 2, 3, 4], [1, 8, 4])
    assert chosen == [0, 1, 1, 0]
    assert rejected == [0, 1, 0]

    chosen, rejected = semantic_delta_token_masks([1, 2, 9, 3], [1, 2, 3])
    assert chosen == [0, 0, 1, 0]
    assert rejected == [0, 1, 0]


def _critical_rows():
    rows = []
    evidence = []
    for index in range(16):
        task_id = f"task_{index:06d}"
        wrong = "sqlite3 -json /workspace/logistics.sqlite 'SELECT amount FROM orders'"
        chosen = "sqlite3 -json /workspace/logistics.sqlite 'SELECT SUM(amount) FROM orders'"
        rows.append(
            {
                "task_id": task_id,
                "sample_id": task_id,
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "task"},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "wrong", "function": {"name": "bash", "arguments": {"command": wrong}}}]},
                    {"role": "tool", "tool_call_id": "wrong", "content": "10"},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "chosen", "function": {"name": "bash", "arguments": {"command": chosen}}}]},
                    {"role": "tool", "tool_call_id": "chosen", "content": "20"},
                    {"role": "assistant", "content": "20"},
                ],
                "supervised_assistant_turn_indices": [1, 2],
                "critical_sql_token_offset": 1,
                "critical_sql_target_id": 42,
                "critical_token_family": "aggregation_function" if index < 9 else "query_start",
            }
        )
        evidence.append(
            {"task_id": task_id, "critical_sql_token_offset": 1, "critical_sql_target_id": 42}
        )
    return rows, {"evidence": evidence}


def test_pair_builder_reuses_identical_error_state_and_actual_wrong_query():
    rows, contract = _critical_rows()
    output, evidence = build_rows(rows, contract)

    assert len(output) == 32
    assert len(evidence) == 16
    first_chosen, first_rejected = output[:2]
    assert first_chosen["messages"][:4] == first_rejected["messages"][:4]
    rejected_candidate = first_rejected["messages"][4]["tool_calls"][0]["function"][
        "arguments"
    ]["command"]
    actual_first_error = first_rejected["messages"][2]["tool_calls"][0]["function"][
        "arguments"
    ]["command"]
    assert rejected_candidate.startswith("sqlite3 -json /workspace/logistics.sqlite ")
    assert sql_from_command(rejected_candidate) == sql_from_command(actual_first_error)
    assert first_chosen["messages"][5]["content"] == "[]"
    assert first_rejected["messages"][5]["content"] == "[]"
    assert first_chosen["messages"][6] == first_rejected["messages"][6] == {
        "role": "assistant",
        "content": "Done.",
    }


def _diagnostic(chosen_preferred: int):
    rows = []
    evidence = []
    for index in range(16):
        task_id = f"task_{index:06d}"
        evidence.append(
            {
                "task_id": task_id,
                "critical_token_family": "aggregation_function" if index < 9 else "query_start",
                "critical_sql_token_offset": 1,
                "critical_sql_target_id": 42,
            }
        )
        for label in ("chosen", "rejected"):
            chosen_wins = index < chosen_preferred
            delta_nll = (1.0 if label == "chosen" else 2.0) if chosen_wins else (
                2.0 if label == "chosen" else 1.0
            )
            rows.append(
                {
                    "task_id": f"{task_id}::{label}",
                    "components": {
                        "semantic_delta": {"mean_nll": delta_nll},
                        "sql_shell": {"mean_nll": delta_nll + 0.1},
                    },
                    "sql_token_rank": {
                        "first_nongreedy_offset": 1,
                        "first_nongreedy_target_id": 42,
                    },
                }
            )
    diagnostic = {
        "contract": "repair-sft-teacher-forced-component-diagnostic-v3",
        "forward_only": True,
        "optimizer_initialized": False,
        "data_sha256": "hash",
        "components": {"semantic_delta": {}},
        "per_task": rows,
    }
    contract = {
        "contract": "semantic-delta-margin-gate-dataset-v1",
        "output_sha256": "hash",
        "evidence": evidence,
    }
    return diagnostic, contract


def test_margin_gate_routes_misranking_to_pairwise_canary_and_preference_to_planner():
    diagnostic, contract = _diagnostic(7)
    result = analyze(diagnostic, contract)
    assert result["semantic_delta_margin"]["chosen_preferred"] == 7
    assert result["decision"]["one_step_training_allowed"] is True
    assert result["decision"]["selected_next_action"] == (
        "one_step_pairwise_chosen_vs_rejected_plan_to_sql_canary"
    )

    diagnostic, contract = _diagnostic(12)
    result = analyze(diagnostic, contract)
    assert result["decision"]["one_step_training_allowed"] is False
    assert result["decision"]["selected_next_action"] == (
        "constrained_sql_planner_and_bash_only_tool_policy"
    )


def test_reference_free_pairwise_loss_prefers_chosen_delta_and_scales_global_pairs():
    log_probs = torch.tensor([-1.0, -9.0, -9.0, -2.0, -9.0, -9.0], requires_grad=True)
    loss, metrics = pairwise_loss_from_flat_sequences(
        log_prob_values=log_probs,
        delta_mask_values=torch.tensor([0, 1, 0, 0, 1, 0]),
        candidate_sign_values=torch.tensor([1, 1, 1, -1, -1, -1]),
        pair_index_values=torch.tensor([0, 0, 0, 0, 0, 0]),
        offsets=torch.tensor([0, 3, 6]),
        beta=1.0,
        global_pair_count=1,
        dp_size=1,
    )

    assert metrics["pairwise/margin_sum"].item() == 1.0
    assert metrics["pairwise/chosen_preferred_count"].item() == 1
    assert loss.item() > 0
    loss.backward()
    assert log_probs.grad[0].item() < 0
    assert log_probs.grad[3].item() > 0


def test_post_canary_gate_requires_preference_improvement_and_no_earlier_regression():
    baseline, contract = _diagnostic(0)
    post, _ = _diagnostic(12)
    baseline_result = analyze(baseline, contract)
    post_result = analyze(post, contract)
    comparison = compare(baseline_result, post_result)
    assert comparison["passed"] is True
    assert comparison["per_task_margin_improved"] == 12

    post_result["frozen_critical_token_audit"][
        "new_earlier_first_nongreedy_regressions"
    ] = 1
    assert compare(baseline_result, post_result)["passed"] is False


def test_margin_launcher_is_forward_only_and_saves_no_checkpoint():
    script = (ROOT / "scripts" / "run_semantic_delta_margin_gate.sh").read_text(
        encoding="utf-8"
    )
    assert "pairs=16" in script
    assert "rows=32" in script
    assert "engine.forward_only=true" in script
    assert "optimizer_initialized=false" in script
    assert "checkpoint_saved=false" in script
    assert "'checkpoint.save_contents=[]'" in script
    assert "trainer.save_freq=-1" in script
    assert "data.train_batch_size=32" in script

    training = (ROOT / "scripts" / "run_semantic_delta_pairwise_canary.sh").read_text(
        encoding="utf-8"
    )
    pipeline = (ROOT / "scripts" / "run_semantic_delta_pairwise_pipeline.sh").read_text(
        encoding="utf-8"
    )
    assert "optimizer_steps=1" in training
    assert "data.train_batch_size=32" in training
    assert "data.micro_batch_size_per_gpu=2" in training
    assert "pair_order=chosen_then_rejected_no_shuffle" in training
    assert "loss_scope=semantic_delta_tokens_only" in training
    assert "'checkpoint.save_contents=[model,extra]'" in training
    assert "trainer.total_training_steps=1" in training
    assert "compare_semantic_delta_margin_canary" in pipeline
    assert "run_semantic_delta_margin_gate.sh" in pipeline
    assert "BASELINE_RUN_NAME" in pipeline
    assert "step120_pairwise_pipeline_baseline" in pipeline
    assert "BASELINE_DIAGNOSTIC" not in pipeline
    assert 'TOKEN_GATE="${BASELINE_OUTPUT_DIR}/token_gate.json"' in pipeline


def test_safe_pretraining_summary_freezes_pairwise_gate_without_raw_assets():
    path = ROOT / "docs" / "semantic_plan_and_delta_pretraining_gate_20260812_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    payload = path.read_text(encoding="utf-8")

    assert summary["semantic_plan_sufficiency_gate"]["arms"]["operator_oracle"]["passed"] is False
    assert summary["semantic_plan_sufficiency_gate"]["arms"]["full_plan_oracle"]["passed"] is False
    assert summary["semantic_delta_margin_gate"]["chosen_preferred"] == 0
    assert summary["semantic_delta_margin_gate"]["frozen_first_nongreedy_token_reconstructed"] == 16
    assert summary["frozen_one_step_gate"]["optimizer_steps"] == 1
    assert summary["frozen_one_step_gate"]["full_replay_before_probability_gate"] is False
    assert summary["scope"]["promotion_allowed"] is False
    assert "/workspace/" not in payload
    assert "SELECT " not in payload


def test_safe_pairwise_canary_summary_records_fail_closed_decision_without_raw_assets():
    path = ROOT / "docs" / "semantic_delta_pairwise_canary_20260812_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    payload = path.read_text(encoding="utf-8")

    assert summary["training_contract"]["optimizer_steps"] == 1
    assert summary["training_contract"]["optimizer_checkpoint_saved"] is False
    assert summary["probability_gate"]["per_task_margin_improved"] == 16
    assert summary["probability_gate"]["post_chosen_preferred"] == 3
    assert summary["probability_gate"]["passed"] is False
    assert summary["decision"]["promotion_allowed"] is False
    assert summary["decision"]["action"] == (
        "stop_no_replay_and_no_additional_pairwise_steps"
    )
    assert "/workspace/" not in payload
    assert "SELECT " not in payload
    assert "task_000" not in payload

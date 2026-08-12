from __future__ import annotations

from pathlib import Path

from scripts.analyze_semantic_delta_margin_gate import analyze
from scripts.analyze_repair_sft_free_run_divergence import sql_from_command
from scripts.prepare_semantic_delta_margin_gate import build_rows
from scripts.teacher_forced_component_masks import semantic_delta_token_masks


ROOT = Path(__file__).resolve().parents[1]


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

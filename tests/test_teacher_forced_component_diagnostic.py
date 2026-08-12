from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.teacher_forced_component_masks import (
    SQLITE_COMMAND_PREFIX,
    assistant_turn_ranges,
    build_repair_component_masks,
    build_sql_weighted_loss_mask,
)
from scripts.analyze_repair_sft_free_run_divergence import analyze_task
from scripts.teacher_forced_token_ranks import (
    ranks_from_full_logits,
    summarize_sql_token_ranks,
)

ROOT = Path(__file__).resolve().parents[1]


def test_assistant_turn_ranges_include_each_closing_marker():
    input_ids = [0, 7, 8, 20, 21, 9, 0, 7, 8, 30, 9]

    ranges = assistant_turn_ranges(input_ids, [7, 8], [9], expected_turns=2)

    assert ranges == [(3, 6), (9, 11)]


def test_component_masks_partition_tool_sql_and_final_answer():
    command = SQLITE_COMMAND_PREFIX + "'SELECT COUNT(*) FROM shipments'"
    rendered = "PRE" + command + " TOOL_END FINAL"
    offsets = [(index, index + 1) for index in range(len(rendered))]
    tool_start = 3
    tool_end = 3 + len(command) + len(" TOOL_END")
    final_start = tool_end
    final_end = len(rendered)

    masks = build_repair_component_masks(
        input_ids=list(range(len(rendered))),
        offsets=offsets,
        rendered_text=rendered,
        command=command,
        turn_ranges=[(tool_start, tool_end), (final_start, final_end)],
    )

    assert sum(masks["sql_shell_mask"]) == len(command) - len(SQLITE_COMMAND_PREFIX)
    assert sum(masks["tool_structure_mask"]) + sum(masks["sql_shell_mask"]) == sum(
        masks["tool_turn_mask"]
    )
    assert not any(
        tool and final
        for tool, final in zip(
            masks["tool_turn_mask"], masks["final_answer_mask"], strict=True
        )
    )


def test_sql_weighted_mask_is_disjoint_and_moves_loss_mass_to_sql():
    weighted = build_sql_weighted_loss_mask(
        tool_structure_mask=[1, 1, 0, 0],
        sql_shell_mask=[0, 0, 1, 0],
        final_answer_mask=[0, 0, 0, 1],
        tool_structure_weight=0.25,
        sql_payload_weight=8.0,
        final_answer_weight=1.0,
    )

    assert weighted == [0.25, 0.25, 8.0, 1.0]
    assert weighted[2] / sum(weighted) > 0.8


def test_sql_weighted_mask_rejects_overlap_and_invalid_weights():
    with pytest.raises(ValueError, match="overlap"):
        build_sql_weighted_loss_mask(
            tool_structure_mask=[1],
            sql_shell_mask=[1],
            final_answer_mask=[0],
            tool_structure_weight=0.25,
            sql_payload_weight=8.0,
            final_answer_weight=1.0,
        )
    with pytest.raises(ValueError, match="weights"):
        build_sql_weighted_loss_mask(
            tool_structure_mask=[1],
            sql_shell_mask=[0],
            final_answer_mask=[0],
            tool_structure_weight=0.0,
            sql_payload_weight=8.0,
            final_answer_weight=1.0,
        )

def test_forward_only_contract_never_initializes_or_saves_optimizer():
    script = (ROOT / "scripts" / "run_repair_sft_teacher_forced_eval.sh").read_text(encoding="utf-8")
    runner = (ROOT / "scripts" / "run_teacher_forced_component_diagnostic.py").read_text(encoding="utf-8")

    assert "engine.forward_only=true" in script
    assert "optimizer_initialized=false" in script
    assert "'checkpoint.save_contents=[]'" in script
    assert script.index('export PYTHONPATH=') < script.index('check_teacher_forced_component_masks.py')
    assert '${PROJECT_ROOT}:/verl:' in script
    assert "training_client.infer_batch" in runner
    assert "training_client.train_batch" not in runner
    assert "trainer.fit()" not in runner
    assert "def component_sft_loss(" in runner
    assert "student_logits=None" in runner


def test_exact_teacher_token_rank_uses_strict_greater_logits_and_vocab_boundary():
    logits = torch.tensor([[[0.0, 3.0, 2.0, 1.0, 99.0], [5.0, 5.0, 1.0, 0.0, 99.0]]])
    labels = torch.tensor([[2, 1]])

    ranks = ranks_from_full_logits(logits, labels, vocab_size=4)

    assert ranks.tolist() == [[2, 1]]


def test_rank_summary_locates_first_nongreedy_sql_token_per_task():
    metrics = {
            "sql_rank/token_count": [2.0, 3.0],
            "sql_rank/rank_sum": [3.0, 3.0],
            "sql_rank/greedy_count": [1.0, 3.0],
            "sql_rank/top5_count": [2.0, 3.0],
            "sql_rank/max_rank": [2.0, 1.0],
            "sql_rank/first_nongreedy_offset": [1.0, -1.0],
            "sql_rank/first_nongreedy_rank": [2.0, -1.0],
            "sql_rank/first_nongreedy_target_id": [42.0, -1.0],
            "sql_rank/first_nongreedy_target_probability": [0.4, -1.0],
    }

    aggregate, per_task = summarize_sql_token_ranks(
        metrics,
        ["a", "b"],
        numbers=lambda value: [float(item) for item in value],
    )

    assert aggregate["greedy_token_count"] == 4
    assert aggregate["tasks_all_sql_tokens_greedy"] == 1
    assert per_task[0]["first_nongreedy_offset"] == 1
    assert per_task[0]["first_nongreedy_target_id"] == 42
    assert per_task[1]["all_tokens_greedy"] is True


def test_unattended_pipeline_compares_both_checkpoints_with_free_rollout():
    script = (ROOT / "scripts" / "run_repair_sft_teacher_forced_prepost_host.sh").read_text(
        encoding="utf-8"
    )

    assert "global_step_120/actor/model/dist_ckpt" in script
    assert "global_step_5/model/dist_ckpt" in script
    assert "compare_teacher_forced_diagnostics.py" in script
    assert "comparison.json" in script
    assert "set_stage done" in script


def test_free_run_divergence_detects_teacher_path_and_continuation():
    command = "sqlite3 -json /workspace/logistics.sqlite 'SELECT COUNT(*) FROM shipments'"
    teacher = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "teacher",
                    "function": {"name": "bash", "arguments": {"command": command}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "teacher", "content": '[{"value":2}]'},
        {"role": "assistant", "content": "answer 2"},
    ]
    exact_rollout = [
        teacher[0],
        teacher[1],
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "rollout",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command": "sqlite3 -json /workspace/logistics.sqlite \'SELECT COUNT(*) FROM shipments\'"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "rollout", "content": '[{"value":2}]'},
        {"role": "assistant", "content": "answer 2"},
    ]

    exact = analyze_task(teacher, exact_rollout)
    continued = analyze_task(
        teacher,
        exact_rollout[:-1]
        + [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "extra",
                        "function": {"name": "bash", "arguments": {"command": "pwd"}},
                    }
                ],
            },
            {"role": "assistant", "content": "answer 2"},
        ],
    )

    assert exact["first_divergence_bucket"] == "teacher_path_matched"
    assert continued["first_divergence_bucket"] == "teacher_evidence_then_continued"


def test_safe_diagnostic_summary_preserves_decision_metrics():
    summary = json.loads(
        (ROOT / "docs" / "repair_sft_teacher_forced_diagnosis_20260811_summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert summary["contract"] == "repair-sft-teacher-forced-diagnosis-summary-v1"
    assert summary["scope"]["task_count"] == 16
    assert summary["execution"]["forward_only"] is True
    assert summary["execution"]["optimizer_initialized"] is False
    assert summary["official_assistant_loss"]["step120"] == 1.8738141059875488
    assert summary["official_assistant_loss"]["post_sft"] == 0.4146054983139038
    assert summary["per_task_probability_distribution"]["post_sft"][
        "sql_tasks_above_0_5"
    ] == 5
    assert summary["free_rollout"]["deltas"]["correct"] == 0
    assert summary["first_divergence"]["post_sft"]["first_sql_diverged"] == 16


def test_pretraining_gate_summary_is_safe_and_keeps_training_disabled():
    path = ROOT / "docs" / "repair_sft_pretraining_gate_20260812_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    payload = path.read_text(encoding="utf-8")

    assert summary["cpu_first_query_semantic_gate"]["step120"]["verified_gold_support"] == 0
    assert summary["cpu_first_query_semantic_gate"]["generic_sft_step5"][
        "teacher_result_equivalent"
    ] == 0
    assert summary["frozen_next_canary"]["intervention"] == "sql_payload_weight_only"
    assert summary["frozen_next_canary"]["training_steps"] == 1
    assert summary["training_started"] is False
    assert summary["promotion_allowed"] is False
    assert "/data/" not in payload
    assert "/workspace/" not in payload


def test_portable_diagnostic_report_is_self_contained_and_source_backed():
    artifact = json.loads(
        (ROOT / "docs" / "repair_sft_teacher_forced_diagnosis_20260811_artifact.json").read_text(
            encoding="utf-8"
        )
    )
    html = (ROOT / "docs" / "repair_sft_teacher_forced_diagnosis_20260811.html").read_text(
        encoding="utf-8"
    )

    assert artifact["manifest"]["title"] == "纠错 SFT 为什么 loss 降但准确率没升"
    assert len(artifact["manifest"]["charts"]) == 1
    assert len(artifact["manifest"]["tables"]) == 1
    assert all(source.get("path") for source in artifact["sources"])
    assert "SELECT component, model" in artifact["sources"][0]["query"]["sql"]
    assert 'data-data-analytics-portable-artifact="true"' in html
    assert "data-analytics-portable-artifact-payload-source" in html
    assert "html,body{max-width:100%;overflow-x:hidden}" in html

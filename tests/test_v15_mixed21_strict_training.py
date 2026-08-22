import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from llin_verl.grpo_group_gate import (
    apply_strict_correctness_group_gate,
    strict_correctness_group_stats,
)
from llin_verl.pi_reward import (
    compute_score_strict_correctness_v3,
    strict_table_answer_match_complete,
)
from scripts.check_v15_mixed21_canary import gate
from scripts.patch_verl_grpo_strict_variance_gate import patch_trainer
from scripts.prepare_v15_mixed21_training import build, validate


ROOT = Path(__file__).resolve().parents[1]


def _numeric_row(index: int, training_allowed: bool = False) -> dict:
    return {
        "data_source": "llin_pi_dwh_v2",
        "prompt": [{"role": "user", "content": f"private task {index}"}],
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "environment_id": f"approved/task-{index}",
                "answer_type": "numeric",
                "expected_value": float(index),
                "verification_sql": f"SELECT {index}",
                "required_tables": [],
            },
        },
        "extra_info": {
            "environment_id": f"approved/task-{index}",
            "instruction_sha256": f"instruction-{index:02d}",
            "gold_sha256": f"gold-{index:02d}",
            "training_allowed": training_allowed,
            "promotion_allowed": False,
        },
    }


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_exact_approval_builds_only_21_tasks_four_times(tmp_path: Path) -> None:
    source = tmp_path / "mixed_approved_candidates.sensitive.parquet"
    audit = tmp_path / "safe_summary.json"
    validation_file = tmp_path / "sealed.parquet"
    canonical = tmp_path / "private" / "train21.sensitive.parquet"
    schedule = tmp_path / "private" / "train21x4.sensitive.parquet"
    safe_summary = tmp_path / "train21x4.safe.json"
    _write_parquet(source, [_numeric_row(index) for index in range(21)])
    _write_parquet(validation_file, [_numeric_row(100), _numeric_row(101)])
    audit.write_text(
        json.dumps(
            {
                "mixed_review": {"approved_candidates": 21, "reviewed": 21},
                "promotion_allowed": False,
            }
        ),
        encoding="utf-8",
    )

    result = build(source, audit, canonical, schedule, validation_file, safe_summary, 20260822)
    checked = validate(source, audit, canonical, schedule, validation_file, safe_summary)

    assert result == checked
    assert result["approved_unique_tasks"] == 21
    assert result["schedule_groups"] == 84
    assert result["online_trajectories"] == 672
    assert result["conditional_reward_repaired_candidates_included"] == 0
    scheduled = pq.read_table(schedule).to_pylist()
    assert {row["extra_info"]["training_exposure_index"] for row in scheduled} == {1, 2, 3, 4}
    assert all(row["extra_info"]["training_allowed"] for row in scheduled)


def test_approval_builder_rejects_renamed_or_pre_authorized_source(tmp_path: Path) -> None:
    wrong_name = tmp_path / "reward_repaired_mixed_candidates.sensitive.parquet"
    audit = tmp_path / "safe_summary.json"
    validation_file = tmp_path / "sealed.parquet"
    _write_parquet(wrong_name, [_numeric_row(index) for index in range(21)])
    _write_parquet(validation_file, [_numeric_row(100)])
    audit.write_text(
        json.dumps(
            {
                "mixed_review": {"approved_candidates": 21, "reviewed": 21},
                "promotion_allowed": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exact independent approval file"):
        build(
            wrong_name,
            audit,
            tmp_path / "canonical.parquet",
            tmp_path / "schedule.parquet",
            validation_file,
            tmp_path / "output.json",
            20260822,
        )


def test_group_gate_masks_all_uniform_groups_and_skips_empty_optimizer_batch() -> None:
    uids = ["mixed"] * 8 + ["all-wrong"] * 8 + ["all-correct"] * 8
    labels = [0, 1] * 4 + [0] * 8 + [1] * 8
    mask, metrics = strict_correctness_group_stats(uids, labels)

    assert mask == [True] * 8 + [False] * 16
    assert metrics["grpo/strict_mixed_groups"] == 1.0
    assert metrics["grpo/skipped_all_wrong_groups"] == 1.0
    assert metrics["grpo/skipped_all_correct_groups"] == 1.0

    batch = SimpleNamespace(
        non_tensor_batch={"uid": ["all-wrong"] * 8, "acc": [0] * 8},
        batch={
            "advantages": torch.ones(8, 3),
            "returns": torch.ones(8, 3),
            "response_mask": torch.ones(8, 3),
        },
        meta_info={},
    )
    gated, empty_metrics = apply_strict_correctness_group_gate(batch)

    assert torch.count_nonzero(gated.batch["advantages"]).item() == 0
    assert torch.count_nonzero(gated.batch["returns"]).item() == 0
    assert torch.count_nonzero(gated.batch["response_mask"]).item() == 0
    assert gated.meta_info["strict_group_should_update_actor"] is False
    assert empty_metrics["grpo/skipped_all_wrong_groups"] == 1.0


def test_group_gate_rejects_incomplete_eight_sample_group() -> None:
    mask, metrics = strict_correctness_group_stats(
        ["prompt"] * 7,
        [0, 1, 0, 1, 0, 1, 0],
        expected_group_size=8,
    )
    assert mask == [False] * 7
    assert metrics["grpo/strict_mixed_groups"] == 0.0
    assert metrics["grpo/skipped_bad_group_size_groups"] == 1.0


def test_runtime_patch_masks_after_kl_and_skips_adam_update(tmp_path: Path) -> None:
    target = tmp_path / "ray_trainer.py"
    target.write_text(
        (ROOT / "reference" / "verl" / "verl" / "experimental" / "separation" / "ray_trainer.py").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )

    assert patch_trainer(target) == "patched"
    text = target.read_text(encoding="utf-8")
    assert text.index("apply_kl_penalty(") < text.index("apply_strict_correctness_group_gate(batch)")
    assert "strict_group_should_update_actor" in text
    assert "return batch\n        metrics[\"actor/update_skipped_no_strict_mixed\"] = 0.0" in text
    assert patch_trainer(target) == "already-patched"
    compile(text, str(target), "exec")


def test_strict_reward_is_binary_and_process_quality_cannot_compensate(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "approved" / "task" / "logistics.sqlite"
    database.parent.mkdir(parents=True)
    import sqlite3

    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE fact_value(value REAL)")
    connection.execute("INSERT INTO fact_value VALUES (21)")
    connection.commit()
    connection.close()
    monkeypatch.setenv("PI_AGENT_SANDBOX_LOWER", str(tmp_path))
    ground_truth = {
        "environment_id": "approved/task",
        "answer_type": "numeric",
        "expected_value": 21.0,
        "verification_sql": "SELECT SUM(value) FROM fact_value",
        "required_tables": ["fact_value"],
    }
    evidence = {
        "pi_tool_events": [
            {
                "name": "bash",
                "arguments": {
                    "command": 'sqlite3 /workspace/logistics.sqlite "SELECT SUM(value) FROM fact_value"'
                },
                "ok": True,
            }
        ]
    }

    wrong = compute_score_strict_correctness_v3(
        "llin_pi_dwh_v2", "已查询并复核，最终答案是 0。", ground_truth, evidence
    )
    correct = compute_score_strict_correctness_v3(
        "llin_pi_dwh_v2", "已查询并复核，最终答案是 21。", ground_truth, evidence
    )

    assert wrong["process_reward_observed"] > 0
    assert wrong["score"] == wrong["acc"] == 0.0
    assert wrong["process_reward_applied"] == 0.0
    assert correct["score"] == correct["acc"] == 1.0
    assert correct["reward_contract"] == "strict-correctness-gated-v3"


def test_general_table_reward_compares_every_ordered_row_and_column() -> None:
    expected = [
        {"lane": "华东", "orders": 10, "share": 0.4},
        {"lane": "华南", "orders": 8, "share": 0.32},
    ]
    correct = """| lane | orders | share |
|---|---:|---:|
| 华东 | 10 | 40% |
| 华南 | 8 | 32% |"""

    assert strict_table_answer_match_complete(correct, expected, 1e-3, 1e-5) == (
        True,
        "markdown_full",
        2,
    )
    assert strict_table_answer_match_complete(
        correct.replace("| 8 |", "| 9 |"), expected, 1e-3, 1e-5
    )[0] is False
    assert strict_table_answer_match_complete(
        correct.replace("华东", "华北"), expected, 1e-3, 1e-5
    )[0] is False
    assert strict_table_answer_match_complete(
        correct + "\n| 华北 | 1 | 4% |", expected, 1e-3, 1e-5
    )[0] is False


def _metric_line(step: int, validation: bool = False, mixed: int = 1) -> str:
    if validation:
        return (
            f"step:{step} - rollouter/validate_time:1 - "
            "val-core/test/acc/mean@1:0.5 - val-core/test/final_answer_correct/mean@1:0.5\n"
        )
    skipped = 2 - mixed
    grad = 0.2 if mixed else 0.0
    update_skipped = 0 if mixed else 1
    return (
        f"step:{step} - training/global_step:{step} - grpo/strict_mixed_groups:{mixed} - "
        f"grpo/skipped_uniform_groups:{skipped} - grpo/total_groups:2 - actor/grad_norm:{grad} - "
        f"actor/update_skipped_no_strict_mixed:{update_skipped} - critic/score/mean:0.5 - critic/kl:0.001\n"
    )


def test_canary_gate_requires_every_validation_checkpoint_and_safe_update(tmp_path: Path) -> None:
    supervisor = tmp_path / "run" / "supervisor"
    checkpoints = tmp_path / "run" / "checkpoints"
    supervisor.mkdir(parents=True)
    lines = [_metric_line(0, validation=True)]
    for step in range(1, 6):
        lines.extend([_metric_line(step), _metric_line(step, validation=True)])
        (checkpoints / f"global_step_{step}").mkdir(parents=True)
    log = supervisor / "canary_driver.log"
    log.write_text("".join(lines), encoding="utf-8")

    result = gate([log], [1, 2, 3, 4, 5], 0, 0.1, 10.0, "canary")

    assert result["passed"] is True
    assert result["checks"]["all_canary_checkpoints_present"] is True
    assert result["checks"]["at_least_one_effective_mixed_group"] is True


def test_shell_contract_is_fixed_to_21x4_with_frozen_kl_and_five_step_gate() -> None:
    runner = (ROOT / "scripts" / "run_pi_v15_mixed21_4x_strict_kl.sh").read_text(encoding="utf-8")
    supervisor = (ROOT / "scripts" / "launch_v15_mixed21_4x_host.sh").read_text(encoding="utf-8")

    for text in (runner, supervisor):
        assert "reward_repaired_mixed_candidates.sensitive.parquet" not in text
    assert "TRAIN_TASKS=21" in runner
    assert "EXPOSURES_PER_TASK=4" in runner
    assert "RESPONSES_PER_GROUP=8" in runner
    assert "CANARY_STEPS=5" in runner
    assert "algorithm.use_kl_in_reward=True" in runner
    assert "compute_score_strict_correctness_v3" in runner
    assert "STALENESS_THRESHOLD=0" in runner
    assert "MODEL_PATH='${MODEL_CONTAINER}'" in supervisor
    assert "SOURCE_CHECKPOINT='${source_checkpoint}'" in supervisor

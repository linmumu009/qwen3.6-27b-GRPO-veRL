from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_qwen38_next_iteration_plan.py"
SPEC = importlib.util.spec_from_file_location("qwen38_next_plan", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _load(name: str):
    return MODULE._load(ROOT / "docs" / name)


def test_plan_reconciles_evidence_and_uses_fresh_native_canary() -> None:
    plan = MODULE.build_plan(
        _load("qwen38_banded_v2_strict_reward_replay_20260818.safe.json"),
        _load("qwen38_native_vs_step70_original70_strict_20260820.safe.json"),
        _load("qwen38_step70_heldout_v15_v20_v21_final_20260820.safe.json"),
        _load("boss_v15_semantic_mismatch_20260818.safe.json"),
    )

    assert all(plan["automatic_checks"].values())
    assert plan["decision"]["next_training_base"] == "qwen38-27b-native-hf-step0"
    assert plan["decision"]["step70_disposition"] == "freeze_for_forensics_not_promotion_or_resume"
    assert plan["evidence"]["native_strict_mixed_tasks"] == 20
    assert plan["evidence"]["step70_strict_mixed_tasks"] == 15
    assert plan["evidence"]["native_heldout_task_passes_at_max_6"] == 2
    assert plan["evidence"]["step70_heldout_task_passes_at_max_6"] == 18


def test_plan_separates_selection_metric_timeout_and_eval() -> None:
    plan = MODULE.build_plan(
        _load("qwen38_banded_v2_strict_reward_replay_20260818.safe.json"),
        _load("qwen38_native_vs_step70_original70_strict_20260820.safe.json"),
        _load("qwen38_step70_heldout_v15_v20_v21_final_20260820.safe.json"),
        _load("boss_v15_semantic_mismatch_20260818.safe.json"),
    )

    assert plan["automatic_data_gate"]["queue_wait_counts_toward_trajectory_timeout"] is False
    assert plan["automatic_data_gate"]["minimum_completed_strict_correct"] == 2
    assert plan["automatic_data_gate"]["minimum_completed_strict_wrong"] == 2
    assert plan["evaluation_kpis"]["data_selection_metric_not_model_quality_metric"] == "strict_mixed_task_count"
    assert plan["fresh_data_plan"]["frozen_new_eval_sandbox"] == "v22"
    assert plan["fresh_data_plan"]["acquisition_sandboxes"] == ["v23", "v24", "v25", "v26"]


def test_report_artifact_matches_decision_record() -> None:
    artifact = json.loads(
        (ROOT / "docs" / "qwen38_next_iteration_plan_20260820.artifact.json").read_text(
            encoding="utf-8"
        )
    )
    assert artifact["surface"] == "report"
    assert artifact["snapshot"]["status"] == "ready"
    evidence = artifact["snapshot"]["datasets"]["model_evidence"]
    assert [(row["模型"], row["原70题严格mixed"]) for row in evidence] == [
        ("Qwen3.8 原生", 20),
        ("Qwen3.8 Step70", 15),
    ]
    phases = artifact["snapshot"]["datasets"]["execution_phases"]
    assert [row["阶段"] for row in phases] == [
        "A 冻结合同",
        "B 重建数据",
        "C 12步金丝雀",
        "D 同seed评测",
        "E 扩量",
    ]

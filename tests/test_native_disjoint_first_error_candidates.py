from pathlib import Path

import pytest

from scripts.analyze_native_disjoint_pair_margin import analyze
from scripts.prepare_native_disjoint_first_error_candidates import (
    filter_pairs,
    forbidden_ids,
    frozen_overlap_audit,
    row_task_id,
)


def pair_rows(tasks: list[str]) -> list[dict]:
    rows = []
    for index, task in enumerate(tasks):
        for label in ("chosen", "rejected"):
            rows.append(
                {
                    "source_task_id": task,
                    "task_id": f"{task}::{label}",
                    "candidate_label": label,
                    "pair_index": index,
                    "messages": [1, 2, 3, {"content": f"observed-{task}"}, label],
                }
            )
    return rows


def test_forbidden_identity_union_and_pair_reindexing():
    eval_contract = {
        "contract": "current-definition-disjoint-first-error-evaluation-v1",
        "evaluation_only": True,
        "may_be_used_as_training_data": False,
        "pairs": 1,
        "evidence": [{"task_id": "task_eval"}],
    }
    additional = [[{"reward_model": {"ground_truth": {"task_id": "task_cal"}}}]]
    forbidden, audits = forbidden_ids(eval_contract, additional)
    rows = pair_rows(["task_eval", "task_keep", "task_cal"])
    evidence = [{"task_id": task} for task in ("task_eval", "task_keep", "task_cal")]

    retained_rows, retained_evidence, excluded = filter_pairs(rows, evidence, forbidden)

    assert forbidden == {"task_eval", "task_cal"}
    assert audits == [{"rows": 1, "unique_task_ids": 1}]
    assert [row["task_id"] for row in retained_evidence] == ["task_keep"]
    assert [row["candidate_label"] for row in retained_rows] == ["chosen", "rejected"]
    assert {row["pair_index"] for row in retained_rows} == {0}
    assert all(row["state_source_checkpoint"] == "native_base" for row in retained_rows)
    assert excluded == 2


def test_row_task_id_prefers_source_then_truth_then_display():
    assert row_task_id({"source_task_id": "source", "task_id": "display::chosen"}) == "source"
    assert row_task_id({"reward_model": {"ground_truth": {"task_id": "truth"}}}) == "truth"
    assert row_task_id({"task_id": "display::rejected"}) == "display"


def test_frozen_overlap_audit_counts_each_source_without_emitting_identities():
    eval_contract = {
        "contract": "current-definition-disjoint-first-error-evaluation-v1",
        "evaluation_only": True,
        "may_be_used_as_training_data": False,
        "pairs": 1,
        "evidence": [{"task_id": "task_eval"}],
    }
    evidence = [{"task_id": task} for task in ("task_eval", "task_a", "task_b")]
    audit = frozen_overlap_audit(
        evidence,
        eval_contract,
        [[{"source_task_id": "task_a"}], [{"source_task_id": "task_eval"}]],
    )
    assert audit == {
        "native_first_error_pairs": 3,
        "eval22_overlap": 1,
        "outside_eval22": 2,
        "additional_overlap_outside_eval22": [1, 0],
        "retained_after_union": 1,
    }


def margin_fixtures(chosen_preferred: int, pairs: int = 11):
    evidence = [{"task_id": f"task_{index:02d}"} for index in range(pairs)]
    per_task = []
    for index in range(pairs):
        preferred = index < chosen_preferred
        for label in ("chosen", "rejected"):
            chosen = label == "chosen"
            nll = 1.0 if chosen == preferred else 2.0
            per_task.append(
                {
                    "task_id": f"task_{index:02d}::{label}",
                    "components": {
                        "semantic_delta": {"mean_nll": nll},
                        "sql_shell": {"mean_nll": nll + 0.1},
                    },
                    "sql_token_rank": {
                        "first_nongreedy_offset": 0,
                        "first_nongreedy_target_id": 1,
                        "first_nongreedy_rank": 2,
                        "first_nongreedy_target_token": "SUM",
                    },
                }
            )
    diagnostic = {
        "contract": "repair-sft-teacher-forced-component-diagnostic-v3",
        "forward_only": True,
        "optimizer_initialized": False,
        "data_sha256": "hash",
        "components": {"semantic_delta": {}},
        "model_label": "step120_native11",
        "per_task": per_task,
    }
    contract = {
        "contract": "current-definition-native-first-error-training-candidates-v1",
        "candidate_role": "real_error_state_training_supply_screen_only",
        "candidate_pair_gate_passed": True,
        "candidate_only": True,
        "may_be_used_as_training_data": False,
        "training_allowed": False,
        "promotion_allowed": False,
        "pairs": pairs,
        "expected_pairs": pairs,
        "output_sha256": "hash",
        "evidence": evidence,
    }
    token = {
        "contract": "current-definition-disjoint-pair-candidate-token-gate-v1",
        "candidate_only": True,
        "may_be_used_as_training_data": False,
        "pairs": pairs,
    }
    return diagnostic, contract, token


def test_native_margin_screen_retains_systematically_misranked_stratum():
    result = analyze(*margin_fixtures(chosen_preferred=2))
    assert result["semantic_delta_margin"]["systematic_misranking_screen_threshold"] == 9
    assert result["screening_decision"]["retain_as_candidate_training_source_stratum"] is True
    assert result["screening_decision"]["remaining_gap_if_retained"] == 37
    assert result["training_allowed"] is False


def test_native_margin_screen_rejects_any_training_authorization():
    diagnostic, contract, token = margin_fixtures(chosen_preferred=2)
    contract["may_be_used_as_training_data"] = True
    with pytest.raises(ValueError, match="fail closed"):
        analyze(diagnostic, contract, token)


def test_native_runner_is_forward_only_and_never_saves_or_trains():
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_native_disjoint_pair_margin.sh"
    ).read_text(encoding="utf-8")
    for needle in (
        "candidate_only=true",
        "may_be_used_as_training_data=false",
        "engine.forward_only=true",
        "checkpoint.load_contents=[]",
        "checkpoint.save_contents=[]",
        "optimizer_initialized=false",
        "training_allowed=false",
    ):
        assert needle in source

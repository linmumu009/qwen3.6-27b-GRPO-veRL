import json

import pytest

from scripts.analyze_disjoint_pair_candidate_pool import sha256_text
from scripts.prepare_disjoint_pair_rollout_candidates import build_candidate_rows


def source_row(task: str, instruction: str) -> dict:
    return {
        "data_source": "old",
        "prompt": [
            {"role": "system", "content": "boss system"},
            {"role": "user", "content": instruction},
        ],
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "task_id": task,
                "environment_id": "sft/v15",
                "answer_type": "numeric",
                "expected_value_json": "999",
                "verification_sql": "SELECT old FROM metric",
            },
        },
        "extra_info": {"split": "train", "tools_kwargs": {"bash": {}}},
    }


def manifest_row(task: str, instruction: str, value: int) -> dict:
    return {
        "task_id": task,
        "natural_language_instruction": instruction,
        "expected_tables": ["Metric"],
        "verification_criteria": {"must_use_fields": ["Amount"]},
        "gold_answer": {
            "answer_type": "numeric",
            "value": value,
            "verification_sql": f"SELECT SUM(amount) FROM metric WHERE id = {value}",
        },
    }


def audit_for(manifests: list[dict], *, passed: bool = True) -> dict:
    records = []
    for index, row in enumerate(manifests):
        records.append(
            {
                "task_id": row["task_id"],
                "tier": "strict_available",
                "semantic_warnings": [],
                "source_instruction_rebuilt": index % 2 == 0,
                "current_instruction_sha256": sha256_text(row["natural_language_instruction"]),
                "current_verification_sql_sha256": sha256_text(
                    row["gold_answer"]["verification_sql"]
                ),
            }
        )
    return {
        "contract": "current-definition-disjoint-pair-pool-audit-v1",
        "strict_available": len(records),
        "data_gate_passed": passed,
        "records": records,
    }


def test_builder_replaces_historical_prompt_and_gold_without_mutating_source():
    manifests = [manifest_row(f"task_{index:03d}", f"当前问题 {index}", index) for index in range(48)]
    sources = [source_row(row["task_id"], f"历史问题 {index}") for index, row in enumerate(manifests)]

    rows, evidence = build_candidate_rows(
        train_rows=sources,
        manifest_by_task={row["task_id"]: row for row in manifests},
        audit=audit_for(manifests),
        row_count=48,
        seed="seed",
    )

    assert len(rows) == len(evidence) == 48
    assert sources[0]["prompt"][1]["content"].startswith("历史问题")
    for row in rows:
        task = row["reward_model"]["ground_truth"]["task_id"]
        current = next(item for item in manifests if item["task_id"] == task)
        truth = row["reward_model"]["ground_truth"]
        assert row["prompt"][1]["content"] == current["natural_language_instruction"]
        assert truth["verification_sql"] == current["gold_answer"]["verification_sql"]
        assert json.loads(truth["expected_value_json"]) == current["gold_answer"]["value"]
        assert truth["required_tables"] == ["metric"]
        assert truth["must_use_fields"] == ["amount"]
        assert row["extra_info"]["pair_acquisition_contract"].endswith("-v1")


def test_builder_fails_closed_on_audit_or_identity_change():
    manifests = [manifest_row(f"task_{index:03d}", f"当前问题 {index}", index) for index in range(48)]
    sources = [source_row(row["task_id"], "历史问题") for row in manifests]
    blocked = audit_for(manifests, passed=False)
    with pytest.raises(ValueError, match="data gate"):
        build_candidate_rows(
            train_rows=sources,
            manifest_by_task={row["task_id"]: row for row in manifests},
            audit=blocked,
            row_count=48,
            seed="seed",
        )

    changed = audit_for(manifests)
    changed["records"][0]["current_instruction_sha256"] = "stale"
    with pytest.raises(ValueError, match="instruction hash changed"):
        build_candidate_rows(
            train_rows=sources,
            manifest_by_task={row["task_id"]: row for row in manifests},
            audit=changed,
            row_count=48,
            seed="seed",
        )

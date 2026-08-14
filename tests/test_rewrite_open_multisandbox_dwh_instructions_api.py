from __future__ import annotations

from scripts.rewrite_open_multisandbox_dwh_instructions_api import (
    build_prompt,
    semantic_payload,
    validate_rewrite,
)


def _task() -> dict:
    instruction = (
        "我在做经营复盘。请从2026年6月、货物类型为“生鲜”的业务里按货品类型比较，"
        "少于3票的不参与，从高到低列出前5项和综合关注分；权重是20%、35%、45%。"
    )
    return {
        "task_id": "llin_open_v15_0001",
        "instruction_role": "company_owner",
        "difficulty_level": 5,
        "natural_language_instruction": instruction,
        "semantic_anchors": ["2026年6月", "生鲜", "货品类型", "20%", "35%", "45%", "3", "5"],
        "semantic_contract": {
            "family": "management_prioritization",
            "explanation_is_open_ended": True,
        },
        "sample_sql": "SELECT secret FROM hidden",
        "gold_answer": {"value": [{"category": "秘密", "value": 99}]},
    }


def test_payload_excludes_sql_gold_and_database_details() -> None:
    task = _task()
    rendered = str(semantic_payload(task)) + build_prompt([task])

    assert "SELECT secret" not in rendered
    assert "秘密" not in rendered
    assert "99" not in rendered
    assert "must_preserve_verbatim" in rendered
    assert "20%" in rendered


def test_validator_requires_all_business_anchors_and_no_technical_terms() -> None:
    task = _task()
    assert validate_rewrite(task, task["natural_language_instruction"]) == []

    invalid = "请查数据库，看看2026年6月生鲜业务，按货品类型从高到低列出前5项。"
    reasons = validate_rewrite(task, invalid)
    assert "technical_term:数据库" in reasons
    assert "anchor_missing:20%" in reasons
    assert "composite_metric_missing" in reasons

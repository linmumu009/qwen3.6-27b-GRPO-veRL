from __future__ import annotations

import sys
import types

from scripts.generate_plan_first_dwh_sandbox import build_plans, build_task, create_database
import scripts.rewrite_plan_first_dwh_instructions_api as rewriter
from scripts.rewrite_plan_first_dwh_instructions_api import (
    build_prompt,
    parse_response,
    semantic_payload,
    validate_rewrite,
)


def _task(tmp_path, index: int):
    database = tmp_path / "logistics.sqlite"
    if not database.exists():
        create_database(database)
    return build_task(database, build_plans()[index])


def test_api_prompt_contains_business_facts_but_no_sql(tmp_path) -> None:
    task = _task(tmp_path, 250)
    payload = semantic_payload(task)
    prompt = build_prompt([task])

    assert payload["must_preserve"]["metric"] == "延误率（百分比）"
    assert "ship_date" not in prompt
    assert task["gold_answer"]["verification_sql"] not in prompt
    assert "target_speaker" in prompt


def test_retry_feedback_contains_only_validation_reason_codes(tmp_path) -> None:
    task = _task(tmp_path, 250)
    task["_rewrite_feedback"] = ["month_missing_or_changed", "top_n_missing"]

    payload = semantic_payload(task)

    assert payload["previous_attempt_rejected_because"] == [
        "month_missing_or_changed",
        "top_n_missing",
    ]
    assert payload["required_corrections_for_retry"] == [
        "月份被遗漏或改错，请逐字保留 conditions 中的2025年具体月份。",
        "遗漏前N名，请按 must_preserve.top_n 明确写出前几名。",
    ]
    rendered = str(payload)
    assert "verification_sql" not in rendered
    assert task["gold_answer"]["verification_sql"] not in rendered


def test_natural_rewrite_must_preserve_every_semantic_constraint(tmp_path) -> None:
    task = _task(tmp_path, 250)
    valid = (
        "我在复盘承运质量。请只看2025年1月发出、已经签收、选择“当日达”服务、"
        "由“战略”级承运商负责且客户属于“大客户”的运单，比较各承运商的延误率。"
        "只统计至少有3票符合条件的承运商，按结果从高到低列出前5名。"
    )
    assert validate_rewrite(task, valid) == []

    invalid = "帮我看看今年各承运商谁表现最好。"
    reasons = validate_rewrite(task, invalid)
    assert "metric_not_explicit" in reasons
    assert "month_missing_or_changed" in reasons
    assert "top_n_missing" in reasons
    assert "minimum_sample_rule_missing" in reasons


def test_service_level_comparison_accepts_the_business_label_itself(tmp_path) -> None:
    task = _task(tmp_path, 100)
    instruction = (
        "帮我看看2025年1月发出且没有取消的运单，按不同服务等级分别统计运单数量，"
        "并按结果从高到低列出来。"
    )

    assert validate_rewrite(task, instruction) == []


def test_warehouse_type_must_not_sound_like_one_specific_warehouse(tmp_path) -> None:
    task = _task(tmp_path, 150)
    explicit = (
        "帮我看下2025年1月发出、已经签收、发货仓类型为中心仓的运单，"
        "各仓库运单数量是多少？按从高到低列出前5名。"
    )
    ambiguous = (
        "帮我看下2025年1月发出、已经签收的中心仓运单，各仓库运单数量是多少？"
        "按从高到低列出前5名。"
    )

    assert validate_rewrite(task, explicit) == []
    assert "warehouse_type_not_explicit" in validate_rewrite(task, ambiguous)


def test_api_response_requires_every_task_id(tmp_path) -> None:
    tasks = [_task(tmp_path, 0), _task(tmp_path, 1)]
    text = (
        '[{"task_id":"llin_dwh_pf_0001","instruction":"自然问题一"},'
        '{"task_id":"llin_dwh_pf_0002","instruction":"自然问题二"}]'
    )
    assert set(parse_response(text, tasks)) == {"llin_dwh_pf_0001", "llin_dwh_pf_0002"}


def test_openai_client_disables_hidden_retries_and_closes(monkeypatch, tmp_path) -> None:
    task = _task(tmp_path, 0)
    observed: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs):
            observed["request"] = kwargs
            content = (
                '[{"task_id":"llin_dwh_pf_0001",'
                '"instruction":"请帮我统计2025年1月发出的运单一共有多少票，给我一个汇总数字。"}]'
            )
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            observed["client_kwargs"] = kwargs
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

        def close(self):
            observed["closed"] = True

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    result = rewriter._call_api(
        [task],
        {
            "_resolved_api_key": "not-a-real-key",
            "base_url": "https://example.invalid/v1",
            "default_model": "fake-model",
            "timeout_seconds": 1,
        },
    )

    assert result[task["task_id"]].startswith("请帮我统计")
    assert observed["client_kwargs"]["max_retries"] == 0
    assert observed["closed"] is True

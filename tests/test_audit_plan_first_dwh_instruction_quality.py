from scripts.audit_plan_first_dwh_instruction_quality import _opening


def test_opening_normalization_removes_varying_business_literals() -> None:
    first = "我在做2025年1月的“当日达”复盘。请给结果。"
    second = "我在做2025 年 9 月的“冷链”复盘。请给结果。"

    assert _opening(first) == _opening(second)

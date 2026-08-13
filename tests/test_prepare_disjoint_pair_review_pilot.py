import pytest

from scripts.prepare_disjoint_pair_review_pilot import risk_key, select_records


def record(task: str, warnings: list[str], tier: str = "review_required") -> dict:
    return {"task_id": task, "semantic_warnings": warnings, "tier": tier}


def test_pilot_selects_lowest_weight_review_required_deterministically():
    rows = [
        record("strict", [], tier="strict_available"),
        record("low-a", ["limit_without_order_by"]),
        record("low-b", ["numeric_gold_without_aggregation"]),
        record("high", ["broad_instruction_reduced_to_row_count"]),
    ]
    selected = select_records(rows, review_count=2, seed="fixed")
    assert {row["task_id"] for row in selected} == {"low-a", "low-b"}
    assert risk_key(selected[0], "fixed") <= risk_key(selected[1], "fixed")


def test_pilot_never_fills_shortfall_with_non_review_tasks():
    with pytest.raises(ValueError, match="only 1 review-required"):
        select_records(
            [record("one", ["limit_without_order_by"]), record("strict", [], "strict_available")],
            review_count=2,
            seed="fixed",
        )

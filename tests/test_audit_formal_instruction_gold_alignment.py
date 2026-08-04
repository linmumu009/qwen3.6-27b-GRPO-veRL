from scripts.audit_formal_instruction_gold_alignment import audit, classify


def record(instruction: str, sql: str, answer_type: str = "numeric"):
    return {
        "task_id": "task",
        "split": "train",
        "instruction": instruction,
        "gold": {"answer_type": answer_type, "verification_sql": sql},
    }


def test_flags_broad_latest_prompt_with_unordered_hidden_target():
    row = record(
        "请根据最近数据分析问题并给出优化建议。",
        "SELECT category, SUM(value) FROM metric GROUP BY category LIMIT 3",
        "table",
    )

    assert classify(row) == [
        "broad_instruction_exact_hidden_target",
        "latest_instruction_without_temporal_sql",
        "limit_without_order_by",
    ]


def test_explicit_aligned_query_is_not_flagged():
    row = record(
        "请计算 metric 表 value 列的总和。",
        "SELECT SUM(value) FROM metric",
    )

    assert classify(row) == []
    result = audit([row])
    assert result["records_with_any_flag"] == 0

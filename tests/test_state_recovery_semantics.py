from scripts.analyze_state_recovery_semantics import (
    critical_token_family,
    semantic_difference,
    sql_signature,
)


def test_semantic_difference_labels_table_filter_aggregation_and_select_changes():
    error = "SELECT COUNT(*) FROM shipments"
    correction = (
        "SELECT carrier, SUM(amount) FROM delivery_costs "
        "WHERE year = 2026 GROUP BY carrier"
    )

    result = semantic_difference(error, correction)

    assert result["primary"] == "table_grounding"
    assert "aggregation_grouping" in result["labels"]
    assert "filter_semantics" in result["labels"]
    assert "select_expression" in result["labels"]


def test_sql_signature_ignores_whitespace_and_extracts_join_temporal_terms():
    signature = sql_signature(
        "SELECT SUM(a.value) FROM alpha a JOIN beta b ON a.id=b.id "
        "WHERE strftime('%Y', a.date)='2026'"
    )

    assert signature["tables"] == ["alpha", "beta"]
    assert signature["aggregates"] == ["SUM"]
    assert signature["join_count"] == 1
    assert signature["temporal_terms"] == ["date", "strftime"]


def test_critical_token_family_separates_query_plan_from_identifiers():
    assert critical_token_family("ĠSUM") == "aggregation_function"
    assert critical_token_family("SELECT") == "query_start"
    assert critical_token_family("ĠFROM") == "clause_keyword"
    assert critical_token_family("Ġqualification") == "identifier_or_literal"

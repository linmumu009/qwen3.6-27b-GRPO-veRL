import pytest

from scripts.analyze_disjoint_pair_review_pilot import analyze


def fixtures():
    packet = [
        {
            "review_index": 0,
            "task_id": "task-a",
            "natural_language_instruction": "instruction",
            "gold_answer": {"answer_type": "table", "verification_sql": "SELECT 1"},
            "semantic_warnings": ["limit_without_order_by"],
        }
    ]
    pilot = {
        "contract": "disjoint-pair-semantic-review-pilot-v1",
        "selected_tasks": 1,
        "review_required_pool": 138,
        "selection_role": "lowest_mechanical_risk_semantic_approval_rate_pilot",
        "training_allowed": False,
        "promotion_allowed": False,
    }
    stability = {
        "contract": "disjoint-pair-review-pilot-query-stability-v1",
        "training_allowed": False,
        "promotion_allowed": False,
    }
    evidence = [
        {
            "review_index": 0,
            "task_id": "task-a",
            "outcome": "stable_under_reverse_unordered_scan_probe",
        }
    ]
    decisions = {
        "contract": "disjoint-pair-semantic-review-decisions-v1",
        "training_allowed": False,
        "promotion_allowed": False,
        "decisions": [
            {
                "review_index": 0,
                "decision": "rejected",
                "instruction_unambiguously_entails_gold": False,
                "verification_sql_fully_answers_instruction": False,
                "expected_value_supported_by_query_result": True,
                "reason_code": "underspecified_metric_or_grouping",
                "confidence": "high",
                "severity": "high",
            }
        ],
    }
    return packet, pilot, stability, evidence, decisions


def test_review_summary_rejects_semantically_misaligned_stable_task():
    packet, pilot, stability, evidence, decisions = fixtures()
    summary, sensitive = analyze(
        packet=packet,
        pilot=pilot,
        stability=stability,
        stability_evidence=evidence,
        decisions=decisions,
    )
    assert summary["semantic_review"]["approved"] == 0
    assert summary["semantic_review"]["rejected"] == 1
    assert summary["semantic_review"]["expected_value_supported_by_query_result"] == 1
    assert summary["data_quality_finding"]["severity"] == "high"
    assert summary["decision"]["stop_reviewing_remaining96_from_same_queue_now"] is True
    assert summary["training_allowed"] is False
    assert sensitive[0]["task_id"] == "task-a"


def test_review_summary_refuses_approval_without_all_semantic_gates():
    packet, pilot, stability, evidence, decisions = fixtures()
    decisions["decisions"][0]["decision"] = "approved"
    with pytest.raises(ValueError, match="lacks all three"):
        analyze(
            packet=packet,
            pilot=pilot,
            stability=stability,
            stability_evidence=evidence,
            decisions=decisions,
        )

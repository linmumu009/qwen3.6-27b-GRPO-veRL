import pytest

from scripts.replay_boss_reward_shadow import index_unique, summarize


def test_replay_refuses_duplicate_task_identity():
    with pytest.raises(ValueError, match="duplicate trajectory task_id"):
        index_unique([{"task_id": "a"}, {"task_id": "a"}], "trajectory")


def test_shadow_summary_exposes_dwh_and_kb_gates():
    rows = [
        {
            "task_family": "dwh",
            "score": 1.0,
            "online_eligible": 1.0,
            "acc": 1.0,
            "requires_semantic_judge": 0.0,
            "gold_sql_verified": 1.0,
            "final_answer_correct": 1.0,
            "sql_evidence_correct": 1.0,
            "safe": 1.0,
            "boss_verdict": "partial",
        },
        {
            "task_family": "kb",
            "score": 0.05,
            "online_eligible": 0.0,
            "acc": 0.0,
            "requires_semantic_judge": 1.0,
            "answerable": 0.0,
            "source_documents_ok": 0.0,
            "gold_numbers_ok": 0.0,
            "gold_anchors_ok": 0.0,
            "abstention_detected": 0.0,
            "boss_verdict": "correct",
        },
    ]

    report = summarize(rows)

    assert report["dwh"]["candidate_strict_but_boss_not_correct"] == 1
    assert report["kb"]["unanswerable_boss_correct_without_abstention"] == 1
    assert report["invariants"]["kb_never_online_eligible_without_semantic_judge"] is True

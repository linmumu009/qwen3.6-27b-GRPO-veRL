import pytest

from scripts.diagnose_boss_exact_step100_step200 import diagnose


def _row(task_id, *, total, answer, correct, process, fields=1.0):
    return {
        "task_id": task_id,
        "reward": {
            "reward_total": total,
            "result_complete": answer,
            "result_has_answer": answer,
            "result_correct_numeric": correct,
            "process_tables_hit": process,
            "process_fields_used": fields,
            "process_docs_hit": None,
            "process_task_fit": process,
        },
        "evidence": {"answer_len": 10, "tables_queried": ["fact_x"]},
    }


def test_diagnose_reconciles_mutually_exclusive_driver_buckets():
    left = [
        _row("completion", total=0.75, answer=1, correct=0, process=1),
        _row("correctness", total=1.0, answer=1, correct=1, process=1),
        _row("process", total=0.75, answer=1, correct=0, process=1),
    ]
    right = [
        _row("completion", total=0.0, answer=0, correct=0, process=1),
        _row("correctness", total=0.75, answer=1, correct=0, process=1),
        _row("process", total=0.5625, answer=1, correct=0, process=0.5),
    ]

    result = diagnose(left, right)

    assert result["decomposition_reconciles"] is True
    assert result["reward_sum_delta"] == pytest.approx(-1.1875)
    assert result["bucket_totals"]["completion_churn"] == pytest.approx(-0.75)
    assert result["bucket_totals"]["numeric_correctness"] == pytest.approx(-0.25)
    assert result["bucket_totals"]["process_quality"] == pytest.approx(-0.1875)


def test_diagnose_rejects_mismatched_task_sets():
    with pytest.raises(ValueError, match="identical task ids"):
        diagnose(
            [_row("left", total=0.0, answer=0, correct=0, process=0)],
            [_row("right", total=0.0, answer=0, correct=0, process=0)],
        )

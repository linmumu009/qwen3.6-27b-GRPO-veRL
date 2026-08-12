from scripts.check_native_anchor_pareto_gate import evaluate


def _attribution() -> dict:
    return {
        "contract": "native-vs-step120-full25-attribution-safe-summary-v1",
        "decision": {
            "future_pareto_gate": {
                "recognized_sqlite_task_floor": 30,
                "complete_count_floor": 40,
                "correct_numeric_count_floor": 7,
                "reward_total_mean_floor": 0.28,
                "wrong_process_ok_ceiling": 8,
            }
        },
    }


def _reward(*, complete: bool, correct: bool, total: float, wrong_ok: bool) -> dict:
    return {
        "task_id": "placeholder",
        "reward": {
            "reward_total": total,
            "result_complete": int(complete),
            "result_has_answer": int(complete),
            "result_correct_numeric": int(correct),
        },
        "evidence": {
            "verdict_fine": "result_wrong_process_ok" if wrong_ok else "timeout"
        },
    }


def _rows() -> list[dict]:
    rows = []
    for index in range(64):
        row = _reward(
            complete=index < 40,
            correct=index < 7,
            total=0.30,
            wrong_ok=index < 8,
        )
        row["task_id"] = f"task_{index}"
        rows.append(row)
    return rows


def test_candidate_must_pass_every_pareto_dimension() -> None:
    passed = evaluate(
        attribution=_attribution(),
        candidate_tools={"rows": 64, "rows_with_recognized_readonly_sqlite": 30},
        candidate_reward_rows=_rows(),
        candidate_label="candidate",
    )
    assert passed["gate_passed"] is True
    assert passed["promotion_allowed"] is True
    assert passed["additional_training_allowed"] is False

    failed = evaluate(
        attribution=_attribution(),
        candidate_tools={"rows": 64, "rows_with_recognized_readonly_sqlite": 29},
        candidate_reward_rows=_rows(),
        candidate_label="candidate",
    )
    assert failed["gate_passed"] is False
    assert failed["checks"]["recognized_sqlite_task_floor"] is False

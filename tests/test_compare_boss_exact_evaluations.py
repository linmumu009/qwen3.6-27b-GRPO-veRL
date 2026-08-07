from scripts.compare_boss_exact_evaluations import summarize


def test_summarize_reproduces_boss_reward_components():
    rows = [
        {
            "task_id": "a",
            "reward": {
                "reward_total": 0.75,
                "result_complete": 1,
                "result_has_answer": 1,
                "result_correct_numeric": 0,
                "process_tables_hit": 1,
                "process_fields_used": 1.0,
                "process_docs_hit": None,
                "process_task_fit": 1,
                "efficiency_n_turns": 10,
                "efficiency_n_sql": 3,
                "efficiency_n_cmds": 4,
                "efficiency_dup_cmd": 1,
            },
            "evidence": {
                "answer_len": 100,
                "verdict": "partial",
                "verdict_fine": "result_wrong_process_ok",
            },
        },
        {
            "task_id": "b",
            "reward": {
                "reward_total": 0.0,
                "result_complete": 0,
                "result_has_answer": 0,
                "result_correct_numeric": 0,
                "process_tables_hit": 0,
                "process_fields_used": 0.0,
                "process_docs_hit": None,
                "process_task_fit": 0,
                "efficiency_n_turns": 20,
                "efficiency_n_sql": 5,
                "efficiency_n_cmds": 8,
                "efficiency_dup_cmd": 2,
            },
            "evidence": {
                "answer_len": 0,
                "verdict": "incomplete",
                "verdict_fine": "timeout",
            },
        },
    ]

    result = summarize(rows)

    assert result["reward_total_mean"] == 0.375
    assert result["result_score_mean"] == 0.25
    assert result["process_score_mean"] == 0.5
    assert result["complete_count"] == 1
    assert result["n_sql_mean"] == 4
    assert result["verdict_counts"] == {"incomplete": 1, "partial": 1}

from scripts.analyze_force_final_sentinel import analyze
from scripts.prepare_force_final_sentinel import DEFAULT_TASK_IDS


def row(task_id: str, reward: float, complete: int, turns: int = 20):
    return {
        "task_id": task_id,
        "reward": {
            "reward_total": reward,
            "result_complete": complete,
            "result_has_answer": complete,
            "efficiency_n_turns": turns,
            "efficiency_n_sql": 3,
            "efficiency_n_cmds": 5,
        },
        "evidence": {"verdict": "partial" if complete else "incomplete", "verdict_fine": "test"},
    }


def test_gate_passes_with_two_rescues_and_unchanged_guardrails():
    baseline = [row(task_id, 0.0, 0, 26) for task_id in DEFAULT_TASK_IDS[:4]] + [
        row(task_id, 0.75, 1, 12) for task_id in DEFAULT_TASK_IDS[4:]
    ]
    sentinel = [row(task_id, 0.6 if index < 2 else 0.0, int(index < 2), 23) for index, task_id in enumerate(DEFAULT_TASK_IDS[:4])] + [
        row(task_id, 0.75, 1, 12) for task_id in DEFAULT_TASK_IDS[4:]
    ]
    result = analyze(baseline, sentinel, {"all_terminal": True})
    assert result["gate"]["passed"]
    assert result["gate"]["decision"] == "proceed_to_5_step_canary"


def test_gate_fails_on_guardrail_reward_regression():
    baseline = [row(task_id, 0.0, 0) for task_id in DEFAULT_TASK_IDS[:4]] + [
        row(task_id, 0.75, 1) for task_id in DEFAULT_TASK_IDS[4:]
    ]
    sentinel = [row(task_id, 0.6, 1) for task_id in DEFAULT_TASK_IDS[:4]] + [
        row(DEFAULT_TASK_IDS[4], 0.5, 1),
        row(DEFAULT_TASK_IDS[5], 0.75, 1),
    ]
    result = analyze(baseline, sentinel, {"all_terminal": True})
    assert not result["gate"]["passed"]
    assert not result["gate"]["guardrails_preserved"]
    assert result["gate"]["decision"] == "refine_force_final_policy_before_training"


def test_gate_fails_closed_without_terminal_adapter_evidence():
    baseline = [row(task_id, 0.0, 0) for task_id in DEFAULT_TASK_IDS[:4]] + [
        row(task_id, 0.75, 1) for task_id in DEFAULT_TASK_IDS[4:]
    ]
    sentinel = [row(task_id, 0.6, 1) for task_id in DEFAULT_TASK_IDS[:4]] + [
        row(task_id, 0.75, 1) for task_id in DEFAULT_TASK_IDS[4:]
    ]
    result = analyze(baseline, sentinel)
    assert not result["gate"]["passed"]
    assert not result["gate"]["adapter_all_terminal"]
    assert result["gate"]["decision"] == "refine_force_final_policy_before_training"

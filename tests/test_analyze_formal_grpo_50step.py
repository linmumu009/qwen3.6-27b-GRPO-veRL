import json
from pathlib import Path

import pytest

from scripts.analyze_formal_grpo_50step import (
    expected_reward,
    extract_bash_commands,
    parse_driver,
    summarize_rollouts,
    unsafe_reasons,
)


def _row(prompt: str, score: float, **components):
    return {
        "input": prompt,
        "output": '<tool_call>{"name":"bash"}</tool_call>',
        "score": score,
        "safe": 1.0,
        "valid_tool_protocol": 1.0,
        "successful_bash": 1.0,
        "required_table_used": 0.0,
        "has_final_answer": 0.0,
        "final_answer_correct": 0.0,
        "sql_evidence_correct": 0.0,
        "acc": 0.0,
        **components,
    }


def test_rollout_audit_finds_zero_variance_groups_and_reward_contract(tmp_path: Path):
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    rows = [_row("a", 0.1, required_table_used=1.0) for _ in range(4)]
    rows += [
        _row("b", 0.05, has_final_answer=1.0),
        _row("b", 0.65, has_final_answer=1.0, final_answer_correct=1.0),
        _row("b", 0.05, has_final_answer=1.0),
        _row("b", 0.65, has_final_answer=1.0, final_answer_correct=1.0),
    ]
    (rollout_dir / "1.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = summarize_rollouts(rollout_dir, expected_steps=1, expected_rows_per_step=8)

    assert result["integrity"]["reward_formula_mismatches"] == 0
    assert result["groups"]["total"] == 2
    assert result["groups"]["zero_reward_variance"] == 1
    assert result["components"]["final_answer_correct"]["count"] == 2


def test_current_boss_primary_reward_is_reconstructed_with_hard_gate():
    row = {
        "safe": 1.0,
        "valid_tool_protocol": 1.0,
        "gold_sql_verified": 1.0,
        "boss_reward": 0.8,
        "evidence_reward": 0.5,
    }

    assert expected_reward(row) == pytest.approx(0.71)
    assert expected_reward({**row, "gold_sql_verified": 0.0}) == 0.0


def test_driver_audit_compares_first_and_last_windows(tmp_path: Path):
    path = tmp_path / "driver.log"
    path.write_text(
        "step:1 - training/global_step:1 - critic/score/mean:np.float64(0.1) - actor/ppo_kl:np.float64(0.001)\n"
        "step:2 - training/global_step:2 - critic/score/mean:np.float64(0.2) - actor/ppo_kl:np.float64(0.002)\n"
        "step:10 - rollouter/validate_time:12.0 - val-core/source/acc/mean@1:np.float64(0.0) "
        "- val-aux/source/reward/mean@1:np.float64(0.08)\n"
        "[LLIN_TRAIN_STAGE] step=2 queue_wait_s=10 deserialize_s=1 assemble_s=2 "
        "reward_s=3 old_log_prob_s=4 ref_log_prob_s=5 adv_s=6 update_actor_s=7 step_s=40\n",
        encoding="utf-8",
    )

    result = parse_driver(path)

    assert result["steps"] == 2
    assert result["metrics"]["critic/score/mean"]["mean"] == pytest.approx(0.15)
    assert result["metrics"]["actor/ppo_kl"]["max"] == 0.002
    assert result["validation"][0]["step"] == 10
    assert result["validation"][0]["val-aux/source/reward/mean@1"] == 0.08
    assert result["fully_async_stage_timing"]["queue_wait_s"]["mean"] == 10.0
    assert result["fully_async_stage_timing"]["queue_wait_share"] == 0.25


def test_tool_call_audit_matches_reward_safety_patterns():
    output = (
        '<tool_call>{"name":"bash","arguments":{"command":"find /data -name x"}}</tool_call>'
        '<tool_call>{"name":"read","arguments":{"path":"a"}}</tool_call>'
    )

    assert extract_bash_commands(output) == ["find /data -name x"]
    assert unsafe_reasons("find /data -name x") == ["host_path_escape"]

    qwen = """<tool_call>
<function=bash>
<parameter=command>
find / -name '*.sqlite' 2>/dev/null
</parameter>
</function>
</tool_call>"""
    assert extract_bash_commands(qwen) == ["find / -name '*.sqlite' 2>/dev/null"]

import pytest

from scripts.analyze_boss_validation import summarize


def row(task_id: str, answer_type: str, boss: float, evidence: float, acc: float = 0.0):
    return {
        "gts": {"task_id": task_id, "answer_type": answer_type},
        "score": 0.7 * boss + 0.3 * evidence,
        "boss_reward": boss,
        "evidence_reward": evidence,
        "acc": acc,
        "safe": 1.0,
        "valid_tool_protocol": 1.0,
        "gold_sql_verified": 1.0,
        "boss_fields_used": None,
        "verifier_error": "",
    }


def test_summary_keeps_none_diagnostic_without_breaking_means():
    result = summarize(
        [
            row("a", "numeric", 0.8, 0.5, 1.0),
            row("b", "table", 0.4, 0.2),
        ]
    )

    assert result["integrity"]["rows"] == 2
    assert result["integrity"]["reward_formula_mismatches"] == 0
    assert result["integrity"]["none_fields"]["boss_fields_used"] == 2
    assert result["metrics"]["score"]["mean"] == pytest.approx(0.525)
    assert result["by_answer_type"]["numeric"]["strict_correct"] == 1
    assert result["safety_reconstruction"]["rows_with_unsafe_command"] == 0


def test_summary_detects_reward_formula_mismatch():
    bad = row("a", "numeric", 1.0, 1.0)
    bad["score"] = 0.0

    result = summarize([bad])

    assert result["integrity"]["reward_formula_mismatches"] == 1


def test_summary_reconstructs_unsafe_bash_command():
    unsafe = row("a", "numeric", 0.0, 0.0)
    unsafe["safe"] = 0.0
    unsafe["score"] = 0.0
    unsafe["output"] = (
        '<tool_call>{"name":"bash","arguments":{"command":"find /data -name x"}}</tool_call>'
    )

    result = summarize([unsafe])

    assert result["safety_reconstruction"]["rows_with_unsafe_command"] == 1
    assert result["safety_reconstruction"]["unsafe_reasons"] == {"host_path_escape": 1}
    assert result["safety_reconstruction"]["safe_field_vs_reconstruction_mismatches"] == 0
    assert result["safety_reconstruction"]["formal_rows_with_unsafe_command"] == 1
    assert result["safety_reconstruction"]["formal_unsafe_reasons"] == {"host_path_escape": 1}
    assert result["safety_reconstruction"]["safe_field_vs_formal_replay_mismatches"] == 0

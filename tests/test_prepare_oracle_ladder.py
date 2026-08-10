from scripts.prepare_oracle_ladder import TASK_CONTRACT, build


def row(task_id: str, value: object = 10) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"question {task_id}"},
        ],
        "reward_model": {
            "ground_truth": {
                "task_id": task_id,
                "expected_value_json": __import__("json").dumps(value),
            }
        },
    }


def test_builds_three_isolated_arms_without_mutating_control():
    source = [row("a", 10), row("b", {"x": 20})]
    result = build(source, ("a", "b"))

    assert list(result) == ["control", "contract", "oracle"]
    assert result["control"][0]["prompt"][-1]["content"] == "question a"
    assert TASK_CONTRACT in result["contract"][0]["prompt"][-1]["content"]
    assert "结构化结果为：10" in result["oracle"][0]["prompt"][-1]["content"]
    assert "结构化结果为：{\"x\": 20}" in result["oracle"][1]["prompt"][-1]["content"]
    assert source[0]["prompt"][-1]["content"] == "question a"


def test_missing_requested_task_fails_closed():
    import pytest

    with pytest.raises(ValueError, match="missing"):
        build([row("a")], ("a", "b"))

from scripts.prepare_pi_dataset import build_records


def test_build_records_creates_verl_tool_metadata():
    prompts = [
        {
            "verifier_id": "env:task",
            "messages": [
                {"role": "system", "content": "old"},
                {"role": "user", "content": "问题"},
            ],
        }
    ]
    verifier = {
        "env:task": {
            "verifier_id": "env:task",
            "environment_id": "sft/version",
            "required_tables": ["fact_table"],
            "gold": {"answer_type": "numeric", "value": 12.5},
        }
    }
    records = build_records(prompts, verifier)
    assert len(records) == 1
    record = records[0]
    assert record["agent_name"] == "tool_agent"
    assert record["prompt"][0] == {"role": "system", "content": "old"}
    assert record["extra_info"]["tools_kwargs"]["query_sqlite"]["create_kwargs"]["environment_id"] == "sft/version"
    assert record["reward_model"]["ground_truth"]["expected_value"] == 12.5


def test_build_records_falls_back_only_when_source_has_no_system_prompt():
    prompts = [
        {
            "verifier_id": "env:task",
            "messages": [{"role": "user", "content": "问题"}],
        }
    ]
    verifier = {
        "env:task": {
            "verifier_id": "env:task",
            "environment_id": "sft/version",
            "required_tables": ["fact_table"],
            "gold": {"answer_type": "numeric", "value": 12.5},
        }
    }

    record = build_records(prompts, verifier)[0]
    assert record["prompt"][0]["role"] == "system"
    assert record["prompt"][1] == {"role": "user", "content": "问题"}

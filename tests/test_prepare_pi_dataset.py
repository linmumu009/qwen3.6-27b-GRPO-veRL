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
    assert record["extra_info"]["tools_kwargs"]["query_sqlite"]["create_kwargs"]["environment_id"] == "sft/version"
    assert record["reward_model"]["ground_truth"]["expected_value"] == 12.5

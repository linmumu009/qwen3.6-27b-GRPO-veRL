from pathlib import Path

import yaml

from llin_verl.boss_pi_contract import canonical_json, load_boss_pi_contract
from llin_verl.pi_tool_contract import (
    command_is_safe,
    command_unsafe_reasons,
    extract_table_names,
    route_sqlite_cli,
)


def test_pi_command_contract_blocks_network_host_escape_and_process_control():
    assert command_is_safe('sqlite3 /workspace/logistics.sqlite "SELECT 1"')
    assert not command_is_safe("curl https://example.com")
    assert not command_is_safe("cat /data/renjunxiang/private")
    assert not command_is_safe("docker ps")
    assert command_unsafe_reasons("curl https://example.com") == ["network"]
    assert command_unsafe_reasons("cat /data/renjunxiang/private") == ["host_path_escape"]
    assert command_unsafe_reasons("docker ps") == ["destructive"]
    assert command_unsafe_reasons("find / -name '*.sqlite'") == ["root_scan"]
    assert command_unsafe_reasons("ls /workspace/") == []


def test_extract_table_names_from_full_pi_bash_command():
    command = 'sqlite3 /workspace/logistics.sqlite "SELECT * FROM fact_a JOIN dim_b USING(id)"'
    assert extract_table_names(command) == ["dim_b", "fact_a"]


def test_missing_image_sqlite_binary_is_transparently_routed():
    command = 'cd /workspace && sqlite3 logistics.sqlite "SELECT 1"'
    assert route_sqlite_cli(command) == (
        'cd /workspace && python3 -m llin_verl.pi_sqlite_cli logistics.sqlite "SELECT 1"'
    )


def test_runtime_tool_schemas_are_byte_semantically_equal_to_boss_contract():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "pi_workspace_tools.yaml").read_text(encoding="utf-8"))
    runtime_schemas = [item["tool_schema"] for item in config["tools"]]

    assert canonical_json(runtime_schemas) == canonical_json(load_boss_pi_contract()["tools"])


def test_runtime_tools_allow_a_server_scoped_sandbox_root_override():
    root = Path(__file__).resolve().parents[1]
    workspace_source = (root / "llin_verl" / "pi_workspace_tools.py").read_text(encoding="utf-8")
    sqlite_source = (root / "llin_verl" / "pi_sqlite_tool.py").read_text(encoding="utf-8")

    assert 'os.environ.get("PI_AGENT_SANDBOX_LOWER")' in workspace_source
    assert 'os.environ.get("PI_AGENT_SANDBOX_LOWER")' in sqlite_source


def test_runtime_persists_exact_tool_response_token_cost_or_fails_closed():
    root = Path(__file__).resolve().parents[1]
    workspace_source = (root / "llin_verl" / "pi_workspace_tools.py").read_text(encoding="utf-8")
    launcher_source = (root / "scripts" / "run_pi_grpo_fully_async_tp4_pp2_cp2.sh").read_text(encoding="utf-8")
    canary_config = yaml.safe_load(
        (root / "configs" / "pi_workspace_tools_relaxed1800.yaml").read_text(encoding="utf-8")
    )

    assert 'Tokenizer.from_file(str(tokenizer_json))' in workspace_source
    assert '.encode(value, add_special_tokens=False).ids' in workspace_source
    assert '"response_token_count": response_token_count' in workspace_source
    assert 'response_token_count = None' in workspace_source
    assert 'PI_AGENT_TOKENIZER_PATH="${PI_AGENT_TOKENIZER_PATH:-${MODEL_PATH}}"' in launcher_source
    assert all(
        item["config"]["response_tokenizer_path"] == "/models/Qwen3.8-27B"
        for item in canary_config["tools"]
    )
    assert "len(response.encode" not in workspace_source


def test_agent_loop_persists_one_request_and_environment_identity_through_reward():
    root = Path(__file__).resolve().parents[1]
    source = (root / "llin_verl" / "pi_agent_loop.py").read_text(encoding="utf-8")
    identity = (root / "llin_verl" / "pi_workspace_identity.py").read_text(encoding="utf-8")
    reward = (root / "llin_verl" / "tiered_query_cost_reward.py").read_text(encoding="utf-8")

    for required in (
        '"request_id": request_id',
        '"pi_trajectory_request_id": request_id',
        '"pi_trajectory_environment_id": environment_id',
        '"pi_environment_id": environment_id',
    ):
        assert required in source
    for required in (
        "workspace request identity changed before reward",
        "workspace environment identity changed before reward",
    ):
        assert required in identity
    for required in (
        'extra_info.get("pi_trajectory_request_id")',
        'extra_info.get("pi_trajectory_environment_id")',
        'event.get("workspace_request_id")',
        'event.get("environment_id")',
        '"runtime_identity_incomplete"',
    ):
        assert required in reward


def test_timeout_workspace_snapshot_is_not_looked_up_after_release():
    root = Path(__file__).resolve().parents[1]
    source = (root / "llin_verl" / "pi_agent_loop.py").read_text(encoding="utf-8")

    assert 'workspace_state = workspace_binding_state(' in source
    assert 'if workspace_state == "live":' in source
    assert 'if not snapshot:' in source
    assert 'raise RuntimeError("live workspace disappeared before reward")' in source

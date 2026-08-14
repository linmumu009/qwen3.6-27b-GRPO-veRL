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

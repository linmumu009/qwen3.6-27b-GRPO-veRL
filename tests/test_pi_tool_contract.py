from llin_verl.pi_tool_contract import command_is_safe, extract_table_names, route_sqlite_cli


def test_pi_command_contract_blocks_network_host_escape_and_process_control():
    assert command_is_safe('sqlite3 /workspace/logistics.sqlite "SELECT 1"')
    assert not command_is_safe("curl https://example.com")
    assert not command_is_safe("cat /data/renjunxiang/private")
    assert not command_is_safe("docker ps")


def test_extract_table_names_from_full_pi_bash_command():
    command = 'sqlite3 /workspace/logistics.sqlite "SELECT * FROM fact_a JOIN dim_b USING(id)"'
    assert extract_table_names(command) == ["dim_b", "fact_a"]


def test_missing_image_sqlite_binary_is_transparently_routed():
    command = 'cd /workspace && sqlite3 logistics.sqlite "SELECT 1"'
    assert route_sqlite_cli(command) == (
        'cd /workspace && python3 -m llin_verl.pi_sqlite_cli logistics.sqlite "SELECT 1"'
    )

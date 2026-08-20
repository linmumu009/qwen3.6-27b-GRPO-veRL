from scripts.prepare_qwen38_fresh_v22_v26 import HOSTS, _balanced_assign


def _row(index: int, difficulty: int) -> dict:
    return {
        "extra_info": {
            "instruction_sha256": f"identity-{index:04d}",
            "difficulty_level": difficulty,
        }
    }


def test_balanced_assignment_uses_all_three_hosts_without_overlap() -> None:
    sizes = {
        "v23_pilot100": 100,
        "v23_rest400": 400,
        "v24": 500,
        "v25": 500,
        "v26": 500,
    }
    arms = {}
    index = 0
    for arm, size in sizes.items():
        arms[arm] = [_row(index + offset, offset % 5 + 1) for offset in range(size)]
        index += size

    assigned = _balanced_assign(arms, seed="fixture")

    totals = {
        host: sum(len(rows) for rows in assigned[host].values()) for host in HOSTS
    }
    identities = [
        row["extra_info"]["instruction_sha256"]
        for host in HOSTS
        for rows in assigned[host].values()
        for row in rows
    ]
    assert sorted(totals.values()) == [666, 667, 667]
    assert len(identities) == len(set(identities)) == 2000
    for arm, size in sizes.items():
        assert sum(len(assigned[host][arm]) for host in HOSTS) == size

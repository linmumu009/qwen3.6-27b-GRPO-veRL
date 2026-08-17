from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.prepare_qwen38_three_host_rerun import partition_version


def row(value: int) -> dict:
    return {
        "prompt": [{"role": "user", "content": f"task {value}"}],
        "extra_info": {"difficulty_level": value % 5 + 1},
    }


def test_three_host_partition_is_exact_disjoint_and_balanced(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([row(i) for i in range(50)]), source)
    manifest = partition_version(
        source,
        tmp_path / "out",
        version="v21",
        allocation={"m05": 18, "m06": 17, "m00": 15},
    )
    assert manifest["source_tasks"] == 50
    assert {host: data["tasks"] for host, data in manifest["partitions"].items()} == {
        "m05": 18,
        "m06": 17,
        "m00": 15,
    }
    seen = set()
    for host, expected in (("m05", 18), ("m06", 17), ("m00", 15)):
        rows = pq.read_table(tmp_path / "out" / f"v21_{host}.sensitive.parquet").to_pylist()
        assert len(rows) == expected
        assert all(item["extra_info"]["qwen38_rerun_host"] == host for item in rows)
        assert all(item["extra_info"]["training_allowed"] is False for item in rows)
        prompts = {item["prompt"][0]["content"] for item in rows}
        assert not seen & prompts
        seen |= prompts
        level_counts = manifest["partitions"][host]["difficulty_level_counts"]
        assert max(level_counts.values()) - min(level_counts.values()) <= 1
    assert len(seen) == 50

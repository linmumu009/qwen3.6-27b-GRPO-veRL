from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.split_sensitive_rollout_dataset import split_dataset


def test_split_is_disjoint_complete_and_balanced(tmp_path: Path):
    source = tmp_path / "full.parquet"
    rows = [
        {"extra_info": {"verifier_id": f"v:{index}", "source_version": f"v{index % 3}"}}
        for index in range(7)
    ]
    pq.write_table(pa.Table.from_pylist(rows), source)

    summary = split_dataset(source, tmp_path / "split", expected_rows=7)

    assert summary["arms"]["m05"]["rows"] == 4
    assert summary["arms"]["m06"]["rows"] == 3
    m05 = pq.read_table(tmp_path / "split" / "boss_multisandbox_dwh_m05.sensitive.parquet").to_pylist()
    m06 = pq.read_table(tmp_path / "split" / "boss_multisandbox_dwh_m06.sensitive.parquet").to_pylist()
    ids05 = {row["extra_info"]["verifier_id"] for row in m05}
    ids06 = {row["extra_info"]["verifier_id"] for row in m06}
    assert not ids05 & ids06
    assert ids05 | ids06 == {f"v:{index}" for index in range(7)}

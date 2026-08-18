from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import scripts.prepare_qwen38_heldout_eval as module


def row(version: str, value: int, *, training: bool = False) -> dict:
    return {
        "prompt": [{"role": "user", "content": f"{version} task {value}"}],
        "extra_info": {
            "source_version": version,
            "difficulty_level": value % 5 + 1,
            "instruction_sha256": f"{version}-instruction-{value}",
            "gold_sha256": f"{version}-gold-{value}",
            "training_allowed": training,
            "promotion_allowed": False,
        },
    }


def test_exact_training_exclusion_and_balanced_two_host_partition(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(module, "EXPECTED_FULL_TASKS", {v: 10 for v in module.VERSIONS})
    monkeypatch.setattr(module, "EXPECTED_TRAIN_TASKS", {"v15": 2, "v20": 3, "v21": 1})
    monkeypatch.setattr(module, "EXPECTED_HOLDOUT_TASKS", {"v15": 8, "v20": 7, "v21": 9})
    sources = []
    training = []
    excluded = {"v15": {1, 8}, "v20": {0, 4, 9}, "v21": {5}}
    for version in module.VERSIONS:
        path = tmp_path / f"{version}.parquet"
        rows = [row(version, value) for value in range(10)]
        pq.write_table(pa.Table.from_pylist(rows), path)
        sources.append((version, path))
        training.extend(row(version, value, training=True) for value in excluded[version])
    training_path = tmp_path / "training.parquet"
    pq.write_table(pa.Table.from_pylist(training), training_path)
    monkeypatch.setattr(module, "EXPECTED_TRAIN_TASKS", dict(Counter(r["extra_info"]["source_version"] for r in training)))

    manifest = module.build(
        sources,
        training_pool=training_path,
        allocations={
            "v15": {"m05": 4, "m06": 4},
            "v20": {"m05": 3, "m06": 4},
            "v21": {"m05": 5, "m06": 4},
        },
        output_dir=tmp_path / "out",
        manifest_path=tmp_path / "out" / "heldout.safe.json",
    )
    assert manifest["heldout_tasks"] == 24
    assert manifest["training_overlap_tasks"] == 0
    for version, expected in (("v15", 8), ("v20", 7), ("v21", 9)):
        holdout = pq.read_table(tmp_path / "out" / f"{version}_heldout.sensitive.parquet").to_pylist()
        assert len(holdout) == expected
        assert all(not item["extra_info"]["training_allowed"] for item in holdout)
        holdout_ids = {module.training_identity(item) for item in holdout}
        training_ids = {module.training_identity(item) for item in training}
        assert not holdout_ids & training_ids

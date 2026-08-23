from __future__ import annotations

import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.stage_bound_pi_sandbox import stage


def _dataset(path: Path, environments: list[str]) -> None:
    rows = [
        {
            "extra_info": {"instruction_sha256": f"{index + 1:064x}"},
            "reward_model": {"ground_truth": {"environment_id": environment}},
        }
        for index, environment in enumerate(environments)
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)


def _environment(root: Path, name: str, value: int) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    with sqlite3.connect(directory / "logistics.sqlite") as connection:
        connection.execute("CREATE TABLE metric(value INTEGER)")
        connection.execute("INSERT INTO metric VALUES (?)", (value,))
    (directory / "README.md").write_text("private task metadata", encoding="utf-8")


def test_stage_copies_only_bound_environment_union_and_hashes_database_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _environment(source, "sft/v1", 1)
    _environment(source, "sft/v2", 2)
    _environment(source, "sft/excluded", 3)
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    _dataset(first, ["sft/v1", "sft/v2"])
    _dataset(second, ["sft/v1"])
    output = tmp_path / "private" / "pi_sandbox"
    summary_path = tmp_path / "audit" / "sandbox.safe.json"

    summary = stage([first, second], source, output, summary_path)

    assert summary["dataset_rows"] == 3
    assert summary["unique_instruction_identities"] == 2
    assert summary["unique_environments"] == 2
    assert summary["database_files"] == 2
    assert len(summary["database_manifest_compound_sha256"]) == 64
    assert (output / "sft/v1/logistics.sqlite").is_file()
    assert (output / "sft/v2/logistics.sqlite").is_file()
    assert not (output / "sft/excluded").exists()
    assert summary_path.is_file()


def test_stage_fails_closed_on_invalid_environment_or_existing_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _environment(source, "sft/v1", 1)
    invalid = tmp_path / "invalid.parquet"
    _dataset(invalid, ["../escape"])
    with pytest.raises(ValueError):
        stage([invalid], source, tmp_path / "out", tmp_path / "safe.json")

    valid = tmp_path / "valid.parquet"
    _dataset(valid, ["sft/v1"])
    output = tmp_path / "out"
    stage([valid], source, output, tmp_path / "first.safe.json")
    with pytest.raises(FileExistsError):
        stage([valid], source, output, tmp_path / "second.safe.json")

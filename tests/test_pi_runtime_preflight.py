from pathlib import Path
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.pi_runtime_preflight import validate_dataset_runtime_environments


def write_dataset(path: Path, environment_id: str) -> None:
    table = pa.Table.from_pylist(
        [{"extra_info": {"environment_id": environment_id, "index": 0}}]
    )
    pq.write_table(table, path)


def make_runtime(root: Path, environment_id: str) -> Path:
    runtime = root.joinpath(*environment_id.split("/"))
    runtime.mkdir(parents=True)
    database = runtime / "logistics.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE fact_metric(category TEXT, value REAL)")
    connection.commit()
    connection.close()
    (runtime / "schema_dictionary.md").write_text(
        "fact_metric(category, value)\n",
        encoding="utf-8",
    )
    (runtime / "documents").mkdir()
    return runtime


def test_runtime_preflight_opens_tool_visible_database(tmp_path: Path) -> None:
    dataset = tmp_path / "tasks.parquet"
    root = tmp_path / "pi_sandbox"
    environment_id = "sft/example_runtime"
    root.mkdir()
    write_dataset(dataset, environment_id)
    make_runtime(root, environment_id)

    result = validate_dataset_runtime_environments(dataset, root)

    assert result["valid"] is True
    assert result["environment_count"] == 1
    assert result["sqlite_opened_read_only"] is True
    assert result["environment_ids_emitted"] is False


def test_runtime_preflight_rejects_database_in_a_different_root(tmp_path: Path) -> None:
    dataset = tmp_path / "tasks.parquet"
    visible_root = tmp_path / "pi_sandbox"
    wrong_root = tmp_path / "project_sandboxes"
    environment_id = "sft/example_runtime"
    visible_root.mkdir()
    write_dataset(dataset, environment_id)
    make_runtime(wrong_root, environment_id)

    with pytest.raises(FileNotFoundError, match="missing_or_invalid=1"):
        validate_dataset_runtime_environments(dataset, visible_root)


def test_runtime_preflight_rejects_unsafe_environment_id(tmp_path: Path) -> None:
    dataset = tmp_path / "tasks.parquet"
    root = tmp_path / "pi_sandbox"
    root.mkdir()
    write_dataset(dataset, "../escape")

    with pytest.raises(ValueError, match="unsafe environment_id"):
        validate_dataset_runtime_environments(dataset, root)

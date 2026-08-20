from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.compare_qwen38_native_step70_strict_train70 import (
    aggregate_safe,
    compare_host,
)


def _row(identity: str, source: str, difficulty: int) -> dict:
    return {
        "extra_info": {
            "instruction_sha256": identity,
            "source_version": source,
            "difficulty_level": difficulty,
        }
    }


def _write(path: Path, rows: list[dict]) -> Path:
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_compare_and_aggregate_strict_transitions(tmp_path: Path) -> None:
    rows = [
        _row("a", "v15", 2),
        _row("b", "v15", 3),
        _row("c", "v20", 4),
        _row("d", "v21", 5),
    ]
    host_safe = tmp_path / "host.safe.json"
    result = compare_host(
        _write(tmp_path / "approved.parquet", rows),
        _write(tmp_path / "native.parquet", rows[:2]),
        _write(tmp_path / "step.parquet", rows[1:3]),
        host_safe,
        expected_approved=4,
        host_label="m05",
    )
    assert result["retained_tasks"] == 1
    assert result["lost_tasks"] == 1
    assert result["gained_tasks"] == 1
    assert result["neither_tasks"] == 1
    assert result["transition_by_source_version"]["retained"] == {"v15": 1}
    assert result["transition_by_difficulty"]["gained"] == {"4": 1}
    assert "instruction_sha256" not in host_safe.read_text(encoding="utf-8")

    aggregate = aggregate_safe(
        [host_safe],
        tmp_path / "aggregate.safe.json",
        expected_tasks=4,
        expected_hosts=1,
    )
    assert aggregate["native_strict_mixed_tasks"] == 2
    assert aggregate["step70_strict_mixed_tasks"] == 2
    assert aggregate["native_to_step70_retention_rate"] == 0.5
    assert aggregate["step70_strict_mixed_change"] == 0


def test_compare_rejects_unapproved_qualified_identity(tmp_path: Path) -> None:
    approved = _write(tmp_path / "approved.parquet", [_row("a", "v15", 2)])
    native = _write(tmp_path / "native.parquet", [_row("outside", "v15", 2)])
    step = _write(tmp_path / "step.parquet", [])
    with pytest.raises(ValueError, match="approved tasks"):
        compare_host(
            approved,
            native,
            step,
            tmp_path / "safe.json",
            expected_approved=1,
            host_label="m05",
        )

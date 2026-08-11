import json
from pathlib import Path

import pytest

from scripts.copy_file_from_ray_resource import validate_payload, write_atomic


def test_validate_payload_checks_jsonl_row_count_and_content():
    payload = b'{"task": 1}\n{"task": 2}\n'

    summary = validate_payload(payload, expected_jsonl_rows=2)

    assert summary["jsonl_rows"] == 2
    assert summary["bytes"] == len(payload)
    assert len(summary["sha256"]) == 64


@pytest.mark.parametrize(
    "payload, rows",
    [
        (b"", None),
        (b'{"task": 1}\n', 2),
        (b'{"task": 1}\nnot-json\n', 2),
    ],
)
def test_validate_payload_rejects_incomplete_artifacts(payload: bytes, rows: int | None):
    with pytest.raises(ValueError):
        validate_payload(payload, expected_jsonl_rows=rows)


def test_write_atomic_replaces_destination(tmp_path: Path):
    destination = tmp_path / "validation" / "125.jsonl"
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps({"old": True}), encoding="utf-8")

    write_atomic(destination, b'{"new": true}\n')

    assert destination.read_bytes() == b'{"new": true}\n'
    assert not (destination.parent / ".125.jsonl.ray-copy.tmp").exists()

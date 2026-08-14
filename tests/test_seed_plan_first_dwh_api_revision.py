from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.generate_plan_first_dwh_sandbox import generate
from scripts.seed_plan_first_dwh_api_revision import seed_revision


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_selective_revision_reuses_only_validated_unchanged_bands(tmp_path) -> None:
    base = generate(tmp_path / "base", "base-v3")
    previous = generate(tmp_path / "previous", "previous-api-v1")
    rows = _read_jsonl(previous / "dwh_tasks.jsonl")
    for row in rows:
        row["instruction_generation"] = {
            "method": "boss_openai_compatible_chat_api",
            "semantic_validation_passed": True,
        }
    with (previous / "dwh_tasks.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    output = tmp_path / "api-v2.incomplete"
    result = seed_revision(base, previous, output, {4})

    seeded = _read_jsonl(output / "dwh_tasks.jsonl")
    assert result["reused_validated_rows"] == 250
    assert result["pending_rewrite_rows"] == 50
    assert {row["difficulty_band"] for row in seeded} == {1, 2, 3, 5, 6}
    assert (output / "logistics.sqlite").exists()


def test_selective_revision_rejects_existing_output(tmp_path) -> None:
    output = tmp_path / "already-there"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        seed_revision(tmp_path / "base", tmp_path / "previous", output, {4})


def test_selective_revision_can_omit_only_rows_rejected_by_new_validator(tmp_path) -> None:
    base = generate(tmp_path / "base", "base-v3")
    previous = generate(tmp_path / "previous", "previous-api-v1")
    rows = _read_jsonl(previous / "dwh_tasks.jsonl")
    rejected_id = rows[0]["task_id"]
    for row in rows:
        row["instruction_generation"] = {
            "method": "boss_openai_compatible_chat_api",
            "semantic_validation_passed": True,
        }
    with (previous / "dwh_tasks.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    def validator(task: dict, _instruction: str) -> list[str]:
        return ["new_gate"] if task["task_id"] == rejected_id else []

    output = tmp_path / "api-v3.incomplete"
    result = seed_revision(
        base,
        previous,
        output,
        set(),
        validator=validator,
        rewrite_invalid=True,
    )

    seeded = _read_jsonl(output / "dwh_tasks.jsonl")
    assert result["reused_validated_rows"] == 299
    assert result["pending_rewrite_rows"] == 1
    assert rejected_id not in {row["task_id"] for row in seeded}

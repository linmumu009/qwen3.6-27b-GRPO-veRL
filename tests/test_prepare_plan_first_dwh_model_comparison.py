import json
from pathlib import Path

from scripts.prepare_plan_first_dwh_model_comparison import (
    create_runtime_projection,
    ordered_tasks,
)


def task(band: int, index: int) -> dict:
    return {
        "difficulty_band": band,
        "natural_language_instruction": f"band {band} task {index}",
    }


def test_ordered_tasks_puts_stratified_48_row_pilot_first_and_freezes_60():
    rows = [task(band, index) for band in range(1, 7) for index in range(50)]
    ordered = ordered_tasks(rows, seed="test")

    assert len(ordered) == 300
    assert sum(pilot for _, _, pilot in ordered) == 48
    assert sum(split == "frozen_evaluation" for _, split, _ in ordered) == 60
    assert sum(split == "training_candidate" for _, split, _ in ordered) == 240
    first = [row for row, _, _ in ordered[:48]]
    assert {
        band: sum(int(row["difficulty_band"]) == band for row in first)
        for band in range(1, 7)
    } == {band: 8 for band in range(1, 7)}


def test_runtime_projection_exposes_only_database_and_schema(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "logistics.sqlite").write_bytes(b"sqlite")
    (source / "schema_dictionary.md").write_text("schema", encoding="utf-8")
    (source / "dwh_tasks.jsonl").write_text(
        json.dumps({"gold_answer": {"value": "hidden"}}), encoding="utf-8"
    )
    runtime = tmp_path / "runtime" / "sft" / "v3_runtime"

    summary = create_runtime_projection(source, runtime)

    assert summary["visible_files"] == ["logistics.sqlite", "schema_dictionary.md"]
    assert summary["hidden_task_manifest_visible"] is False
    assert (runtime / "documents").is_dir()
    assert not (runtime / "dwh_tasks.jsonl").exists()
    assert {
        str(path.relative_to(runtime)).replace("\\", "/")
        for path in runtime.rglob("*")
        if path.is_file()
    } == {"logistics.sqlite", "schema_dictionary.md"}


def test_runtime_projection_fails_closed_on_extra_visible_file(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "logistics.sqlite").write_bytes(b"sqlite")
    (source / "schema_dictionary.md").write_text("schema", encoding="utf-8")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "logistics.sqlite").write_bytes(b"sqlite")
    (runtime / "schema_dictionary.md").write_text("schema", encoding="utf-8")
    (runtime / "leak.jsonl").write_text("hidden", encoding="utf-8")

    try:
        create_runtime_projection(source, runtime)
    except ValueError as exc:
        assert "unexpected files" in str(exc)
    else:
        raise AssertionError("expected runtime projection to reject the leak")

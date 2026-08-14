import json
from pathlib import Path

from scripts.standalone_rollout_shards import (
    completed_shard_rows,
    padded_rows_for_equal_chunks,
    shard_path,
    shard_ranges,
    write_jsonl_atomic,
)


def test_shard_ranges_cover_remainder_once():
    assert shard_ranges(281, 32)[0] == (0, 32)
    assert shard_ranges(281, 32)[-1] == (256, 281)
    assert sum(stop - start for start, stop in shard_ranges(281, 32)) == 281


def test_tail_batch_padding_only_reaches_next_equal_worker_chunk():
    assert padded_rows_for_equal_chunks(128, 16) == 128
    assert padded_rows_for_equal_chunks(104, 16) == 112
    assert padded_rows_for_equal_chunks(96, 16) == 96


def test_tail_batch_padding_rejects_invalid_shapes():
    for rows, chunks in ((0, 16), (104, 0), (-1, 16), (104, -1)):
        try:
            padded_rows_for_equal_chunks(rows, chunks)
        except ValueError:
            pass
        else:
            raise AssertionError((rows, chunks))


def test_atomic_shard_requires_exact_task_sample_shape(tmp_path: Path):
    path = shard_path(tmp_path, 0, 2)
    rows = [
        {"source_task_index": task, "sample_index": sample, "output": "x"}
        for task in range(2)
        for sample in range(3)
    ]
    assert write_jsonl_atomic(path, rows) == 6
    assert completed_shard_rows(path, start=0, stop=2, samples_per_task=3) == 6
    assert path.stat().st_size > 0
    assert not list(path.parent.glob("*.tmp.*"))

    path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    assert completed_shard_rows(path, start=0, stop=2, samples_per_task=3) == 0

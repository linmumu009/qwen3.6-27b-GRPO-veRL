import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.replay_strict_table_reward_gate import replay


def _task(identity: str) -> dict:
    return {
        "data_source": "test",
        "agent_name": "pi_agent",
        "prompt": [{"role": "user", "content": "sensitive"}],
        "ability": "boss_pi_dwh",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "answer_type": "table",
                "expected_value_json": json.dumps(
                    [{"category": "A", "value": 10}, {"category": "B", "value": 20}]
                ),
                "verification_sql": "SELECT category, value FROM fact_rank",
                "abs_tol": 1e-3,
                "rel_tol": 1e-5,
            },
        },
        "extra_info": {
            "instruction_sha256": identity,
            "training_allowed": False,
            "promotion_allowed": False,
        },
    }


def _trajectory(index: int, sample: int, output: str) -> dict:
    return {
        "source_task_index": index,
        "sample_index": sample,
        "output": output,
        "trajectory_timeout": False,
        "runtime_error": False,
    }


def test_replay_fails_closed_and_writes_only_strict_mixed_rows(tmp_path: Path):
    approved = tmp_path / "approved.parquet"
    dataset = tmp_path / "dataset.parquet"
    rows = [_task("id-a"), _task("id-b")]
    pq.write_table(pa.Table.from_pylist(rows), approved)
    pq.write_table(pa.Table.from_pylist(rows), dataset)
    run_dir = tmp_path / "run"
    shards = run_dir / "shards"
    shards.mkdir(parents=True)
    trajectories = [
        _trajectory(0, 0, "|类别|数值|\n|---|---:|\n|A|10|\n|B|20|"),
        _trajectory(0, 1, "|类别|数值|\n|---|---:|\n|A|0|\n|B|0|"),
        _trajectory(1, 0, "A=20\nB=10"),
        _trajectory(1, 1, "A=0\nB=0"),
    ]
    (shards / "tasks_00000_00002.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trajectories),
        encoding="utf-8",
    )
    safe = tmp_path / "safe.json"
    qualified = tmp_path / "qualified.parquet"

    summary = replay(
        approved,
        [("v20-wave2", dataset, run_dir)],
        safe,
        qualified,
        expected_approved=2,
        host_label="fixture",
    )

    assert summary["legacy_mixed_tasks"] == 2
    assert summary["strict_mixed_tasks"] == 1
    assert summary["tasks_lost_to_strict_judge"] == 1
    assert summary["qualified_private_rows"] == 1
    assert summary["gate_passed"] is False
    assert pq.read_table(qualified).num_rows == 1
    qualified_extra = pq.read_table(qualified).to_pylist()[0]["extra_info"]
    assert qualified_extra["strict_reward_replay_passed"] is True
    assert qualified_extra["training_allowed"] is False
    rendered = safe.read_text(encoding="utf-8")
    assert "id-a" not in rendered
    assert "SELECT" not in rendered
    assert "A=20" not in rendered

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.prepare_qwen38_topology_benchmark import prepare


ROOT = Path(__file__).resolve().parents[1]


def row(value: int) -> dict:
    return {
        "prompt": [{"role": "user", "content": f"task {value}"}],
        "extra_info": {"difficulty_level": value % 5 + 1},
    }


def test_benchmark_subset_is_deterministic_private_and_non_trainable(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([row(i) for i in range(10)]), source)
    output = tmp_path / "bench.parquet"
    manifest_path = tmp_path / "bench.safe.json"
    manifest = prepare(source, output, manifest_path, tasks=4)
    rows = pq.read_table(output).to_pylist()
    assert manifest["source_tasks"] == 10
    assert manifest["selected_tasks"] == 4
    assert manifest["training_allowed"] is False
    assert len(rows) == 4
    assert all(item["extra_info"]["training_allowed"] is False for item in rows)
    assert manifest == json.loads(manifest_path.read_text(encoding="utf-8"))


def test_benchmark_ray_launcher_requires_exact_visible_npus_and_isolated_ports() -> None:
    script = (ROOT / "scripts" / "start_ray_qwen38_topology_benchmark.sh").read_text(
        encoding="utf-8"
    )
    for fragment in (
        'EXPECTED_NPUS="${EXPECTED_NPUS:?EXPECTED_NPUS is required}"',
        "torch_npu.npu.device_count()",
        "expected %s visible NPUs, observed %s",
        '--resources="{\\"${RAY_RESOURCE}\\": 1}"',
        '--min-worker-port="${RAY_MIN_WORKER_PORT}"',
        '--max-worker-port="${RAY_MAX_WORKER_PORT}"',
        '--temp-dir="${RAY_TEMP_DIR}"',
    ):
        assert fragment in script

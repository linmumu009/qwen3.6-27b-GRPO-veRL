from __future__ import annotations

from datetime import date, timedelta
import json
import sqlite3

from scripts.audit_open_multisandbox_dwh import audit_sandbox
from scripts.generate_open_multisandbox_dwh import generate_one, generate_tasks


def _database(path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE fact_waybill (
            waybill_no TEXT PRIMARY KEY,
            created_at TEXT,
            updated_at TEXT,
            status TEXT,
            goods_type TEXT,
            cargo_type TEXT,
            cargo_desc_type TEXT,
            cargo_weight_kg REAL,
            cargo_volume_m3 REAL
        );
        CREATE TABLE fact_pickup (
            pickup_id TEXT, waybill_no TEXT, timestamp TEXT, operator_id TEXT,
            location TEXT, duration_hours REAL, status TEXT, remark TEXT
        );
        CREATE TABLE fact_sorting (
            sorting_id TEXT, waybill_no TEXT, timestamp TEXT, operator_id TEXT,
            location TEXT, duration_hours REAL, status TEXT, remark TEXT
        );
        CREATE TABLE fact_transit (
            transit_id TEXT, waybill_no TEXT, timestamp TEXT, operator_id TEXT,
            location TEXT, duration_hours REAL, status TEXT, remark TEXT
        );
        CREATE TABLE fact_delivery (
            delivery_id TEXT, waybill_no TEXT, timestamp TEXT, operator_id TEXT,
            location TEXT, duration_hours REAL, status TEXT, remark TEXT
        );
        CREATE TABLE fact_pod (
            pod_id TEXT, waybill_no TEXT, timestamp TEXT, operator_id TEXT,
            location TEXT, duration_hours REAL, status TEXT, remark TEXT
        );
        """
    )
    start = date(2025, 1, 1)
    statuses = ("已签收", "运输中", "已取消")
    goods_values = ("普货", "生鲜", "医药", "高值")
    cargo_values = ("文件", "包裹", "托盘", "木箱")
    description_values = ("日用品", "食品", "药品", "设备")
    events = ("pickup", "sorting", "transit", "delivery", "pod")
    for index in range(2400):
        waybill = f"WB{index:05d}"
        day = start + timedelta(days=index % 360)
        status = statuses[index % len(statuses)]
        goods = goods_values[(index // 3) % len(goods_values)]
        cargo = cargo_values[(index // 7) % len(cargo_values)]
        description = description_values[(index // 11) % len(description_values)]
        connection.execute(
            "INSERT INTO fact_waybill VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                waybill,
                day.isoformat(),
                day.isoformat(),
                status,
                goods,
                cargo,
                description,
                10.0 + index % 200,
                0.1 + (index % 50) / 10,
            ),
        )
        for offset, event in enumerate(events, start=1):
            connection.execute(
                f"INSERT INTO fact_{event} VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"{event}-{index}",
                    waybill,
                    day.isoformat(),
                    "operator",
                    "location",
                    float(offset + (index % 11) / 5),
                    "完成",
                    "",
                ),
            )
    connection.commit()
    connection.close()


def test_generate_500_tasks_with_old_band5_as_level1(tmp_path) -> None:
    database = tmp_path / "logistics.sqlite"
    _database(database)

    tasks = generate_tasks(database, source_version="20260628_v15", seed=17)

    assert len(tasks) == 500
    assert {level: sum(row["difficulty_level"] == level for row in tasks) for level in range(1, 6)} == {
        1: 100,
        2: 100,
        3: 100,
        4: 100,
        5: 100,
    }
    assert len({row["natural_language_instruction"] for row in tasks}) == 500
    assert len({row["sample_sql"] for row in tasks}) == 500
    assert all(row["training_allowed"] is False for row in tasks)
    assert all(row["validation"]["difficulty_calibration_status"] == "awaiting_8x_rollout" for row in tasks)
    level1 = [row for row in tasks if row["difficulty_level"] == 1]
    assert all(row["difficulty_baseline"] == "previous_plan_first_band_5" for row in level1)
    assert all(row["query_plan"]["feature_counts"]["essential_joins"] == 2 for row in level1)
    assert all(row["query_plan"]["feature_counts"]["redundant_joins"] == 0 for row in tasks)


def test_generate_one_snapshots_source_and_writes_summary(tmp_path) -> None:
    source = tmp_path / "source.sqlite"
    _database(source)
    output = generate_one(
        source,
        tmp_path / "outputs",
        source_version="20260628_v15",
        output_version="llin_test_v15",
        seed=19,
    )

    rows = [json.loads(line) for line in (output / "dwh_tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((output / "generation_summary.json").read_text(encoding="utf-8"))
    assert len(rows) == 500
    assert summary["task_count"] == 500
    assert summary["difficulty_distribution"] == {str(level): 100 for level in range(1, 6)}
    assert summary["training_allowed"] is False
    assert (output / "logistics.sqlite").read_bytes() == source.read_bytes()
    audit = audit_sandbox(output)
    assert audit["sql_gold_replay_passed_rows"] == 500
    assert audit["semantic_anchor_passed_rows"] == 500
    assert audit["api_naturalized_rows"] == 0

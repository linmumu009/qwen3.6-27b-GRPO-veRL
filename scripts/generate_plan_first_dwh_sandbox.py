#!/usr/bin/env python3
"""Generate one deterministic, plan-first logistics DWH sandbox.

The generator deliberately has no network or model dependency.  A structured
query plan is the single source of truth for both the natural-language
instruction and the executable verification SQL, preventing instruction/gold
semantic drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_VERSION = "20260814_llin_dwh_planfirst_v1"
GENERATOR_CONTRACT = "llin-plan-first-dwh-v1"
ALLOWED_TASK_TYPES = {
    "aggregate_query",
    "single_metric_query",
    "comparison_analysis",
}
FORBIDDEN_SQL = re.compile(
    r"\b(?:ATTACH|DETACH|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|VACUUM|PRAGMA)\b",
    re.IGNORECASE,
)


REGIONS = (
    ("R01", "华东一区", "核心区"),
    ("R02", "华东二区", "核心区"),
    ("R03", "华南一区", "核心区"),
    ("R04", "华南二区", "成长区"),
    ("R05", "华北一区", "核心区"),
    ("R06", "华北二区", "成长区"),
    ("R07", "西南区", "成长区"),
    ("R08", "华中区", "成长区"),
)
WAREHOUSE_TYPES = ("中心仓", "区域仓", "前置仓")
CARRIER_LEVELS = ("战略", "核心", "标准")
CUSTOMER_SEGMENTS = ("大客户", "成长客户", "标准客户")
SERVICE_LEVELS = ("当日达", "次日达", "经济件", "冷链")
STATUSES = ("已签收", "运输中", "已取消")


@dataclass(frozen=True)
class FilterSpec:
    sql: str
    description: str


@dataclass(frozen=True)
class QueryPlan:
    plan_id: str
    difficulty_band: int
    task_type: str
    metric_key: str
    metric_sql: str
    metric_description: str
    joins: tuple[str, ...]
    filters: tuple[FilterSpec, ...]
    group_sql: str | None = None
    group_description: str | None = None
    order_direction: str | None = None
    limit: int | None = None
    having_sql: str | None = None
    having_description: str | None = None


JOIN_SQL = {
    "warehouses": "JOIN warehouses AS w ON w.warehouse_id = s.warehouse_id",
    "regions": "JOIN regions AS r ON r.region_id = w.region_id",
    "carriers": "JOIN carriers AS c ON c.carrier_id = s.carrier_id",
    "customers": "JOIN customers AS u ON u.customer_id = s.customer_id",
}

METRICS = {
    "shipment_count": ("COUNT(*)", "运单数量"),
    "customer_count": ("COUNT(DISTINCT s.customer_id)", "去重客户数量"),
    "freight_sum": ("ROUND(SUM(s.freight_amount), 2)", "运费总额"),
    "weight_sum": ("ROUND(SUM(s.weight_kg), 2)", "货物总重量（千克）"),
    "delivery_avg": ("ROUND(AVG(s.delivery_hours), 2)", "平均配送时长（小时）"),
    "freight_avg": ("ROUND(AVG(s.freight_amount), 2)", "平均单票运费"),
    "delay_rate": (
        "ROUND(100.0 * SUM(s.delayed_flag) / COUNT(*), 2)",
        "延误率（百分比）",
    ),
}


def _sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _month_window(month: int) -> tuple[str, str, str]:
    start = date(2025, month, 1)
    end = date(2026, 1, 1) if month == 12 else date(2025, month + 1, 1)
    # DATE(...) is explicit both to SQLite and to the repository's temporal
    # semantic gate; the stored values are ISO dates, so the meaning is unchanged.
    sql = f"DATE(s.ship_date) >= '{start.isoformat()}' AND DATE(s.ship_date) < '{end.isoformat()}'"
    return sql, start.isoformat(), (end - timedelta(days=1)).isoformat()


def _filter(sql: str, description: str) -> FilterSpec:
    return FilterSpec(sql=sql, description=description)


def _metric(key: str) -> tuple[str, str]:
    return METRICS[key]


def build_plans() -> list[QueryPlan]:
    """Build exactly 300 plans: six auditable bands of 50 tasks."""

    plans: list[QueryPlan] = []
    scalar_metrics = (
        "shipment_count",
        "customer_count",
        "freight_sum",
        "weight_sum",
        "delivery_avg",
    )

    for band in range(1, 7):
        for offset in range(50):
            month = offset // 5 + 1
            metric_key = scalar_metrics[offset % len(scalar_metrics)]
            metric_sql, metric_description = _metric(metric_key)
            month_sql, start, end = _month_window(month)
            plan_id = f"llin_dwh_pf_{(band - 1) * 50 + offset + 1:04d}"

            if band == 1:
                plans.append(
                    QueryPlan(
                        plan_id=plan_id,
                        difficulty_band=band,
                        task_type="single_metric_query",
                        metric_key=metric_key,
                        metric_sql=metric_sql,
                        metric_description=metric_description,
                        joins=(),
                        filters=(
                            _filter(month_sql, f"发运日期在 {start} 至 {end}（含首尾）"),
                        ),
                    )
                )
                continue

            service = SERVICE_LEVELS[offset % len(SERVICE_LEVELS)]
            if band == 2:
                plans.append(
                    QueryPlan(
                        plan_id=plan_id,
                        difficulty_band=band,
                        task_type="aggregate_query",
                        metric_key=metric_key,
                        metric_sql=metric_sql,
                        metric_description=metric_description,
                        joins=(),
                        filters=(
                            _filter(month_sql, f"发运日期在 {start} 至 {end}（含首尾）"),
                            _filter("s.status = '已签收'", "运单状态为“已签收”"),
                            _filter(
                                f"s.service_level = {_sql_quote(service)}",
                                f"服务等级为“{service}”",
                            ),
                        ),
                    )
                )
                continue

            if band == 3:
                plans.append(
                    QueryPlan(
                        plan_id=plan_id,
                        difficulty_band=band,
                        task_type="comparison_analysis",
                        metric_key=metric_key,
                        metric_sql=metric_sql,
                        metric_description=metric_description,
                        joins=(),
                        filters=(
                            _filter(month_sql, f"发运日期在 {start} 至 {end}（含首尾）"),
                            _filter("s.status <> '已取消'", "排除状态为“已取消”的运单"),
                        ),
                        group_sql="s.service_level",
                        group_description="服务等级",
                        order_direction="DESC" if offset % 2 == 0 else "ASC",
                    )
                )
                continue

            warehouse_type = WAREHOUSE_TYPES[offset % len(WAREHOUSE_TYPES)]
            if band == 4:
                plans.append(
                    QueryPlan(
                        plan_id=plan_id,
                        difficulty_band=band,
                        task_type="comparison_analysis",
                        metric_key=metric_key,
                        metric_sql=metric_sql,
                        metric_description=metric_description,
                        joins=("warehouses",),
                        filters=(
                            _filter(month_sql, f"发运日期在 {start} 至 {end}（含首尾）"),
                            _filter("s.status = '已签收'", "运单状态为“已签收”"),
                            _filter(
                                f"w.warehouse_type = {_sql_quote(warehouse_type)}",
                                f"仓库类型为“{warehouse_type}”",
                            ),
                        ),
                        group_sql="w.warehouse_name",
                        group_description="仓库名称",
                        order_direction="DESC",
                        limit=5,
                    )
                )
                continue

            region_tier = REGIONS[offset % len(REGIONS)][2]
            if band == 5:
                plans.append(
                    QueryPlan(
                        plan_id=plan_id,
                        difficulty_band=band,
                        task_type="comparison_analysis",
                        metric_key=metric_key,
                        metric_sql=metric_sql,
                        metric_description=metric_description,
                        joins=("warehouses", "regions"),
                        filters=(
                            _filter(month_sql, f"发运日期在 {start} 至 {end}（含首尾）"),
                            _filter("s.status = '已签收'", "运单状态为“已签收”"),
                            _filter(
                                f"s.service_level = {_sql_quote(service)}",
                                f"服务等级为“{service}”",
                            ),
                            _filter(
                                f"r.region_tier = {_sql_quote(region_tier)}",
                                f"区域层级为“{region_tier}”",
                            ),
                        ),
                        group_sql="r.region_name",
                        group_description="区域名称",
                        order_direction="DESC",
                        limit=5,
                    )
                )
                continue

            carrier_level = CARRIER_LEVELS[offset % len(CARRIER_LEVELS)]
            segment = CUSTOMER_SEGMENTS[(offset // 2) % len(CUSTOMER_SEGMENTS)]
            delay_sql, delay_description = _metric("delay_rate")
            plans.append(
                QueryPlan(
                    plan_id=plan_id,
                    difficulty_band=band,
                    task_type="comparison_analysis",
                    metric_key="delay_rate",
                    metric_sql=delay_sql,
                    metric_description=delay_description,
                    joins=("warehouses", "carriers", "customers"),
                    filters=(
                        _filter(month_sql, f"发运日期在 {start} 至 {end}（含首尾）"),
                        _filter("s.status = '已签收'", "运单状态为“已签收”"),
                        _filter(
                            f"s.service_level = {_sql_quote(service)}",
                            f"服务等级为“{service}”",
                        ),
                        _filter(
                            f"c.carrier_level = {_sql_quote(carrier_level)}",
                            f"承运商等级为“{carrier_level}”",
                        ),
                        _filter(
                            f"u.customer_segment = {_sql_quote(segment)}",
                            f"客户分层为“{segment}”",
                        ),
                    ),
                    group_sql="c.carrier_name",
                    group_description="承运商名称",
                    order_direction="DESC",
                    limit=5,
                    having_sql="COUNT(*) >= 3",
                    having_description="每个承运商至少有 3 票满足条件的运单",
                )
            )

    if len(plans) != 300:
        raise AssertionError(f"expected 300 plans, got {len(plans)}")
    return plans


def compile_sql(plan: QueryPlan) -> str:
    select = plan.metric_sql
    if plan.group_sql:
        select = f"{plan.group_sql} AS category, {plan.metric_sql} AS value"
    else:
        select = f"{plan.metric_sql} AS value"
    lines = [f"SELECT {select}", "FROM shipments AS s"]
    lines.extend(JOIN_SQL[name] for name in plan.joins)
    if plan.filters:
        lines.append("WHERE " + "\n  AND ".join(item.sql for item in plan.filters))
    if plan.group_sql:
        lines.append(f"GROUP BY {plan.group_sql}")
    if plan.having_sql:
        lines.append(f"HAVING {plan.having_sql}")
    if plan.group_sql and plan.order_direction:
        lines.append(f"ORDER BY value {plan.order_direction}, category ASC")
    if plan.limit is not None:
        lines.append(f"LIMIT {plan.limit}")
    return "\n".join(lines)


def render_instruction(plan: QueryPlan) -> str:
    conditions = "；".join(item.description for item in plan.filters)
    if plan.having_description:
        conditions += f"；并要求{plan.having_description}"
    if plan.group_sql:
        direction = "从高到低" if plan.order_direction == "DESC" else "从低到高"
        limit_text = f"，只保留前 {plan.limit} 行" if plan.limit else ""
        return (
            f"请查询物流数据仓库，按{plan.group_description}分组计算{plan.metric_description}。"
            f"统计条件：{conditions}。按指标值{direction}排序{limit_text}。"
            "请返回 category 和 value 两列；不要估算，必须使用数据库中的精确结果。"
        )
    return (
        f"请查询物流数据仓库并计算{plan.metric_description}。统计条件：{conditions}。"
        "只返回一个精确数值，不要分组，也不要估算。"
    )


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE regions (
            region_id TEXT PRIMARY KEY,
            region_name TEXT NOT NULL UNIQUE,
            region_tier TEXT NOT NULL
        );
        CREATE TABLE warehouses (
            warehouse_id TEXT PRIMARY KEY,
            warehouse_name TEXT NOT NULL UNIQUE,
            region_id TEXT NOT NULL REFERENCES regions(region_id),
            warehouse_type TEXT NOT NULL,
            capacity_tons INTEGER NOT NULL
        );
        CREATE TABLE carriers (
            carrier_id TEXT PRIMARY KEY,
            carrier_name TEXT NOT NULL UNIQUE,
            carrier_level TEXT NOT NULL
        );
        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            customer_name TEXT NOT NULL UNIQUE,
            customer_segment TEXT NOT NULL,
            home_region_id TEXT NOT NULL REFERENCES regions(region_id)
        );
        CREATE TABLE shipments (
            shipment_id TEXT PRIMARY KEY,
            ship_date TEXT NOT NULL,
            warehouse_id TEXT NOT NULL REFERENCES warehouses(warehouse_id),
            carrier_id TEXT NOT NULL REFERENCES carriers(carrier_id),
            customer_id TEXT NOT NULL REFERENCES customers(customer_id),
            service_level TEXT NOT NULL,
            status TEXT NOT NULL,
            weight_kg REAL NOT NULL,
            freight_amount REAL NOT NULL,
            delivery_hours REAL NOT NULL,
            delayed_flag INTEGER NOT NULL CHECK (delayed_flag IN (0, 1))
        );
        CREATE INDEX idx_shipments_date ON shipments(ship_date);
        CREATE INDEX idx_shipments_warehouse ON shipments(warehouse_id);
        CREATE INDEX idx_shipments_carrier ON shipments(carrier_id);
        CREATE INDEX idx_shipments_customer ON shipments(customer_id);
        CREATE INDEX idx_shipments_service_status ON shipments(service_level, status);
        """
    )


def create_database(path: Path, seed: int = 20260814) -> None:
    rng = random.Random(seed)
    connection = sqlite3.connect(path)
    try:
        _create_schema(connection)
        connection.executemany("INSERT INTO regions VALUES (?, ?, ?)", REGIONS)

        warehouses: list[tuple[str, str, str, str, int]] = []
        for idx in range(24):
            region = REGIONS[idx % len(REGIONS)]
            warehouses.append(
                (
                    f"W{idx + 1:03d}",
                    f"{region[1]}-{idx // len(REGIONS) + 1}号仓",
                    region[0],
                    WAREHOUSE_TYPES[idx % len(WAREHOUSE_TYPES)],
                    1400 + (idx * 173) % 2600,
                )
            )
        connection.executemany("INSERT INTO warehouses VALUES (?, ?, ?, ?, ?)", warehouses)

        carriers = [
            (
                f"C{idx + 1:03d}",
                f"承运商{idx + 1:02d}",
                CARRIER_LEVELS[idx % len(CARRIER_LEVELS)],
            )
            for idx in range(18)
        ]
        connection.executemany("INSERT INTO carriers VALUES (?, ?, ?)", carriers)

        customers = [
            (
                f"U{idx + 1:04d}",
                f"客户{idx + 1:03d}",
                CUSTOMER_SEGMENTS[idx % len(CUSTOMER_SEGMENTS)],
                REGIONS[(idx * 5) % len(REGIONS)][0],
            )
            for idx in range(120)
        ]
        connection.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", customers)

        start = date(2025, 1, 1)
        shipment_rows: list[tuple[Any, ...]] = []
        service_base_hours = {"当日达": 12.0, "次日达": 28.0, "经济件": 54.0, "冷链": 22.0}
        for idx in range(18000):
            ship_date = start + timedelta(days=(idx * 37 + idx // 23) % 365)
            warehouse = warehouses[(idx * 7 + idx // 29) % len(warehouses)]
            carrier = carriers[(idx * 11 + idx // 31) % len(carriers)]
            customer = customers[(idx * 13 + idx // 17) % len(customers)]
            service = SERVICE_LEVELS[(idx * 3 + idx // 41) % len(SERVICE_LEVELS)]
            status_roll = (idx * 19 + idx // 7) % 100
            status = "已签收" if status_roll < 86 else ("运输中" if status_roll < 95 else "已取消")
            weight = round(8.0 + rng.random() * 690.0 + (idx % 9) * 4.5, 2)
            freight = round(18.0 + weight * (0.82 + (idx % 7) * 0.07) + (idx % 5) * 9.0, 2)
            normal_hours = service_base_hours[service]
            delay_threshold = 23 + (idx % 11)
            delayed = 1 if ((idx * 17 + idx // 5) % 100) < delay_threshold else 0
            delivery_hours = round(normal_hours + rng.random() * 8.0 + delayed * (9 + idx % 13), 2)
            shipment_rows.append(
                (
                    f"S{idx + 1:06d}",
                    ship_date.isoformat(),
                    warehouse[0],
                    carrier[0],
                    customer[0],
                    service,
                    status,
                    weight,
                    freight,
                    delivery_hours,
                    delayed,
                )
            )
        connection.executemany("INSERT INTO shipments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", shipment_rows)
        connection.commit()
        result = connection.execute("PRAGMA integrity_check").fetchone()
        if result != ("ok",):
            raise RuntimeError(f"SQLite integrity check failed: {result}")
    finally:
        connection.close()


def _readonly_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _validate_sql(sql: str) -> None:
    stripped = sql.strip()
    if not stripped.upper().startswith("SELECT "):
        raise ValueError("verification SQL must start with SELECT")
    if ";" in stripped:
        raise ValueError("verification SQL must be one statement without semicolons")
    if FORBIDDEN_SQL.search(stripped):
        raise ValueError("verification SQL contains a forbidden write or control keyword")


def execute_plan(database: Path, plan: QueryPlan) -> tuple[list[dict[str, Any]], str]:
    sql = compile_sql(plan)
    _validate_sql(sql)
    with _readonly_connection(database) as connection:
        first = [dict(row) for row in connection.execute(sql).fetchall()]
        second = [dict(row) for row in connection.execute(sql).fetchall()]
    if first != second:
        raise ValueError(f"non-deterministic result for {plan.plan_id}")
    if not first:
        raise ValueError(f"empty result for {plan.plan_id}")
    if any(value is None for row in first for value in row.values()):
        raise ValueError(f"NULL result for {plan.plan_id}")
    result_hash = hashlib.sha256(
        json.dumps(first, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return first, result_hash


def _table_names(plan: QueryPlan) -> list[str]:
    return ["shipments", *plan.joins]


def _plan_payload(plan: QueryPlan) -> dict[str, Any]:
    payload = asdict(plan)
    payload["filters"] = [asdict(item) for item in plan.filters]
    payload["output_shape"] = "category_value_table" if plan.group_sql else "scalar"
    payload["feature_counts"] = {
        "joins": len(plan.joins),
        "filters": len(plan.filters),
        "group_by": int(plan.group_sql is not None),
        "having": int(plan.having_sql is not None),
        "top_k": int(plan.limit is not None),
        "derived_metric": int(plan.metric_key == "delay_rate"),
    }
    return payload


def build_task(database: Path, plan: QueryPlan) -> dict[str, Any]:
    instruction = render_instruction(plan)
    rows, result_hash = execute_plan(database, plan)
    sql = compile_sql(plan)
    if plan.group_sql:
        gold_value: Any = [
            {"category": row["category"], "value": row["value"]}
            for row in rows
        ]
        answer_type = "table"
    else:
        gold_value = rows[0]["value"]
        answer_type = "numeric"
    return {
        "task_id": plan.plan_id,
        "task_type": plan.task_type,
        "task_category": "answerable",
        "scenario_type": "dwh_query",
        "business_domain": "logistics",
        "difficulty": f"band_{plan.difficulty_band}",
        "difficulty_level": plan.difficulty_band,
        "difficulty_band": plan.difficulty_band,
        "natural_language_instruction": instruction,
        "instruction_variants": [instruction],
        "expected_tables": _table_names(plan),
        "expected_operations": [
            "filter",
            "aggregate",
            *("join" for _ in plan.joins[:1]),
            *("group_by" for _ in [0] if plan.group_sql),
            *("order_by" for _ in [0] if plan.order_direction),
        ],
        "answerability_label": {
            "is_answerable": True,
            "reason": "all referenced tables, fields, filters, and dates exist in logistics.sqlite",
        },
        "query_plan": _plan_payload(plan),
        "sample_sql": sql,
        "gold_answer": {
            "answer_type": answer_type,
            "value": gold_value,
            "verification_sql": sql,
        },
        "validation": {
            "checked": True,
            "checked_against_database": True,
            "expected_result_exists": True,
            "read_only": True,
            "deterministic": True,
            "nonempty": True,
            "semantic_source": "query_plan",
            "result_sha256": result_hash,
            "result_row_count": len(rows),
        },
        "_qa_status": "passed",
        "generation_contract": GENERATOR_CONTRACT,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_dictionary() -> str:
    return """# 物流 DWH 数据字典

数据库为 SQLite，业务日期范围为 2025-01-01 至 2025-12-31。

## 表与字段

- `shipments`：运单事实表。主键 `shipment_id`；`ship_date` 为 `YYYY-MM-DD`；外键为 `warehouse_id`、`carrier_id`、`customer_id`；度量字段为 `weight_kg`、`freight_amount`、`delivery_hours`、`delayed_flag`。
- `warehouses`：仓库维表。含 `warehouse_name`、`region_id`、`warehouse_type`、`capacity_tons`。
- `regions`：区域维表。含 `region_name`、`region_tier`。
- `carriers`：承运商维表。含 `carrier_name`、`carrier_level`。
- `customers`：客户维表。含 `customer_name`、`customer_segment`、`home_region_id`。

## 连接关系

- `shipments.warehouse_id = warehouses.warehouse_id`
- `warehouses.region_id = regions.region_id`
- `shipments.carrier_id = carriers.carrier_id`
- `shipments.customer_id = customers.customer_id`

## 枚举值

- `service_level`：当日达、次日达、经济件、冷链
- `status`：已签收、运输中、已取消
- `warehouse_type`：中心仓、区域仓、前置仓
- `region_tier`：核心区、成长区
- `carrier_level`：战略、核心、标准
- `customer_segment`：大客户、成长客户、标准客户
- `delayed_flag`：1 表示延误，0 表示未延误
"""


def _calibration_manifest() -> dict[str, Any]:
    return {
        "contract": "llin-dwh-rollout-calibration-v1",
        "generation_bands": 6,
        "tasks_per_band": 50,
        "pilot": {
            "tasks_per_band": 8,
            "rollouts_per_task": 4,
            "total_pilot_rollouts": 192,
            "selection": "deterministic_stratified",
        },
        "target_success_rate": {"minimum": 0.20, "maximum": 0.80},
        "decisions": {
            "below_minimum": "simplify_or_replace_before_training",
            "above_maximum": "increase_constraints_or_replace_before_training",
            "inside_range": "eligible_for_training",
        },
        "note": "Generation difficulty is structural. Final eligibility requires model rollout evidence.",
    }


def validate_task_set(tasks: Sequence[dict[str, Any]]) -> None:
    if len(tasks) != 300:
        raise ValueError(f"expected 300 tasks, got {len(tasks)}")
    task_ids = [task["task_id"] for task in tasks]
    instructions = [task["natural_language_instruction"] for task in tasks]
    sqls = [task["gold_answer"]["verification_sql"] for task in tasks]
    for label, values in (("task ID", task_ids), ("instruction", instructions), ("SQL", sqls)):
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {label} detected")
    band_counts = {band: 0 for band in range(1, 7)}
    for task in tasks:
        band_counts[task["difficulty_band"]] += 1
        if task["task_type"] not in ALLOWED_TASK_TYPES:
            raise ValueError(f"unsupported task type: {task['task_type']}")
        if task["_qa_status"] != "passed" or not task["validation"]["checked"]:
            raise ValueError(f"unvalidated task: {task['task_id']}")
    if set(band_counts.values()) != {50}:
        raise ValueError(f"unbalanced difficulty bands: {band_counts}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def verify_existing(output_dir: Path) -> dict[str, Any]:
    """Verify one exact sandbox directory without scanning sibling versions."""

    required = {
        "logistics.sqlite",
        "dwh_tasks.jsonl",
        "schema_dictionary.md",
        "rollout_calibration.json",
        "generation_summary.json",
    }
    missing = sorted(name for name in required if not (output_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"sandbox is missing required files: {missing}")

    tasks = _read_jsonl(output_dir / "dwh_tasks.jsonl")
    validate_task_set(tasks)
    database = output_dir / "logistics.sqlite"
    with _readonly_connection(database) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"SQLite integrity check failed: {integrity}")
        for task in tasks:
            sql = str(task["gold_answer"]["verification_sql"])
            _validate_sql(sql)
            rows = [dict(row) for row in connection.execute(sql).fetchall()]
            gold = task["gold_answer"]
            expected = (
                [{"value": gold["value"]}]
                if gold["answer_type"] == "numeric"
                else gold["value"]
            )
            if rows != expected:
                raise ValueError(f"gold replay mismatch: {task['task_id']}")

    summary = json.loads((output_dir / "generation_summary.json").read_text(encoding="utf-8"))
    for name, expected_hash in summary.get("files", {}).items():
        actual_hash = _sha256(output_dir / name)
        if actual_hash != expected_hash:
            raise ValueError(f"SHA-256 mismatch for {name}")
    return {
        "environment_id": summary.get("environment_id"),
        "task_count": len(tasks),
        "difficulty_band_counts": summary.get("difficulty_band_counts"),
        "database_integrity_check": integrity,
        "gold_replay_rows": len(tasks),
        "file_hashes_verified": len(summary.get("files", {})),
    }


def generate(output_root: Path, version: str = DEFAULT_VERSION) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", version):
        raise ValueError("version must match [A-Za-z0-9._-]+")
    output_dir = output_root / version
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing sandbox: {output_dir}")
    output_dir.mkdir(parents=True)

    database = output_dir / "logistics.sqlite"
    create_database(database)
    plans = build_plans()
    tasks = [build_task(database, plan) for plan in plans]
    validate_task_set(tasks)

    tasks_path = output_dir / "dwh_tasks.jsonl"
    _write_jsonl(tasks_path, tasks)
    (output_dir / "schema_dictionary.md").write_text(_schema_dictionary(), encoding="utf-8")
    _write_json(output_dir / "rollout_calibration.json", _calibration_manifest())
    summary = {
        "contract": GENERATOR_CONTRACT,
        "environment_id": f"sft/{version}",
        "task_count": len(tasks),
        "difficulty_band_counts": {str(band): 50 for band in range(1, 7)},
        "database_integrity_check": "ok",
        "semantic_source": "query_plan",
        "external_api_used": False,
        "files": {
            "logistics.sqlite": _sha256(database),
            "dwh_tasks.jsonl": _sha256(tasks_path),
            "schema_dictionary.md": _sha256(output_dir / "schema_dictionary.md"),
            "rollout_calibration.json": _sha256(output_dir / "rollout_calibration.json"),
        },
    }
    _write_json(output_dir / "generation_summary.json", summary)
    return output_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-root", type=Path)
    mode.add_argument("--verify-existing", type=Path)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_existing is not None:
        print(json.dumps(verify_existing(args.verify_existing), ensure_ascii=False, indent=2))
        return 0
    output_dir = generate(args.output_root, args.version)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

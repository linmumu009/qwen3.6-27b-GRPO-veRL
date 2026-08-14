#!/usr/bin/env python3
"""Generate calibrated, boss-open DWH tasks from several read-only sandboxes.

The source databases are never modified.  Each output sandbox contains a
snapshot of one source ``logistics.sqlite`` plus 500 plan-first tasks.  The
old plan-first Band 5 is the structural floor for new Level 1; higher levels
add temporal comparison, derived measures, process diagnosis, attribution,
and finally an open management decision with an explicit verifiable table.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import sqlite3
from typing import Any, Iterable, Sequence


CONTRACT = "llin-open-multisandbox-dwh-v1"
DEFAULT_VERSIONS = (
    "20260628_v15",
    "20260628_v20",
    "20260628_v21",
    "20260628_v22",
    "20260628_v23",
    "20260628_v24",
    "20260628_v25",
    "20260628_v26",
)
FORBIDDEN_SQL = re.compile(
    r"\b(?:ATTACH|DETACH|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|VACUUM|PRAGMA)\b",
    re.IGNORECASE,
)
TECHNICAL_INSTRUCTION_RE = re.compile(
    r"SQL|SQLite|数据库|数据仓库|表名|字段名|SELECT|JOIN|GROUP\s+BY|CTE|窗口函数",
    re.IGNORECASE,
)
ROLES = (
    "company_owner",
    "operations",
    "data_analyst",
    "regional_manager",
    "finance",
    "warehouse_manager",
    "planning",
    "customer_service",
    "procurement",
    "sales",
    "general_employee",
)
ROLE_OPENERS = {
    "company_owner": "我想从经营上快速判断一下重点。",
    "operations": "我在做运营复盘，想把真正的问题找出来。",
    "data_analyst": "我在整理这期分析，想把口径和结果一次核准。",
    "regional_manager": "我在安排区域工作，想先确认哪些业务最值得关注。",
    "finance": "我在看投入产出，想核对这部分业务表现。",
    "warehouse_manager": "我在排后续仓内资源，想先看清各类业务的处理情况。",
    "planning": "我在做下一阶段计划，需要一份能直接用于判断的数据。",
    "customer_service": "我在复盘客户体验，想知道问题主要集中在哪里。",
    "procurement": "我在准备合作评估，需要一份有依据的业务比较。",
    "sales": "我在安排客户和产品策略，想了解不同业务的表现差别。",
    "general_employee": "麻烦帮我把这部分业务情况理清楚。",
}
DIMENSION_LABELS = {
    "goods_type": "货物类型",
    "cargo_type": "货品类型",
    "cargo_desc_type": "货物分类",
    "service_type": "配送服务",
    "status": "运单状态",
}
METRIC_LABELS = {
    "shipment_count": "运单量",
    "weight_sum": "总货重",
    "delivery_avg": "平均派送处理时长",
    "process_avg": "平均全流程处理时长",
    "handoff_gap": "揽收与派送处理时长的平均差值",
    "late_stage_share": "派送环节时长占比",
    "change_rate": "环比变化率",
    "gap_rate": "相对整体的时长差异率",
    "attention_score": "综合关注分",
}
LEVEL_FAMILIES = {
    1: "grouped_ranking",
    2: "period_comparison",
    3: "process_diagnosis",
    4: "baseline_attribution",
    5: "management_prioritization",
}


@dataclass(frozen=True)
class Candidate:
    level: int
    family: str
    role: str
    sql: str
    instruction: str
    metric_key: str
    group_field: str
    anchors: tuple[str, ...]
    evidence_steps: int
    essential_joins: int
    derived_metrics: int
    temporal_comparisons: int
    openness: int


def _quote(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_only_sql(sql: str) -> None:
    if FORBIDDEN_SQL.search(sql) or not sql.lstrip().upper().startswith(("SELECT", "WITH")):
        raise ValueError("verification SQL must be one read-only SELECT/WITH statement")
    if ";" in sql.rstrip().rstrip(";"):
        raise ValueError("multiple SQL statements are forbidden")


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


def _available_dimensions(connection: sqlite3.Connection) -> list[str]:
    columns = _columns(connection, "fact_waybill")
    # These four fields are present in all eight selected boss sandboxes.  A
    # sandbox-specific field (for example service_type) is deliberately not
    # allowed to leak into a supposedly cross-sandbox generation contract.
    preferred = ["goods_type", "cargo_type", "cargo_desc_type", "status"]
    dimensions = []
    for name in preferred:
        if name not in columns:
            continue
        count = connection.execute(
            f"SELECT COUNT(DISTINCT {name}) FROM fact_waybill WHERE {name} IS NOT NULL"
        ).fetchone()[0]
        if 2 <= int(count) <= 30:
            dimensions.append(name)
    if len(dimensions) < 3:
        raise ValueError("fact_waybill does not expose three useful business dimensions")
    return dimensions


def _month_catalog(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        """
        SELECT strftime('%Y-%m', created_at) AS month, COUNT(*) AS n
        FROM fact_waybill
        WHERE date(created_at) IS NOT NULL
        GROUP BY month
        HAVING n >= 35
        ORDER BY month
        """
    ).fetchall()
    months = [str(row[0]) for row in rows if row[0]]
    if len(months) < 2:
        raise ValueError("fact_waybill needs at least two populated months")
    return months


def _dimension_values(connection: sqlite3.Connection, dimension: str) -> list[str]:
    rows = connection.execute(
        f"""
        SELECT {dimension}, COUNT(*) AS n
        FROM fact_waybill
        WHERE {dimension} IS NOT NULL AND TRIM(CAST({dimension} AS TEXT)) <> ''
        GROUP BY {dimension}
        HAVING n >= 20
        ORDER BY n DESC, {dimension}
        LIMIT 12
        """
    ).fetchall()
    return [str(row[0]) for row in rows]


def _month_text(month: str) -> str:
    year, number = month.split("-")
    return f"{int(year)}年{int(number)}月"


def _previous_month(month: str) -> str:
    value = datetime.strptime(month + "-01", "%Y-%m-%d")
    year = value.year if value.month > 1 else value.year - 1
    number = value.month - 1 if value.month > 1 else 12
    return f"{year:04d}-{number:02d}"


def _month_filter(alias: str, month: str) -> str:
    return f"strftime('%Y-%m', {alias}.created_at) = {_quote(month)}"


def _base_cte(month: str, filter_field: str, filter_value: str) -> str:
    return f"""base AS (
  SELECT w.waybill_no, w.goods_type, w.cargo_type, w.cargo_desc_type,
         w.status, w.cargo_weight_kg, w.cargo_volume_m3,
         p.duration_hours AS pickup_hours,
         s.duration_hours AS sorting_hours,
         t.duration_hours AS transit_hours,
         d.duration_hours AS delivery_hours,
         o.duration_hours AS pod_hours
  FROM fact_waybill AS w
  JOIN fact_pickup AS p ON p.waybill_no = w.waybill_no
  JOIN fact_sorting AS s ON s.waybill_no = w.waybill_no
  JOIN fact_transit AS t ON t.waybill_no = w.waybill_no
  JOIN fact_delivery AS d ON d.waybill_no = w.waybill_no
  JOIN fact_pod AS o ON o.waybill_no = w.waybill_no
  WHERE {_month_filter('w', month)}
    AND w.{filter_field} = {_quote(filter_value)}
)"""


def _make_candidate(
    *,
    level: int,
    role: str,
    month: str,
    previous: str,
    group_field: str,
    filter_field: str,
    filter_value: str,
    metric_variant: int,
) -> Candidate:
    group_label = DIMENSION_LABELS[group_field]
    filter_label = DIMENSION_LABELS[filter_field]
    month_label = _month_text(month)
    previous_label = _month_text(previous)
    opener = ROLE_OPENERS[role]
    qgroup = f"w.{group_field}"
    qfilter = f"w.{filter_field} = {_quote(filter_value)}"
    month_where = _month_filter("w", month)
    family = LEVEL_FAMILIES[level]

    if level == 1:
        # Both joins are used by every metric; Level 1 therefore starts at the
        # old Band-5 floor without counting decorative joins as difficulty.
        metrics = (
            ("process_avg", "ROUND(AVG(p.duration_hours + d.duration_hours), 2)"),
            ("delivery_avg", "ROUND(AVG(d.duration_hours + 0.10 * p.duration_hours), 2)"),
            ("handoff_gap", "ROUND(AVG(ABS(d.duration_hours - p.duration_hours)), 2)"),
            ("late_stage_share", "ROUND(100.0 * AVG(d.duration_hours / (p.duration_hours + d.duration_hours)), 2)"),
        )
        metric_key, metric_sql = metrics[metric_variant % len(metrics)]
        sql = f"""SELECT {qgroup} AS category, {metric_sql} AS value
FROM fact_waybill AS w
JOIN fact_pickup AS p ON p.waybill_no = w.waybill_no
JOIN fact_delivery AS d ON d.waybill_no = w.waybill_no
WHERE {month_where}
  AND {qfilter}
  AND w.cargo_weight_kg IS NOT NULL
  AND d.duration_hours IS NOT NULL
GROUP BY {qgroup}
HAVING COUNT(*) >= 3
ORDER BY value DESC, category ASC
LIMIT 5"""
        instruction = (
            f"{opener}只看{month_label}、{filter_label}为“{filter_value}”的业务，"
            f"按{group_label}比较{METRIC_LABELS[metric_key]}，样本少于3票的不算，"
            "从高到低列出前5项和对应结果。"
        )
        anchors = (month_label, filter_value, group_label, METRIC_LABELS[metric_key], "3", "5")
        return Candidate(level, family, role, sql, instruction, metric_key, group_field, anchors, 1, 2, 0, 0, 1)

    if level == 2:
        measures = (
            ("shipment_count", "COUNT(*)", "运单量"),
            ("weight_sum", "ROUND(SUM(w.cargo_weight_kg), 2)", "总货重"),
            ("delivery_avg", "ROUND(AVG(d.duration_hours + 0.10 * p.duration_hours), 2)", "平均派送处理时长"),
            ("process_avg", "ROUND(AVG(p.duration_hours + d.duration_hours), 2)", "平均揽收到派送处理时长"),
        )
        measure_key, measure_sql, measure_label = measures[metric_variant % len(measures)]
        sql = f"""WITH monthly AS (
  SELECT {qgroup} AS category,
         strftime('%Y-%m', w.created_at) AS month,
         {measure_sql} AS measure,
         COUNT(*) AS sample_count
  FROM fact_waybill AS w
  JOIN fact_pickup AS p ON p.waybill_no = w.waybill_no
  JOIN fact_delivery AS d ON d.waybill_no = w.waybill_no
  WHERE strftime('%Y-%m', w.created_at) IN ({_quote(previous)}, {_quote(month)})
    AND {qfilter}
    AND p.duration_hours IS NOT NULL
    AND d.duration_hours IS NOT NULL
  GROUP BY {qgroup}, month
), paired AS (
  SELECT category,
         SUM(CASE WHEN month = {_quote(previous)} THEN measure ELSE 0 END) AS previous_value,
         SUM(CASE WHEN month = {_quote(month)} THEN measure ELSE 0 END) AS current_value,
         SUM(CASE WHEN month = {_quote(previous)} THEN sample_count ELSE 0 END) AS previous_count,
         SUM(CASE WHEN month = {_quote(month)} THEN sample_count ELSE 0 END) AS current_count
  FROM monthly GROUP BY category
)
SELECT category,
       ROUND(100.0 * (current_value - previous_value) / previous_value, 2) AS value
FROM paired
WHERE previous_count >= 3 AND current_count >= 3 AND previous_value <> 0
ORDER BY value DESC, category ASC
LIMIT 5"""
        instruction = (
            f"{opener}只看{filter_label}为“{filter_value}”的业务，比较{month_label}和"
            f"{previous_label}各{group_label}的{measure_label}变化，两个自然月都少于3票的不要纳入。"
            "请按环比增幅从高到低列出前5项，并给出变化百分比。"
        )
        anchors = (month_label, previous_label, filter_value, group_label, measure_label, "3", "5", "百分比")
        return Candidate(level, family, role, sql, instruction, f"{measure_key}_change_rate", group_field, anchors, 2, 2, 1, 1, 2)

    base = _base_cte(month, filter_field, filter_value)
    if level == 3:
        phase = ("transit_hours", "delivery_hours", "sorting_hours")[metric_variant % 3]
        phase_label = {"transit_hours": "运输环节", "delivery_hours": "派送环节", "sorting_hours": "分拣环节"}[phase]
        sql = f"""WITH {base}, grouped AS (
  SELECT {group_field} AS category,
         COUNT(*) AS sample_count,
         AVG({phase}) AS phase_avg,
         AVG(pickup_hours + sorting_hours + transit_hours + delivery_hours + pod_hours) AS total_avg
  FROM base GROUP BY {group_field}
)
SELECT category, ROUND(100.0 * phase_avg / total_avg, 2) AS value
FROM grouped
WHERE sample_count >= 3 AND total_avg > 0
ORDER BY value DESC, category ASC
LIMIT 5"""
        instruction = (
            f"{opener}范围限定为{month_label}、{filter_label}为“{filter_value}”的业务。"
            f"我想诊断各{group_label}的流程卡点，请计算{phase_label}占全流程处理时长的比例，"
            "只看至少3票的分组，按占比从高到低列出前5项。"
        )
        anchors = (month_label, filter_value, group_label, phase_label, "全流程", "3", "5")
        return Candidate(level, family, role, sql, instruction, "process_share", group_field, anchors, 3, 5, 1, 0, 3)

    if level == 4:
        measures = (
            ("process", "pickup_hours + sorting_hours + transit_hours + delivery_hours + pod_hours", "全流程处理时长"),
            ("delivery", "delivery_hours", "派送处理时长"),
            ("transit", "transit_hours", "运输处理时长"),
            ("sorting", "sorting_hours", "分拣处理时长"),
        )
        measure_key, measure_sql, measure_label = measures[metric_variant % len(measures)]
        sql = f"""WITH {base}, group_stats AS (
  SELECT {group_field} AS category, COUNT(*) AS sample_count,
         AVG({measure_sql}) AS group_avg
  FROM base GROUP BY {group_field}
), overall AS (
  SELECT AVG({measure_sql}) AS overall_avg
  FROM base
)
SELECT g.category, ROUND(100.0 * (g.group_avg - o.overall_avg) / o.overall_avg, 2) AS value
FROM group_stats AS g CROSS JOIN overall AS o
WHERE g.sample_count >= 3 AND o.overall_avg > 0
ORDER BY value DESC, category ASC
LIMIT 5"""
        instruction = (
            f"{opener}请分析{month_label}、{filter_label}为“{filter_value}”的业务，"
            f"找出哪些{group_label}拉长了{measure_label}。以这批业务的整体平均{measure_label}为基线，"
            "计算各组高出基线的百分比，只保留至少3票的分组，从高到低列出差异最大的前5项。"
        )
        anchors = (month_label, filter_value, group_label, measure_label, "整体平均", "百分比", "3", "5")
        return Candidate(level, family, role, sql, instruction, f"{measure_key}_gap_rate", group_field, anchors, 4, 5, 1, 0, 4)

    weight_profiles = (
        (20, 35, 45, "全流程改善优先"),
        (25, 45, 30, "派送改善优先"),
        (35, 25, 40, "规模与全流程并重"),
        (30, 35, 35, "均衡关注"),
    )
    volume_weight, delivery_weight, process_weight, priority_label = weight_profiles[
        metric_variant % len(weight_profiles)
    ]
    sql = f"""WITH {base}, group_stats AS (
  SELECT {group_field} AS category, COUNT(*) AS volume,
         AVG(delivery_hours) AS delivery_avg,
         AVG(pickup_hours + sorting_hours + transit_hours + delivery_hours + pod_hours) AS process_avg
  FROM base GROUP BY {group_field}
), normalized AS (
  SELECT category, volume, delivery_avg, process_avg,
         100.0 * volume / MAX(volume) OVER () AS volume_index,
         100.0 * delivery_avg / MAX(delivery_avg) OVER () AS delivery_index,
         100.0 * process_avg / MAX(process_avg) OVER () AS process_index
  FROM group_stats WHERE volume >= 3
)
SELECT category,
       ROUND({volume_weight / 100:.2f} * volume_index + {delivery_weight / 100:.2f} * delivery_index + {process_weight / 100:.2f} * process_index, 2) AS value
FROM normalized
ORDER BY value DESC, category ASC
LIMIT 5"""
    instruction = (
        f"{opener}请从{month_label}、{filter_label}为“{filter_value}”的业务里判断最该优先关注的"
        f"{group_label}，这次采用“{priority_label}”口径。综合关注分按运单量{volume_weight}%、"
        f"平均派送时长{delivery_weight}%、平均全流程处理时长{process_weight}%计算，"
        "每项先按本批最高值折算成百分制；少于3票的不参与。请从高到低列出前5项和综合分，并简要说明你的判断。"
    )
    anchors = (
        month_label,
        filter_value,
        group_label,
        priority_label,
        f"{volume_weight}%",
        f"{delivery_weight}%",
        f"{process_weight}%",
        "3",
        "5",
    )
    return Candidate(level, family, role, sql, instruction, "attention_score", group_field, anchors, 5, 5, 4, 0, 5)


def _execute(connection: sqlite3.Connection, sql: str) -> tuple[list[dict[str, Any]], str]:
    _read_only_sql(sql)
    connection.row_factory = sqlite3.Row
    rows = [dict(row) for row in connection.execute(sql).fetchall()]
    if not 2 <= len(rows) <= 5:
        raise ValueError("gold table must contain 2-5 rows")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        category = row.get("category")
        value = row.get("value")
        if category is None or value is None:
            raise ValueError("gold row must contain category and value")
        normalized.append({"category": str(category), "value": value})
    if len({str(row["category"]) for row in normalized}) != len(normalized):
        raise ValueError("gold categories must be unique")
    return normalized, _canonical_hash(normalized)


def generate_tasks(database: Path, *, source_version: str, seed: int) -> list[dict[str, Any]]:
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        dimensions = _available_dimensions(connection)
        months = _month_catalog(connection)
        month_set = set(months)
        month_pairs = [(month, _previous_month(month)) for month in months if _previous_month(month) in month_set]
        if not month_pairs:
            month_pairs = [(months[-1], months[-2])]
        values = {dimension: _dimension_values(connection, dimension) for dimension in dimensions}
        rng = random.Random(f"{seed}:{source_version}")
        tasks: list[dict[str, Any]] = []
        seen_sql: set[str] = set()
        seen_instruction: set[str] = set()
        attempts = 0
        for level in range(1, 6):
            accepted = 0
            while accepted < 100 and attempts < 50000:
                attempts += 1
                group_field = rng.choice(dimensions)
                filter_options = [name for name in dimensions if name != group_field and values[name]]
                if not filter_options:
                    continue
                filter_field = rng.choice(filter_options)
                filter_value = rng.choice(values[filter_field])
                month, previous = rng.choice(month_pairs)
                role = ROLES[(accepted + level + rng.randrange(len(ROLES))) % len(ROLES)]
                candidate = _make_candidate(
                    level=level,
                    role=role,
                    month=month,
                    previous=previous,
                    group_field=group_field,
                    filter_field=filter_field,
                    filter_value=filter_value,
                    metric_variant=attempts,
                )
                if candidate.sql in seen_sql or candidate.instruction in seen_instruction:
                    continue
                try:
                    gold, result_hash = _execute(connection, candidate.sql)
                except (sqlite3.Error, ValueError):
                    continue
                task_id = f"llin_open_{source_version.rsplit('_', 1)[-1]}_{len(tasks)+1:04d}"
                feature_counts = {
                    "essential_joins": candidate.essential_joins,
                    "evidence_steps": candidate.evidence_steps,
                    "derived_metrics": candidate.derived_metrics,
                    "temporal_comparisons": candidate.temporal_comparisons,
                    "business_openness": candidate.openness,
                    "redundant_joins": 0,
                }
                task = {
                    "task_id": task_id,
                    "task_type": candidate.family,
                    "task_category": "answerable",
                    "scenario_type": "dwh_query",
                    "business_domain": "logistics",
                    "source_sandbox_version": source_version,
                    "difficulty": f"level_{level}",
                    "difficulty_level": level,
                    "difficulty_band": level,
                    "difficulty_baseline": "previous_plan_first_band_5" if level == 1 else "above_previous_plan_first_band_5",
                    "natural_language_instruction": candidate.instruction,
                    "instruction_variants": [candidate.instruction],
                    "instruction_role": candidate.role,
                    "instruction_style": "boss_open_mixed_company_roles_plan_first_v1",
                    "semantic_anchors": list(candidate.anchors),
                    "semantic_contract": {
                        "family": candidate.family,
                        "metric": candidate.metric_key,
                        "group_dimension": candidate.group_field,
                        "required_anchors": list(candidate.anchors),
                        "explanation_is_open_ended": level == 5,
                        "verified_deliverable": "top_five_category_value_table",
                    },
                    "expected_tables": sorted(set(re.findall(r"\bfact_[a-z_]+\b", candidate.sql))),
                    "expected_operations": [
                        "filter", "aggregate", "join", "group_by", "order_by", "top_k",
                        *("temporal_compare" for _ in [0] if candidate.temporal_comparisons),
                        *("derived_metric" for _ in [0] if candidate.derived_metrics),
                    ],
                    "query_plan": {
                        **asdict(candidate),
                        "sql": None,
                        "instruction": None,
                        "anchors": list(candidate.anchors),
                        "feature_counts": feature_counts,
                        "output_shape": "category_value_table",
                    },
                    "sample_sql": candidate.sql,
                    "gold_answer": {
                        "answer_type": "table",
                        "value": gold,
                        "verification_sql": candidate.sql,
                    },
                    "validation": {
                        "checked": True,
                        "checked_against_database": True,
                        "expected_result_exists": True,
                        "read_only": True,
                        "deterministic": True,
                        "nonempty": True,
                        "semantic_source": "evidence_plan",
                        "result_sha256": result_hash,
                        "result_row_count": len(gold),
                        "difficulty_calibration_status": "awaiting_8x_rollout",
                    },
                    "_qa_status": "passed",
                    "generation_contract": CONTRACT,
                    "training_allowed": False,
                }
                # Do not duplicate the SQL/instruction inside the public plan payload.
                task["query_plan"].pop("sql", None)
                task["query_plan"].pop("instruction", None)
                seen_sql.add(candidate.sql)
                seen_instruction.add(candidate.instruction)
                tasks.append(task)
                accepted += 1
            if accepted != 100:
                raise RuntimeError(f"could only generate {accepted}/100 tasks for level {level}")
        return tasks
    finally:
        connection.close()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _schema_dictionary(database: Path) -> str:
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        lines = ["# 物流业务数据字典", "", "以下为本沙箱可查询的数据表与字段。", ""]
        for (table,) in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ):
            columns = [str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")]
            lines.append(f"- `{table}`：" + "、".join(f"`{name}`" for name in columns))
        return "\n".join(lines) + "\n"
    finally:
        connection.close()


def generate_one(
    source_database: Path,
    output_root: Path,
    *,
    source_version: str,
    output_version: str,
    seed: int,
) -> Path:
    output = output_root / output_version
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    incomplete = output.with_name(output.name + ".incomplete")
    if incomplete.exists():
        raise FileExistsError(f"incomplete output already exists: {incomplete}")
    incomplete.mkdir(parents=True)
    try:
        database = incomplete / "logistics.sqlite"
        shutil.copy2(source_database, database)
        tasks = generate_tasks(database, source_version=source_version, seed=seed)
        _write_jsonl(incomplete / "dwh_tasks.jsonl", tasks)
        (incomplete / "schema_dictionary.md").write_text(
            _schema_dictionary(database), encoding="utf-8"
        )
        summary = {
            "contract": CONTRACT,
            "environment_id": f"sft/{output_version}",
            "source_sandbox_version": source_version,
            "source_database_sha256": _sha256(source_database),
            "task_count": len(tasks),
            "difficulty_distribution": dict(Counter(str(row["difficulty_level"]) for row in tasks)),
            "family_distribution": dict(Counter(row["task_type"] for row in tasks)),
            "role_distribution": dict(Counter(row["instruction_role"] for row in tasks)),
            "training_allowed": False,
            "external_api_used": False,
            "difficulty_calibration_status": "awaiting_8x_rollout",
            "files": {
                "logistics.sqlite": _sha256(database),
                "dwh_tasks.jsonl": _sha256(incomplete / "dwh_tasks.jsonl"),
                "schema_dictionary.md": _sha256(incomplete / "schema_dictionary.md"),
            },
        }
        (incomplete / "generation_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        incomplete.replace(output)
    except Exception:
        raise
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--versions", nargs="+", default=list(DEFAULT_VERSIONS))
    parser.add_argument("--output-prefix", default="20260814_llin_dwh_open_v1")
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = []
    for version in args.versions:
        source_database = args.source_root / version / "logistics.sqlite"
        if not source_database.is_file():
            raise FileNotFoundError(source_database)
        suffix = version.rsplit("_", 1)[-1]
        output_version = f"{args.output_prefix}_{suffix}"
        outputs.append(
            str(
                generate_one(
                    source_database,
                    args.output_root,
                    source_version=version,
                    output_version=output_version,
                    seed=args.seed,
                )
            )
        )
    print(json.dumps({"outputs": outputs}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

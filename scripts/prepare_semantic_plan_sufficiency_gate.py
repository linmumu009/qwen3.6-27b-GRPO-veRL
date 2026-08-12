#!/usr/bin/env python3
"""Build a three-arm, one-generation semantic-plan sufficiency gate."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable
from urllib.parse import quote

import pandas as pd

from llin_verl.boss_pi_contract import canonical_json
from scripts.analyze_repair_sft_all_query_semantics import classify_query_sequence
from scripts.analyze_repair_sft_first_query_semantics import ground_truth_by_task
from scripts.analyze_repair_sft_free_run_divergence import normalize_container, sql_from_command
from scripts.analyze_state_recovery_semantics import AGGREGATES, clause, semantic_difference


ARMS = ("control", "operator_oracle", "full_plan_oracle")
HINT_PREFIX = "SEMANTIC_PLAN_GATE_V1\n"
SQL_KEYWORDS = frozenset(
    {
        "all", "and", "as", "asc", "between", "by", "case", "desc", "distinct",
        "else", "end", "false", "from", "full", "group", "having", "in", "inner",
        "is", "join", "left", "like", "limit", "not", "null", "on", "or", "order",
        "outer", "right", "select", "then", "true", "when", "where", "with",
        *[value.casefold() for value in AGGREGATES],
    }
)
IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_$]*"
QUALIFIED_IDENTIFIER_RE = re.compile(rf"\b({IDENTIFIER})(?:\.({IDENTIFIER}))?\b")
STRING_LITERAL_RE = re.compile(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"")
COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)


def _hash_stable_value(value: Any) -> Any:
    """Ignore nullable struct fields added by an Arrow/Parquet round trip."""

    if isinstance(value, dict):
        return {
            str(key): _hash_stable_value(child)
            for key, child in value.items()
            if child is not None
        }
    if isinstance(value, (list, tuple)):
        return [_hash_stable_value(child) for child in value]
    return value


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(_hash_stable_value(value)).encode("utf-8")).hexdigest()


def _without_literals(sql: str) -> str:
    return STRING_LITERAL_RE.sub(" ", COMMENT_RE.sub(" ", sql))


def _split_top_level(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote_char = ""
    index = 0
    while index < len(value):
        character = value[index]
        if quote_char:
            if character == quote_char:
                if index + 1 < len(value) and value[index + 1] == quote_char:
                    index += 1
                else:
                    quote_char = ""
        elif character in {"'", '"'}:
            quote_char = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth = max(0, depth - 1)
        elif character == "," and depth == 0:
            if value[start:index].strip():
                parts.append(value[start:index].strip())
            start = index + 1
        index += 1
    if value[start:].strip():
        parts.append(value[start:].strip())
    return parts


def _table_aliases(sql: str) -> tuple[list[str], dict[str, str]]:
    tables: list[str] = []
    aliases: dict[str, str] = {}
    clean = _without_literals(sql)
    pattern = re.compile(
        rf"\b(?:FROM|JOIN)\s+({IDENTIFIER}(?:\.{IDENTIFIER})?)", re.IGNORECASE
    )
    for match in pattern.finditer(clean):
        table = match.group(1).casefold()
        trailing_alias = re.match(
            rf"\s+(?:AS\s+)?({IDENTIFIER})\b", clean[match.end() :], re.IGNORECASE
        )
        candidate = str(trailing_alias.group(1) if trailing_alias else "").casefold()
        if table not in tables:
            tables.append(table)
        aliases[table] = table
        aliases[table.split(".")[-1]] = table
        if candidate and candidate not in SQL_KEYWORDS:
            aliases[candidate] = table
    return sorted(tables), aliases


def _schema_columns(database: Path, tables: Iterable[str]) -> dict[str, set[str]]:
    resolved = database.resolve(strict=True)
    uri = f"file:{quote(resolved.as_posix(), safe='/')}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True, timeout=5.0)
    try:
        connection.execute("PRAGMA query_only=ON")
        result: dict[str, set[str]] = {}
        for table in tables:
            bare = table.split(".")[-1]
            escaped = bare.replace('"', '""')
            rows = connection.execute(f'PRAGMA table_info("{escaped}")').fetchall()
            if not rows:
                raise ValueError(f"semantic plan references unknown table: {table}")
            result[table] = {str(row[1]).casefold() for row in rows}
        return result
    finally:
        connection.close()


def _resolve_identifier(
    owner: str | None,
    name: str,
    *,
    aliases: dict[str, str],
    schema: dict[str, set[str]],
) -> str | None:
    normalized = name.casefold()
    if normalized in SQL_KEYWORDS or normalized.isdigit():
        return None
    if owner:
        table = aliases.get(owner.casefold())
        if table and normalized in schema.get(table, set()):
            return f"{table}.{normalized}"
        return None
    matches = [table for table, columns in schema.items() if normalized in columns]
    if len(matches) == 1:
        return f"{matches[0]}.{normalized}"
    if matches:
        return f"*.{normalized}"
    return None


def _columns_in(
    sql_fragment: str,
    *,
    aliases: dict[str, str],
    schema: dict[str, set[str]],
) -> list[str]:
    clean = _without_literals(sql_fragment)
    result = {
        resolved
        for match in QUALIFIED_IDENTIFIER_RE.finditer(clean)
        if (
            resolved := _resolve_identifier(
                match.group(1) if match.group(2) else None,
                match.group(2) or match.group(1),
                aliases=aliases,
                schema=schema,
            )
        )
    }
    return sorted(result)


def operator_plan(sql: str) -> dict[str, Any]:
    select = clause(sql, "SELECT", ("FROM",))
    where = clause(sql, "WHERE", ("GROUP BY", "HAVING", "ORDER BY", "LIMIT"))
    group_by = clause(sql, "GROUP BY", ("HAVING", "ORDER BY", "LIMIT"))
    having = clause(sql, "HAVING", ("ORDER BY", "LIMIT"))
    order_by = clause(sql, "ORDER BY", ("LIMIT",))
    limit = clause(sql, "LIMIT", ())
    aggregate_calls: set[tuple[str, bool]] = set()
    for match in re.finditer(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*(DISTINCT\b)?", sql, re.IGNORECASE
    ):
        function = match.group(1).upper()
        if function in AGGREGATES:
            aggregate_calls.add((function, bool(match.group(2))))
    comparison_operators = sorted(
        {
            match.group(0).upper().replace("  ", " ")
            for match in re.finditer(
                r"(?:<=|>=|<>|!=|=|<|>|\bLIKE\b|\bIN\b|\bBETWEEN\b|\bIS\s+(?:NOT\s+)?NULL\b)",
                _without_literals(where),
                re.IGNORECASE,
            )
        }
    )
    temporal_ops = sorted(
        {
            match.group(1).upper()
            for match in re.finditer(
                r"\b(DATE|DATETIME|JULIANDAY|STRFTIME|TIME)\s*\(", sql, re.IGNORECASE
            )
        }
    )
    directions = sorted(
        {match.group(1).upper() for match in re.finditer(r"\b(ASC|DESC)\b", order_by, re.IGNORECASE)}
    )
    return {
        "scope": "operator_only",
        "projection_item_count": len(_split_top_level(select)),
        "aggregation": {
            "calls": [
                {"function": function, "distinct": distinct}
                for function, distinct in sorted(aggregate_calls)
            ],
            "grouping_required": bool(group_by),
            "group_key_count": len(_split_top_level(group_by)),
            "having_required": bool(having),
        },
        "filter": {
            "required": bool(where),
            "comparison_operators": comparison_operators,
            "and_count": len(re.findall(r"\bAND\b", where, re.IGNORECASE)),
            "or_count": len(re.findall(r"\bOR\b", where, re.IGNORECASE)),
        },
        "temporal": {"operators": temporal_ops},
        "ordering": {
            "required": bool(order_by),
            "directions": directions,
            "limit_required": bool(limit),
        },
    }


def full_semantic_plan(sql: str, database: Path) -> dict[str, Any]:
    tables, aliases = _table_aliases(sql)
    if not tables:
        raise ValueError("verified correction SQL has no bounded FROM/JOIN table")
    schema = _schema_columns(database, tables)
    role_fragments = {
        "projection": clause(sql, "SELECT", ("FROM",)),
        "filter": clause(sql, "WHERE", ("GROUP BY", "HAVING", "ORDER BY", "LIMIT")),
        "grouping": clause(sql, "GROUP BY", ("HAVING", "ORDER BY", "LIMIT")),
        "having": clause(sql, "HAVING", ("ORDER BY", "LIMIT")),
        "ordering": clause(sql, "ORDER BY", ("LIMIT",)),
    }
    columns_by_role = {
        role: _columns_in(fragment, aliases=aliases, schema=schema)
        for role, fragment in role_fragments.items()
    }
    join_columns: set[str] = set()
    join_edges: set[tuple[str, str]] = set()
    for match in re.finditer(
        rf"\b({IDENTIFIER})(?:\.({IDENTIFIER}))?\s*=\s*"
        rf"({IDENTIFIER})(?:\.({IDENTIFIER}))?",
        _without_literals(sql),
        re.IGNORECASE,
    ):
        left = _resolve_identifier(
            match.group(1) if match.group(2) else None,
            match.group(2) or match.group(1),
            aliases=aliases,
            schema=schema,
        )
        right = _resolve_identifier(
            match.group(3) if match.group(4) else None,
            match.group(4) or match.group(3),
            aliases=aliases,
            schema=schema,
        )
        if left and right and left != right:
            edge = tuple(sorted((left, right)))
            join_edges.add(edge)
            join_columns.update(edge)
    columns_by_role["join"] = sorted(join_columns)
    plan = operator_plan(sql)
    plan["scope"] = "full_semantic_plan"
    plan["grounding"] = {
        "tables": tables,
        "columns_by_role": columns_by_role,
        "join_count": len(re.findall(r"\bJOIN\b", _without_literals(sql), re.IGNORECASE)),
        "equality_join_edges": [list(edge) for edge in sorted(join_edges)],
    }
    return plan


def hint_message(plan: dict[str, Any] | None) -> dict[str, str]:
    payload = {
        "instruction": (
            "Continue from the observed insufficient SQL evidence. Return exactly one bash tool call "
            "whose command contains one read-only SELECT or WITH query against "
            "/workspace/logistics.sqlite, then stop. Do not call read/edit/write and do not provide a final answer."
        ),
        "semantic_plan": plan,
    }
    return {
        "role": "user",
        "content": HINT_PREFIX
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    }


def replay_rows(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    frame = pd.read_parquet(path)
    order: list[str] = []
    rows: dict[str, dict[str, Any]] = {}
    for _, series in frame.iterrows():
        row = normalize_container(series.to_dict())
        truth = (row.get("reward_model") or {}).get("ground_truth") or {}
        task_id = str(truth.get("task_id") or "")
        if not task_id or task_id in rows:
            raise ValueError(f"replay contains missing or duplicate task_id: {task_id!r}")
        order.append(task_id)
        rows[task_id] = row
    return order, rows


def semantic_audit_families(path: Path) -> dict[str, str]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("contract") != "repair-sft-state-recovery-semantic-audit-v1":
        raise ValueError("semantic-plan gate requires state recovery semantic audit v1")
    return {
        str(row["task_id"]): str(row["critical_token"]["family"])
        for row in result["per_task"]
    }


def build_gate_rows(
    *,
    state_rows: list[dict[str, Any]],
    replay_by_task: dict[str, dict[str, Any]],
    truths: dict[str, dict[str, Any]],
    critical_families: dict[str, str],
    database: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for state_source in state_rows:
        state = normalize_container(state_source)
        task_id = str(state.get("task_id") or "")
        if task_id not in replay_by_task or task_id not in truths or task_id not in critical_families:
            raise ValueError(f"state task is absent from a gate input: {task_id!r}")
        messages = state.get("messages") or []
        if [message.get("role") for message in messages[:5]] != [
            "system", "user", "assistant", "tool", "assistant"
        ]:
            raise ValueError(f"{task_id}: unexpected state-conditioned message shape")
        error_command = messages[2]["tool_calls"][0]["function"]["arguments"]["command"]
        correction_command = messages[4]["tool_calls"][0]["function"]["arguments"]["command"]
        error_sql = sql_from_command(error_command)
        correction_sql = sql_from_command(correction_command)
        if error_sql is None or correction_sql is None:
            raise ValueError(f"{task_id}: missing error or correction SQL")
        correction_check = classify_query_sequence(
            database=database,
            messages=[messages[4]],
            truth=truths[task_id],
        )
        if not correction_check["verified_or_equivalent_anywhere"]:
            raise ValueError(f"{task_id}: correction SQL is not mechanically verified")
        difference = semantic_difference(error_sql, correction_sql)
        plans = {
            "control": None,
            "operator_oracle": operator_plan(correction_sql),
            "full_plan_oracle": full_semantic_plan(correction_sql, database),
        }
        base_prompt = copy.deepcopy(messages[:4])
        for arm in ARMS:
            gate_id = f"{task_id}::{arm}"
            row = copy.deepcopy(replay_by_task[task_id])
            row["prompt"] = base_prompt + [hint_message(plans[arm])]
            row["sample_id"] = f"semantic-plan-gate-{gate_id}"
            row["data_source"] = "semantic_plan_sufficiency_gate"
            row["semantic_plan_gate_id"] = gate_id
            row["semantic_plan_gate_arm"] = arm
            row["semantic_plan_gate_source_task_id"] = task_id
            row["semantic_plan_gate_plan_json"] = json.dumps(
                plans[arm], ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            extra_info = copy.deepcopy(row.get("extra_info") or {})
            extra_info["tool_selection"] = ["bash"]
            extra_info["semantic_plan_gate_id"] = gate_id
            row["extra_info"] = extra_info
            reward_model = copy.deepcopy(row.get("reward_model") or {})
            ground_truth = copy.deepcopy(reward_model.get("ground_truth") or {})
            ground_truth["semantic_plan_gate_id"] = gate_id
            ground_truth["semantic_plan_gate_arm"] = arm
            ground_truth["semantic_plan_gate_source_task_id"] = task_id
            ground_truth["semantic_plan_gate_aggregation_critical"] = (
                critical_families[task_id] == "aggregation_function"
            )
            reward_model["ground_truth"] = ground_truth
            row["reward_model"] = reward_model
            output.append(row)
            evidence.append(
                {
                    "gate_id": gate_id,
                    "task_id": task_id,
                    "arm": arm,
                    "aggregation_critical": critical_families[task_id]
                    == "aggregation_function",
                    "base_prompt_sha256": sha256_value(base_prompt),
                    "hint_sha256": sha256_value(row["prompt"][-1]),
                    "error_query_sha256": sha256_value(error_sql),
                    "correction_query_sha256": sha256_value(correction_sql),
                    "semantic_difference_labels": difference["labels"],
                }
            )
    return output, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-parquet", type=Path, required=True)
    parser.add_argument("--replay-parquet", type=Path, required=True)
    parser.add_argument("--semantic-audit", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    state_frame = pd.read_parquet(args.state_parquet)
    states = [normalize_container(row) for row in state_frame.to_dict(orient="records")]
    replay_order, replays = replay_rows(args.replay_parquet)
    truth_order, truths = ground_truth_by_task(args.replay_parquet)
    families = semantic_audit_families(args.semantic_audit)
    state_order = [str(row.get("task_id") or "") for row in states]
    if state_order != replay_order or state_order != truth_order or set(state_order) != set(families):
        raise ValueError("state, replay and semantic-audit task IDs/order differ")

    rows, evidence = build_gate_rows(
        state_rows=states,
        replay_by_task=replays,
        truths=truths,
        critical_families=families,
        database=args.database,
    )
    task_count = len(state_order)
    if task_count != 16 or len(rows) != 48:
        raise ValueError(f"frozen gate requires 16 tasks / 48 rows, got {task_count}/{len(rows)}")
    aggregation_critical = sum(family == "aggregation_function" for family in families.values())
    if aggregation_critical != 9:
        raise ValueError(
            f"frozen operator gate requires 9 aggregation-critical tasks, got {aggregation_critical}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "semantic_plan_sufficiency_gate.parquet"
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    contract = {
        "contract": "semantic-plan-sufficiency-gate-dataset-v1",
        "source_checkpoint": "step120",
        "source_split": "train236_same_task_development_gate",
        "task_count": task_count,
        "rows": len(rows),
        "arms": list(ARMS),
        "rows_per_arm": dict(Counter(item["arm"] for item in evidence)),
        "aggregation_critical_tasks": aggregation_critical,
        "generation_policy": {
            "greedy_n": 1,
            "max_assistant_turns": 1,
            "tool_execution_after_generated_call": False,
            "allowed_generated_tool": "bash",
            "required_payload": "one read-only SELECT/WITH query",
        },
        "gate_thresholds": {
            "control_expected_recovery_near": "2/16 within historical second query",
            "operator_aggregation_critical_min": 4,
            "operator_control_success_regressions_max": 0,
            "full_verified_or_equivalent_min": 8,
        },
        "oracle_policy": {
            "control": "no semantic plan",
            "operator_oracle": "operators/counts only; no table, column, literal, SQL, result or answer",
            "full_plan_oracle": "table/join/column roles plus operators; no literal, SQL, result or answer",
        },
        "output": output_path.name,
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "evidence": evidence,
        "optimizer_initialized": False,
        "checkpoint_saved": False,
        "promotion_allowed": False,
    }
    contract_path = args.output_dir / "contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in contract.items() if key != "evidence"}, indent=2))


if __name__ == "__main__":
    main()

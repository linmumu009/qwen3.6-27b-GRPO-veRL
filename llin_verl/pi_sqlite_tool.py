"""A read-only SQLite tool for PI warehouse trajectories."""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

_ENVIRONMENT_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
_TABLE_RE = re.compile(
    r"\b(?:from|join)\s+[\"`\[]?([A-Za-z_][A-Za-z0-9_]*)",
    flags=re.IGNORECASE,
)


def resolve_database(sandbox_root: str | Path, environment_id: str, database_name: str) -> Path:
    """Resolve a database path and prove that it remains below the sandbox root."""
    if not environment_id or not _ENVIRONMENT_RE.fullmatch(environment_id):
        raise ValueError("invalid environment_id")
    if ".." in Path(environment_id).parts:
        raise ValueError("environment_id cannot contain '..'")

    root = Path(sandbox_root).resolve(strict=True)
    database = (root / environment_id / database_name).resolve(strict=True)
    if not database.is_relative_to(root):
        raise ValueError("database path escapes the configured sandbox root")
    if not database.is_file():
        raise ValueError(f"database does not exist: {database}")
    return database


def extract_table_names(sql: str) -> list[str]:
    """Extract FROM/JOIN table names for reward evidence."""
    return sorted({match.group(1).lower() for match in _TABLE_RE.finditer(sql)})


def _run_query(database: Path, sql: str, max_rows: int, progress_steps: int) -> tuple[list[str], list[list[Any]], bool]:
    if not sql or len(sql) > 20_000:
        raise ValueError("SQL must contain between 1 and 20,000 characters")

    uri = f"file:{quote(database.as_posix(), safe='/')}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True, timeout=5.0)
    try:
        connection.execute("PRAGMA query_only=ON")
        remaining = progress_steps

        def progress() -> int:
            nonlocal remaining
            remaining -= 1
            return 1 if remaining <= 0 else 0

        connection.set_progress_handler(progress, 1_000)
        cursor = connection.execute(sql)
        if cursor.description is None:
            raise ValueError("only read-only SELECT/WITH queries are allowed")
        columns = [str(column[0]) for column in cursor.description]
        rows = [list(row) for row in cursor.fetchmany(max_rows + 1)]
        truncated = len(rows) > max_rows
        return columns, rows[:max_rows], truncated
    finally:
        connection.close()


class ReadOnlySQLiteTool(BaseTool):
    """Execute bounded, read-only SQL against one trajectory environment."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.sandbox_root = config.get("sandbox_root", "/pi_sandbox")
        self.database_name = config.get("database_name", "logistics.sqlite")
        self.max_rows = int(config.get("max_rows", 80))
        self.progress_steps = int(config.get("progress_steps", 100_000))

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[ToolResponse, float, dict]:
        del instance_id
        sql = str(parameters.get("sql", "")).strip()
        agent_data = kwargs.get("agent_data")
        create_kwargs: dict[str, Any] = {}
        if agent_data is not None:
            tool_kwargs = agent_data.tools_kwargs.get(self.name, {})
            create_kwargs = tool_kwargs.get("create_kwargs", {})
        environment_id = str(create_kwargs.get("environment_id", ""))
        tables = extract_table_names(sql)

        try:
            database = resolve_database(self.sandbox_root, environment_id, self.database_name)
            columns, rows, truncated = await asyncio.to_thread(
                _run_query,
                database,
                sql,
                self.max_rows,
                self.progress_steps,
            )
            payload = {
                "columns": columns,
                "rows": rows,
                "truncated": truncated,
            }
            ok = True
            response = json.dumps(payload, ensure_ascii=False, default=str)
        except (ValueError, OSError, sqlite3.Error) as exc:
            ok = False
            response = f"只读查询失败：{exc}"

        if agent_data is not None:
            query_log = agent_data.extra_fields.setdefault("llin_sql_queries", [])
            query_log.append({"sql": sql, "tables": tables, "ok": ok})

        metrics = {
            "query_ok": float(ok),
            "tables_used": len(tables),
        }
        return ToolResponse(text=response), 0.0, metrics

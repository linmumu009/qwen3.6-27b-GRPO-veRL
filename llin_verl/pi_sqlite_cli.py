"""Read-only sqlite3 CLI compatibility layer for the Ascend veRL image."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from pathlib import Path
from urllib.parse import quote


_READ_ONLY = re.compile(r"^\s*(?:SELECT|WITH|PRAGMA|EXPLAIN)\b", re.IGNORECASE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-header", action="store_true")
    parser.add_argument("-csv", action="store_true")
    parser.add_argument("-json", action="store_true")
    parser.add_argument("-readonly", action="store_true")
    parser.add_argument("-separator", default="|")
    parser.add_argument("database")
    parser.add_argument("command", nargs="?")
    return parser.parse_args(argv)


def _connect(path: str) -> sqlite3.Connection:
    database = Path(path).resolve(strict=True)
    uri = f"file:{quote(database.as_posix(), safe='/')}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True, timeout=5.0)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _dot_command(connection: sqlite3.Connection, command: str) -> tuple[list[str], list[tuple]]:
    parts = command.strip().split(maxsplit=1)
    name = parts[0]
    value = parts[1] if len(parts) > 1 else ""
    if name == ".tables":
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return [], [(" ".join(str(row[0]) for row in rows),)]
    if name == ".schema":
        if value:
            rows = connection.execute(
                "SELECT sql FROM sqlite_master WHERE name = ? AND sql IS NOT NULL ORDER BY type, name",
                (value,),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name"
            ).fetchall()
        return [], rows
    raise ValueError(f"unsupported sqlite3 dot command: {name}")


def _execute(connection: sqlite3.Connection, command: str) -> tuple[list[str], list[tuple]]:
    if command.lstrip().startswith("."):
        return _dot_command(connection, command)
    statements = [part.strip() for part in command.split(";") if part.strip()]
    if not statements:
        return [], []
    columns: list[str] = []
    rows: list[tuple] = []
    for statement in statements:
        if not _READ_ONLY.match(statement):
            raise ValueError("only SELECT/WITH/PRAGMA/EXPLAIN statements are allowed")
        cursor = connection.execute(statement)
        columns = [str(item[0]) for item in cursor.description or []]
        rows = cursor.fetchmany(10_001)
        if len(rows) > 10_000:
            raise ValueError("query output exceeds 10,000 rows")
    return columns, rows


def _print_rows(args: argparse.Namespace, columns: list[str], rows: list[tuple]) -> None:
    if args.json:
        print(json.dumps([dict(zip(columns, row)) for row in rows], ensure_ascii=False, default=str))
        return
    writer = csv.writer(sys.stdout, delimiter="," if args.csv else args.separator, lineterminator="\n")
    if args.header and columns:
        writer.writerow(columns)
    writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = args.command if args.command is not None else sys.stdin.read()
    try:
        connection = _connect(args.database)
        try:
            columns, rows = _execute(connection, command)
        finally:
            connection.close()
        _print_rows(args, columns, rows)
        return 0
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


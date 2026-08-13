import sqlite3
from pathlib import Path

from scripts.audit_review_pilot_query_stability import execute_probe


def database(tmp_path: Path) -> Path:
    path = tmp_path / "probe.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("create table items(value integer)")
    connection.executemany("insert into items values (?)", [(1,), (2,), (3,)])
    connection.commit()
    connection.close()
    return path


def test_reverse_probe_changes_unordered_limit(tmp_path: Path):
    path = database(tmp_path)
    _, normal = execute_probe(path, "SELECT value FROM items LIMIT 2", False)
    _, reversed_rows = execute_probe(path, "SELECT value FROM items LIMIT 2", True)
    assert normal != reversed_rows


def test_reverse_probe_preserves_explicit_order(tmp_path: Path):
    path = database(tmp_path)
    sql = "SELECT value FROM items ORDER BY value DESC LIMIT 2"
    assert execute_probe(path, sql, False) == execute_probe(path, sql, True)

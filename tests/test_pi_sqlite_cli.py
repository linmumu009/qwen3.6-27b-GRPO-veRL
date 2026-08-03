import sqlite3

from llin_verl.pi_sqlite_cli import main


def test_read_only_sqlite_cli_supports_tables_select_and_rejects_write(tmp_path, capsys):
    database = tmp_path / "data.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("create table metric(value integer)")
    connection.execute("insert into metric values (7)")
    connection.commit()
    connection.close()

    assert main([str(database), ".tables"]) == 0
    assert "metric" in capsys.readouterr().out
    assert main(["-header", str(database), "SELECT value FROM metric"]) == 0
    assert capsys.readouterr().out.splitlines() == ["value", "7"]
    assert main([str(database), "DELETE FROM metric"]) == 1
    assert "only SELECT" in capsys.readouterr().err

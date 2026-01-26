from pathlib import Path
import pytest
from chinvex.storage import Storage, SCHEMA_VERSION


def test_schema_has_version_in_meta_table(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    cur = storage.conn.execute("SELECT value FROM meta WHERE key = 'schema_version'")
    row = cur.fetchone()
    assert row is not None
    assert int(row["value"]) == SCHEMA_VERSION


def test_mismatched_schema_version_errors(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    # Manually set wrong version
    storage.conn.execute("UPDATE meta SET value = '999' WHERE key = 'schema_version'")
    storage.conn.commit()
    storage.close()

    # Reopen should error
    with pytest.raises(RuntimeError, match="schema version mismatch"):
        storage2 = Storage(db_path)
        storage2.ensure_schema()

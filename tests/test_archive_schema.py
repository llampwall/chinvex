"""Test archive schema migration."""
import pytest
import sqlite3


def test_documents_table_has_archived_column(tmp_path):
    """Test that documents table has archived column."""
    from chinvex.storage import Storage

    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("PRAGMA table_info(documents)")
    columns = [row[1] for row in cursor.fetchall()]

    assert "archived" in columns
    assert "archived_at" in columns
    conn.close()


def test_archived_index_exists(tmp_path):
    """Test that archived index is created."""
    from chinvex.storage import Storage

    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_documents_archived'"
    )

    result = cursor.fetchone()
    conn.close()
    assert result is not None

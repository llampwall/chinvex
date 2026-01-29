"""Test chunk_key storage and migration."""
import pytest
import sqlite3
from pathlib import Path


def test_chunks_table_has_chunk_key_column(tmp_path):
    """Test that chunks table includes chunk_key column."""
    from chinvex.storage import Storage

    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    cursor = storage.conn.execute("PRAGMA table_info(chunks)")
    columns = [row[1] for row in cursor.fetchall()]

    assert "chunk_key" in columns


def test_chunk_key_index_exists(tmp_path):
    """Test that chunk_key index is created."""
    from chinvex.storage import Storage

    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    cursor = storage.conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_chunks_chunk_key'")

    assert cursor.fetchone() is not None

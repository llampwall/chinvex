"""Test archive commands."""
import pytest
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typer.testing import CliRunner
from chinvex.cli import app

runner = CliRunner()


def test_archive_run_dry_run_default(tmp_path):
    """Test that archive_old_documents dry-run doesn't modify data."""
    from chinvex.archive import archive_old_documents
    from chinvex.storage import Storage

    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    # Insert old document
    now = datetime.utcnow()
    old_date = (now - timedelta(days=200)).isoformat()

    storage._execute(
        "INSERT INTO documents (doc_id, source_type, source_uri, title, updated_at, archived) VALUES (?, ?, ?, ?, ?, ?)",
        ("doc_old", "repo", "file:///old.txt", "Old Doc", old_date, 0)
    )
    storage.conn.commit()

    # Run dry-run
    count = archive_old_documents(storage, age_threshold_days=180, dry_run=True)

    # Should report 1 would be archived
    assert count == 1

    # But archived flag should still be 0
    cursor = storage.conn.execute("SELECT archived FROM documents WHERE doc_id = ?", ("doc_old",))
    row = cursor.fetchone()
    assert row["archived"] == 0


def test_archive_run_force_executes(tmp_path):
    """Test that archive_old_documents without dry_run modifies data."""
    from chinvex.archive import archive_old_documents
    from chinvex.storage import Storage

    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    # Insert old document
    now = datetime.utcnow()
    old_date = (now - timedelta(days=200)).isoformat()

    storage._execute(
        "INSERT INTO documents (doc_id, source_type, source_uri, title, updated_at, archived) VALUES (?, ?, ?, ?, ?, ?)",
        ("doc_old", "repo", "file:///old.txt", "Old Doc", old_date, 0)
    )
    storage.conn.commit()

    # Run archive (not dry-run)
    count = archive_old_documents(storage, age_threshold_days=180, dry_run=False)

    # Should report 1 archived
    assert count == 1

    # And archived flag should be 1
    cursor = storage.conn.execute("SELECT archived, archived_at FROM documents WHERE doc_id = ?", ("doc_old",))
    row = cursor.fetchone()
    assert row["archived"] == 1
    assert row["archived_at"] is not None


def test_archive_run_respects_threshold(tmp_path):
    """Test that only docs older than threshold are archived."""
    from chinvex.archive import archive_old_documents
    from chinvex.storage import Storage

    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    # Insert test documents
    now = datetime.utcnow()
    old_date = (now - timedelta(days=200)).isoformat()
    recent_date = (now - timedelta(days=30)).isoformat()

    storage._execute(
        "INSERT INTO documents (doc_id, source_type, source_uri, title, updated_at, archived) VALUES (?, ?, ?, ?, ?, ?)",
        ("doc_old", "repo", "file:///old.txt", "Old Doc", old_date, 0)
    )
    storage._execute(
        "INSERT INTO documents (doc_id, source_type, source_uri, title, updated_at, archived) VALUES (?, ?, ?, ?, ?, ?)",
        ("doc_recent", "repo", "file:///recent.txt", "Recent Doc", recent_date, 0)
    )
    storage.conn.commit()

    # Run archive
    count = archive_old_documents(storage, age_threshold_days=180)

    # Verify only old doc archived
    assert count == 1

    cursor = storage.conn.execute("SELECT doc_id, archived FROM documents ORDER BY doc_id")
    rows = cursor.fetchall()
    assert rows[0]["doc_id"] == "doc_old"
    assert rows[0]["archived"] == 1
    assert rows[1]["doc_id"] == "doc_recent"
    assert rows[1]["archived"] == 0

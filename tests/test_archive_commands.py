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


def test_archive_list_shows_archived_docs(tmp_path):
    """Test that list_archived_documents returns archived documents."""
    from chinvex.archive import list_archived_documents
    from chinvex.storage import Storage

    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    # Insert archived and active docs
    storage._execute(
        "INSERT INTO documents (doc_id, source_type, source_uri, title, archived, archived_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("doc_archived", "repo", "file:///archived.txt", "Archived Doc", 1, "2026-01-01T00:00:00")
    )
    storage._execute(
        "INSERT INTO documents (doc_id, source_type, source_uri, title, archived) VALUES (?, ?, ?, ?, ?)",
        ("doc_active", "repo", "file:///active.txt", "Active Doc", 0)
    )
    storage.conn.commit()

    # List archived
    docs = list_archived_documents(storage, limit=50)

    # Should only get archived doc
    assert len(docs) == 1
    assert docs[0]["doc_id"] == "doc_archived"
    assert docs[0]["title"] == "Archived Doc"


def test_archive_restore_unarchives_doc(tmp_path):
    """Test that restore flips archived flag."""
    from chinvex.archive import restore_document
    from chinvex.storage import Storage

    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    # Insert archived doc
    storage._execute(
        "INSERT INTO documents (doc_id, source_type, source_uri, title, archived, archived_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("doc_archived", "repo", "file:///archived.txt", "Archived Doc", 1, "2026-01-01T00:00:00")
    )
    storage.conn.commit()

    # Restore
    success = restore_document(storage, "doc_archived")

    # Should succeed
    assert success == True

    # Verify unarchived
    cursor = storage.conn.execute("SELECT archived, archived_at FROM documents WHERE doc_id = ?", ("doc_archived",))
    row = cursor.fetchone()
    assert row["archived"] == 0
    assert row["archived_at"] is None


def test_archive_restore_nonexistent_doc(tmp_path):
    """Test that restore returns False for nonexistent doc."""
    from chinvex.archive import restore_document
    from chinvex.storage import Storage

    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    # Try to restore nonexistent doc
    success = restore_document(storage, "nonexistent")

    # Should fail
    assert success == False


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


def test_archive_purge_dry_run_default(tmp_path):
    """Test that purge is dry-run by default."""
    from chinvex.archive import purge_archived_documents
    from chinvex.storage import Storage

    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    # Insert very old archived doc
    now = datetime.utcnow()
    very_old = (now - timedelta(days=400)).isoformat()

    storage._execute(
        "INSERT INTO documents (doc_id, source_type, source_uri, title, archived, archived_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("doc_very_old", "repo", "file:///old.txt", "Very Old", 1, very_old)
    )
    storage.conn.commit()

    # Purge dry-run
    count = purge_archived_documents(storage, age_threshold_days=365, dry_run=True)

    # Should report 1 would be purged
    assert count == 1

    # But document should still exist
    cursor = storage.conn.execute("SELECT COUNT(*) as count FROM documents WHERE doc_id = ?", ("doc_very_old",))
    assert cursor.fetchone()["count"] == 1


def test_archive_purge_deletes_permanently(tmp_path):
    """Test that purge --force deletes archived docs."""
    from chinvex.archive import purge_archived_documents
    from chinvex.storage import Storage

    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    # Insert very old archived doc with chunks
    now = datetime.utcnow()
    very_old = (now - timedelta(days=400)).isoformat()

    storage._execute(
        "INSERT INTO documents (doc_id, source_type, source_uri, title, archived, archived_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("doc_very_old", "repo", "file:///old.txt", "Very Old", 1, very_old)
    )
    storage._execute(
        "INSERT INTO chunks (chunk_id, doc_id, source_type, ordinal, text, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("chunk_1", "doc_very_old", "repo", 0, "test chunk", very_old)
    )
    storage.conn.commit()

    # Purge with force
    count = purge_archived_documents(storage, age_threshold_days=365, dry_run=False)

    # Verify deleted
    assert count == 1
    cursor = storage.conn.execute("SELECT COUNT(*) as count FROM documents WHERE doc_id = ?", ("doc_very_old",))
    assert cursor.fetchone()["count"] == 0

    # Verify chunks also deleted
    cursor = storage.conn.execute("SELECT COUNT(*) as count FROM chunks WHERE doc_id = ?", ("doc_very_old",))
    assert cursor.fetchone()["count"] == 0


def test_archive_purge_only_purges_archived(tmp_path):
    """Test that purge only affects archived documents."""
    from chinvex.archive import purge_archived_documents
    from chinvex.storage import Storage

    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    # Insert very old active doc (should NOT be purged)
    now = datetime.utcnow()
    very_old = (now - timedelta(days=400)).isoformat()

    storage._execute(
        "INSERT INTO documents (doc_id, source_type, source_uri, title, archived) VALUES (?, ?, ?, ?, ?)",
        ("doc_active_old", "repo", "file:///old.txt", "Old Active", 0)
    )
    storage.conn.commit()

    # Purge
    count = purge_archived_documents(storage, age_threshold_days=365, dry_run=False)

    # Should not purge active doc
    assert count == 0
    cursor = storage.conn.execute("SELECT COUNT(*) as count FROM documents WHERE doc_id = ?", ("doc_active_old",))
    assert cursor.fetchone()["count"] == 1

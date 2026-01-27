# tests/state/test_extractors.py
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime, timedelta, timezone
from chinvex.state.extractors import extract_recently_changed


def test_extract_recently_changed():
    """Test extracting recently changed documents."""
    # Create temporary DB file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        db_path = f.name

    try:
        # Create DB with schema
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE source_fingerprints (
                source_uri TEXT NOT NULL,
                context_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                size_bytes INTEGER,
                mtime_unix INTEGER,
                content_sha256 TEXT,
                thread_updated_at TEXT,
                last_turn_id TEXT,
                parser_version TEXT NOT NULL,
                chunker_version TEXT NOT NULL,
                embedded_model TEXT,
                last_ingested_at_unix INTEGER NOT NULL,
                last_status TEXT NOT NULL,
                last_error TEXT,
                PRIMARY KEY (source_uri, context_name)
            )
        """)

        # Insert test data
        now = datetime.now(timezone.utc)
        recent_time = now - timedelta(minutes=30)

        conn.execute("""
            INSERT INTO source_fingerprints
            (source_uri, context_name, source_type, doc_id, parser_version, chunker_version,
             last_ingested_at_unix, last_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            "C:\\test\\recent.py",
            "TestContext",
            "repo",
            "doc_recent",
            "v1",
            "v1",
            int(recent_time.timestamp()),
            "ok"
        ])
        conn.commit()
        conn.close()

        # Test extraction
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        changed = extract_recently_changed(
            context="TestContext",
            since=cutoff,
            limit=20,
            db_path=db_path
        )

        assert isinstance(changed, list)
        assert len(changed) == 1
        assert changed[0].doc_id == "doc_recent"
        assert changed[0].source_type == "repo"
        assert changed[0].change_type == "modified"

    finally:
        # Cleanup
        Path(db_path).unlink(missing_ok=True)

from pathlib import Path
from chinvex.storage import Storage


def test_fingerprints_table_exists(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    cur = storage.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='source_fingerprints'"
    )
    assert cur.fetchone() is not None


def test_upsert_fingerprint_for_file(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    storage.upsert_fingerprint(
        source_uri="C:\\Code\\test.py",
        context_name="TestContext",
        source_type="repo",
        doc_id="doc123",
        size_bytes=1024,
        mtime_unix=1234567890,
        content_sha256="abc123",
        parser_version="v1",
        chunker_version="v1",
        embedded_model="mxbai-embed-large",
        last_status="ok",
        last_error=None,
    )

    fp = storage.get_fingerprint("C:\\Code\\test.py", "TestContext")
    assert fp is not None
    assert fp["source_type"] == "repo"
    assert fp["size_bytes"] == 1024
    assert fp["last_status"] == "ok"


def test_upsert_fingerprint_for_thread(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    storage.upsert_fingerprint(
        source_uri="thread-abc",
        context_name="TestContext",
        source_type="codex_session",
        doc_id="doc456",
        thread_updated_at="2026-01-26T10:00:00Z",
        last_turn_id="turn-999",
        parser_version="v1",
        chunker_version="v1",
        embedded_model="mxbai-embed-large",
        last_status="ok",
        last_error=None,
    )

    fp = storage.get_fingerprint("thread-abc", "TestContext")
    assert fp is not None
    assert fp["source_type"] == "codex_session"
    assert fp["thread_updated_at"] == "2026-01-26T10:00:00Z"
    assert fp["last_turn_id"] == "turn-999"

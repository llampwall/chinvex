"""Test auto-archive on ingest."""
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from chinvex.storage import Storage
from chinvex.archive import archive_old_documents
from unittest.mock import patch, MagicMock


def test_auto_archive_runs_after_ingest(tmp_path):
    """Test that auto-archive runs when enabled in config."""
    from chinvex.ingest import ingest_context
    from chinvex.context import ContextConfig, ContextIncludes, ContextIndex, OllamaConfig, RankingConfig, ArchiveConfig

    # Create test repo
    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()
    (test_repo / "file.txt").write_text("test content", encoding="utf-8")

    # Create old document in database
    db_path = tmp_path / "hybrid.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    now = datetime.utcnow()
    old_date = (now - timedelta(days=200)).isoformat()
    storage._execute(
        "INSERT INTO documents (doc_id, source_type, source_uri, title, updated_at, archived) VALUES (?, ?, ?, ?, ?, ?)",
        ("doc_old", "repo", "file:///old.txt", "Old Doc", old_date, 0)
    )
    storage.conn.commit()
    storage.close()

    # Create context with archive enabled
    ctx = ContextConfig(
        schema_version=2,
        name="test",
        aliases=[],
        includes=ContextIncludes(
            repos=[test_repo],
            chat_roots=[],
            codex_session_roots=[],
            note_roots=[]
        ),
        index=ContextIndex(
            sqlite_path=db_path,
            chroma_dir=tmp_path / "chroma"
        ),
        weights={"repo": 1.0},
        ollama=OllamaConfig(
            base_url="http://127.0.0.1:11434",
            embed_model="mxbai-embed-large"
        ),
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
        ranking=RankingConfig(
            recency_enabled=True,
            recency_half_life_days=90
        ),
        archive=ArchiveConfig(
            enabled=True,
            auto_archive_on_ingest=True,
            age_threshold_days=180,
            archive_penalty=0.8
        )
    )

    # Mock OllamaEmbedder to avoid network calls
    with patch('chinvex.ingest.OllamaEmbedder') as mock_embedder:
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.embed.return_value = [[0.1] * 1024]  # Mock embedding
        mock_embedder_instance.model = "mxbai-embed-large"  # Mock model attribute
        mock_embedder.return_value = mock_embedder_instance

        # Run ingest
        result = ingest_context(ctx)

    # Verify old document was archived
    storage = Storage(db_path)
    cursor = storage.conn.execute("SELECT archived FROM documents WHERE doc_id = ?", ("doc_old",))
    row = cursor.fetchone()
    assert row is not None
    assert row["archived"] == 1


def test_auto_archive_respects_enabled_flag(tmp_path):
    """Test that auto-archive only runs when enabled."""
    from chinvex.ingest import ingest_context
    from chinvex.context import ContextConfig, ContextIncludes, ContextIndex, OllamaConfig, RankingConfig, ArchiveConfig

    # Create test repo
    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()
    (test_repo / "file.txt").write_text("test content", encoding="utf-8")

    # Create old document in database
    db_path = tmp_path / "hybrid.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    now = datetime.utcnow()
    old_date = (now - timedelta(days=200)).isoformat()
    storage._execute(
        "INSERT INTO documents (doc_id, source_type, source_uri, title, updated_at, archived) VALUES (?, ?, ?, ?, ?, ?)",
        ("doc_old", "repo", "file:///old.txt", "Old Doc", old_date, 0)
    )
    storage.conn.commit()
    storage.close()

    # Create context with archive disabled
    ctx = ContextConfig(
        schema_version=2,
        name="test",
        aliases=[],
        includes=ContextIncludes(
            repos=[test_repo],
            chat_roots=[],
            codex_session_roots=[],
            note_roots=[]
        ),
        index=ContextIndex(
            sqlite_path=db_path,
            chroma_dir=tmp_path / "chroma"
        ),
        weights={"repo": 1.0},
        ollama=OllamaConfig(
            base_url="http://127.0.0.1:11434",
            embed_model="mxbai-embed-large"
        ),
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
        ranking=RankingConfig(
            recency_enabled=True,
            recency_half_life_days=90
        ),
        archive=ArchiveConfig(
            enabled=False,
            auto_archive_on_ingest=True,
            age_threshold_days=180,
            archive_penalty=0.8
        )
    )

    # Mock OllamaEmbedder to avoid network calls
    with patch('chinvex.ingest.OllamaEmbedder') as mock_embedder:
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.embed.return_value = [[0.1] * 1024]
        mock_embedder_instance.model = "mxbai-embed-large"
        mock_embedder.return_value = mock_embedder_instance

        # Run ingest
        result = ingest_context(ctx)

    # Verify old document was NOT archived
    storage = Storage(db_path)
    cursor = storage.conn.execute("SELECT archived FROM documents WHERE doc_id = ?", ("doc_old",))
    row = cursor.fetchone()
    assert row is not None
    assert row["archived"] == 0


def test_auto_archive_logs_count(tmp_path, capsys):
    """Test that auto-archive logs archived count."""
    from chinvex.ingest import ingest_context
    from chinvex.context import ContextConfig, ContextIncludes, ContextIndex, OllamaConfig, RankingConfig, ArchiveConfig

    # Create test repo
    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()
    (test_repo / "file.txt").write_text("test content", encoding="utf-8")

    # Create old document in database
    db_path = tmp_path / "hybrid.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    now = datetime.utcnow()
    old_date = (now - timedelta(days=200)).isoformat()
    storage._execute(
        "INSERT INTO documents (doc_id, source_type, source_uri, title, updated_at, archived) VALUES (?, ?, ?, ?, ?, ?)",
        ("doc_old", "repo", "file:///old.txt", "Old Doc", old_date, 0)
    )
    storage.conn.commit()
    storage.close()

    # Create context with archive enabled
    ctx = ContextConfig(
        schema_version=2,
        name="test",
        aliases=[],
        includes=ContextIncludes(
            repos=[test_repo],
            chat_roots=[],
            codex_session_roots=[],
            note_roots=[]
        ),
        index=ContextIndex(
            sqlite_path=db_path,
            chroma_dir=tmp_path / "chroma"
        ),
        weights={"repo": 1.0},
        ollama=OllamaConfig(
            base_url="http://127.0.0.1:11434",
            embed_model="mxbai-embed-large"
        ),
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
        ranking=RankingConfig(
            recency_enabled=True,
            recency_half_life_days=90
        ),
        archive=ArchiveConfig(
            enabled=True,
            auto_archive_on_ingest=True,
            age_threshold_days=180,
            archive_penalty=0.8
        )
    )

    # Mock OllamaEmbedder to avoid network calls
    with patch('chinvex.ingest.OllamaEmbedder') as mock_embedder:
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.embed.return_value = [[0.1] * 1024]
        mock_embedder_instance.model = "mxbai-embed-large"
        mock_embedder.return_value = mock_embedder_instance

        # Run ingest
        result = ingest_context(ctx)

    # Check stdout for archive message
    captured = capsys.readouterr()
    assert "Archived 1 docs" in captured.out or "Archived 1 document" in captured.out

# tests/test_ingest_result.py
from datetime import datetime
from pathlib import Path
import json
from chinvex.ingest import IngestRunResult, ingest_context
from chinvex.context import ContextConfig

def test_ingest_run_result_creation():
    """Test IngestRunResult can be created with all required fields."""
    result = IngestRunResult(
        run_id="test_run_123",
        context="TestContext",
        started_at=datetime.now(),
        finished_at=datetime.now(),
        new_doc_ids=["doc1", "doc2"],
        updated_doc_ids=["doc3"],
        new_chunk_ids=["chunk1", "chunk2", "chunk3"],
        skipped_doc_ids=["doc4"],
        error_doc_ids=["doc5"],
        stats={"files_scanned": 10, "total_chunks": 25}
    )

    assert result.run_id == "test_run_123"
    assert result.context == "TestContext"
    assert len(result.new_doc_ids) == 2
    assert len(result.new_chunk_ids) == 3
    assert result.stats["files_scanned"] == 10


def test_ingest_context_returns_result(tmp_path: Path, monkeypatch):
    """Test that ingest_context returns IngestRunResult."""
    # Mock OllamaEmbedder
    class FakeEmbedder:
        def __init__(self, host: str, model: str, fallback_host: str | None = None):
            self.model = model
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr("chinvex.ingest.OllamaEmbedder", FakeEmbedder)

    # Create minimal test context
    test_repo = tmp_path / "repo"
    test_repo.mkdir()
    (test_repo / "test.txt").write_text("test content")

    db_path = tmp_path / "hybrid.db"
    chroma_dir = tmp_path / "chroma"

    ctx_dict = {
        "schema_version": 1,
        "name": "TestContext",
        "aliases": [],
        "includes": {
            "repos": [test_repo],
            "chat_roots": [],
            "codex_session_roots": [],
            "note_roots": []
        },
        "index": {
            "sqlite_path": db_path,
            "chroma_dir": chroma_dir
        },
        "weights": {"repo": 1.0, "chat": 0.8, "codex_session": 0.9, "note": 0.7},
        "ollama": {"base_url": "http://127.0.0.1:11434", "embed_model": "mxbai-embed-large"},
        "created_at": "2026-01-26T00:00:00Z",
        "updated_at": "2026-01-26T00:00:00Z"
    }

    from chinvex.context import ContextConfig
    ctx = ContextConfig.from_dict(ctx_dict)

    result = ingest_context(ctx)

    assert isinstance(result, IngestRunResult)
    assert result.run_id is not None
    assert result.context == "TestContext"
    assert isinstance(result.new_chunk_ids, list)
    assert isinstance(result.stats, dict)


def test_ingest_calls_post_hook(tmp_path: Path, monkeypatch):
    """Test that ingest calls post-ingest hook."""
    # Mock OllamaEmbedder
    class FakeEmbedder:
        def __init__(self, host: str, model: str, fallback_host: str | None = None):
            self.model = model
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr("chinvex.ingest.OllamaEmbedder", FakeEmbedder)

    # Track hook calls
    hook_called = []

    def mock_hook(context, result):
        hook_called.append((context, result))

    monkeypatch.setattr('chinvex.ingest.post_ingest_hook', mock_hook)

    # Create minimal test context
    test_repo = tmp_path / "repo"
    test_repo.mkdir()
    (test_repo / "test.txt").write_text("test content")

    db_path = tmp_path / "hybrid.db"
    chroma_dir = tmp_path / "chroma"

    ctx_dict = {
        "schema_version": 1,
        "name": "TestContext",
        "aliases": [],
        "includes": {
            "repos": [test_repo],
            "chat_roots": [],
            "codex_session_roots": [],
            "note_roots": []
        },
        "index": {
            "sqlite_path": db_path,
            "chroma_dir": chroma_dir
        },
        "weights": {"repo": 1.0, "chat": 0.8, "codex_session": 0.9, "note": 0.7},
        "ollama": {"base_url": "http://127.0.0.1:11434", "embed_model": "mxbai-embed-large"},
        "created_at": "2026-01-26T00:00:00Z",
        "updated_at": "2026-01-26T00:00:00Z"
    }

    from chinvex.context import ContextConfig
    ctx = ContextConfig.from_dict(ctx_dict)

    result = ingest_context(ctx)

    assert len(hook_called) == 1
    assert hook_called[0][0] == ctx
    assert hook_called[0][1] == result

"""Tests for mixed embedding provider detection in cross-context search."""

import pytest
from pathlib import Path
from chinvex.context import ContextConfig, ContextIncludes, ContextIndex, OllamaConfig, EmbeddingConfig
from chinvex.search import search_multi_context
from datetime import datetime, timezone


def test_mixed_embedding_providers_raises_error(tmp_path, monkeypatch):
    """Cross-context search with mixed embedding providers should raise clear error."""
    # Setup two contexts with different embedding providers
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"
    contexts_root.mkdir()
    indexes_root.mkdir()

    # Context 1: OpenAI embeddings
    ctx1_dir = contexts_root / "Context1"
    ctx1_dir.mkdir()
    ctx1_index = indexes_root / "Context1"
    ctx1_index.mkdir()

    ctx1_data = {
        "schema_version": 2,
        "name": "Context1",
        "aliases": [],
        "includes": {"repos": [], "chat_roots": [], "codex_session_roots": [], "note_roots": []},
        "index": {"sqlite_path": str(ctx1_index / "hybrid.db"), "chroma_dir": str(ctx1_index / "chroma")},
        "weights": {"repo": 1.0, "chat": 0.8},
        "ollama": {"base_url": "http://localhost:11434", "embed_model": "mxbai-embed-large"},
        "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    (ctx1_dir / "context.json").write_text(
        __import__("json").dumps(ctx1_data, indent=2)
    )

    # Context 2: Ollama embeddings
    ctx2_dir = contexts_root / "Context2"
    ctx2_dir.mkdir()
    ctx2_index = indexes_root / "Context2"
    ctx2_index.mkdir()

    ctx2_data = {
        "schema_version": 2,
        "name": "Context2",
        "aliases": [],
        "includes": {"repos": [], "chat_roots": [], "codex_session_roots": [], "note_roots": []},
        "index": {"sqlite_path": str(ctx2_index / "hybrid.db"), "chroma_dir": str(ctx2_index / "chroma")},
        "weights": {"repo": 1.0, "chat": 0.8},
        "ollama": {"base_url": "http://localhost:11434", "embed_model": "mxbai-embed-large"},
        "embedding": {"provider": "ollama", "model": "mxbai-embed-large"},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    (ctx2_dir / "context.json").write_text(
        __import__("json").dumps(ctx2_data, indent=2)
    )

    # Set environment variable for contexts root
    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))

    # Attempt cross-context search with mixed providers
    with pytest.raises(ValueError) as exc_info:
        search_multi_context(
            contexts=["Context1", "Context2"],
            query="test query",
            k=5
        )

    # Verify error message is clear
    error_msg = str(exc_info.value)
    assert "mixed embedding providers" in error_msg.lower()
    assert "openai" in error_msg.lower()
    assert "ollama" in error_msg.lower()
    assert "Context1" in error_msg or "Context2" in error_msg


def test_same_embedding_provider_allowed(tmp_path, monkeypatch):
    """Cross-context search with same embedding provider should succeed."""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"
    contexts_root.mkdir()
    indexes_root.mkdir()

    # Both contexts use OpenAI
    for ctx_name in ["ContextA", "ContextB"]:
        ctx_dir = contexts_root / ctx_name
        ctx_dir.mkdir()
        ctx_index = indexes_root / ctx_name
        ctx_index.mkdir()

        # Create minimal index files
        from chinvex.storage import Storage
        db_path = ctx_index / "hybrid.db"
        storage = Storage(db_path)
        storage.ensure_schema()
        storage.close()

        ctx_data = {
            "schema_version": 2,
            "name": ctx_name,
            "aliases": [],
            "includes": {"repos": [], "chat_roots": [], "codex_session_roots": [], "note_roots": []},
            "index": {"sqlite_path": str(db_path), "chroma_dir": str(ctx_index / "chroma")},
            "weights": {"repo": 1.0, "chat": 0.8},
            "ollama": {"base_url": "http://localhost:11434", "embed_model": "mxbai-embed-large"},
            "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        (ctx_dir / "context.json").write_text(
            __import__("json").dumps(ctx_data, indent=2)
        )

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Should not raise error (will fail on actual embedding, but that's expected)
    try:
        search_multi_context(
            contexts=["ContextA", "ContextB"],
            query="test query",
            k=5
        )
    except ValueError as e:
        # Should not be mixed embedding error
        assert "mixed embedding" not in str(e).lower()

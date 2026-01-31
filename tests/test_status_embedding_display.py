# tests/test_status_embedding_display.py
import pytest
import json
from pathlib import Path
from datetime import datetime, timezone
from chinvex.cli_status import generate_status_from_contexts, ContextStatus


def test_status_shows_embedding_provider(tmp_path):
    """Status output should include embedding provider per context."""
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    # Create contexts with different embedding providers
    contexts_data = [
        ("Chinvex", "openai", "text-embedding-3-small", 1200),
        ("OldContext", "ollama", "mxbai-embed-large", 500),
        ("AnotherContext", "openai", "text-embedding-3-large", 800),
    ]

    for ctx_name, provider, model, chunks in contexts_data:
        ctx_dir = contexts_root / ctx_name
        ctx_dir.mkdir()

        # Create context.json with embedding config
        ctx_config = {
            "schema_version": 2,
            "name": ctx_name,
            "aliases": [],
            "includes": {"repos": [], "chat_roots": [], "codex_session_roots": [], "note_roots": []},
            "index": {"sqlite_path": f"path/to/{ctx_name}.db", "chroma_dir": f"path/to/{ctx_name}/chroma"},
            "weights": {"repo": 1.0, "chat": 0.8},
            "ollama": {"base_url": "http://localhost:11434", "embed_model": "mxbai-embed-large"},
            "embedding": {"provider": provider, "model": model},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        (ctx_dir / "context.json").write_text(json.dumps(ctx_config, indent=2))

        # Create STATUS.json
        status_data = {
            "context": ctx_name,
            "chunks": chunks,
            "last_sync": datetime.now(timezone.utc).isoformat(),
            "freshness": {
                "is_stale": False,
                "hours_since_sync": 2.5
            }
        }
        (ctx_dir / "STATUS.json").write_text(json.dumps(status_data, indent=2))

    # Generate status output
    output = generate_status_from_contexts(contexts_root)

    # Verify embedding provider is shown
    assert "openai" in output.lower()
    assert "ollama" in output.lower()
    assert "Chinvex" in output
    assert "OldContext" in output
    assert "AnotherContext" in output

    # Verify it's in a table format with headers
    assert "Embedding" in output or "Provider" in output


def test_status_legacy_context_shows_ollama_default(tmp_path):
    """Legacy contexts without embedding config should show 'ollama (default)'."""
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    ctx_dir = contexts_root / "LegacyContext"
    ctx_dir.mkdir()

    # Context without embedding field (schema v1 or early v2)
    ctx_config = {
        "schema_version": 1,
        "name": "LegacyContext",
        "aliases": [],
        "includes": {"repos": [], "chat_roots": [], "codex_session_roots": [], "note_roots": []},
        "index": {"sqlite_path": "path/to/db", "chroma_dir": "path/to/chroma"},
        "weights": {"repo": 1.0, "chat": 0.8},
        "ollama": {"base_url": "http://localhost:11434", "embed_model": "mxbai-embed-large"},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    (ctx_dir / "context.json").write_text(json.dumps(ctx_config, indent=2))

    status_data = {
        "context": "LegacyContext",
        "chunks": 300,
        "last_sync": datetime.now(timezone.utc).isoformat(),
        "freshness": {"is_stale": False, "hours_since_sync": 1.0}
    }
    (ctx_dir / "STATUS.json").write_text(json.dumps(status_data, indent=2))

    output = generate_status_from_contexts(contexts_root)

    # Should show ollama as default for legacy contexts
    assert "ollama" in output.lower()
    assert "LegacyContext" in output

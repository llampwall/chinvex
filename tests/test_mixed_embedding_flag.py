# tests/test_mixed_embedding_flag.py
import pytest
from typer.testing import CliRunner
from chinvex.cli import app
from pathlib import Path
import json
from datetime import datetime, timezone


runner = CliRunner()


def test_allow_mixed_embeddings_flag_not_yet_supported(tmp_path, monkeypatch):
    """Flag --allow-mixed-embeddings should exist but return 'not yet supported' error."""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"
    contexts_root.mkdir()
    indexes_root.mkdir()

    # Create two contexts with different providers
    for ctx_name, provider in [("Ctx1", "openai"), ("Ctx2", "ollama")]:
        ctx_dir = contexts_root / ctx_name
        ctx_dir.mkdir()
        ctx_index = indexes_root / ctx_name
        ctx_index.mkdir()

        ctx_data = {
            "schema_version": 2,
            "name": ctx_name,
            "aliases": [],
            "includes": {"repos": [], "chat_roots": [], "codex_session_roots": [], "note_roots": []},
            "index": {"sqlite_path": str(ctx_index / "hybrid.db"), "chroma_dir": str(ctx_index / "chroma")},
            "weights": {"repo": 1.0, "chat": 0.8},
            "ollama": {"base_url": "http://localhost:11434", "embed_model": "mxbai-embed-large"},
            "embedding": {"provider": provider},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        (ctx_dir / "context.json").write_text(json.dumps(ctx_data, indent=2))

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))

    # Test CLI with flag
    result = runner.invoke(app, [
        "search",
        "test query",
        "--contexts", "Ctx1,Ctx2",
        "--allow-mixed-embeddings"
    ])

    # Should exit with error
    assert result.exit_code != 0
    assert "not yet supported" in result.stdout.lower() or "not yet supported" in str(result.exception).lower()


def test_allow_mixed_embeddings_flag_exists():
    """Verify --allow-mixed-embeddings flag is recognized by CLI."""
    result = runner.invoke(app, ["search", "--help"])

    # Help should mention the flag
    assert "--allow-mixed-embeddings" in result.stdout

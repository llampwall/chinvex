"""Test watch history CLI commands."""
import pytest
import json
import os
from pathlib import Path
from typer.testing import CliRunner
from chinvex.cli import app

runner = CliRunner()


@pytest.fixture
def test_context(tmp_path, monkeypatch):
    """Create a test context with mock base directory."""
    # Set up contexts root
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    indexes_root = tmp_path / "indexes"
    indexes_root.mkdir()

    context_dir = contexts_root / "TestContext"
    context_dir.mkdir()

    index_dir = indexes_root / "TestContext"
    index_dir.mkdir()

    # Create context.json with all required fields
    config = {
        "name": "TestContext",
        "schema_version": 1,
        "includes": {
            "repos": [],
            "chat_roots": [],
            "codex_session_roots": [],
            "note_roots": []
        },
        "index": {
            "sqlite_path": str(index_dir / "hybrid.db"),
            "chroma_dir": str(index_dir / "chroma")
        }
    }
    (context_dir / "context.json").write_text(json.dumps(config))

    # Set environment variables
    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(indexes_root))

    return {"root": contexts_root, "dir": context_dir}


def test_watch_history_command(test_context):
    """Test chinvex watch history command."""
    result = runner.invoke(app, ["watch", "history", "--context", "TestContext"])
    assert result.exit_code == 0
    assert "No watch history found" in result.stdout


def test_watch_history_with_filters(test_context):
    """Test watch history with filters."""
    result = runner.invoke(app, [
        "watch", "history",
        "--context", "TestContext",
        "--since", "7d",
        "--id", "test_watch",
        "--limit", "10"
    ])
    assert result.exit_code == 0


def test_watch_history_formats(test_context):
    """Test watch history output formats."""
    # JSON format
    result = runner.invoke(app, [
        "watch", "history",
        "--context", "TestContext",
        "--format", "json"
    ])
    assert result.exit_code == 0
    assert "[]" in result.stdout or "No watch history" in result.stdout

    # Table format (default)
    result = runner.invoke(app, [
        "watch", "history",
        "--context", "TestContext",
        "--format", "table"
    ])
    assert result.exit_code == 0

"""Tests for CLI commands required by strap integration.

These commands were added to support the strap-chinvex integration:
- context create --idempotent
- context exists
- context list --json
- context rename --to
- context remove-repo --repo [--prune]
- ingest --register-only
"""

from pathlib import Path
import json
from typer.testing import CliRunner
from chinvex.cli import app

runner = CliRunner()


# === context create --idempotent ===

def test_context_create_idempotent_creates_new(tmp_path: Path, monkeypatch) -> None:
    """--idempotent creates context if it doesn't exist."""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(indexes_root))

    result = runner.invoke(app, ["context", "create", "NewContext", "--idempotent"])
    assert result.exit_code == 0
    assert "Created context: NewContext" in result.stdout

    # Verify created
    ctx_file = contexts_root / "NewContext" / "context.json"
    assert ctx_file.exists()


def test_context_create_idempotent_noop_if_exists(tmp_path: Path, monkeypatch) -> None:
    """--idempotent is a no-op if context already exists."""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(indexes_root))

    # Create first
    runner.invoke(app, ["context", "create", "ExistingContext"])

    # Idempotent create should succeed
    result = runner.invoke(app, ["context", "create", "ExistingContext", "--idempotent"])
    assert result.exit_code == 0
    assert "already exists" in result.stdout


# === context exists ===

def test_context_exists_returns_0_when_exists(tmp_path: Path, monkeypatch) -> None:
    """context exists returns exit code 0 when context exists."""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(indexes_root))

    runner.invoke(app, ["context", "create", "TestContext"])

    result = runner.invoke(app, ["context", "exists", "TestContext"])
    assert result.exit_code == 0
    assert "exists" in result.stdout


def test_context_exists_returns_1_when_missing(tmp_path: Path, monkeypatch) -> None:
    """context exists returns exit code 1 when context doesn't exist."""
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))

    result = runner.invoke(app, ["context", "exists", "NonExistent"])
    assert result.exit_code == 1
    assert "does not exist" in result.stdout


# === context list --json ===

def test_context_list_json_empty(tmp_path: Path, monkeypatch) -> None:
    """context list --json returns empty array when no contexts."""
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))

    result = runner.invoke(app, ["context", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data == []


def test_context_list_json_returns_contexts(tmp_path: Path, monkeypatch) -> None:
    """context list --json returns array of context objects."""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(indexes_root))

    runner.invoke(app, ["context", "create", "Alpha"])
    runner.invoke(app, ["context", "create", "Beta"])

    result = runner.invoke(app, ["context", "list", "--json"])
    assert result.exit_code == 0

    data = json.loads(result.stdout)
    assert len(data) == 2

    names = [c["name"] for c in data]
    assert "Alpha" in names
    assert "Beta" in names

    # Verify structure
    for ctx in data:
        assert "name" in ctx
        assert "aliases" in ctx
        assert "updated_at" in ctx


# === context rename --to ===

def test_context_rename_success(tmp_path: Path, monkeypatch) -> None:
    """context rename moves directories and updates config."""
    import sys
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(indexes_root))

    runner.invoke(app, ["context", "create", "OldName"])

    result = runner.invoke(app, ["context", "rename", "OldName", "--to", "NewName"])

    # On Windows, ChromaDB may hold file locks - accept either success or lock error
    if result.exit_code == 1 and "locked" in result.stdout and sys.platform == "win32":
        import pytest
        pytest.skip("Windows file locking prevents rename in test environment")

    assert result.exit_code == 0
    assert "Renamed" in result.stdout

    # Old should not exist
    assert not (contexts_root / "OldName").exists()
    assert not (indexes_root / "OldName").exists()

    # New should exist
    assert (contexts_root / "NewName").exists()
    assert (indexes_root / "NewName").exists()

    # Config should be updated
    ctx_file = contexts_root / "NewName" / "context.json"
    data = json.loads(ctx_file.read_text())
    assert data["name"] == "NewName"
    assert "NewName" in data["index"]["sqlite_path"]
    assert "NewName" in data["index"]["chroma_dir"]


def test_context_rename_source_not_found(tmp_path: Path, monkeypatch) -> None:
    """context rename fails if source doesn't exist."""
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))

    result = runner.invoke(app, ["context", "rename", "NonExistent", "--to", "NewName"])
    assert result.exit_code == 1
    assert "does not exist" in result.stdout


def test_context_rename_target_exists(tmp_path: Path, monkeypatch) -> None:
    """context rename fails if target already exists."""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(indexes_root))

    runner.invoke(app, ["context", "create", "Source"])
    runner.invoke(app, ["context", "create", "Target"])

    result = runner.invoke(app, ["context", "rename", "Source", "--to", "Target"])
    assert result.exit_code == 1
    assert "already exists" in result.stdout


# === context remove-repo ===

def test_context_remove_repo_success(tmp_path: Path, monkeypatch) -> None:
    """context remove-repo removes path from config."""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"
    repo_path = tmp_path / "myrepo"
    repo_path.mkdir()

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(indexes_root))

    # Create context and register repo
    runner.invoke(app, ["context", "create", "TestContext"])
    runner.invoke(app, [
        "ingest", "--context", "TestContext",
        "--repo", str(repo_path),
        "--register-only"
    ])

    # Verify repo is registered
    ctx_file = contexts_root / "TestContext" / "context.json"
    data = json.loads(ctx_file.read_text())
    assert len(data["includes"]["repos"]) == 1

    # Remove repo
    result = runner.invoke(app, [
        "context", "remove-repo", "TestContext",
        "--repo", str(repo_path)
    ])
    assert result.exit_code == 0
    assert "Removed" in result.stdout

    # Verify removed
    data = json.loads(ctx_file.read_text())
    assert len(data["includes"]["repos"]) == 0


def test_context_remove_repo_not_found(tmp_path: Path, monkeypatch) -> None:
    """context remove-repo fails if path not in config."""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(indexes_root))

    runner.invoke(app, ["context", "create", "TestContext"])

    result = runner.invoke(app, [
        "context", "remove-repo", "TestContext",
        "--repo", "C:\\nonexistent\\path"
    ])
    assert result.exit_code == 1
    assert "not found" in result.stdout


def test_context_remove_repo_prune_shows_warning(tmp_path: Path, monkeypatch) -> None:
    """context remove-repo --prune shows not implemented warning."""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"
    repo_path = tmp_path / "myrepo"
    repo_path.mkdir()

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(indexes_root))

    runner.invoke(app, ["context", "create", "TestContext"])
    runner.invoke(app, [
        "ingest", "--context", "TestContext",
        "--repo", str(repo_path),
        "--register-only"
    ])

    result = runner.invoke(app, [
        "context", "remove-repo", "TestContext",
        "--repo", str(repo_path),
        "--prune"
    ])
    assert result.exit_code == 0
    assert "not yet implemented" in result.stdout


# === ingest --register-only ===

def test_ingest_register_only_creates_context(tmp_path: Path, monkeypatch) -> None:
    """ingest --register-only creates context if missing."""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"
    repo_path = tmp_path / "myrepo"
    repo_path.mkdir()

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(indexes_root))

    result = runner.invoke(app, [
        "ingest", "--context", "NewContext",
        "--repo", str(repo_path),
        "--register-only"
    ])
    assert result.exit_code == 0
    assert "Registered" in result.stdout

    # Context should exist
    ctx_file = contexts_root / "NewContext" / "context.json"
    assert ctx_file.exists()

    # Repo should be in config
    data = json.loads(ctx_file.read_text())
    assert len(data["includes"]["repos"]) == 1


def test_ingest_register_only_adds_to_existing(tmp_path: Path, monkeypatch) -> None:
    """ingest --register-only adds to existing context."""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"
    repo1 = tmp_path / "repo1"
    repo2 = tmp_path / "repo2"
    repo1.mkdir()
    repo2.mkdir()

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(indexes_root))

    # Create context with first repo
    runner.invoke(app, [
        "ingest", "--context", "TestContext",
        "--repo", str(repo1),
        "--register-only"
    ])

    # Add second repo
    result = runner.invoke(app, [
        "ingest", "--context", "TestContext",
        "--repo", str(repo2),
        "--register-only"
    ])
    assert result.exit_code == 0

    # Both repos should be in config
    ctx_file = contexts_root / "TestContext" / "context.json"
    data = json.loads(ctx_file.read_text())
    assert len(data["includes"]["repos"]) == 2


def test_ingest_register_only_requires_context(tmp_path: Path, monkeypatch) -> None:
    """ingest --register-only requires --context flag."""
    repo_path = tmp_path / "myrepo"
    repo_path.mkdir()

    result = runner.invoke(app, [
        "ingest",
        "--repo", str(repo_path),
        "--register-only"
    ])
    assert result.exit_code == 2
    # Error can be either the generic "must provide --context" or the register-only specific one
    assert "--context" in result.stdout


def test_ingest_register_only_requires_path(tmp_path: Path, monkeypatch) -> None:
    """ingest --register-only requires --repo or --chat-root."""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(indexes_root))

    result = runner.invoke(app, [
        "ingest", "--context", "TestContext",
        "--register-only"
    ])
    assert result.exit_code == 2
    assert "requires --repo or --chat-root" in result.stdout


def test_ingest_register_only_validates_path_exists(tmp_path: Path, monkeypatch) -> None:
    """ingest --register-only validates that path exists."""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(indexes_root))

    result = runner.invoke(app, [
        "ingest", "--context", "TestContext",
        "--repo", "C:\\nonexistent\\path",
        "--register-only"
    ])
    assert result.exit_code == 2
    assert "does not exist" in result.stdout


def test_ingest_register_only_deduplicates(tmp_path: Path, monkeypatch) -> None:
    """ingest --register-only doesn't add duplicate paths."""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"
    repo_path = tmp_path / "myrepo"
    repo_path.mkdir()

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(indexes_root))

    # Register same path twice
    runner.invoke(app, [
        "ingest", "--context", "TestContext",
        "--repo", str(repo_path),
        "--register-only"
    ])
    runner.invoke(app, [
        "ingest", "--context", "TestContext",
        "--repo", str(repo_path),
        "--register-only"
    ])

    # Should only have one entry
    ctx_file = contexts_root / "TestContext" / "context.json"
    data = json.loads(ctx_file.read_text())
    assert len(data["includes"]["repos"]) == 1

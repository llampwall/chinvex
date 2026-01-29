import pytest
from pathlib import Path
from chinvex.context import load_context
from chinvex.context_cli import get_contexts_root

def test_ingest_creates_context_if_missing(tmp_path, monkeypatch):
    """Test that ingest auto-creates context with --repo flag."""
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(tmp_path / "indexes"))

    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    (repo_path / "test.txt").write_text("hello")

    # Context doesn't exist yet
    assert not (contexts_root / "NewContext" / "context.json").exists()

    # Run ingest with --repo
    from chinvex.cli import app
    from typer.testing import CliRunner
    runner = CliRunner()

    result = runner.invoke(app, [
        "ingest",
        "--context", "NewContext",
        "--repo", str(repo_path)
    ])

    assert result.exit_code == 0
    assert (contexts_root / "NewContext" / "context.json").exists()

    # Verify context.json has repo in includes
    ctx = load_context("NewContext", contexts_root)
    assert len(ctx.includes.repos) == 1

def test_ingest_deduplicates_repo_paths(tmp_path, monkeypatch):
    """Test that duplicate --repo paths are ignored."""
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()
    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(tmp_path / "indexes"))

    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    (repo_path / "test.txt").write_text("hello")

    from chinvex.cli import app
    from typer.testing import CliRunner
    runner = CliRunner()

    # Run ingest with duplicate --repo (different forms)
    result = runner.invoke(app, [
        "ingest",
        "--context", "NewContext",
        "--repo", str(repo_path),
        "--repo", str(repo_path.resolve())  # Same path, different form
    ])

    assert result.exit_code == 0
    ctx = load_context("NewContext", contexts_root)
    assert len(ctx.includes.repos) == 1  # Deduplicated

def test_ingest_fails_if_repo_doesnt_exist(tmp_path, monkeypatch):
    """Test that --repo path that doesn't exist fails immediately."""
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()
    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(tmp_path / "indexes"))

    from chinvex.cli import app
    from typer.testing import CliRunner
    runner = CliRunner()

    result = runner.invoke(app, [
        "ingest",
        "--context", "NewContext",
        "--repo", "/nonexistent/path"
    ])

    assert result.exit_code != 0
    assert "does not exist" in result.stdout.lower()

def test_ingest_no_write_context_flag(tmp_path, monkeypatch):
    """Test --no-write-context ingests without persisting context.json."""
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()
    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("CHINVEX_INDEXES_ROOT", str(tmp_path / "indexes"))

    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    (repo_path / "test.txt").write_text("hello")

    from chinvex.cli import app
    from typer.testing import CliRunner
    runner = CliRunner()

    # Run with --no-write-context
    result = runner.invoke(app, [
        "ingest",
        "--context", "TempContext",
        "--repo", str(repo_path),
        "--no-write-context"
    ])

    # Should succeed but not create context.json
    assert result.exit_code == 0
    assert not (contexts_root / "TempContext" / "context.json").exists()

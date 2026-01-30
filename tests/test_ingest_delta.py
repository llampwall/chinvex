# tests/test_ingest_delta.py
import pytest
from pathlib import Path
from typer.testing import CliRunner
from chinvex.cli import app

runner = CliRunner()


def test_ingest_with_paths_flag(tmp_path: Path, monkeypatch):
    """--paths should trigger delta ingest for specific files"""
    # Setup minimal context
    contexts_root = tmp_path / "contexts"
    ctx_dir = contexts_root / "TestCtx"
    ctx_dir.mkdir(parents=True)

    # Create repo source
    repo = tmp_path / "repo"
    repo.mkdir()
    file1 = repo / "file1.txt"
    file1.write_text("content1")
    file2 = repo / "file2.txt"
    file2.write_text("content2")

    # Create context.json
    import json
    ctx_config = {
        "schema_version": 2,
        "name": "TestCtx",
        "includes": {"repos": [str(repo)]},
        "index": {
            "sqlite_path": str(ctx_dir / "hybrid.db"),
            "chroma_dir": str(ctx_dir / "chroma")
        }
    }
    (ctx_dir / "context.json").write_text(json.dumps(ctx_config))

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))

    # Run delta ingest for only file1
    result = runner.invoke(app, [
        "ingest",
        "--context", "TestCtx",
        "--paths", str(file1)
    ])

    assert result.exit_code == 0
    # Should have processed file1 only
    # (verification via stats will be added in implementation)


def test_ingest_with_multiple_paths(tmp_path: Path, monkeypatch):
    """--paths with multiple files should process all"""
    contexts_root = tmp_path / "contexts"
    ctx_dir = contexts_root / "TestCtx"
    ctx_dir.mkdir(parents=True)

    repo = tmp_path / "repo"
    repo.mkdir()
    files = [repo / f"file{i}.txt" for i in range(3)]
    for f in files:
        f.write_text(f"content_{f.name}")

    import json
    ctx_config = {
        "schema_version": 2,
        "name": "TestCtx",
        "includes": {"repos": [str(repo)]},
        "index": {
            "sqlite_path": str(ctx_dir / "hybrid.db"),
            "chroma_dir": str(ctx_dir / "chroma")
        }
    }
    (ctx_dir / "context.json").write_text(json.dumps(ctx_config))

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))

    # Pass multiple paths
    paths_arg = ",".join(str(f) for f in files)
    result = runner.invoke(app, [
        "ingest",
        "--context", "TestCtx",
        "--paths", paths_arg
    ])

    assert result.exit_code == 0


def test_ingest_without_paths_does_full(tmp_path: Path, monkeypatch):
    """Without --paths, should do full ingest"""
    contexts_root = tmp_path / "contexts"
    ctx_dir = contexts_root / "TestCtx"
    ctx_dir.mkdir(parents=True)

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "file1.txt").write_text("content1")
    (repo / "file2.txt").write_text("content2")

    import json
    ctx_config = {
        "schema_version": 2,
        "name": "TestCtx",
        "includes": {"repos": [str(repo)]},
        "index": {
            "sqlite_path": str(ctx_dir / "hybrid.db"),
            "chroma_dir": str(ctx_dir / "chroma")
        }
    }
    (ctx_dir / "context.json").write_text(json.dumps(ctx_config))

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))

    # No --paths = full ingest
    result = runner.invoke(app, [
        "ingest",
        "--context", "TestCtx"
    ])

    assert result.exit_code == 0

# tests/test_status_json.py
import pytest
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from chinvex.status import write_status_json, compute_freshness


def test_write_status_json_basic(tmp_path: Path):
    """Should write STATUS.json with required fields"""
    ctx_dir = tmp_path / "contexts" / "TestCtx"
    ctx_dir.mkdir(parents=True)

    stats = {
        "documents": 10,
        "chunks": 42,
        "last_sync": datetime.now(timezone.utc).isoformat()
    }

    sources = [
        {"type": "repo", "path": "/code/repo1", "watching": True}
    ]

    embedding_info = {
        "provider": "ollama",
        "model": "mxbai-embed-large",
        "dimensions": 1024
    }

    write_status_json(
        context_dir=ctx_dir,
        stats=stats,
        sources=sources,
        embedding=embedding_info
    )

    status_file = ctx_dir / "STATUS.json"
    assert status_file.exists()

    data = json.loads(status_file.read_text())
    assert data["context"] == "TestCtx"
    assert data["chunks"] == 42
    assert data["embedding"]["provider"] == "ollama"


def test_status_includes_freshness(tmp_path: Path):
    """STATUS.json should include freshness calculation"""
    ctx_dir = tmp_path / "contexts" / "TestCtx"
    ctx_dir.mkdir(parents=True)

    now = datetime.now(timezone.utc)
    stats = {
        "chunks": 100,
        "last_sync": now.isoformat()
    }

    write_status_json(
        context_dir=ctx_dir,
        stats=stats,
        sources=[],
        embedding={"provider": "ollama", "model": "test"}
    )

    data = json.loads((ctx_dir / "STATUS.json").read_text())

    assert "freshness" in data
    assert "stale_after_hours" in data["freshness"]
    assert "is_stale" in data["freshness"]
    assert "hours_since_sync" in data["freshness"]


def test_compute_freshness_not_stale(tmp_path: Path):
    """Recent sync should not be stale"""
    now = datetime.now(timezone.utc)

    freshness = compute_freshness(
        last_sync=now.isoformat(),
        stale_after_hours=6
    )

    assert freshness["is_stale"] is False
    assert freshness["hours_since_sync"] < 1


def test_compute_freshness_stale(tmp_path: Path):
    """Old sync should be stale"""
    old_time = datetime.now(timezone.utc) - timedelta(hours=7)

    freshness = compute_freshness(
        last_sync=old_time.isoformat(),
        stale_after_hours=6
    )

    assert freshness["is_stale"] is True
    assert freshness["hours_since_sync"] > 6

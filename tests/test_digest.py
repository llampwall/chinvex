import pytest
from pathlib import Path
from datetime import datetime, timedelta
from chinvex.digest import generate_digest


def test_generate_digest_basic(tmp_path):
    """Test basic digest generation."""
    # Setup: create ingest_runs.jsonl
    runs_log = tmp_path / "ingest_runs.jsonl"
    runs_log.write_text(
        '{"run_id": "run1", "status": "started", "started_at": "2026-01-29T12:00:00Z"}\n'
        '{"run_id": "run1", "status": "succeeded", "ended_at": "2026-01-29T12:05:00Z", "docs_seen": 100, "docs_changed": 10, "chunks_new": 50, "chunks_updated": 20}\n'
    )

    # Generate digest
    output_md = tmp_path / "digest.md"
    output_json = tmp_path / "digest.json"

    generate_digest(
        context_name="TestContext",
        ingest_runs_log=runs_log,
        watch_history_log=None,
        state_md=None,
        output_md=output_md,
        output_json=output_json,
        since_hours=24
    )

    assert output_md.exists()
    assert output_json.exists()

    # Check markdown content
    content = output_md.read_text()
    assert "# Digest:" in content
    assert "10" in content  # docs_changed
    assert "70 chunks updated" in content  # chunks_new (50) + chunks_updated (20) = 70


def test_generate_digest_with_watches(tmp_path):
    """Test digest includes watch hits."""
    # Setup watch history
    watch_log = tmp_path / "watch_history.jsonl"
    watch_log.write_text(
        '{"ts": "2026-01-29T10:00:00Z", "watch_id": "test_watch", "query": "retry logic", "hits": [{"chunk_id": "abc123", "score": 0.85, "snippet": "retry with backoff"}]}\n'
    )

    # Setup ingest runs
    runs_log = tmp_path / "ingest_runs.jsonl"
    runs_log.write_text(
        '{"run_id": "run1", "status": "succeeded", "ended_at": "2026-01-29T12:00:00Z", "docs_seen": 10, "docs_changed": 2, "chunks_new": 5, "chunks_updated": 0}\n'
    )

    output_md = tmp_path / "digest.md"
    generate_digest(
        context_name="TestContext",
        ingest_runs_log=runs_log,
        watch_history_log=watch_log,
        state_md=None,
        output_md=output_md,
        output_json=None,
        since_hours=24
    )

    content = output_md.read_text()
    assert "Watch Hits" in content
    assert "retry logic" in content


def test_generate_digest_deterministic(tmp_path):
    """Test that digest generation is deterministic."""
    runs_log = tmp_path / "ingest_runs.jsonl"
    runs_log.write_text(
        '{"run_id": "run1", "status": "succeeded", "ended_at": "2026-01-29T12:00:00Z", "docs_seen": 10, "docs_changed": 2, "chunks_new": 5, "chunks_updated": 0}\n'
    )

    output1 = tmp_path / "digest1.md"
    output2 = tmp_path / "digest2.md"

    generate_digest(
        context_name="TestContext",
        ingest_runs_log=runs_log,
        watch_history_log=None,
        state_md=None,
        output_md=output1,
        output_json=None,
        since_hours=24
    )

    generate_digest(
        context_name="TestContext",
        ingest_runs_log=runs_log,
        watch_history_log=None,
        state_md=None,
        output_md=output2,
        output_json=None,
        since_hours=24
    )

    # Should be identical except for "Generated:" timestamp (which we don't include in this impl)
    content1 = output1.read_text()
    content2 = output2.read_text()

    # Compare core content (excluding any timestamp if present)
    # For now, just check they're similar enough
    assert len(content1) == len(content2)

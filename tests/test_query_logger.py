import json
import time
from datetime import datetime, timedelta
from pathlib import Path

from chinvex.query_logger import (
    QueryLogger,
    QueryLogEntry,
    log_search_query,
    rotate_old_logs,
)


def test_query_logger_creates_log_directory(tmp_path):
    """QueryLogger should create .chinvex/logs/ directory if missing."""
    chinvex_dir = tmp_path / ".chinvex"
    logger = QueryLogger(chinvex_dir)

    log_dir = chinvex_dir / "logs"
    assert log_dir.exists()
    assert log_dir.is_dir()


def test_query_logger_appends_to_jsonl(tmp_path):
    """QueryLogger should append search queries to queries.jsonl."""
    chinvex_dir = tmp_path / ".chinvex"
    logger = QueryLogger(chinvex_dir)

    entry = QueryLogEntry(
        timestamp=datetime.now().isoformat(),
        context="Chinvex",
        query="test query",
        k=5,
        num_results=3,
        top_chunk_ids=["chunk1", "chunk2", "chunk3"],
        top_scores=[0.95, 0.87, 0.72],
        latency_ms=125.5,
    )

    logger.log(entry)

    log_file = chinvex_dir / "logs" / "queries.jsonl"
    assert log_file.exists()

    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 1

    logged = json.loads(lines[0])
    assert logged["context"] == "Chinvex"
    assert logged["query"] == "test query"
    assert logged["k"] == 5
    assert logged["num_results"] == 3
    assert logged["top_chunk_ids"] == ["chunk1", "chunk2", "chunk3"]
    assert logged["top_scores"] == [0.95, 0.87, 0.72]
    assert logged["latency_ms"] == 125.5


def test_query_logger_multiple_entries(tmp_path):
    """QueryLogger should append multiple entries correctly."""
    chinvex_dir = tmp_path / ".chinvex"
    logger = QueryLogger(chinvex_dir)

    for i in range(3):
        entry = QueryLogEntry(
            timestamp=datetime.now().isoformat(),
            context="Chinvex",
            query=f"query {i}",
            k=5,
            num_results=2,
            top_chunk_ids=[f"chunk{i}a", f"chunk{i}b"],
            top_scores=[0.9, 0.8],
            latency_ms=100.0 + i,
        )
        logger.log(entry)

    log_file = chinvex_dir / "logs" / "queries.jsonl"
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 3

    # Verify each entry
    for i, line in enumerate(lines):
        logged = json.loads(line)
        assert logged["query"] == f"query {i}"
        assert logged["latency_ms"] == 100.0 + i


def test_rotate_old_logs_removes_30_day_old_entries(tmp_path):
    """rotate_old_logs should remove entries older than 30 days."""
    chinvex_dir = tmp_path / ".chinvex"
    log_dir = chinvex_dir / "logs"
    log_dir.mkdir(parents=True)
    log_file = log_dir / "queries.jsonl"

    now = datetime.now()
    old_timestamp = (now - timedelta(days=31)).isoformat()
    recent_timestamp = (now - timedelta(days=10)).isoformat()

    # Write entries with different timestamps
    with open(log_file, "w") as f:
        f.write(json.dumps({"timestamp": old_timestamp, "query": "old"}) + "\n")
        f.write(json.dumps({"timestamp": recent_timestamp, "query": "recent"}) + "\n")
        f.write(json.dumps({"timestamp": now.isoformat(), "query": "current"}) + "\n")

    rotate_old_logs(chinvex_dir, retention_days=30)

    # Only recent entries should remain
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 2

    queries = [json.loads(line)["query"] for line in lines]
    assert "old" not in queries
    assert "recent" in queries
    assert "current" in queries


def test_rotate_old_logs_handles_empty_file(tmp_path):
    """rotate_old_logs should handle empty log file gracefully."""
    chinvex_dir = tmp_path / ".chinvex"
    log_dir = chinvex_dir / "logs"
    log_dir.mkdir(parents=True)
    log_file = log_dir / "queries.jsonl"
    log_file.touch()

    # Should not raise
    rotate_old_logs(chinvex_dir, retention_days=30)
    assert log_file.exists()


def test_rotate_old_logs_handles_missing_file(tmp_path):
    """rotate_old_logs should handle missing log file gracefully."""
    chinvex_dir = tmp_path / ".chinvex"

    # Should not raise
    rotate_old_logs(chinvex_dir, retention_days=30)


def test_log_search_query_integration(tmp_path):
    """log_search_query should record search with all metadata."""
    chinvex_dir = tmp_path / ".chinvex"

    start = time.time()
    log_search_query(
        chinvex_dir=chinvex_dir,
        context="Chinvex",
        query="integration test",
        results=[
            {"chunk_id": "c1", "score": 0.95},
            {"chunk_id": "c2", "score": 0.82},
        ],
        k=5,
    )
    elapsed_ms = (time.time() - start) * 1000

    log_file = chinvex_dir / "logs" / "queries.jsonl"
    assert log_file.exists()

    logged = json.loads(log_file.read_text())
    assert logged["context"] == "Chinvex"
    assert logged["query"] == "integration test"
    assert logged["k"] == 5
    assert logged["num_results"] == 2
    assert logged["top_chunk_ids"] == ["c1", "c2"]
    assert logged["top_scores"] == [0.95, 0.82]
    assert logged["latency_ms"] > 0
    assert logged["latency_ms"] < elapsed_ms + 100  # reasonable upper bound

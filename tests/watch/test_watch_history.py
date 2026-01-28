"""Test watch history logging."""
import pytest
import json
from pathlib import Path


def test_append_watch_history(tmp_path):
    """Test that watch hits are appended to history log."""
    from chinvex.watch.runner import append_watch_history

    history_file = tmp_path / "watch_history.jsonl"

    entry = {
        "ts": "2026-01-28T12:00:00Z",
        "run_id": "run_123",
        "watch_id": "test_watch",
        "query": "test query",
        "hits": [
            {"chunk_id": "abc", "score": 0.85, "snippet": "test snippet"}
        ]
    }

    append_watch_history(str(history_file), entry)

    # Verify file exists and contains entry
    assert history_file.exists()
    with open(history_file) as f:
        lines = f.readlines()
    assert len(lines) == 1
    logged = json.loads(lines[0])
    assert logged["watch_id"] == "test_watch"


def test_append_watch_history_multiple_entries(tmp_path):
    """Test that multiple watch hits are appended."""
    from chinvex.watch.runner import append_watch_history

    history_file = tmp_path / "watch_history.jsonl"

    for i in range(3):
        entry = {
            "ts": f"2026-01-28T12:0{i}:00Z",
            "run_id": f"run_{i}",
            "watch_id": "test_watch",
            "query": "test",
            "hits": []
        }
        append_watch_history(str(history_file), entry)

    with open(history_file) as f:
        lines = f.readlines()
    assert len(lines) == 3


def test_watch_history_caps_hits_at_10(tmp_path):
    """Test that watch history caps hits at 10 per entry."""
    from chinvex.watch.runner import create_watch_history_entry

    # Create entry with 20 hits
    hits = [{"chunk_id": f"chunk_{i}", "score": 0.9 - i*0.01, "snippet": f"text {i}"}
            for i in range(20)]

    entry = create_watch_history_entry(
        watch_id="test",
        query="test",
        hits=hits,
        run_id="run_123"
    )

    # Should cap at 10 and mark truncated
    assert len(entry["hits"]) == 10
    assert entry["truncated"] is True

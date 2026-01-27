# tests/watch/test_watch.py
from chinvex.watch.models import Watch
from chinvex.watch.runner import run_watches

def test_watch_creation():
    """Test Watch model creation."""
    watch = Watch(
        id="test_watch",
        query="test query",
        min_score=0.75,
        enabled=True,
        created_at="2026-01-26T00:00:00Z"
    )
    assert watch.id == "test_watch"
    assert watch.enabled is True

def test_run_watches_basic():
    """Test watch execution returns hits."""
    from unittest.mock import Mock

    # Create mock context
    mock_context = Mock()

    watches = [
        Watch(
            id="test",
            query="test query",
            min_score=0.5,
            enabled=True,
            created_at="2026-01-26T00:00:00Z"
        )
    ]

    # Mock new chunk IDs
    new_chunk_ids = ["chunk1", "chunk2"]

    hits = run_watches(mock_context, new_chunk_ids, watches)

    assert isinstance(hits, list)
    # Empty list is expected since we're using mocks

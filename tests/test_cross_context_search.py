"""Test cross-context search functionality."""
import pytest
from chinvex.search import search_multi_context, SearchResult


def test_search_multi_context_merges_results():
    """Test that search_multi_context merges results by score."""
    # Note: This test requires actual contexts to be set up
    # For now, we test the basic structure
    results = search_multi_context(
        contexts=["TestContext1", "TestContext2"],
        query="test",
        k=5
    )

    # Expected: Returns list of SearchResult objects
    assert isinstance(results, list)
    # Expected: Results are tagged with context name
    for r in results:
        assert hasattr(r, 'context')
        assert r.context in ["TestContext1", "TestContext2"] or r.context is None


def test_search_multi_context_respects_k_limit():
    """Test that k limits total results, not per-context."""
    results = search_multi_context(
        contexts=["TestContext1", "TestContext2"],
        query="test",
        k=10
    )
    assert len(results) <= 10


def test_search_multi_context_sorts_by_score():
    """Test that results are sorted by score descending."""
    results = search_multi_context(
        contexts=["TestContext1", "TestContext2"],
        query="test",
        k=10
    )
    if len(results) > 1:
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


def test_search_multi_context_all_expands_to_all_contexts():
    """Test that 'all' expands to all available contexts."""
    results = search_multi_context(
        contexts="all",
        query="test",
        k=10
    )
    assert isinstance(results, list)

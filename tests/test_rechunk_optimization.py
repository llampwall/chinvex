"""Test rechunk optimization with embedding reuse."""
import pytest
from chinvex.chunking import chunk_key


def test_chunk_key_stable():
    """Test that chunk_key is stable for same text."""
    text1 = "This is test content."
    text2 = "This is test content."  # Same content

    assert chunk_key(text1) == chunk_key(text2)


def test_chunk_key_normalizes_whitespace():
    """Test that chunk_key normalizes whitespace."""
    text1 = "This  is   test content."
    text2 = "This is test content."

    # Should produce same key (whitespace normalized)
    assert chunk_key(text1) == chunk_key(text2)


def test_chunk_key_different_content():
    """Test that different content produces different keys."""
    text1 = "This is test content."
    text2 = "This is different content."

    assert chunk_key(text1) != chunk_key(text2)


def test_chunk_key_length():
    """Test that chunk_key produces 16-char hex string."""
    text = "Test content"
    key = chunk_key(text)

    assert len(key) == 16
    assert all(c in '0123456789abcdef' for c in key)

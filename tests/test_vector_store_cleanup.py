"""Test VectorStore connection cleanup."""

import tempfile
from pathlib import Path

import pytest

from chinvex.vectors import VectorStore


def test_vector_store_close():
    """Test that VectorStore.close() works without errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        persist_dir = Path(tmpdir) / "chroma"
        persist_dir.mkdir()

        # Create and close VectorStore
        vec_store = VectorStore(persist_dir)
        vec_store.close()

        # Verify client and collection are cleared
        assert vec_store.client is None
        assert vec_store.collection is None


def test_vector_store_context_manager():
    """Test that VectorStore works as a context manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        persist_dir = Path(tmpdir) / "chroma"
        persist_dir.mkdir()

        # Use VectorStore as context manager
        with VectorStore(persist_dir) as vec_store:
            # Should be usable within context
            assert vec_store.client is not None
            assert vec_store.collection is not None
            count = vec_store.count()
            assert count == 0

        # Should be cleaned up after context
        assert vec_store.client is None
        assert vec_store.collection is None


def test_vector_store_close_idempotent():
    """Test that calling close() multiple times is safe."""
    with tempfile.TemporaryDirectory() as tmpdir:
        persist_dir = Path(tmpdir) / "chroma"
        persist_dir.mkdir()

        vec_store = VectorStore(persist_dir)
        vec_store.close()
        # Should not raise error on second close
        vec_store.close()


def test_vector_store_operations_after_close():
    """Test that operations fail gracefully after close."""
    with tempfile.TemporaryDirectory() as tmpdir:
        persist_dir = Path(tmpdir) / "chroma"
        persist_dir.mkdir()

        vec_store = VectorStore(persist_dir)
        vec_store.close()

        # Operations should fail because collection is None
        with pytest.raises(AttributeError):
            vec_store.count()

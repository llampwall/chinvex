import pytest
from pathlib import Path
from chinvex.index_meta import IndexMeta, read_index_meta, write_index_meta

def test_index_meta_creation(tmp_path):
    """Test creating index metadata."""
    meta = IndexMeta(
        schema_version=2,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        created_at="2026-01-29T12:00:00Z"
    )

    assert meta.embedding_provider == "openai"
    assert meta.embedding_dimensions == 1536

def test_write_and_read_index_meta(tmp_path):
    """Test writing and reading index metadata."""
    meta_path = tmp_path / "meta.json"

    meta = IndexMeta(
        schema_version=2,
        embedding_provider="ollama",
        embedding_model="mxbai-embed-large",
        embedding_dimensions=1024,
        created_at="2026-01-29T12:00:00Z"
    )

    write_index_meta(meta_path, meta)
    assert meta_path.exists()

    loaded = read_index_meta(meta_path)
    assert loaded.embedding_provider == "ollama"
    assert loaded.embedding_dimensions == 1024

def test_read_missing_meta_returns_none(tmp_path):
    """Test reading non-existent meta.json returns None."""
    meta_path = tmp_path / "meta.json"
    result = read_index_meta(meta_path)
    assert result is None

def test_dimension_mismatch_check():
    """Test checking dimension mismatch."""
    meta = IndexMeta(
        schema_version=2,
        embedding_provider="ollama",
        embedding_model="mxbai-embed-large",
        embedding_dimensions=1024,
        created_at="2026-01-29T12:00:00Z"
    )

    # Match
    assert meta.matches_provider("ollama", "mxbai-embed-large", 1024) is True

    # Mismatch on dimensions
    assert meta.matches_provider("ollama", "mxbai-embed-large", 1536) is False

    # Mismatch on provider
    assert meta.matches_provider("openai", "text-embedding-3-small", 1024) is False

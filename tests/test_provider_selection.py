import pytest
import os
from pathlib import Path
from chinvex.embedding_providers import get_provider
from chinvex.context import ContextConfig


def test_provider_selection_precedence_cli():
    """Test CLI flag takes precedence."""
    provider = get_provider(
        cli_provider="openai",
        context_config=None,
        env_provider="ollama"
    )
    assert provider.model_name == "text-embedding-3-small"


def test_provider_selection_precedence_context():
    """Test context.json takes precedence over env."""
    context_config = {
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small"
        }
    }
    provider = get_provider(
        cli_provider=None,
        context_config=context_config,
        env_provider="ollama"
    )
    assert provider.model_name == "text-embedding-3-small"


def test_provider_selection_precedence_env():
    """Test env var used if CLI and context not set."""
    provider = get_provider(
        cli_provider=None,
        context_config=None,
        env_provider="openai"
    )
    assert provider.model_name == "text-embedding-3-small"


def test_provider_selection_default_ollama():
    """Test default is ollama."""
    provider = get_provider(
        cli_provider=None,
        context_config=None,
        env_provider=None
    )
    assert provider.model_name == "mxbai-embed-large"


def test_dimension_mismatch_fails(tmp_path):
    """Test that dimension mismatch fails ingest."""
    # Create meta.json with ollama dims
    from chinvex.index_meta import IndexMeta, write_index_meta
    meta_path = tmp_path / "meta.json"
    meta = IndexMeta(
        schema_version=2,
        embedding_provider="ollama",
        embedding_model="mxbai-embed-large",
        embedding_dimensions=1024,
        created_at="2026-01-29T12:00:00Z"
    )
    write_index_meta(meta_path, meta)

    # Read it back and verify
    from chinvex.index_meta import read_index_meta
    existing_meta = read_index_meta(meta_path)

    # Check that it doesn't match openai provider
    assert not existing_meta.matches_provider("openai", "text-embedding-3-small", 1536)

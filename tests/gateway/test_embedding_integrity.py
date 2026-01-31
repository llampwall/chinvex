"""Tests for embedding integrity enforcement in gateway."""

import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from chinvex.gateway.app import load_embedding_config_from_contexts, validate_embedding_provider_available
from chinvex.index_meta import IndexMeta
from datetime import datetime


def test_load_embedding_config_reads_meta_json():
    """Gateway startup should read embedding config from first context's meta.json."""
    # Mock context structure
    mock_context = MagicMock()
    mock_context.name = "Chinvex"
    mock_context.index.sqlite_path = "P:/ai_memory/indexes/Chinvex/hybrid.db"

    # Mock meta.json
    mock_meta = IndexMeta(
        schema_version=1,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        created_at="2026-01-27T10:00:00Z"
    )

    with patch("chinvex.gateway.app.list_contexts") as mock_list:
        with patch("chinvex.gateway.app.load_context") as mock_load:
            with patch("chinvex.gateway.app.read_index_meta") as mock_read_meta:
                mock_list.return_value = [mock_context]
                mock_load.return_value = mock_context
                mock_read_meta.return_value = mock_meta

                config = load_embedding_config_from_contexts()

    assert config["embedding_provider"] == "openai"
    assert config["embedding_model"] == "text-embedding-3-small"
    assert config["contexts_loaded"] == 1


def test_load_embedding_config_handles_missing_meta():
    """Should use safe defaults when meta.json is missing."""
    mock_context = MagicMock()
    mock_context.name = "Chinvex"
    mock_context.index.sqlite_path = "P:/ai_memory/indexes/Chinvex/hybrid.db"

    with patch("chinvex.gateway.app.list_contexts") as mock_list:
        with patch("chinvex.gateway.app.load_context") as mock_load:
            with patch("chinvex.gateway.app.read_index_meta") as mock_read_meta:
                mock_list.return_value = [mock_context]
                mock_load.return_value = mock_context
                mock_read_meta.return_value = None  # meta.json missing

                config = load_embedding_config_from_contexts()

    # Should default to OpenAI (P5 spec default)
    assert config["embedding_provider"] == "openai"
    assert config["embedding_model"] == "text-embedding-3-small"
    assert config["contexts_loaded"] == 1


def test_load_embedding_config_detects_mixed_providers():
    """Should detect when contexts use different embedding providers."""
    mock_ctx1 = MagicMock()
    mock_ctx1.name = "Chinvex"
    mock_ctx1.index.sqlite_path = "P:/ai_memory/indexes/Chinvex/hybrid.db"

    mock_ctx2 = MagicMock()
    mock_ctx2.name = "Personal"
    mock_ctx2.index.sqlite_path = "P:/ai_memory/indexes/Personal/hybrid.db"

    mock_meta1 = IndexMeta(
        schema_version=1,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        created_at="2026-01-27T10:00:00Z"
    )

    mock_meta2 = IndexMeta(
        schema_version=1,
        embedding_provider="ollama",
        embedding_model="mxbai-embed-large",
        embedding_dimensions=1024,
        created_at="2026-01-27T10:00:00Z"
    )

    with patch("chinvex.gateway.app.list_contexts") as mock_list:
        with patch("chinvex.gateway.app.load_context") as mock_load:
            with patch("chinvex.gateway.app.read_index_meta") as mock_read_meta:
                mock_list.return_value = [mock_ctx1, mock_ctx2]

                # Return different metas based on context
                def side_effect(path):
                    if "Chinvex" in str(path):
                        return mock_meta1
                    else:
                        return mock_meta2

                mock_read_meta.side_effect = side_effect
                mock_load.side_effect = [mock_ctx1, mock_ctx2]

                config = load_embedding_config_from_contexts()

    # Should use first context's config but flag mixed providers
    assert config["embedding_provider"] == "openai"
    assert config["embedding_model"] == "text-embedding-3-small"
    assert config["contexts_loaded"] == 2
    assert config["mixed_providers"] is True


def test_validate_embedding_provider_openai_with_key():
    """Should pass validation when OpenAI provider configured and API key present."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
        # Should not raise
        validate_embedding_provider_available("openai", "text-embedding-3-small")


def test_validate_embedding_provider_openai_missing_key():
    """Should raise error when OpenAI provider configured but API key missing."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError, match="OpenAI API key required"):
            validate_embedding_provider_available("openai", "text-embedding-3-small")


def test_validate_embedding_provider_ollama_available():
    """Should pass validation when Ollama provider configured and service responds."""
    import requests

    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("requests.get") as mock_get:
        mock_get.return_value = mock_response
        # Should not raise
        validate_embedding_provider_available("ollama", "mxbai-embed-large")


def test_validate_embedding_provider_ollama_unavailable():
    """Should raise error when Ollama provider configured but service unavailable."""
    import requests

    with patch("requests.get") as mock_get:
        mock_get.side_effect = requests.ConnectionError("Connection refused")

        with pytest.raises(RuntimeError, match="Ollama service unavailable"):
            validate_embedding_provider_available("ollama", "mxbai-embed-large")


def test_validate_embedding_provider_unknown():
    """Should raise error for unknown provider."""
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        validate_embedding_provider_available("unknown-provider", "some-model")

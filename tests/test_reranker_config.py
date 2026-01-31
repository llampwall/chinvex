# tests/test_reranker_config.py
import json
from pathlib import Path

import pytest

from chinvex.reranker_config import (
    RerankerConfig,
    load_reranker_config,
    validate_reranker_config,
)


def test_reranker_config_dataclass_structure():
    """RerankerConfig should have required fields."""
    config = RerankerConfig(
        provider="cohere",
        model="rerank-english-v3.0",
        candidates=20,
        top_k=5
    )

    assert config.provider == "cohere"
    assert config.model == "rerank-english-v3.0"
    assert config.candidates == 20
    assert config.top_k == 5


def test_reranker_config_defaults():
    """RerankerConfig should provide sensible defaults."""
    config = RerankerConfig(
        provider="cohere",
        model="rerank-english-v3.0"
    )

    assert config.candidates == 20  # default
    assert config.top_k == 5  # default


def test_load_reranker_config_from_context_json(tmp_path):
    """load_reranker_config should parse reranker field from context.json."""
    context_file = tmp_path / "context.json"
    context_data = {
        "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
        "reranker": {
            "provider": "cohere",
            "model": "rerank-english-v3.0",
            "candidates": 30,
            "top_k": 8
        }
    }
    context_file.write_text(json.dumps(context_data))

    config = load_reranker_config(context_file)

    assert config is not None
    assert config.provider == "cohere"
    assert config.model == "rerank-english-v3.0"
    assert config.candidates == 30
    assert config.top_k == 8


def test_load_reranker_config_returns_none_when_missing(tmp_path):
    """load_reranker_config should return None if reranker field absent."""
    context_file = tmp_path / "context.json"
    context_data = {
        "embedding": {"provider": "openai", "model": "text-embedding-3-small"}
    }
    context_file.write_text(json.dumps(context_data))

    config = load_reranker_config(context_file)

    assert config is None


def test_load_reranker_config_returns_none_when_null(tmp_path):
    """load_reranker_config should return None if reranker is null."""
    context_file = tmp_path / "context.json"
    context_data = {
        "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
        "reranker": None
    }
    context_file.write_text(json.dumps(context_data))

    config = load_reranker_config(context_file)

    assert config is None


def test_validate_reranker_config_accepts_valid_providers():
    """validate_reranker_config should accept cohere, jina, local providers."""
    for provider in ["cohere", "jina", "local"]:
        config = RerankerConfig(provider=provider, model="test-model")
        # Should not raise
        validate_reranker_config(config)


def test_validate_reranker_config_rejects_unknown_provider():
    """validate_reranker_config should reject unknown providers."""
    config = RerankerConfig(provider="unknown", model="test-model")

    with pytest.raises(ValueError, match="Unknown reranker provider"):
        validate_reranker_config(config)


def test_validate_reranker_config_requires_positive_candidates():
    """validate_reranker_config should require candidates > 0."""
    config = RerankerConfig(provider="cohere", model="test", candidates=0)

    with pytest.raises(ValueError, match="candidates must be positive"):
        validate_reranker_config(config)


def test_validate_reranker_config_requires_positive_top_k():
    """validate_reranker_config should require top_k > 0."""
    config = RerankerConfig(provider="cohere", model="test", top_k=0)

    with pytest.raises(ValueError, match="top_k must be positive"):
        validate_reranker_config(config)


def test_validate_reranker_config_requires_top_k_le_candidates():
    """validate_reranker_config should require top_k <= candidates."""
    config = RerankerConfig(provider="cohere", model="test", candidates=10, top_k=15)

    with pytest.raises(ValueError, match="top_k must be <= candidates"):
        validate_reranker_config(config)


def test_validate_reranker_config_caps_candidates_at_50():
    """validate_reranker_config should cap candidates at 50 (budget guardrail)."""
    config = RerankerConfig(provider="cohere", model="test", candidates=100)

    with pytest.raises(ValueError, match="candidates cannot exceed 50"):
        validate_reranker_config(config)

# src/chinvex/reranker_config.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RerankerConfig:
    """Reranker configuration for two-stage retrieval.

    Attributes:
        provider: Reranker provider ('cohere', 'jina', 'local')
        model: Model name/identifier for the provider
        candidates: Number of candidates to retrieve in stage 1 (default 20)
        top_k: Number of results to return after reranking (default 5)
    """
    provider: str
    model: str
    candidates: int = 20
    top_k: int = 5


def load_reranker_config(context_file: Path) -> RerankerConfig | None:
    """Load reranker config from context.json.

    Args:
        context_file: Path to context.json

    Returns:
        RerankerConfig if present and valid, None if absent or null
    """
    with open(context_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    reranker_data = data.get("reranker")

    if reranker_data is None:
        return None

    return RerankerConfig(
        provider=reranker_data["provider"],
        model=reranker_data["model"],
        candidates=reranker_data.get("candidates", 20),
        top_k=reranker_data.get("top_k", 5),
    )


def validate_reranker_config(config: RerankerConfig) -> None:
    """Validate reranker configuration.

    Args:
        config: RerankerConfig to validate

    Raises:
        ValueError: If configuration is invalid
    """
    valid_providers = {"cohere", "jina", "local"}
    if config.provider not in valid_providers:
        raise ValueError(
            f"Unknown reranker provider: {config.provider}. "
            f"Valid providers: {valid_providers}"
        )

    if config.candidates <= 0:
        raise ValueError("candidates must be positive")

    if config.top_k <= 0:
        raise ValueError("top_k must be positive")

    if config.top_k > config.candidates:
        raise ValueError("top_k must be <= candidates")

    # Budget guardrail: max 50 candidates
    if config.candidates > 50:
        raise ValueError(
            f"candidates cannot exceed 50 (budget guardrail), got {config.candidates}"
        )

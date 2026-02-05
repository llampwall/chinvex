from __future__ import annotations

import logging
import os
import time
from typing import Protocol

from openai import OpenAI, RateLimitError, APIError
from prometheus_client import Counter, Histogram

log = logging.getLogger(__name__)

# Metrics
EMBEDDINGS_TOTAL = Counter(
    "chinvex_embeddings_total",
    "Total embedding requests",
    ["provider"]
)

EMBEDDINGS_LATENCY = Histogram(
    "chinvex_embeddings_latency_seconds",
    "Embedding request latency",
    ["provider"]
)

EMBEDDINGS_RETRIES = Counter(
    "chinvex_embeddings_retries_total",
    "Total embedding retries",
    ["provider"]
)


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        ...

    @property
    def dimensions(self) -> int:
        """Return the dimensionality of embeddings."""
        ...

    @property
    def model_name(self) -> str:
        """Return the model name."""
        ...


class OllamaProvider:
    """Ollama embedding provider."""

    # Model dimensions (hardcoded for now, could query Ollama API)
    MODEL_DIMS = {
        "mxbai-embed-large": 1024,
    }

    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model

        if model not in self.MODEL_DIMS:
            raise ValueError(f"Unknown Ollama model: {model}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        EMBEDDINGS_TOTAL.labels(provider="ollama").inc()

        with EMBEDDINGS_LATENCY.labels(provider="ollama").time():
            # Implementation will use existing OllamaEmbedder
            from .embed import OllamaEmbedder
            embedder = OllamaEmbedder(self.host, self.model)
            return embedder.embed(texts)

    @property
    def dimensions(self) -> int:
        return self.MODEL_DIMS[self.model]

    @property
    def model_name(self) -> str:
        return self.model


class OpenAIProvider:
    """OpenAI embedding provider."""

    # Model dimensions
    MODEL_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    MAX_BATCH_SIZE = 2048
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0

    def __init__(self, api_key: str | None, model: str):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

        self.model = model
        if model not in self.MODEL_DIMS:
            raise ValueError(f"Unknown OpenAI model: {model}")

        self.client = OpenAI(api_key=self.api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings using OpenAI API.
        Handles batching (max 2048 texts) and retries (3x with backoff).
        Filters out empty strings as OpenAI API rejects them.
        """
        EMBEDDINGS_TOTAL.labels(provider="openai").inc()

        with EMBEDDINGS_LATENCY.labels(provider="openai").time():
            # Filter out empty/whitespace-only strings and track indices
            non_empty_texts = []
            non_empty_indices = []
            for i, text in enumerate(texts):
                if text and text.strip():
                    non_empty_texts.append(text)
                    non_empty_indices.append(i)

            # If all texts are empty, return zero vectors
            if not non_empty_texts:
                return [[0.0] * self.dimensions for _ in texts]

            # Generate embeddings for non-empty texts
            all_embeddings = []
            for i in range(0, len(non_empty_texts), self.MAX_BATCH_SIZE):
                batch = non_empty_texts[i:i + self.MAX_BATCH_SIZE]
                embeddings = self._embed_batch(batch)
                all_embeddings.extend(embeddings)

            # Map embeddings back to original indices (use zero vectors for empty strings)
            result = []
            embedding_idx = 0
            for i in range(len(texts)):
                if i in non_empty_indices:
                    result.append(all_embeddings[embedding_idx])
                    embedding_idx += 1
                else:
                    result.append([0.0] * self.dimensions)

            return result

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch with retry logic."""
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model
                )
                return [item.embedding for item in response.data]
            except RateLimitError as e:
                EMBEDDINGS_RETRIES.labels(provider="openai").inc()
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    log.warning(f"Rate limit hit, retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"OpenAI rate limit exceeded after {self.MAX_RETRIES} attempts") from e
            except APIError as e:
                raise RuntimeError(f"OpenAI API error: {e}") from e

    @property
    def dimensions(self) -> int:
        return self.MODEL_DIMS[self.model]

    @property
    def model_name(self) -> str:
        return self.model


def get_provider(
    cli_provider: str | None,
    context_config: dict | None,
    env_provider: str | None,
    ollama_host: str = "http://localhost:11434"
) -> EmbeddingProvider:
    """
    Select embedding provider based on precedence:
    1. CLI flag (--embed-provider)
    2. context.json embedding config
    3. Environment variable (CHINVEX_EMBED_PROVIDER)
    4. Default: OpenAI text-embedding-3-small (P5 spec)

    Args:
        cli_provider: Provider from CLI flag
        context_config: Context config dict (may contain embedding.provider)
        env_provider: Provider from environment variable
        ollama_host: Ollama service URL (default: http://localhost:11434)

    Returns:
        Configured embedding provider instance

    Raises:
        ValueError: If provider is unknown or configuration is invalid
        RuntimeError: If OpenAI selected but API key is missing
    """
    provider_name = None
    model = None

    # 1. CLI flag (highest priority)
    if cli_provider:
        provider_name = cli_provider
    # 2. context.json
    elif context_config and "embedding" in context_config:
        provider_name = context_config["embedding"].get("provider")
        model = context_config["embedding"].get("model")
    # 3. Environment variable
    elif env_provider:
        provider_name = env_provider
    # 4. Default: OpenAI (P5 spec - was Ollama in P4)
    else:
        provider_name = "openai"

    # Instantiate provider
    if provider_name == "openai":
        model = model or "text-embedding-3-small"
        return OpenAIProvider(api_key=None, model=model)
    elif provider_name == "ollama":
        model = model or "mxbai-embed-large"
        return OllamaProvider(ollama_host, model)
    else:
        raise ValueError(f"Unknown embedding provider: {provider_name}")

# P5 Implementation Plan: Reliability + Retrieval Quality

> **For Claude:** REQUIRED SUB-SKILL: Use batch-exec to execute this plan.

**Goal:** Turn Chinvex from "works if you babysit it" into a trustworthy appliance with embedding integrity enforced, retrieval quality measurable and improvable, memory files maintained automatically, and operational annoyances fixed.

**Architecture:**
- P5.1: Gateway validates embedding provider matches index metadata, hard fails on mismatch
- P5.2: Memory system (STATE.md/CONSTRAINTS.md/DECISIONS.md) maintained via `update-memory` command, briefs updated for project state visibility
- P5.3: Golden query eval suite with baseline metrics and CI gates
- P5.4: Two-stage retrieval with configurable reranker (Cohere/Jina/local cross-encoder)

**Tech Stack:** Python 3.12, ChromaDB, OpenAI embeddings (default), sentence-transformers (optional local reranker), pytest

**Implementation Order:**
1. P5.1 Embedding Integrity (batches 1-3)
2. P5.2.2 + P5.2.3 Brief/Morning Brief (batch 4)
3. P5.2.1 + P5.2.4 Memory Maintainer + Hook (batches 5-6)
4. P5.3 Eval Suite (batches 7-8)
5. P5.4 Reranker (batches 9-10)

---

<!-- Tasks will be appended by batch-plan subagents -->
## Task 1: Gateway /health endpoint returns embedding configuration

**Files:**
- Edit: `C:\Code\chinvex\src\chinvex\gateway\endpoints\health.py`
- Edit: `C:\Code\chinvex\src\chinvex\gateway\app.py`
- Test: `C:\Code\chinvex\tests\gateway\endpoints\test_health.py`

**Step 1: Write the failing test**

```python
# Edit C:\Code\chinvex\tests\gateway\endpoints\test_health.py
from fastapi.testclient import TestClient
from fastapi import FastAPI
from chinvex.gateway.endpoints.health import router
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone


app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_health_endpoint_returns_ok():
    """Should return status ok without auth."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "contexts_available" in data


def test_health_endpoint_returns_embedding_config():
    """Should return embedding provider, model, and uptime in health check."""
    # Mock the gateway state to have embedding config loaded
    mock_meta = MagicMock()
    mock_meta.embedding_provider = "openai"
    mock_meta.embedding_model = "text-embedding-3-small"
    
    # Mock contexts and startup time
    mock_contexts = [MagicMock(name="Chinvex"), MagicMock(name="Personal")]
    mock_startup_time = datetime.now(timezone.utc)
    
    with patch("chinvex.gateway.endpoints.health.get_gateway_state") as mock_state:
        mock_state.return_value = {
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "contexts_loaded": 2,
            "startup_time": mock_startup_time
        }
        
        response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify embedding config is present
    assert data["embedding_provider"] == "openai"
    assert data["embedding_model"] == "text-embedding-3-small"
    assert data["contexts_loaded"] == 2
    assert "uptime_seconds" in data
    assert data["uptime_seconds"] >= 0
```

**Step 2: Run test to verify it fails**

Run: `pwsh -Command "cd C:\Code\chinvex; pytest tests/gateway/endpoints/test_health.py::test_health_endpoint_returns_embedding_config -v"`

Expected: FAIL with "ImportError: cannot import name 'get_gateway_state'" or "KeyError: 'embedding_provider'"

**Step 3: Write minimal implementation**

```python
# Edit C:\Code\chinvex\src\chinvex\gateway\endpoints\health.py
"""Health check endpoint."""

from fastapi import APIRouter
from datetime import datetime, timezone
from chinvex.gateway import __version__
from chinvex.context import list_contexts
from chinvex.context_cli import get_contexts_root

router = APIRouter()

# Module-level state set during startup
_gateway_state = {
    "embedding_provider": None,
    "embedding_model": None,
    "contexts_loaded": 0,
    "startup_time": None
}


def set_gateway_state(embedding_provider: str, embedding_model: str, contexts_loaded: int):
    """Set gateway state during startup. Called from app.py startup event."""
    _gateway_state["embedding_provider"] = embedding_provider
    _gateway_state["embedding_model"] = embedding_model
    _gateway_state["contexts_loaded"] = contexts_loaded
    _gateway_state["startup_time"] = datetime.now(timezone.utc)


def get_gateway_state() -> dict:
    """Get current gateway state for testing."""
    return _gateway_state.copy()


@router.get("/health")
async def health():
    """
    Health check endpoint. No authentication required.

    Returns:
        Status information including version, context count, and embedding configuration
    """
    try:
        contexts_root = get_contexts_root()
        contexts = list_contexts(contexts_root)
        contexts_available = len(contexts)
    except Exception:
        # If context listing fails, don't fail health check
        contexts_available = 0

    # Calculate uptime
    uptime_seconds = 0
    if _gateway_state["startup_time"]:
        uptime_seconds = int((datetime.now(timezone.utc) - _gateway_state["startup_time"]).total_seconds())

    response = {
        "status": "ok",
        "version": __version__,
        "contexts_available": contexts_available
    }

    # Include embedding config if available
    if _gateway_state["embedding_provider"]:
        response["embedding_provider"] = _gateway_state["embedding_provider"]
        response["embedding_model"] = _gateway_state["embedding_model"]
        response["contexts_loaded"] = _gateway_state["contexts_loaded"]
        response["uptime_seconds"] = uptime_seconds

    return response
```

**Step 4: Run test to verify it passes**

Run: `pwsh -Command "cd C:\Code\chinvex; pytest tests/gateway/endpoints/test_health.py::test_health_endpoint_returns_embedding_config -v"`

Expected: PASS

**Step 5: Commit**

```bash
pwsh -Command "cd C:\Code\chinvex; git add src/chinvex/gateway/endpoints/health.py tests/gateway/endpoints/test_health.py"
pwsh -Command "cd C:\Code\chinvex; git commit -m `"$(cat <<'EOF'
feat: add embedding config to /health endpoint

- /health now returns embedding_provider, embedding_model
- Tracks contexts_loaded and uptime_seconds
- Module-level state set during gateway startup
- Test coverage for new fields

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)`""
```

---

## Task 2: Gateway reads embedding config from meta.json on startup

**Files:**
- Edit: `C:\Code\chinvex\src\chinvex\gateway\app.py`
- Test: `C:\Code\chinvex\tests\gateway\test_embedding_integrity.py` (create)

**Step 1: Write the failing test**

```python
# Create C:\Code\chinvex\tests\gateway\test_embedding_integrity.py
"""Tests for embedding integrity enforcement in gateway."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from chinvex.gateway.app import load_embedding_config_from_contexts
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
```

**Step 2: Run test to verify it fails**

Run: `pwsh -Command "cd C:\Code\chinvex; pytest tests/gateway/test_embedding_integrity.py -v"`

Expected: FAIL with "ImportError: cannot import name 'load_embedding_config_from_contexts'"

**Step 3: Write minimal implementation**

```python
# Edit C:\Code\chinvex\src\chinvex\gateway\app.py
# Add to imports section:
from chinvex.index_meta import read_index_meta
from chinvex.context_cli import get_contexts_root
from chinvex.context import list_contexts, load_context

# Add new function before startup_warmup:
def load_embedding_config_from_contexts() -> dict:
    """
    Load embedding configuration from contexts.

    Reads meta.json from each context to determine embedding provider.
    Uses first context's config as the gateway's default.
    Detects mixed providers across contexts.

    Returns:
        dict with keys: embedding_provider, embedding_model, contexts_loaded, mixed_providers
    """
    from pathlib import Path

    try:
        contexts_root = get_contexts_root()
        contexts = list_contexts(contexts_root)

        if not contexts:
            # No contexts loaded - use safe default (OpenAI per P5 spec)
            return {
                "embedding_provider": "openai",
                "embedding_model": "text-embedding-3-small",
                "contexts_loaded": 0,
                "mixed_providers": False
            }

        # Read meta.json from each context
        providers_seen = set()
        first_provider = None
        first_model = None

        for ctx_info in contexts:
            context = load_context(ctx_info.name, contexts_root)

            # Derive meta.json path from sqlite_path
            index_dir = Path(context.index.sqlite_path).parent
            meta_path = index_dir / "meta.json"

            meta = read_index_meta(meta_path)

            if meta:
                providers_seen.add(f"{meta.embedding_provider}:{meta.embedding_model}")
                if first_provider is None:
                    first_provider = meta.embedding_provider
                    first_model = meta.embedding_model
            else:
                # No meta.json - assume OpenAI default (P5 spec)
                if first_provider is None:
                    first_provider = "openai"
                    first_model = "text-embedding-3-small"
                providers_seen.add("openai:text-embedding-3-small")

        # Use first context's config or default
        if first_provider is None:
            first_provider = "openai"
            first_model = "text-embedding-3-small"

        return {
            "embedding_provider": first_provider,
            "embedding_model": first_model,
            "contexts_loaded": len(contexts),
            "mixed_providers": len(providers_seen) > 1
        }

    except Exception as e:
        logger.error(f"Failed to load embedding config: {e}", exc_info=True)
        # Fail safe to OpenAI default
        return {
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "contexts_loaded": 0,
            "mixed_providers": False
        }


# Modify startup_warmup function to call load_embedding_config_from_contexts:
@app.on_event("startup")
async def startup_warmup():
    """Warm up the gateway by preloading contexts and initializing storage."""
    logger.info("Starting gateway warmup...")

    # Load embedding config from contexts
    embedding_config = load_embedding_config_from_contexts()

    # Set gateway state for /health endpoint
    from chinvex.gateway.endpoints.health import set_gateway_state
    set_gateway_state(
        embedding_provider=embedding_config["embedding_provider"],
        embedding_model=embedding_config["embedding_model"],
        contexts_loaded=embedding_config["contexts_loaded"]
    )

    # Log warning if mixed providers detected
    if embedding_config["mixed_providers"]:
        logger.warning(
            "Mixed embedding providers detected across contexts. "
            "Cross-context search will be restricted."
        )

    logger.info(
        f"Gateway configured with {embedding_config['embedding_provider']} "
        f"({embedding_config['embedding_model']}), "
        f"{embedding_config['contexts_loaded']} contexts loaded"
    )

    try:
        from chinvex.context_cli import get_contexts_root, list_contexts
        from chinvex.context import load_context
        from chinvex.storage import Storage
        from chinvex.vectors import VectorStore

        # Load context registry
        contexts_root = get_contexts_root()
        contexts = list_contexts(contexts_root)
        logger.info(f"Loaded {len(contexts)} contexts")

        # Preload first context to initialize storage and vector store
        if contexts:
            context = load_context(contexts[0].name, contexts_root)
            # Touch SQLite
            storage = Storage(context.index.sqlite_path)
            storage._execute("SELECT 1")
            # Touch Chroma
            vec_store = VectorStore(context.index.chroma_dir)
            vec_store.collection.count()
            logger.info(f"Warmed up context: {contexts[0].name}")

        logger.info("Gateway warmup complete")
    except Exception as e:
        logger.error(f"Warmup failed (non-fatal): {e}", exc_info=True)
```

**Step 4: Run test to verify it passes**

Run: `pwsh -Command "cd C:\Code\chinvex; pytest tests/gateway/test_embedding_integrity.py -v"`

Expected: PASS

**Step 5: Commit**

```bash
pwsh -Command "cd C:\Code\chinvex; git add src/chinvex/gateway/app.py tests/gateway/test_embedding_integrity.py; git commit -m 'feat: gateway reads embedding config from meta.json on startup

- load_embedding_config_from_contexts() reads meta.json from all contexts
- Uses first context config as gateway default
- Detects mixed providers across contexts
- Defaults to OpenAI text-embedding-3-small when meta.json missing
- Logs warning when mixed providers detected
- Test coverage for config loading, missing meta, mixed providers

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>'"
```

---

## Task 3: Validate embedding provider is available before accepting queries

**Files:**
- Edit: `C:\Code\chinvex\src\chinvex\gateway\app.py`
- Edit: `C:\Code\chinvex\src\chinvex\embedding_providers.py`
- Test: `C:\Code\chinvex\tests\gateway\test_embedding_integrity.py`

**Step 1: Write the failing test**

```python
# Edit C:\Code\chinvex\tests\gateway\test_embedding_integrity.py
# Add to existing file:

import os
from unittest.mock import patch
from chinvex.gateway.app import validate_embedding_provider_available


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
    from unittest.mock import MagicMock

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
```

**Step 2: Run test to verify it fails**

Run: `pwsh -Command "cd C:\Code\chinvex; pytest tests/gateway/test_embedding_integrity.py::test_validate_embedding_provider_openai_with_key -v"`

Expected: FAIL with "ImportError: cannot import name 'validate_embedding_provider_available'"

**Step 3: Write minimal implementation**

```python
# Edit C:\Code\chinvex\src\chinvex\gateway\app.py
# Add after load_embedding_config_from_contexts:

def validate_embedding_provider_available(provider: str, model: str) -> None:
    """
    Validate that the specified embedding provider is available and can be used.

    For OpenAI: checks that OPENAI_API_KEY is set
    For Ollama: checks that service is reachable

    Raises:
        RuntimeError: If provider is configured but not available
        ValueError: If provider is unknown
    """
    import os
    import requests

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"OpenAI API key required for query embedding. "
                f"Gateway index uses {provider} ({model}), but OPENAI_API_KEY is not set. "
                f"Set the environment variable or switch to a different embedding provider."
            )
        # API key is set - assume it's valid (will fail at query time if not)

    elif provider == "ollama":
        # Check Ollama service is reachable
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        try:
            response = requests.get(f"{ollama_host}/api/tags", timeout=2)
            response.raise_for_status()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            raise RuntimeError(
                f"Ollama service unavailable. "
                f"Gateway index uses {provider} ({model}), but service at {ollama_host} is not responding. "
                f"Start Ollama or switch to a different embedding provider."
            ) from e

    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


# Modify startup_warmup to validate provider:
@app.on_event("startup")
async def startup_warmup():
    """Warm up the gateway by preloading contexts and initializing storage."""
    logger.info("Starting gateway warmup...")

    # Load embedding config from contexts
    embedding_config = load_embedding_config_from_contexts()

    # Validate provider is available before accepting queries
    try:
        validate_embedding_provider_available(
            embedding_config["embedding_provider"],
            embedding_config["embedding_model"]
        )
        logger.info(
            f"Validated embedding provider: {embedding_config['embedding_provider']} "
            f"({embedding_config['embedding_model']})"
        )
    except (RuntimeError, ValueError) as e:
        # CRITICAL: Gateway cannot serve queries without valid embedding provider
        logger.error(f"Embedding provider validation failed: {e}")
        logger.error("Gateway startup aborted. Fix embedding configuration and restart.")
        raise

    # Set gateway state for /health endpoint
    from chinvex.gateway.endpoints.health import set_gateway_state
    set_gateway_state(
        embedding_provider=embedding_config["embedding_provider"],
        embedding_model=embedding_config["embedding_model"],
        contexts_loaded=embedding_config["contexts_loaded"]
    )

    # Log warning if mixed providers detected
    if embedding_config["mixed_providers"]:
        logger.warning(
            "Mixed embedding providers detected across contexts. "
            "Cross-context search will be restricted."
        )

    logger.info(
        f"Gateway configured with {embedding_config['embedding_provider']} "
        f"({embedding_config['embedding_model']}), "
        f"{embedding_config['contexts_loaded']} contexts loaded"
    )

    try:
        from chinvex.context_cli import get_contexts_root, list_contexts
        from chinvex.context import load_context
        from chinvex.storage import Storage
        from chinvex.vectors import VectorStore

        # Load context registry
        contexts_root = get_contexts_root()
        contexts = list_contexts(contexts_root)
        logger.info(f"Loaded {len(contexts)} contexts")

        # Preload first context to initialize storage and vector store
        if contexts:
            context = load_context(contexts[0].name, contexts_root)
            # Touch SQLite
            storage = Storage(context.index.sqlite_path)
            storage._execute("SELECT 1")
            # Touch Chroma
            vec_store = VectorStore(context.index.chroma_dir)
            vec_store.collection.count()
            logger.info(f"Warmed up context: {contexts[0].name}")

        logger.info("Gateway warmup complete")
    except Exception as e:
        logger.error(f"Warmup failed (non-fatal): {e}", exc_info=True)
```

**Step 4: Run test to verify it passes**

Run: `pwsh -Command "cd C:\Code\chinvex; pytest tests/gateway/test_embedding_integrity.py -v"`

Expected: PASS

**Step 5: Commit**

```bash
pwsh -Command "cd C:\Code\chinvex; git add src/chinvex/gateway/app.py tests/gateway/test_embedding_integrity.py; git commit -m 'feat: validate embedding provider availability on gateway startup

- validate_embedding_provider_available() checks provider is ready
- OpenAI: validates OPENAI_API_KEY environment variable is set
- Ollama: validates service is reachable at configured host
- Gateway startup fails fast with clear error if provider unavailable
- Prevents silent degradation to wrong embedding space
- Test coverage for OpenAI (with/without key), Ollama (up/down), unknown providers

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>'"
```

---

## Task 4: Set OpenAI as default embedding provider for new contexts

**Files:**
- Edit: `C:\Code\chinvex\src\chinvex\embedding_providers.py`
- Test: `C:\Code\chinvex\tests\test_embedding_providers.py` (create)

**Step 1: Write the failing test**

```python
# Create C:\Code\chinvex\tests\test_embedding_providers.py
"""Tests for embedding provider selection and defaults."""

import pytest
import os
from unittest.mock import patch
from chinvex.embedding_providers import get_provider, OpenAIProvider, OllamaProvider


def test_get_provider_defaults_to_openai():
    """When no provider specified, should default to OpenAI (P5 spec)."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
        provider = get_provider(
            cli_provider=None,
            context_config=None,
            env_provider=None
        )

    assert isinstance(provider, OpenAIProvider)
    assert provider.model_name == "text-embedding-3-small"
    assert provider.dimensions == 1536


def test_get_provider_cli_overrides_all():
    """CLI flag should override context.json and env."""
    context_config = {
        "embedding": {
            "provider": "ollama",
            "model": "mxbai-embed-large"
        }
    }

    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key", "CHINVEX_EMBED_PROVIDER": "ollama"}):
        provider = get_provider(
            cli_provider="openai",
            context_config=context_config,
            env_provider="ollama"
        )

    assert isinstance(provider, OpenAIProvider)
    assert provider.model_name == "text-embedding-3-small"


def test_get_provider_context_config_overrides_env():
    """context.json should override environment variable."""
    context_config = {
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small"
        }
    }

    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key", "CHINVEX_EMBED_PROVIDER": "ollama"}):
        provider = get_provider(
            cli_provider=None,
            context_config=context_config,
            env_provider="ollama"
        )

    assert isinstance(provider, OpenAIProvider)


def test_get_provider_env_overrides_default():
    """Environment variable should override default."""
    with patch.dict(os.environ, {"CHINVEX_EMBED_PROVIDER": "ollama"}):
        provider = get_provider(
            cli_provider=None,
            context_config=None,
            env_provider="ollama"
        )

    assert isinstance(provider, OllamaProvider)
    assert provider.model_name == "mxbai-embed-large"


def test_get_provider_explicit_ollama():
    """Should still support explicit Ollama selection."""
    provider = get_provider(
        cli_provider="ollama",
        context_config=None,
        env_provider=None
    )

    assert isinstance(provider, OllamaProvider)
    assert provider.model_name == "mxbai-embed-large"
    assert provider.dimensions == 1024


def test_openai_provider_requires_api_key():
    """OpenAI provider should fail if API key is missing."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="OpenAI API key required"):
            OpenAIProvider(api_key=None, model="text-embedding-3-small")


def test_openai_provider_with_explicit_key():
    """OpenAI provider should accept explicit API key."""
    provider = OpenAIProvider(api_key="sk-test-key", model="text-embedding-3-small")

    assert provider.model_name == "text-embedding-3-small"
    assert provider.dimensions == 1536
```

**Step 2: Run test to verify it fails**

Run: `pwsh -Command "cd C:\Code\chinvex; pytest tests/test_embedding_providers.py::test_get_provider_defaults_to_openai -v"`

Expected: FAIL with "AssertionError: assert isinstance(<OllamaProvider>, OpenAIProvider)" (current default is Ollama)

**Step 3: Write minimal implementation**

```python
# Edit C:\Code\chinvex\src\chinvex\embedding_providers.py
# Modify get_provider function (around line 156):

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
```

**Step 4: Run test to verify it passes**

Run: `pwsh -Command "cd C:\Code\chinvex; pytest tests/test_embedding_providers.py -v"`

Expected: PASS

**Step 5: Commit**

```bash
pwsh -Command "cd C:\Code\chinvex; git add src/chinvex/embedding_providers.py tests/test_embedding_providers.py; git commit -m 'feat: change default embedding provider to OpenAI

- get_provider() now defaults to OpenAI text-embedding-3-small (P5 spec)
- Previous default was Ollama (P4 and earlier)
- Rationale: 45x faster, consistent quality, negligible cost for personal use
- Ollama still available via --embed-provider ollama flag
- Provider precedence: CLI > context.json > env > default (OpenAI)
- Test coverage for default, precedence, explicit Ollama, API key validation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>'"
```

---
## Task 5: Detect mixed embedding providers in cross-context search and refuse with clear error

**Files:**
- Edit: `C:\Code\chinvex\src\chinvex\search.py`
- Test: `C:\Code\chinvex\tests\test_mixed_embedding_detection.py`

**Step 1: Write the failing test**

```python
# tests/test_mixed_embedding_detection.py
import pytest
from pathlib import Path
from chinvex.context import ContextConfig, ContextIncludes, ContextIndex, OllamaConfig, EmbeddingConfig
from chinvex.search import search_multi_context
from datetime import datetime, timezone


def test_mixed_embedding_providers_raises_error(tmp_path, monkeypatch):
    """Cross-context search with mixed embedding providers should raise clear error."""
    # Setup two contexts with different embedding providers
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"
    contexts_root.mkdir()
    indexes_root.mkdir()

    # Context 1: OpenAI embeddings
    ctx1_dir = contexts_root / "Context1"
    ctx1_dir.mkdir()
    ctx1_index = indexes_root / "Context1"
    ctx1_index.mkdir()

    ctx1_data = {
        "schema_version": 2,
        "name": "Context1",
        "aliases": [],
        "includes": {"repos": [], "chat_roots": [], "codex_session_roots": [], "note_roots": []},
        "index": {"sqlite_path": str(ctx1_index / "hybrid.db"), "chroma_dir": str(ctx1_index / "chroma")},
        "weights": {"repo": 1.0, "chat": 0.8},
        "ollama": {"base_url": "http://localhost:11434", "embed_model": "mxbai-embed-large"},
        "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    (ctx1_dir / "context.json").write_text(
        __import__("json").dumps(ctx1_data, indent=2)
    )

    # Context 2: Ollama embeddings
    ctx2_dir = contexts_root / "Context2"
    ctx2_dir.mkdir()
    ctx2_index = indexes_root / "Context2"
    ctx2_index.mkdir()

    ctx2_data = {
        "schema_version": 2,
        "name": "Context2",
        "aliases": [],
        "includes": {"repos": [], "chat_roots": [], "codex_session_roots": [], "note_roots": []},
        "index": {"sqlite_path": str(ctx2_index / "hybrid.db"), "chroma_dir": str(ctx2_index / "chroma")},
        "weights": {"repo": 1.0, "chat": 0.8},
        "ollama": {"base_url": "http://localhost:11434", "embed_model": "mxbai-embed-large"},
        "embedding": {"provider": "ollama", "model": "mxbai-embed-large"},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    (ctx2_dir / "context.json").write_text(
        __import__("json").dumps(ctx2_data, indent=2)
    )

    # Set environment variable for contexts root
    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))

    # Attempt cross-context search with mixed providers
    with pytest.raises(ValueError) as exc_info:
        search_multi_context(
            contexts=["Context1", "Context2"],
            query="test query",
            k=5
        )

    # Verify error message is clear
    error_msg = str(exc_info.value)
    assert "mixed embedding providers" in error_msg.lower()
    assert "openai" in error_msg.lower()
    assert "ollama" in error_msg.lower()
    assert "Context1" in error_msg or "Context2" in error_msg


def test_same_embedding_provider_allowed(tmp_path, monkeypatch):
    """Cross-context search with same embedding provider should succeed."""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"
    contexts_root.mkdir()
    indexes_root.mkdir()

    # Both contexts use OpenAI
    for ctx_name in ["ContextA", "ContextB"]:
        ctx_dir = contexts_root / ctx_name
        ctx_dir.mkdir()
        ctx_index = indexes_root / ctx_name
        ctx_index.mkdir()

        # Create minimal index files
        from chinvex.storage import Storage
        db_path = ctx_index / "hybrid.db"
        storage = Storage(db_path)
        storage.ensure_schema()
        storage.close()

        ctx_data = {
            "schema_version": 2,
            "name": ctx_name,
            "aliases": [],
            "includes": {"repos": [], "chat_roots": [], "codex_session_roots": [], "note_roots": []},
            "index": {"sqlite_path": str(db_path), "chroma_dir": str(ctx_index / "chroma")},
            "weights": {"repo": 1.0, "chat": 0.8},
            "ollama": {"base_url": "http://localhost:11434", "embed_model": "mxbai-embed-large"},
            "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        (ctx_dir / "context.json").write_text(
            __import__("json").dumps(ctx_data, indent=2)
        )

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Should not raise error (will fail on actual embedding, but that's expected)
    try:
        search_multi_context(
            contexts=["ContextA", "ContextB"],
            query="test query",
            k=5
        )
    except ValueError as e:
        # Should not be mixed embedding error
        assert "mixed embedding" not in str(e).lower()
```

**Step 2: Run test to verify it fails**

Run: `pwsh -Command "pytest C:\Code\chinvex\tests\test_mixed_embedding_detection.py -v"`

Expected: FAIL with "AttributeError" or "search_multi_context does not check for mixed embeddings"

**Step 3: Write minimal implementation**

```python
# src/chinvex/search.py
# Add this function before search_multi_context

def _detect_mixed_embedding_providers(contexts: list[str], contexts_root: Path) -> None:
    """
    Detect if contexts use different embedding providers and raise error if so.

    Args:
        contexts: List of context names
        contexts_root: Root directory for contexts

    Raises:
        ValueError: If contexts use different embedding providers
    """
    from .context import load_context

    providers = {}
    for ctx_name in contexts:
        try:
            ctx = load_context(ctx_name, contexts_root)
            # Determine provider from embedding config or default to ollama
            if ctx.embedding:
                provider = ctx.embedding.provider
                model = ctx.embedding.model or "unknown"
            else:
                # Legacy contexts default to ollama
                provider = "ollama"
                model = ctx.ollama.embed_model

            providers[ctx_name] = (provider, model)
        except Exception:
            # Skip contexts that fail to load
            continue

    # Check if all providers are the same
    if not providers:
        return  # No contexts loaded successfully

    unique_providers = set(p[0] for p in providers.values())
    if len(unique_providers) > 1:
        # Build detailed error message
        provider_details = ", ".join(
            f"{ctx}={prov}" for ctx, (prov, _) in sorted(providers.items())
        )
        raise ValueError(
            f"Cross-context search with mixed embedding providers is not allowed. "
            f"Contexts have different providers: {provider_details}. "
            f"Use --context to search a single context, or ensure all contexts use the same embedding provider."
        )


# Update search_multi_context function
def search_multi_context(
    contexts: list[str] | str,
    query: str,
    k: int = 10,
    min_score: float = 0.35,
    source: str = "all",
    ollama_host: str | None = None,
    recency_enabled: bool = True,
) -> list[SearchResult]:
    """
    Search across multiple contexts, merge results by score.

    Args:
        contexts: List of context names, or "all" for all contexts
        query: Search query
        k: Total number of results to return (not per-context)
        min_score: Minimum score threshold
        source: Filter by source type (all/repo/chat/codex_session)
        ollama_host: Ollama host override
        recency_enabled: Enable recency decay

    Returns:
        List of SearchResult objects sorted by score descending
    """
    from pathlib import Path
    import os

    # Get contexts root
    contexts_root_str = os.getenv("CHINVEX_CONTEXTS_ROOT", "P:/ai_memory/contexts")
    contexts_root = Path(contexts_root_str)

    # Expand "all" to all available contexts
    if contexts == "all":
        from .context import list_contexts
        contexts = [c.name for c in list_contexts(contexts_root)]

    # Cap contexts to prevent slowdown
    max_contexts = 10  # TODO: Make configurable
    if len(contexts) > max_contexts:
        contexts = contexts[:max_contexts]

    # CHANGE: Detect mixed embedding providers before search
    _detect_mixed_embedding_providers(contexts, contexts_root)

    # Per-context cap: fetch more than k to ensure good merged results
    k_per_context = min(k * 2, 20)

    # [Rest of function remains unchanged...]
    # Gather results from each context
    all_results = []
    for ctx_name in contexts:
        try:
            from .context import load_context
            ctx = load_context(ctx_name, contexts_root)
            results = search_context(
                ctx=ctx,
                query=query,
                k=k_per_context,
                min_score=min_score,
                source=source,
                ollama_host_override=ollama_host,
                recency_enabled=recency_enabled,
            )
            # Tag each result with source context
            for r in results:
                r.context = ctx_name
            all_results.extend(results)
        except Exception as e:
            # Log warning but continue with other contexts
            print(f"Warning: Failed to search context {ctx_name}: {e}")
            continue

    # Sort by score descending, take top k
    all_results.sort(key=lambda r: r.score, reverse=True)
    final_results = all_results[:k]

    # Debug logging: score distribution across contexts
    if final_results:
        score_min = min(r.score for r in final_results)
        score_max = max(r.score for r in final_results)
        context_counts = {}
        for r in final_results:
            context_counts[r.context] = context_counts.get(r.context, 0) + 1
        print(f"[DEBUG] Cross-context scores: min={score_min:.3f}, max={score_max:.3f}, by_context={context_counts}")

    return final_results
```

**Step 4: Run test to verify it passes**

Run: `pwsh -Command "pytest C:\Code\chinvex\tests\test_mixed_embedding_detection.py -v"`

Expected: PASS

**Step 5: Commit**

```bash
pwsh -Command "git add C:\Code\chinvex\src\chinvex\search.py C:\Code\chinvex\tests\test_mixed_embedding_detection.py; git commit -m 'feat(P5.1): detect mixed embedding providers in cross-context search

- Add _detect_mixed_embedding_providers validation function
- Raise clear error when contexts use different embedding providers
- Allow cross-context search only when all contexts use same provider
- Test coverage for mixed and same-provider scenarios

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>'"
```

---
## Task 6: Add `--allow-mixed-embeddings` flag that returns "not yet supported" error

**Files:**
- Edit: `C:\Code\chinvex\src\chinvex\cli.py`
- Edit: `C:\Code\chinvex\src\chinvex\search.py`
- Test: `C:\Code\chinvex\tests\test_mixed_embedding_flag.py`

**Step 1: Write the failing test**

```python
# tests/test_mixed_embedding_flag.py
import pytest
from typer.testing import CliRunner
from chinvex.cli import app
from pathlib import Path
import json
from datetime import datetime, timezone


runner = CliRunner()


def test_allow_mixed_embeddings_flag_not_yet_supported(tmp_path, monkeypatch):
    """Flag --allow-mixed-embeddings should exist but return 'not yet supported' error."""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"
    contexts_root.mkdir()
    indexes_root.mkdir()

    # Create two contexts with different providers
    for ctx_name, provider in [("Ctx1", "openai"), ("Ctx2", "ollama")]:
        ctx_dir = contexts_root / ctx_name
        ctx_dir.mkdir()
        ctx_index = indexes_root / ctx_name
        ctx_index.mkdir()

        ctx_data = {
            "schema_version": 2,
            "name": ctx_name,
            "aliases": [],
            "includes": {"repos": [], "chat_roots": [], "codex_session_roots": [], "note_roots": []},
            "index": {"sqlite_path": str(ctx_index / "hybrid.db"), "chroma_dir": str(ctx_index / "chroma")},
            "weights": {"repo": 1.0, "chat": 0.8},
            "ollama": {"base_url": "http://localhost:11434", "embed_model": "mxbai-embed-large"},
            "embedding": {"provider": provider},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        (ctx_dir / "context.json").write_text(json.dumps(ctx_data, indent=2))

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))

    # Test CLI with flag
    result = runner.invoke(app, [
        "search",
        "test query",
        "--contexts", "Ctx1,Ctx2",
        "--allow-mixed-embeddings"
    ])

    # Should exit with error
    assert result.exit_code != 0
    assert "not yet supported" in result.stdout.lower() or "not yet supported" in str(result.exception).lower()


def test_allow_mixed_embeddings_flag_exists():
    """Verify --allow-mixed-embeddings flag is recognized by CLI."""
    result = runner.invoke(app, ["search", "--help"])

    # Help should mention the flag
    assert "--allow-mixed-embeddings" in result.stdout
```

**Step 2: Run test to verify it fails**

Run: `pwsh -Command "pytest C:\Code\chinvex\tests\test_mixed_embedding_flag.py -v"`

Expected: FAIL with "no such option: --allow-mixed-embeddings"

**Step 3: Write minimal implementation**

```python
# src/chinvex/cli.py
# Update the search_cmd function signature (around line 203)

@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="Search query"),
    context: str | None = typer.Option(None, "--context", "-c", help="Context name to search (deprecated for multi-context)"),
    contexts: str | None = typer.Option(None, "--contexts", help="Comma-separated context names"),
    all_contexts: bool = typer.Option(False, "--all", help="Search all contexts"),
    exclude: str | None = typer.Option(None, "--exclude", help="Comma-separated contexts to exclude (with --all)"),
    config: Path | None = typer.Option(None, "--config", help="Path to old config.json (deprecated)"),
    k: int = typer.Option(8, "--k", help="Top K results"),
    min_score: float = typer.Option(0.35, "--min-score", help="Minimum score threshold"),
    source: str = typer.Option("all", "--source", help="all|repo|chat|codex_session"),
    project: str | None = typer.Option(None, "--project", help="Filter by project"),
    repo: str | None = typer.Option(None, "--repo", help="Filter by repo"),
    ollama_host: str | None = typer.Option(None, "--ollama-host", help="Override Ollama host"),
    no_recency: bool = typer.Option(False, "--no-recency", help="Disable recency decay"),
    allow_mixed_embeddings: bool = typer.Option(False, "--allow-mixed-embeddings", help="Allow mixed embedding providers (P6+)"),
) -> None:
    if not in_venv():
        typer.secho("Warning: Not running inside a virtual environment.", fg=typer.colors.YELLOW)

    if source not in {"all", "repo", "chat", "codex_session"}:
        raise typer.BadParameter("source must be one of: all, repo, chat, codex_session")

    # Check for --allow-mixed-embeddings flag
    if allow_mixed_embeddings:
        typer.secho(
            "Error: Mixed-space embedding merge is not yet supported. "
            "This feature is planned for P6+. "
            "For now, ensure all contexts use the same embedding provider.",
            fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    # [Rest of function remains unchanged...]
```

```python
# src/chinvex/search.py
# Update search_multi_context signature to accept the flag

def search_multi_context(
    contexts: list[str] | str,
    query: str,
    k: int = 10,
    min_score: float = 0.35,
    source: str = "all",
    ollama_host: str | None = None,
    recency_enabled: bool = True,
    allow_mixed_embeddings: bool = False,  # NEW PARAMETER
) -> list[SearchResult]:
    """
    Search across multiple contexts, merge results by score.

    Args:
        contexts: List of context names, or "all" for all contexts
        query: Search query
        k: Total number of results to return (not per-context)
        min_score: Minimum score threshold
        source: Filter by source type (all/repo/chat/codex_session)
        ollama_host: Ollama host override
        recency_enabled: Enable recency decay
        allow_mixed_embeddings: Allow mixed embedding providers (not yet supported in P5)

    Returns:
        List of SearchResult objects sorted by score descending

    Raises:
        ValueError: If contexts use mixed embedding providers and allow_mixed_embeddings=False
        NotImplementedError: If allow_mixed_embeddings=True (not supported in P5)
    """
    from pathlib import Path
    import os

    # Check for mixed embeddings flag
    if allow_mixed_embeddings:
        raise NotImplementedError(
            "Mixed-space embedding merge is not yet supported. "
            "This feature is planned for P6+. "
            "For now, ensure all contexts use the same embedding provider."
        )

    # Get contexts root
    contexts_root_str = os.getenv("CHINVEX_CONTEXTS_ROOT", "P:/ai_memory/contexts")
    contexts_root = Path(contexts_root_str)

    # [Rest of function remains unchanged...]
```

**Step 4: Run test to verify it passes**

Run: `pwsh -Command "pytest C:\Code\chinvex\tests\test_mixed_embedding_flag.py -v"`

Expected: PASS

**Step 5: Commit**

```bash
pwsh -Command "git add C:\Code\chinvex\src\chinvex\cli.py C:\Code\chinvex\src\chinvex\search.py C:\Code\chinvex\tests\test_mixed_embedding_flag.py; git commit -m 'feat(P5.1): add --allow-mixed-embeddings flag (placeholder for P6)

- Add --allow-mixed-embeddings CLI flag to search command
- Raise NotImplementedError with clear message when flag is used
- Placeholder for future P6+ mixed-space merge feature
- Test coverage for flag existence and error message

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>'"
```

---

## Task 7: Update `chinvex status` command to show embedding provider per context

**Files:**
- Edit: `C:\Code\chinvex\src\chinvex\cli_status.py`
- Edit: `C:\Code\chinvex\src\chinvex\cli.py` (if needed for status command)
- Test: `C:\Code\chinvex\tests\test_status_embedding_display.py`

**Step 1: Write the failing test**

```python
# tests/test_status_embedding_display.py
import pytest
import json
from pathlib import Path
from datetime import datetime, timezone
from chinvex.cli_status import generate_status_from_contexts, ContextStatus


def test_status_shows_embedding_provider(tmp_path):
    """Status output should include embedding provider per context."""
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    # Create contexts with different embedding providers
    contexts_data = [
        ("Chinvex", "openai", "text-embedding-3-small", 1200),
        ("OldContext", "ollama", "mxbai-embed-large", 500),
        ("AnotherContext", "openai", "text-embedding-3-large", 800),
    ]

    for ctx_name, provider, model, chunks in contexts_data:
        ctx_dir = contexts_root / ctx_name
        ctx_dir.mkdir()

        # Create context.json with embedding config
        ctx_config = {
            "schema_version": 2,
            "name": ctx_name,
            "aliases": [],
            "includes": {"repos": [], "chat_roots": [], "codex_session_roots": [], "note_roots": []},
            "index": {"sqlite_path": f"path/to/{ctx_name}.db", "chroma_dir": f"path/to/{ctx_name}/chroma"},
            "weights": {"repo": 1.0, "chat": 0.8},
            "ollama": {"base_url": "http://localhost:11434", "embed_model": "mxbai-embed-large"},
            "embedding": {"provider": provider, "model": model},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        (ctx_dir / "context.json").write_text(json.dumps(ctx_config, indent=2))

        # Create STATUS.json
        status_data = {
            "context": ctx_name,
            "chunks": chunks,
            "last_sync": datetime.now(timezone.utc).isoformat(),
            "freshness": {
                "is_stale": False,
                "hours_since_sync": 2.5
            }
        }
        (ctx_dir / "STATUS.json").write_text(json.dumps(status_data, indent=2))

    # Generate status output
    output = generate_status_from_contexts(contexts_root)

    # Verify embedding provider is shown
    assert "openai" in output.lower()
    assert "ollama" in output.lower()
    assert "Chinvex" in output
    assert "OldContext" in output
    assert "AnotherContext" in output

    # Verify it's in a table format with headers
    assert "Embedding" in output or "Provider" in output


def test_status_legacy_context_shows_ollama_default(tmp_path):
    """Legacy contexts without embedding config should show 'ollama (default)'."""
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    ctx_dir = contexts_root / "LegacyContext"
    ctx_dir.mkdir()

    # Context without embedding field (schema v1 or early v2)
    ctx_config = {
        "schema_version": 1,
        "name": "LegacyContext",
        "aliases": [],
        "includes": {"repos": [], "chat_roots": [], "codex_session_roots": [], "note_roots": []},
        "index": {"sqlite_path": "path/to/db", "chroma_dir": "path/to/chroma"},
        "weights": {"repo": 1.0, "chat": 0.8},
        "ollama": {"base_url": "http://localhost:11434", "embed_model": "mxbai-embed-large"},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    (ctx_dir / "context.json").write_text(json.dumps(ctx_config, indent=2))

    status_data = {
        "context": "LegacyContext",
        "chunks": 300,
        "last_sync": datetime.now(timezone.utc).isoformat(),
        "freshness": {"is_stale": False, "hours_since_sync": 1.0}
    }
    (ctx_dir / "STATUS.json").write_text(json.dumps(status_data, indent=2))

    output = generate_status_from_contexts(contexts_root)

    # Should show ollama as default for legacy contexts
    assert "ollama" in output.lower()
    assert "LegacyContext" in output
```

**Step 2: Run test to verify it fails**

Run: `pwsh -Command "pytest C:\Code\chinvex\tests\test_status_embedding_display.py -v"`

Expected: FAIL with "AssertionError: 'Embedding' not found in output" or similar

**Step 3: Write minimal implementation**

```python
# src/chinvex/cli_status.py (COMPLETE FILE REPLACEMENT for clarity)
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ContextStatus:
    name: str
    chunks: int
    last_sync: str
    is_stale: bool
    hours_since_sync: float
    watcher_running: bool
    embedding_provider: str  # NEW FIELD
    embedding_model: str | None = None  # NEW FIELD


def format_status_output(contexts: list[ContextStatus], watcher_running: bool) -> str:
    """
    Format status output as table.

    Args:
        contexts: List of context statuses
        watcher_running: Whether watcher is running globally

    Returns:
        Formatted status string
    """
    lines = ["# Chinvex Global Status", ""]

    # Table header (UPDATED to include Embedding column)
    lines.append("| Context | Chunks | Last Sync | Embedding | Status |")
    lines.append("|---------|--------|-----------|-----------|--------|")

    # Rows
    for ctx in contexts:
        status_icon = "[OK]" if not ctx.is_stale else "[STALE]"
        hours_str = f"{int(ctx.hours_since_sync)}h ago"

        # Format embedding info
        if ctx.embedding_model:
            embed_str = f"{ctx.embedding_provider}/{ctx.embedding_model}"
        else:
            embed_str = ctx.embedding_provider

        # Truncate if too long
        if len(embed_str) > 18:
            embed_str = embed_str[:15] + "..."

        lines.append(
            f"| {ctx.name:<15} | {ctx.chunks:<6} | {hours_str:<9} | {embed_str:<17} | {status_icon:<6} |"
        )

    lines.append("")
    lines.append(f"Watcher: {'Running' if watcher_running else 'Stopped'}")

    return "\n".join(lines)


def read_global_status(contexts_root: Path) -> str:
    """
    Read GLOBAL_STATUS.md if it exists.

    Args:
        contexts_root: Root directory for contexts

    Returns:
        Contents of GLOBAL_STATUS.md
    """
    global_status = contexts_root / "GLOBAL_STATUS.md"
    if not global_status.exists():
        return "GLOBAL_STATUS.md not found. Run ingest to generate."

    return global_status.read_text(encoding="utf-8")


def generate_status_from_contexts(contexts_root: Path) -> str:
    """
    Generate status by reading all STATUS.json files.

    Args:
        contexts_root: Root directory for contexts

    Returns:
        Formatted status string
    """
    statuses = []

    for ctx_dir in contexts_root.iterdir():
        if not ctx_dir.is_dir():
            continue
        if ctx_dir.name.startswith("_"):
            continue

        status_json = ctx_dir / "STATUS.json"
        context_json = ctx_dir / "context.json"  # NEW: Read context.json for embedding info

        if not status_json.exists():
            continue

        try:
            data = json.loads(status_json.read_text(encoding="utf-8"))
            freshness = data.get("freshness", {})

            # NEW: Read embedding provider from context.json
            embedding_provider = "ollama"  # Default
            embedding_model = None

            if context_json.exists():
                try:
                    ctx_data = json.loads(context_json.read_text(encoding="utf-8"))
                    if "embedding" in ctx_data:
                        embedding_provider = ctx_data["embedding"].get("provider", "ollama")
                        embedding_model = ctx_data["embedding"].get("model")
                    # If no embedding field, check ollama config for model name
                    elif "ollama" in ctx_data:
                        embedding_model = ctx_data["ollama"].get("embed_model")
                except (json.JSONDecodeError, KeyError):
                    pass  # Use defaults

            statuses.append(ContextStatus(
                name=ctx_dir.name,
                chunks=data.get("chunks", 0),
                last_sync=data.get("last_sync", "unknown"),
                is_stale=freshness.get("is_stale", False),
                hours_since_sync=freshness.get("hours_since_sync", 0),
                watcher_running=False,  # Determined globally
                embedding_provider=embedding_provider,
                embedding_model=embedding_model
            ))
        except (json.JSONDecodeError, KeyError):
            continue

    # Check watcher status
    watcher_running = _check_watcher_running()

    return format_status_output(statuses, watcher_running)


def _check_watcher_running() -> bool:
    """Check if sync watcher is running via daemon state."""
    try:
        from .sync.cli import get_state_dir
        from .sync.daemon import DaemonManager, DaemonState

        state_dir = get_state_dir()
        dm = DaemonManager(state_dir)
        state = dm.get_state()

        return state == DaemonState.RUNNING
    except Exception:
        return False
```

**Step 4: Run test to verify it passes**

Run: `pwsh -Command "pytest C:\Code\chinvex\tests\test_status_embedding_display.py -v"`

Expected: PASS

**Step 5: Commit**

```bash
pwsh -Command "git add C:\Code\chinvex\src\chinvex\cli_status.py C:\Code\chinvex\tests\test_status_embedding_display.py; git commit -m 'feat(P5.1): show embedding provider in chinvex status output

- Add embedding_provider and embedding_model fields to ContextStatus
- Read embedding config from context.json when generating status
- Display embedding provider/model in status table
- Legacy contexts show ollama with model name
- Test coverage for OpenAI, Ollama, and legacy contexts

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>'"
```

---

## Task 8: Warn when querying Ollama contexts ("Consider migrating to OpenAI")

**Files:**
- Edit: `C:\Code\chinvex\src\chinvex\search.py`
- Test: `C:\Code\chinvex\tests\test_ollama_migration_warning.py`

**Step 1: Write the failing test**

```python
# tests/test_ollama_migration_warning.py
import pytest
from pathlib import Path
import json
from datetime import datetime, timezone
from chinvex.context import load_context, ContextConfig
from chinvex.search import search_context
import sys
from io import StringIO


def test_ollama_context_shows_migration_warning(tmp_path, monkeypatch, capsys):
    """Searching an Ollama context should print migration warning to stderr."""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"
    contexts_root.mkdir()
    indexes_root.mkdir()

    # Create context with Ollama embeddings
    ctx_dir = contexts_root / "OllamaContext"
    ctx_dir.mkdir()
    ctx_index = indexes_root / "OllamaContext"
    ctx_index.mkdir()

    # Create minimal index
    from chinvex.storage import Storage
    db_path = ctx_index / "hybrid.db"
    storage = Storage(db_path)
    storage.ensure_schema()
    storage.close()

    ctx_data = {
        "schema_version": 2,
        "name": "OllamaContext",
        "aliases": [],
        "includes": {"repos": [], "chat_roots": [], "codex_session_roots": [], "note_roots": []},
        "index": {"sqlite_path": str(db_path), "chroma_dir": str(ctx_index / "chroma")},
        "weights": {"repo": 1.0, "chat": 0.8},
        "ollama": {"base_url": "http://localhost:11434", "embed_model": "mxbai-embed-large"},
        "embedding": {"provider": "ollama", "model": "mxbai-embed-large"},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    (ctx_dir / "context.json").write_text(json.dumps(ctx_data, indent=2))

    # Load context
    ctx = load_context("OllamaContext", contexts_root)

    # Capture stderr
    old_stderr = sys.stderr
    sys.stderr = StringIO()

    try:
        # Attempt search (will fail on embedding, but warning should print first)
        try:
            search_context(
                ctx=ctx,
                query="test query",
                k=5
            )
        except Exception:
            pass  # Expected to fail on actual embedding

        # Get stderr output
        stderr_output = sys.stderr.getvalue()
    finally:
        sys.stderr = old_stderr

    # Verify warning message
    assert "ollama" in stderr_output.lower()
    assert "openai" in stderr_output.lower()
    assert "migrat" in stderr_output.lower() or "consider" in stderr_output.lower()


def test_openai_context_no_migration_warning(tmp_path, capsys):
    """Searching an OpenAI context should not print migration warning."""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"
    contexts_root.mkdir()
    indexes_root.mkdir()

    # Create context with OpenAI embeddings
    ctx_dir = contexts_root / "OpenAIContext"
    ctx_dir.mkdir()
    ctx_index = indexes_root / "OpenAIContext"
    ctx_index.mkdir()

    from chinvex.storage import Storage
    db_path = ctx_index / "hybrid.db"
    storage = Storage(db_path)
    storage.ensure_schema()
    storage.close()

    ctx_data = {
        "schema_version": 2,
        "name": "OpenAIContext",
        "aliases": [],
        "includes": {"repos": [], "chat_roots": [], "codex_session_roots": [], "note_roots": []},
        "index": {"sqlite_path": str(db_path), "chroma_dir": str(ctx_index / "chroma")},
        "weights": {"repo": 1.0, "chat": 0.8},
        "ollama": {"base_url": "http://localhost:11434", "embed_model": "mxbai-embed-large"},
        "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    (ctx_dir / "context.json").write_text(json.dumps(ctx_data, indent=2))

    ctx = load_context("OpenAIContext", contexts_root)

    # Capture stderr
    old_stderr = sys.stderr
    sys.stderr = StringIO()

    try:
        try:
            search_context(ctx=ctx, query="test", k=5)
        except Exception:
            pass

        stderr_output = sys.stderr.getvalue()
    finally:
        sys.stderr = old_stderr

    # Should NOT contain migration warning
    migration_keywords = ["migrat", "consider switching", "consider moving"]
    has_migration_warning = any(kw in stderr_output.lower() for kw in migration_keywords)
    assert not has_migration_warning
```

**Step 2: Run test to verify it fails**

Run: `pwsh -Command "pytest C:\Code\chinvex\tests\test_ollama_migration_warning.py -v"`

Expected: FAIL with "AssertionError: 'ollama' not found in stderr_output"

**Step 3: Write minimal implementation**

```python
# src/chinvex/search.py
import sys  # Add at top of file if not already imported


def search_context(
    ctx,
    query: str,
    *,
    k: int = 8,
    min_score: float = 0.35,
    source: str = "all",
    project: str | None = None,
    repo: str | None = None,
    ollama_host_override: str | None = None,
    recency_enabled: bool = True,
) -> list[SearchResult]:
    """
    Search within a context using context-aware weights.
    """
    from .context import ContextConfig

    # NEW: Warn if context uses Ollama embeddings
    if ctx.embedding and ctx.embedding.provider == "ollama":
        print(
            f"Warning: Context '{ctx.name}' uses Ollama embeddings. "
            f"Consider migrating to OpenAI for faster and more consistent results. "
            f"Run: chinvex ingest --context {ctx.name} --embed-provider openai --rebuild-index",
            file=sys.stderr
        )
    elif not ctx.embedding:
        # Legacy context without embedding field defaults to Ollama
        print(
            f"Warning: Context '{ctx.name}' uses Ollama embeddings (legacy default). "
            f"Consider migrating to OpenAI for faster and more consistent results. "
            f"Run: chinvex ingest --context {ctx.name} --embed-provider openai --rebuild-index",
            file=sys.stderr
        )

    db_path = ctx.index.sqlite_path
    chroma_dir = ctx.index.chroma_dir

    storage = Storage(db_path)
    storage.ensure_schema()

    # Use Ollama config from context with localhost fallback
    ollama_host = ollama_host_override or ctx.ollama.base_url
    embedding_model = ctx.ollama.embed_model
    fallback_host = "http://127.0.0.1:11434" if ollama_host != "http://127.0.0.1:11434" else None

    embedder = OllamaEmbedder(ollama_host, embedding_model, fallback_host=fallback_host)
    vectors = VectorStore(chroma_dir)

    # Use context weights for source-type prioritization
    scored = search_chunks(
        storage,
        vectors,
        embedder,
        query,
        k=k,
        min_score=min_score,
        source=source,
        project=project,
        repo=repo,
        weights=ctx.weights,
    )

    results = [
        SearchResult(
            chunk_id=item.chunk_id,
            score=item.score,
            source_type=item.row["source_type"],
            title=_title_from_row(item.row),
            citation=_citation_from_row(item.row),
            snippet=make_snippet(item.row["text"], limit=200),
        )
        for item in scored
    ]

    storage.close()
    return results[:k]
```

**Step 4: Run test to verify it passes**

Run: `pwsh -Command "pytest C:\Code\chinvex\tests\test_ollama_migration_warning.py -v"`

Expected: PASS

**Step 5: Commit**

```bash
pwsh -Command "git add C:\Code\chinvex\src\chinvex\search.py C:\Code\chinvex\tests\test_ollama_migration_warning.py; git commit -m 'feat(P5.1): warn when querying Ollama contexts

- Add migration warning to stderr when searching Ollama contexts
- Suggest chinvex ingest --embed-provider openai --rebuild-index
- Warn for both explicit Ollama configs and legacy defaults
- No warning for OpenAI contexts
- Test coverage for Ollama and OpenAI scenarios

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>'"
```

---
## Task 9: Update `chinvex brief` to use exact header matching for CONSTRAINTS.md

**Files:**
- Edit: `C:\Code\chinvex\src\chinvex\brief.py`
- Test: `C:\Code\chinvex\tests\test_brief.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_brief.py

def test_constraints_exact_headers_only(tmp_path):
    """Test brief extracts only Infrastructure, Rules, Hazards sections from CONSTRAINTS.md."""
    constraints_md = tmp_path / "CONSTRAINTS.md"
    constraints_md.write_text("""# Constraints

## Infrastructure
- ChromaDB batch limit: 5000
- Gateway port: 7778

## Rules
- Schema stays v2
- No migrations

## Key Facts
- Gateway: localhost:7778
- Token env var: CHINVEX_API_TOKEN

## Hazards
- Batch size exceeded = silent failure
- Don't delete meta.json

## Security
- API keys in env only
""")

    output = tmp_path / "SESSION_BRIEF.md"
    from chinvex.brief import generate_brief

    generate_brief(
        context_name="TestContext",
        state_md=None,
        constraints_md=constraints_md,
        decisions_md=None,
        latest_digest=None,
        watch_history_log=None,
        output=output
    )

    content = output.read_text()

    # Should include exact headers
    assert "## Infrastructure" in content
    assert "## Rules" in content
    assert "## Hazards" in content

    # Should include content under exact headers
    assert "ChromaDB batch limit: 5000" in content
    assert "Schema stays v2" in content
    assert "Batch size exceeded = silent failure" in content

    # Should NOT include Key Facts or Security (not in exact header list)
    assert "## Key Facts" not in content
    assert "Token env var: CHINVEX_API_TOKEN" not in content
    assert "## Security" not in content
    assert "API keys in env only" not in content


def test_constraints_missing_some_exact_headers(tmp_path):
    """Test brief gracefully handles CONSTRAINTS.md missing some exact headers."""
    constraints_md = tmp_path / "CONSTRAINTS.md"
    constraints_md.write_text("""# Constraints

## Infrastructure
- Gateway port: 7778

## Key Facts
- This should not appear
""")

    output = tmp_path / "SESSION_BRIEF.md"
    from chinvex.brief import generate_brief

    generate_brief(
        context_name="TestContext",
        state_md=None,
        constraints_md=constraints_md,
        decisions_md=None,
        latest_digest=None,
        watch_history_log=None,
        output=output
    )

    content = output.read_text()

    # Should include Infrastructure
    assert "## Infrastructure" in content
    assert "Gateway port: 7778" in content

    # Should NOT include Key Facts
    assert "## Key Facts" not in content
    assert "This should not appear" not in content
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_brief.py::test_constraints_exact_headers_only -v`

Expected: FAIL with assertion error showing "## Key Facts" or "Token env var" appears in content

**Step 3: Write minimal implementation**

```python
# Edit src/chinvex/brief.py

def _extract_constraints_top(constraints_md: Path) -> list[str]:
    """Extract only Infrastructure, Rules, and Hazards sections from CONSTRAINTS.md."""
    content = constraints_md.read_text()
    lines = content.split("\n")

    # Exact headers to extract
    target_headers = {"## Infrastructure", "## Rules", "## Hazards"}

    result = []
    in_target_section = False
    current_section_lines = []

    for line in lines:
        if line.startswith("# Constraints"):
            continue  # Skip title

        if line.startswith("## "):
            # Save previous section if it was a target
            if in_target_section and current_section_lines:
                result.extend(current_section_lines)
                result.append("")  # Blank line between sections

            # Check if this is a target section
            if line in target_headers:
                in_target_section = True
                current_section_lines = [line]
            else:
                in_target_section = False
                current_section_lines = []
        elif in_target_section:
            current_section_lines.append(line)

    # Save last section if it was a target
    if in_target_section and current_section_lines:
        result.extend(current_section_lines)

    return result
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_brief.py::test_constraints_exact_headers_only tests/test_brief.py::test_constraints_missing_some_exact_headers -v`

Expected: PASS (both tests pass)

**Step 5: Commit**

```bash
git add src/chinvex/brief.py tests/test_brief.py
git commit -m "feat(brief): use exact header matching for CONSTRAINTS.md

Update _extract_constraints_top to extract only Infrastructure, Rules,
and Hazards sections per P5.2.2 spec. No longer fuzzy matches or
includes all sections until first ##.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Update `chinvex brief` to include DECISIONS.md Recent rollup + last 7 days

**Files:**
- Edit: `C:\Code\chinvex\src\chinvex\brief.py`
- Test: `C:\Code\chinvex\tests\test_brief.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_brief.py

def test_decisions_recent_rollup_section(tmp_path):
    """Test brief includes Recent rollup from DECISIONS.md."""
    recent_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

    decisions_md = tmp_path / "DECISIONS.md"
    decisions_md.write_text(f"""# Decisions

## Recent (last 30 days)
- Switched to OpenAI embeddings for speed
- Fixed ChromaDB batch limit issue
- Implemented cross-context search

## 2026-01

### {recent_date} — Recent decision

- **Why:** Testing
- **Impact:** Should appear in dated section
- **Evidence:** commit abc123
""")

    output = tmp_path / "SESSION_BRIEF.md"
    from chinvex.brief import generate_brief

    generate_brief(
        context_name="TestContext",
        state_md=None,
        constraints_md=None,
        decisions_md=decisions_md,
        latest_digest=None,
        watch_history_log=None,
        output=output
    )

    content = output.read_text()

    # Should include Recent rollup section
    assert "## Recent (last 30 days)" in content
    assert "Switched to OpenAI embeddings for speed" in content
    assert "Fixed ChromaDB batch limit issue" in content

    # Should also include dated entries from last 7 days
    assert "Recent decision" in content
    assert "Should appear in dated section" in content


def test_decisions_rollup_plus_last_7_days(tmp_path):
    """Test brief includes both rollup and dated entries from last 7 days."""
    recent_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    old_date = (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d")

    decisions_md = tmp_path / "DECISIONS.md"
    decisions_md.write_text(f"""# Decisions

## Recent (last 30 days)
- Summary line 1
- Summary line 2

## 2026-01

### {recent_date} — Recent dated entry

- **Why:** Testing recent
- **Impact:** Should appear
- **Evidence:** commit abc123

### {old_date} — Old dated entry

- **Why:** Testing old
- **Impact:** Should NOT appear (>7 days)
- **Evidence:** commit def456
""")

    output = tmp_path / "SESSION_BRIEF.md"
    from chinvex.brief import generate_brief

    generate_brief(
        context_name="TestContext",
        state_md=None,
        constraints_md=None,
        decisions_md=decisions_md,
        latest_digest=None,
        watch_history_log=None,
        output=output
    )

    content = output.read_text()

    # Rollup section
    assert "## Recent (last 30 days)" in content
    assert "Summary line 1" in content

    # Recent dated entry (within 7 days)
    assert "Recent dated entry" in content
    assert "Should appear" in content

    # Old dated entry should NOT appear
    assert "Old dated entry" not in content
    assert "Should NOT appear" not in content


def test_decisions_no_rollup_section(tmp_path):
    """Test brief handles DECISIONS.md without Recent rollup section."""
    recent_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    decisions_md = tmp_path / "DECISIONS.md"
    decisions_md.write_text(f"""# Decisions

## 2026-01

### {recent_date} — Decision without rollup

- **Why:** No rollup section exists
- **Impact:** Should still appear
- **Evidence:** commit abc123
""")

    output = tmp_path / "SESSION_BRIEF.md"
    from chinvex.brief import generate_brief

    generate_brief(
        context_name="TestContext",
        state_md=None,
        constraints_md=None,
        decisions_md=decisions_md,
        latest_digest=None,
        watch_history_log=None,
        output=output
    )

    content = output.read_text()

    # Should include dated entry even without rollup
    assert "Decision without rollup" in content
    assert "Should still appear" in content
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_brief.py::test_decisions_recent_rollup_section -v`

Expected: FAIL with assertion error showing rollup section is missing

**Step 3: Write minimal implementation**

```python
# Edit src/chinvex/brief.py

def _extract_recent_decisions(decisions_md: Path, days: int) -> list[str]:
    """Extract Recent rollup section + dated entries from last N days."""
    content = decisions_md.read_text()
    lines = content.split("\n")

    cutoff_date = datetime.now() - timedelta(days=days)
    result = []

    # Extract Recent rollup section first
    in_recent_rollup = False
    rollup_lines = []

    for i, line in enumerate(lines):
        if line.strip() == "## Recent (last 30 days)":
            in_recent_rollup = True
            rollup_lines.append(line)
            continue

        if in_recent_rollup:
            # Stop at next ## heading or ### entry
            if line.startswith("## ") or line.startswith("### "):
                break
            rollup_lines.append(line)

    # Add rollup section if found
    if rollup_lines:
        result.extend(rollup_lines)
        result.append("")

    # Extract dated entries from last N days
    current_decision = []
    current_date = None

    for line in lines:
        # Match decision heading: ### YYYY-MM-DD — Title
        match = re.match(r"^### (\d{4}-\d{2}-\d{2}) — (.+)", line)
        if match:
            date_str, title = match.groups()
            decision_date = datetime.strptime(date_str, "%Y-%m-%d")

            # Save previous decision if within window
            if current_decision and current_date and current_date >= cutoff_date:
                result.extend(current_decision)
                result.append("")

            # Start new decision
            current_date = decision_date
            current_decision = [line]
        elif current_decision:
            current_decision.append(line)

    # Save last decision
    if current_decision and current_date and current_date >= cutoff_date:
        result.extend(current_decision)

    return result
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_brief.py::test_decisions_recent_rollup_section tests/test_brief.py::test_decisions_rollup_plus_last_7_days tests/test_brief.py::test_decisions_no_rollup_section -v`

Expected: PASS (all three tests pass)

**Step 5: Commit**

```bash
git add src/chinvex/brief.py tests/test_brief.py
git commit -m "feat(brief): include DECISIONS.md Recent rollup + last 7 days

Update _extract_recent_decisions to extract both the Recent rollup
section and dated entries from the last 7 days, per P5.2.2 spec.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 11: Implement morning brief active/stale context detection (7-day threshold)

**Files:**
- Create: `C:\Code\chinvex\src\chinvex\morning_brief.py`
- Test: `C:\Code\chinvex\tests\test_morning_brief.py`

**Step 1: Write the failing test**

```python
# Create tests/test_morning_brief.py

import json
import pytest
from pathlib import Path
from datetime import datetime, timedelta


def test_detect_active_contexts_7_day_threshold(tmp_path):
    """Test active context detection using 7-day threshold."""
    from chinvex.morning_brief import detect_active_stale_contexts

    # Create contexts with different last_sync timestamps
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    # Active: 3 days ago
    active_ctx = contexts_root / "ActiveContext"
    active_ctx.mkdir()
    (active_ctx / "STATUS.json").write_text(json.dumps({
        "context": "ActiveContext",
        "chunks": 100,
        "last_sync": (datetime.now() - timedelta(days=3)).isoformat()
    }))

    # Stale: 10 days ago
    stale_ctx = contexts_root / "StaleContext"
    stale_ctx.mkdir()
    (stale_ctx / "STATUS.json").write_text(json.dumps({
        "context": "StaleContext",
        "chunks": 50,
        "last_sync": (datetime.now() - timedelta(days=10)).isoformat()
    }))

    # Edge case: exactly 7 days ago (should be active)
    edge_ctx = contexts_root / "EdgeContext"
    edge_ctx.mkdir()
    (edge_ctx / "STATUS.json").write_text(json.dumps({
        "context": "EdgeContext",
        "chunks": 75,
        "last_sync": (datetime.now() - timedelta(days=7, seconds=-60)).isoformat()
    }))

    active, stale = detect_active_stale_contexts(contexts_root)

    assert len(active) == 2
    assert len(stale) == 1

    active_names = {ctx["context"] for ctx in active}
    assert "ActiveContext" in active_names
    assert "EdgeContext" in active_names

    stale_names = {ctx["context"] for ctx in stale}
    assert "StaleContext" in stale_names


def test_detect_contexts_sorted_by_last_sync(tmp_path):
    """Test active contexts are sorted by last_sync (most recent first)."""
    from chinvex.morning_brief import detect_active_stale_contexts

    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    # Create 3 active contexts with different timestamps
    for i, days_ago in [(1, 1), (2, 5), (3, 2)]:
        ctx_dir = contexts_root / f"Context{i}"
        ctx_dir.mkdir()
        (ctx_dir / "STATUS.json").write_text(json.dumps({
            "context": f"Context{i}",
            "chunks": 100,
            "last_sync": (datetime.now() - timedelta(days=days_ago)).isoformat()
        }))

    active, stale = detect_active_stale_contexts(contexts_root)

    assert len(active) == 3
    assert active[0]["context"] == "Context1"  # 1 day ago (most recent)
    assert active[1]["context"] == "Context3"  # 2 days ago
    assert active[2]["context"] == "Context2"  # 5 days ago


def test_detect_contexts_missing_last_sync(tmp_path):
    """Test contexts without last_sync are treated as stale."""
    from chinvex.morning_brief import detect_active_stale_contexts

    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    # Context with no last_sync field
    no_sync_ctx = contexts_root / "NoSyncContext"
    no_sync_ctx.mkdir()
    (no_sync_ctx / "STATUS.json").write_text(json.dumps({
        "context": "NoSyncContext",
        "chunks": 100
    }))

    active, stale = detect_active_stale_contexts(contexts_root)

    assert len(active) == 0
    assert len(stale) == 1
    assert stale[0]["context"] == "NoSyncContext"


def test_detect_contexts_cap_at_top_5(tmp_path):
    """Test active contexts are capped at top 5 by recent activity."""
    from chinvex.morning_brief import detect_active_stale_contexts

    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    # Create 8 active contexts
    for i in range(8):
        ctx_dir = contexts_root / f"Context{i}"
        ctx_dir.mkdir()
        (ctx_dir / "STATUS.json").write_text(json.dumps({
            "context": f"Context{i}",
            "chunks": 100,
            "last_sync": (datetime.now() - timedelta(days=i+1)).isoformat()
        }))

    active, stale = detect_active_stale_contexts(contexts_root, max_active=5)

    # Should return only top 5 most recent
    assert len(active) == 5
    assert active[0]["context"] == "Context0"  # 1 day ago
    assert active[4]["context"] == "Context4"  # 5 days ago

    # Context5-7 are still "active" but not in the top 5, not stale
    assert len(stale) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_morning_brief.py::test_detect_active_contexts_7_day_threshold -v`

Expected: FAIL with ModuleNotFoundError or ImportError for chinvex.morning_brief

**Step 3: Write minimal implementation**

```python
# Create src/chinvex/morning_brief.py

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path


def detect_active_stale_contexts(
    contexts_root: Path,
    max_active: int = 5,
    active_threshold_days: int = 7
) -> tuple[list[dict], list[dict]]:
    """
    Detect active and stale contexts based on last_sync timestamp.

    Args:
        contexts_root: Path to contexts directory
        max_active: Maximum number of active contexts to return (sorted by recency)
        active_threshold_days: Days threshold for active vs stale

    Returns:
        (active_contexts, stale_contexts) where each is a list of status dicts
    """
    cutoff = datetime.now() - timedelta(days=active_threshold_days)

    all_contexts = []

    # Collect all STATUS.json files
    for ctx_dir in contexts_root.iterdir():
        if not ctx_dir.is_dir():
            continue

        status_file = ctx_dir / "STATUS.json"
        if not status_file.exists():
            continue

        try:
            status = json.loads(status_file.read_text())
            all_contexts.append(status)
        except (json.JSONDecodeError, OSError):
            continue

    # Separate into active and stale
    active = []
    stale = []

    for status in all_contexts:
        last_sync_str = status.get("last_sync")

        if not last_sync_str:
            stale.append(status)
            continue

        try:
            last_sync = datetime.fromisoformat(last_sync_str.replace("Z", "+00:00"))
            # Remove timezone info for comparison
            if last_sync.tzinfo:
                last_sync = last_sync.replace(tzinfo=None)

            if last_sync >= cutoff:
                active.append(status)
            else:
                stale.append(status)
        except (ValueError, AttributeError):
            stale.append(status)

    # Sort active by last_sync (most recent first)
    active.sort(key=lambda s: s.get("last_sync", ""), reverse=True)

    # Cap at max_active
    active = active[:max_active]

    return active, stale
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_morning_brief.py -v`

Expected: PASS (all tests pass)

**Step 5: Commit**

```bash
git add src/chinvex/morning_brief.py tests/test_morning_brief.py
git commit -m "feat(morning-brief): implement active/stale context detection

Add detect_active_stale_contexts function with 7-day threshold and
top-5 sorting by last_sync timestamp, per P5.2.3 spec.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 12: Parse STATE.md to extract Current Objective and Next Actions

**Files:**
- Edit: `C:\Code\chinvex\src\chinvex\morning_brief.py`
- Test: `C:\Code\chinvex\tests\test_morning_brief.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_morning_brief.py

def test_parse_state_md_objective_and_actions(tmp_path):
    """Test extracting Current Objective and Next Actions from STATE.md."""
    from chinvex.morning_brief import parse_state_md

    state_md = tmp_path / "STATE.md"
    state_md.write_text("""# State

## Current Objective
P5 implementation - reliability and retrieval quality

## Active Work
- Brief generation updates
- Morning brief overhaul

## Blockers
None

## Next Actions
- [ ] Update CONSTRAINTS.md extraction
- [ ] Add Recent rollup to DECISIONS.md
- [ ] Implement active/stale detection
- [ ] Parse STATE.md for objectives
- [ ] Format morning brief with ntfy

## Out of Scope (for now)
- Multi-user auth
- Smart scheduling agent
""")

    objective, actions = parse_state_md(state_md)

    assert objective == "P5 implementation - reliability and retrieval quality"
    assert len(actions) == 5
    assert "Update CONSTRAINTS.md extraction" in actions
    assert "Format morning brief with ntfy" in actions


def test_parse_state_md_max_5_actions(tmp_path):
    """Test Next Actions are capped at 5 bullets."""
    from chinvex.morning_brief import parse_state_md

    state_md = tmp_path / "STATE.md"
    state_md.write_text("""# State

## Current Objective
Test objective

## Next Actions
- [ ] Action 1
- [ ] Action 2
- [ ] Action 3
- [ ] Action 4
- [ ] Action 5
- [ ] Action 6
- [ ] Action 7
- [ ] Action 8
""")

    objective, actions = parse_state_md(state_md, max_actions=5)

    assert objective == "Test objective"
    assert len(actions) == 5
    assert "Action 1" in actions
    assert "Action 5" in actions
    assert "Action 6" not in actions


def test_parse_state_md_missing_sections(tmp_path):
    """Test graceful handling when sections are missing."""
    from chinvex.morning_brief import parse_state_md

    state_md = tmp_path / "STATE.md"
    state_md.write_text("""# State

## Active Work
- Some work

## Blockers
None
""")

    objective, actions = parse_state_md(state_md)

    assert objective is None
    assert actions == []


def test_parse_state_md_file_not_exists(tmp_path):
    """Test handling when STATE.md doesn't exist."""
    from chinvex.morning_brief import parse_state_md

    state_md = tmp_path / "nonexistent.md"

    objective, actions = parse_state_md(state_md)

    assert objective is None
    assert actions == []


def test_parse_state_md_multiline_objective(tmp_path):
    """Test objective extraction takes only first line."""
    from chinvex.morning_brief import parse_state_md

    state_md = tmp_path / "STATE.md"
    state_md.write_text("""# State

## Current Objective
First line is the objective
Second line should be ignored
Third line too

## Next Actions
- [ ] Action 1
""")

    objective, actions = parse_state_md(state_md)

    assert objective == "First line is the objective"
    assert len(actions) == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_morning_brief.py::test_parse_state_md_objective_and_actions -v`

Expected: FAIL with AttributeError or ImportError for parse_state_md

**Step 3: Write minimal implementation**

```python
# Edit src/chinvex/morning_brief.py

import re


def parse_state_md(
    state_md_path: Path,
    max_actions: int = 5
) -> tuple[str | None, list[str]]:
    """
    Parse STATE.md to extract Current Objective and Next Actions.

    Args:
        state_md_path: Path to STATE.md file
        max_actions: Maximum number of actions to return

    Returns:
        (objective, actions) where objective is a string or None,
        and actions is a list of action strings (without checkbox prefix)
    """
    if not state_md_path.exists():
        return None, []

    try:
        content = state_md_path.read_text()
    except OSError:
        return None, []

    lines = content.split("\n")

    objective = None
    actions = []

    in_objective_section = False
    in_actions_section = False

    for line in lines:
        # Detect sections
        if line.strip() == "## Current Objective":
            in_objective_section = True
            in_actions_section = False
            continue

        if line.strip() == "## Next Actions":
            in_objective_section = False
            in_actions_section = True
            continue

        # Stop at next section
        if line.startswith("## "):
            in_objective_section = False
            in_actions_section = False
            continue

        # Extract objective (first non-empty line after header)
        if in_objective_section and not objective:
            stripped = line.strip()
            if stripped:
                objective = stripped
                in_objective_section = False  # Only take first line

        # Extract actions (checkbox items)
        if in_actions_section:
            # Match "- [ ] Action text" or "- [x] Action text"
            match = re.match(r"^-\s*\[[ xX]\]\s*(.+)", line.strip())
            if match:
                action_text = match.group(1).strip()
                actions.append(action_text)

                if len(actions) >= max_actions:
                    break

    return objective, actions
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_morning_brief.py::test_parse_state_md_objective_and_actions tests/test_morning_brief.py::test_parse_state_md_max_5_actions tests/test_morning_brief.py::test_parse_state_md_missing_sections tests/test_morning_brief.py::test_parse_state_md_file_not_exists tests/test_morning_brief.py::test_parse_state_md_multiline_objective -v`

Expected: PASS (all tests pass)

**Step 5: Commit**

```bash
git add src/chinvex/morning_brief.py tests/test_morning_brief.py
git commit -m "feat(morning-brief): parse STATE.md for objective and actions

Add parse_state_md function to extract Current Objective (first line)
and Next Actions (max 5 bullets) from STATE.md, per P5.2.3 spec.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 13: Format morning brief with objectives, next actions, and ntfy integration

**Files:**
- Edit: `C:\Code\chinvex\src\chinvex\morning_brief.py`
- Edit: `C:\Code\chinvex\scripts\morning_brief.ps1`
- Test: `C:\Code\chinvex\tests\test_morning_brief.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_morning_brief.py

def test_generate_morning_brief_with_objectives(tmp_path):
    """Test morning brief includes objectives and next actions from STATE.md."""
    from chinvex.morning_brief import generate_morning_brief

    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    # Active context with STATE.md
    ctx1 = contexts_root / "ProjectAlpha"
    ctx1.mkdir()
    (ctx1 / "STATUS.json").write_text(json.dumps({
        "context": "ProjectAlpha",
        "chunks": 150,
        "last_sync": (datetime.now() - timedelta(days=2)).isoformat()
    }))

    # Create docs/memory/STATE.md (simulate repo structure)
    docs_memory = ctx1 / "docs" / "memory"
    docs_memory.mkdir(parents=True)
    (docs_memory / "STATE.md").write_text("""# State

## Current Objective
Implement P5 reliability features

## Next Actions
- [ ] Add embedding integrity checks
- [ ] Update brief generation
- [ ] Test cross-context search
""")

    # Active context without STATE.md
    ctx2 = contexts_root / "ProjectBeta"
    ctx2.mkdir()
    (ctx2 / "STATUS.json").write_text(json.dumps({
        "context": "ProjectBeta",
        "chunks": 75,
        "last_sync": (datetime.now() - timedelta(days=1)).isoformat()
    }))

    # Stale context
    ctx3 = contexts_root / "ProjectGamma"
    ctx3.mkdir()
    (ctx3 / "STATUS.json").write_text(json.dumps({
        "context": "ProjectGamma",
        "chunks": 50,
        "last_sync": (datetime.now() - timedelta(days=10)).isoformat()
    }))

    output = tmp_path / "MORNING_BRIEF.md"

    brief_text, ntfy_body = generate_morning_brief(contexts_root, output)

    # Check output file
    assert output.exists()
    content = output.read_text()

    # System Health section (compact)
    assert "## System Health" in content
    assert "Contexts: 3" in content
    assert "1 stale" in content

    # Active Projects section with objectives
    assert "## Active Projects" in content
    assert "### ProjectAlpha" in content
    assert "**Objective:** Implement P5 reliability features" in content
    assert "**Next Actions:**" in content
    assert "Add embedding integrity checks" in content
    assert "Update brief generation" in content

    # ProjectBeta should appear but without objective (no STATE.md)
    assert "### ProjectBeta" in content
    assert "No STATE.md found" in content or "Objective:" not in content.split("### ProjectBeta")[1].split("###")[0]

    # Stale Contexts section
    assert "## Stale Contexts" in content
    assert "ProjectGamma" in content
    assert "hours since sync" in content or "days since sync" in content

    # ntfy body should include top objective
    assert "ProjectAlpha" in ntfy_body
    assert "Implement P5 reliability features" in ntfy_body


def test_generate_morning_brief_no_stale_contexts(tmp_path):
    """Test morning brief when all contexts are active."""
    from chinvex.morning_brief import generate_morning_brief

    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    # Only active contexts
    for i in range(2):
        ctx = contexts_root / f"Context{i}"
        ctx.mkdir()
        (ctx / "STATUS.json").write_text(json.dumps({
            "context": f"Context{i}",
            "chunks": 100,
            "last_sync": (datetime.now() - timedelta(days=i+1)).isoformat()
        }))

    output = tmp_path / "MORNING_BRIEF.md"

    brief_text, ntfy_body = generate_morning_brief(contexts_root, output)

    content = output.read_text()

    # Should not have Stale Contexts section
    assert "## Stale Contexts" not in content or "None" in content.split("## Stale Contexts")[1].split("##")[0]


def test_generate_morning_brief_cap_5_active(tmp_path):
    """Test morning brief caps active contexts at 5."""
    from chinvex.morning_brief import generate_morning_brief

    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    # Create 8 active contexts
    for i in range(8):
        ctx = contexts_root / f"Context{i}"
        ctx.mkdir()
        (ctx / "STATUS.json").write_text(json.dumps({
            "context": f"Context{i}",
            "chunks": 100,
            "last_sync": (datetime.now() - timedelta(days=i+1)).isoformat()
        }))

    output = tmp_path / "MORNING_BRIEF.md"

    brief_text, ntfy_body = generate_morning_brief(contexts_root, output)

    content = output.read_text()

    # Should only show top 5
    assert "### Context0" in content
    assert "### Context4" in content
    assert "### Context5" not in content
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_morning_brief.py::test_generate_morning_brief_with_objectives -v`

Expected: FAIL with AttributeError or ImportError for generate_morning_brief

**Step 3: Write minimal implementation**

```python
# Edit src/chinvex/morning_brief.py

def generate_morning_brief(
    contexts_root: Path,
    output_path: Path
) -> tuple[str, str]:
    """
    Generate morning brief with active project objectives.

    Args:
        contexts_root: Path to contexts directory
        output_path: Path to write MORNING_BRIEF.md

    Returns:
        (brief_markdown, ntfy_body) where ntfy_body is suitable for push notification
    """
    active_contexts, stale_contexts = detect_active_stale_contexts(contexts_root)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Morning Brief",
        f"Generated: {timestamp}",
        "",
        "## System Health",
        f"- Contexts: {len(active_contexts) + len(stale_contexts)} ({len(stale_contexts)} stale)",
        "",
    ]

    # Active Projects section
    if active_contexts:
        lines.append("## Active Projects")

        for ctx_status in active_contexts:
            ctx_name = ctx_status["context"]
            lines.append(f"### {ctx_name}")

            # Try to find STATE.md in context directory
            ctx_dir = contexts_root / ctx_name
            state_md_path = ctx_dir / "docs" / "memory" / "STATE.md"

            objective, actions = parse_state_md(state_md_path)

            if objective:
                lines.append(f"**Objective:** {objective}")
            else:
                lines.append("**Objective:** (no STATE.md)")

            if actions:
                lines.append("**Next Actions:**")
                for action in actions:
                    lines.append(f"- [ ] {action}")

            lines.append("")

    # Stale Contexts section
    if stale_contexts:
        lines.append("## Stale Contexts")

        for ctx_status in stale_contexts:
            ctx_name = ctx_status["context"]
            last_sync_str = ctx_status.get("last_sync", "unknown")

            if last_sync_str != "unknown":
                try:
                    last_sync = datetime.fromisoformat(last_sync_str.replace("Z", "+00:00"))
                    if last_sync.tzinfo:
                        last_sync = last_sync.replace(tzinfo=None)

                    hours_ago = (datetime.now() - last_sync).total_seconds() / 3600

                    if hours_ago < 48:
                        time_desc = f"{int(hours_ago)} hours since sync"
                    else:
                        days_ago = int(hours_ago / 24)
                        time_desc = f"{days_ago} days since sync"
                except (ValueError, AttributeError):
                    time_desc = "unknown"
            else:
                time_desc = "never synced"

            lines.append(f"- **{ctx_name}**: {time_desc}")

        lines.append("")

    brief_text = "\n".join(lines)

    # Generate ntfy body (first 1-2 objectives)
    ntfy_lines = []
    ntfy_lines.append(f"Contexts: {len(active_contexts)} active, {len(stale_contexts)} stale")

    if active_contexts:
        ntfy_lines.append("")
        for i, ctx_status in enumerate(active_contexts[:2]):
            ctx_name = ctx_status["context"]
            ctx_dir = contexts_root / ctx_name
            state_md_path = ctx_dir / "docs" / "memory" / "STATE.md"

            objective, _ = parse_state_md(state_md_path)

            if objective:
                ntfy_lines.append(f"{ctx_name}: {objective}")

    ntfy_body = "\n".join(ntfy_lines)

    # Write output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(brief_text)

    return brief_text, ntfy_body
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_morning_brief.py::test_generate_morning_brief_with_objectives tests/test_morning_brief.py::test_generate_morning_brief_no_stale_contexts tests/test_morning_brief.py::test_generate_morning_brief_cap_5_active -v`

Expected: PASS (all tests pass)

**Step 5: Commit**

```bash
git add src/chinvex/morning_brief.py tests/test_morning_brief.py scripts/morning_brief.ps1
git commit -m "feat(morning-brief): format with objectives and ntfy integration

Implement generate_morning_brief to output System Health, Active
Projects (with objectives/actions from STATE.md), and Stale Contexts.
Update morning_brief.ps1 to use Python module. Per P5.2.3 spec.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

"""Main FastAPI application for Chinvex Gateway."""

import logging
import traceback
from pathlib import Path

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from uuid import uuid4
from time import time

from . import __version__
from .auth import verify_token
from .rate_limit import RateLimiter
from .audit import AuditLogger
from .error_log import RotatingErrorLogger
from .config import load_gateway_config
from .metrics import MetricsCollector
from .endpoints import health, healthz, search, evidence, chunks, contexts, metrics

from chinvex.index_meta import read_index_meta
from chinvex.context_cli import get_contexts_root
from chinvex.context import list_contexts, load_context

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize app
app = FastAPI(
    title="Chinvex Memory API",
    version=__version__,
    description="Query personal knowledge base with grounded retrieval"
)

# Load config
config = load_gateway_config()
rate_limiter = RateLimiter({
    'requests_per_minute': config.rate_limit.requests_per_minute,
    'requests_per_hour': config.rate_limit.requests_per_hour
})
audit_logger = AuditLogger(config.audit_log_path)
error_logger = RotatingErrorLogger("P:/ai_memory/gateway_errors.jsonl", max_size_mb=50, max_files=5)
metrics_collector = MetricsCollector()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


# Request middleware for audit logging and request_id
@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    """Log all requests to audit log and add request_id header."""
    request_id = str(uuid4())
    request.state.request_id = request_id

    start_time = time()
    response = await call_next(request)
    latency_ms = int((time() - start_time) * 1000)

    # Add request_id to response headers
    response.headers["X-Request-ID"] = request_id

    # Extract context from request state if available
    context = getattr(request.state, "context", None)

    audit_logger.log(
        request_id=request_id,
        endpoint=request.url.path,
        status=response.status_code,
        latency_ms=latency_ms,
        context=context,
        client_ip=request.client.host if request.client else None
    )

    # Record metrics
    metrics_collector.record_request(
        endpoint=request.url.path,
        status_code=response.status_code,
        duration=latency_ms / 1000.0  # Convert to seconds
    )

    return response


# Exception handler for 5xx errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Log stack traces for unhandled exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")

    # Log full stack trace to console
    logger.error(
        f"Unhandled exception [request_id={request_id}] {request.method} {request.url.path}",
        exc_info=exc
    )

    # Log to rotating file
    error_logger.log(
        request_id=request_id,
        method=request.method,
        path=str(request.url.path),
        error=exc,
        timestamp=time()
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "request_id": request_id
        }
    )


# Rate limiting dependency
async def check_rate_limit(token: str = Depends(verify_token)):
    """Check rate limit for authenticated token."""
    rate_limiter.check_limit(token)
    return token


def load_embedding_config_from_contexts() -> dict:
    """
    Load embedding configuration from contexts.

    Reads meta.json from each context to determine embedding provider.
    Uses first context's config as the gateway's default.
    Detects mixed providers across contexts.

    Returns:
        dict with keys: embedding_provider, embedding_model, contexts_loaded, mixed_providers
    """
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


# Startup event for warmup
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
            # Touch Chroma (and close to avoid lingering connections)
            vec_store = VectorStore(context.index.chroma_dir)
            vec_store.collection.count()
            vec_store.close()  # Clean up warmup connection
            logger.info(f"Warmed up context: {contexts[0].name}")

        logger.info("Gateway warmup complete")
    except Exception as e:
        logger.error(f"Warmup failed (non-fatal): {e}", exc_info=True)


@app.on_event("shutdown")
async def shutdown_cleanup():
    """Clean up database connections on gateway shutdown."""
    logger.info("Starting gateway shutdown cleanup...")

    try:
        from chinvex.storage import Storage

        # Force close global SQLite connection
        Storage.force_close_global_connection()
        logger.info("Closed global SQLite connection")

        # Note: ChromaDB clients are created per-request and cleaned up by GC.
        # The VectorStore.close() method is available for explicit cleanup
        # in long-running processes or tests.

        logger.info("Gateway shutdown cleanup complete")
    except Exception as e:
        logger.error(f"Shutdown cleanup failed (non-fatal): {e}", exc_info=True)


# Routers - health endpoints are public for monitoring
app.include_router(health.router, tags=["Health"])
app.include_router(healthz.router, tags=["Health"])
app.include_router(metrics.router, tags=["Metrics"])

# Protected routers with auth + rate limiting
app.include_router(
    evidence.router,
    prefix="/v1",
    tags=["Evidence"],
    dependencies=[Depends(check_rate_limit)]
)
app.include_router(
    search.router,
    prefix="/v1",
    tags=["Search"],
    dependencies=[Depends(check_rate_limit)]
)
app.include_router(
    chunks.router,
    prefix="/v1",
    tags=["Chunks"],
    dependencies=[Depends(check_rate_limit)]
)
app.include_router(
    contexts.router,
    prefix="/v1",
    tags=["Contexts"],
    dependencies=[Depends(check_rate_limit)]
)

# Optional answer endpoint (flag-gated)
if config.enable_server_llm:
    from .endpoints import answer
    app.include_router(
        answer.router,
        prefix="/v1",
        tags=["Answer"],
        dependencies=[Depends(check_rate_limit)]
    )


def custom_openapi():
    """Customize OpenAPI schema for ChatGPT Actions compatibility."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Chinvex Memory API",
        version=__version__,
        description="Query personal knowledge base with grounded retrieval",
        routes=app.routes,
    )

    # Ensure descriptions are under 300 chars (OpenAI limit)
    for path in openapi_schema.get("paths", {}).values():
        for operation in path.values():
            if isinstance(operation, dict) and "description" in operation:
                if len(operation["description"]) > 300:
                    operation["description"] = operation["description"][:297] + "..."

    # Add security scheme
    openapi_schema.setdefault("components", {})
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "description": "Enter your Chinvex API token"
        }
    }

    openapi_schema["security"] = [{"bearerAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

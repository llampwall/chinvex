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


# Startup event for warmup
@app.on_event("startup")
async def startup_warmup():
    """Warm up the gateway by preloading contexts and initializing storage."""
    logger.info("Starting gateway warmup...")
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

"""Main FastAPI application for Chinvex Gateway."""

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from uuid import uuid4
from time import time

from . import __version__
from .auth import verify_token
from .rate_limit import RateLimiter
from .audit import AuditLogger
from .config import load_gateway_config
from .endpoints import health, search, evidence, chunks, contexts


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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


# Request middleware for audit logging
@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    """Log all requests to audit log."""
    request_id = str(uuid4())
    request.state.request_id = request_id

    start_time = time()
    response = await call_next(request)
    latency_ms = int((time() - start_time) * 1000)

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

    return response


# Rate limiting dependency
async def check_rate_limit(token: str = Depends(verify_token)):
    """Check rate limit for authenticated token."""
    rate_limiter.check_limit(token)
    return token


# Routers - health endpoint has no auth
app.include_router(health.router, tags=["Health"])

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

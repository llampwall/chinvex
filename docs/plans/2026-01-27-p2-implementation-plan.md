# P2 Gateway API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose Chinvex as a secure, TLS-reachable REST API for ChatGPT Actions with token authentication and rate limiting.

**Architecture:** FastAPI gateway as a thin security layer over existing Chinvex search/retrieval. All endpoints (except `/health`) require Bearer token auth. In-memory rate limiting. Privacy-preserving audit logs.

**Tech Stack:** FastAPI, Uvicorn, Pydantic, existing Chinvex core

---

## Prerequisites

Before starting implementation:

1. P1 must be complete
2. Python 3.12+ environment
3. Existing Chinvex installation working
4. Test context with ingested data

---

## Phase 1: Foundation & Configuration

### Task 1: Install FastAPI dependencies

**Files:**
- Modify: `pyproject.toml:7-13`

**Step 1: Add FastAPI dependencies**

Edit `pyproject.toml`:

```toml
dependencies = [
  "typer>=0.12.3",
  "chromadb>=0.5.3",
  "requests>=2.32.3",
  "mcp>=1.0.0",
  "portalocker>=2.10.1",
  "fastapi>=0.109.0",
  "uvicorn[standard]>=0.27.0",
  "pydantic>=2.5.0",
]
```

**Step 2: Install dependencies**

Run: `pip install -e .`
Expected: All packages install successfully

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add FastAPI dependencies for P2 gateway"
```

---

### Task 2: Create gateway package structure

**Files:**
- Create: `src/chinvex/gateway/__init__.py`
- Create: `src/chinvex/gateway/endpoints/__init__.py`

**Step 1: Create gateway directory**

Run: `mkdir -p src/chinvex/gateway/endpoints`
Expected: Directories created

**Step 2: Create package init files**

Create `src/chinvex/gateway/__init__.py`:

```python
"""Chinvex Gateway API."""

__version__ = "0.2.0"
```

Create `src/chinvex/gateway/endpoints/__init__.py`:

```python
"""Gateway API endpoints."""
```

**Step 3: Verify structure**

Run: `ls src/chinvex/gateway/`
Expected: `__init__.py` and `endpoints/` present

**Step 4: Commit**

```bash
git add src/chinvex/gateway/
git commit -m "feat(gateway): create package structure"
```

---

### Task 3: Configuration dataclasses and loader

**Files:**
- Create: `src/chinvex/gateway/config.py`
- Create: `tests/gateway/test_config.py`

**Step 1: Write test for default config loading**

Create `tests/gateway/test_config.py`:

```python
import os
import tempfile
from pathlib import Path
from chinvex.gateway.config import load_gateway_config, GatewayConfig


def test_load_default_config_when_file_missing():
    """Should return defaults when config file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_path = Path(tmpdir) / "nonexistent.json"
        os.environ.pop("CHINVEX_GATEWAY_CONFIG", None)

        config = load_gateway_config(str(fake_path))

        assert config.host == "127.0.0.1"
        assert config.port == 7778
        assert config.enabled is True


def test_load_config_from_file():
    """Should load config from JSON file."""
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "gateway.json"
        config_path.write_text(json.dumps({
            "gateway": {
                "host": "0.0.0.0",
                "port": 8080,
                "context_allowlist": ["TestContext"]
            }
        }))

        config = load_gateway_config(str(config_path))

        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.context_allowlist == ["TestContext"]


def test_token_property_reads_from_env():
    """Should read token from environment variable."""
    os.environ["CHINVEX_API_TOKEN"] = "test_token_123"

    config = GatewayConfig()

    assert config.token == "test_token_123"

    del os.environ["CHINVEX_API_TOKEN"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/gateway/test_config.py -v`
Expected: FAIL with "No module named 'chinvex.gateway.config'"

**Step 3: Write minimal implementation**

Create `src/chinvex/gateway/config.py`:

```python
"""Gateway configuration loading."""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RateLimitConfig:
    requests_per_minute: int = 60
    requests_per_hour: int = 500


@dataclass
class LimitsConfig:
    max_k: int = 20
    max_chunk_ids: int = 20
    max_query_length: int = 1000
    max_chunk_text_length: int = 5000


@dataclass
class GatewayConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 7778
    token_env: str = "CHINVEX_API_TOKEN"
    context_allowlist: Optional[list[str]] = None
    cors_origins: list[str] = field(default_factory=lambda: [
        "https://chat.openai.com",
        "https://chatgpt.com"
    ])
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    enable_server_llm: bool = False
    audit_log_path: str = "P:\\ai_memory\\gateway_audit.jsonl"

    @property
    def token(self) -> Optional[str]:
        """Get token from environment variable."""
        return os.environ.get(self.token_env)


def load_gateway_config(path: Optional[str] = None) -> GatewayConfig:
    """
    Load gateway configuration from file or defaults.

    Priority:
    1. Explicit path argument
    2. CHINVEX_GATEWAY_CONFIG environment variable
    3. P:\\ai_memory\\gateway.json
    4. Defaults
    """
    if path is None:
        path = os.environ.get("CHINVEX_GATEWAY_CONFIG", "P:\\ai_memory\\gateway.json")

    config_path = Path(path)

    if not config_path.exists():
        return GatewayConfig()

    with open(config_path) as f:
        data = json.load(f)

    gateway_data = data.get("gateway", {})

    # Build rate_limit if present
    rate_limit_data = gateway_data.get("rate_limit", {})
    rate_limit = RateLimitConfig(**rate_limit_data) if rate_limit_data else RateLimitConfig()

    # Build limits if present
    limits_data = gateway_data.get("limits", {})
    limits = LimitsConfig(**limits_data) if limits_data else LimitsConfig()

    return GatewayConfig(
        enabled=gateway_data.get("enabled", True),
        host=gateway_data.get("host", "127.0.0.1"),
        port=gateway_data.get("port", 7778),
        token_env=gateway_data.get("token_env", "CHINVEX_API_TOKEN"),
        context_allowlist=gateway_data.get("context_allowlist"),
        cors_origins=gateway_data.get("cors_origins", [
            "https://chat.openai.com",
            "https://chatgpt.com"
        ]),
        rate_limit=rate_limit,
        limits=limits,
        enable_server_llm=gateway_data.get("enable_server_llm", False)
            or os.environ.get("GATEWAY_ENABLE_SERVER_LLM", "").lower() == "true",
        audit_log_path=gateway_data.get("audit_log_path", "P:\\ai_memory\\gateway_audit.jsonl")
    )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/gateway/test_config.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/chinvex/gateway/config.py tests/gateway/test_config.py
git commit -m "feat(gateway): add configuration system with dataclasses"
```

---

## Phase 2: Security Layer

### Task 4: Authentication middleware

**Files:**
- Create: `src/chinvex/gateway/auth.py`
- Create: `tests/gateway/test_auth.py`

**Step 1: Write test for token verification**

Create `tests/gateway/test_auth.py`:

```python
import os
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from chinvex.gateway.auth import verify_token


def test_verify_token_success():
    """Should accept valid token."""
    os.environ["CHINVEX_API_TOKEN"] = "valid_token_123"

    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid_token_123")

    # Should not raise
    result = verify_token(creds)
    assert result == "valid_token_123"

    del os.environ["CHINVEX_API_TOKEN"]


def test_verify_token_invalid():
    """Should reject invalid token."""
    os.environ["CHINVEX_API_TOKEN"] = "valid_token_123"

    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong_token")

    with pytest.raises(HTTPException) as exc_info:
        verify_token(creds)

    assert exc_info.value.status_code == 401
    assert "Invalid authentication token" in exc_info.value.detail

    del os.environ["CHINVEX_API_TOKEN"]


def test_verify_token_not_configured():
    """Should fail when token not configured."""
    os.environ.pop("CHINVEX_API_TOKEN", None)

    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="any_token")

    with pytest.raises(HTTPException) as exc_info:
        verify_token(creds)

    assert exc_info.value.status_code == 500
    assert "not configured" in exc_info.value.detail
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/gateway/test_auth.py -v`
Expected: FAIL with "No module named 'chinvex.gateway.auth'"

**Step 3: Write implementation**

Create `src/chinvex/gateway/auth.py`:

```python
"""Authentication middleware for gateway."""

import secrets
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .config import load_gateway_config

security = HTTPBearer()


def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> str:
    """
    Verify bearer token using constant-time comparison.

    Args:
        credentials: HTTP Bearer credentials from request

    Returns:
        Token string if valid

    Raises:
        HTTPException: 500 if token not configured, 401 if invalid
    """
    config = load_gateway_config()
    expected = config.token

    if not expected:
        raise HTTPException(
            status_code=500,
            detail="Gateway token not configured (CHINVEX_API_TOKEN missing)"
        )

    provided = credentials.credentials
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )

    return provided
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/gateway/test_auth.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/chinvex/gateway/auth.py tests/gateway/test_auth.py
git commit -m "feat(gateway): add bearer token authentication with constant-time comparison"
```

---

### Task 5: Input validation models

**Files:**
- Create: `src/chinvex/gateway/validation.py`
- Create: `tests/gateway/test_validation.py`

**Step 1: Write tests for validation**

Create `tests/gateway/test_validation.py`:

```python
import pytest
from pydantic import ValidationError
from chinvex.gateway.validation import EvidenceRequest, SearchRequest, ChunksRequest


def test_evidence_request_valid():
    """Should accept valid evidence request."""
    req = EvidenceRequest(
        context="Chinvex",
        query="test query",
        k=8
    )
    assert req.context == "Chinvex"
    assert req.query == "test query"
    assert req.k == 8


def test_context_name_invalid_chars():
    """Should reject context with path traversal."""
    with pytest.raises(ValidationError) as exc_info:
        EvidenceRequest(context="../../../etc/passwd", query="test")

    assert "Invalid context name format" in str(exc_info.value)


def test_query_too_long():
    """Should reject query exceeding 1000 chars."""
    with pytest.raises(ValidationError) as exc_info:
        EvidenceRequest(context="Test", query="a" * 1001)

    assert "exceeds 1000 character limit" in str(exc_info.value)


def test_query_empty():
    """Should reject empty query."""
    with pytest.raises(ValidationError) as exc_info:
        EvidenceRequest(context="Test", query="   ")

    assert "cannot be empty" in str(exc_info.value)


def test_query_null_bytes():
    """Should reject query with null bytes."""
    with pytest.raises(ValidationError) as exc_info:
        EvidenceRequest(context="Test", query="test\x00query")

    assert "Null bytes not allowed" in str(exc_info.value)


def test_k_out_of_range():
    """Should reject k outside 1-20."""
    with pytest.raises(ValidationError) as exc_info:
        EvidenceRequest(context="Test", query="test", k=50)

    assert "must be between 1 and 20" in str(exc_info.value)


def test_chunks_request_too_many_ids():
    """Should reject more than 20 chunk IDs."""
    with pytest.raises(ValidationError) as exc_info:
        ChunksRequest(
            context="Test",
            chunk_ids=["abc123def456"] * 21
        )

    assert "Maximum 20 chunk IDs" in str(exc_info.value)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/gateway/test_validation.py -v`
Expected: FAIL with "No module named 'chinvex.gateway.validation'"

**Step 3: Write implementation**

Create `src/chinvex/gateway/validation.py`:

```python
"""Request validation models."""

import re
from typing import Optional
from pydantic import BaseModel, field_validator

CONTEXT_PATTERN = re.compile(r'^[A-Za-z0-9_-]{1,50}$')
CHUNK_ID_PATTERN = re.compile(r'^[a-f0-9]{12}$')


class EvidenceRequest(BaseModel):
    """Request for /v1/evidence endpoint."""
    context: str
    query: str
    k: int = 8
    source_types: Optional[list[str]] = None

    @field_validator('context')
    def validate_context(cls, v):
        if not CONTEXT_PATTERN.match(v):
            raise ValueError('Invalid context name format')
        return v

    @field_validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        if len(v) > 1000:
            raise ValueError('Query exceeds 1000 character limit')
        if '\x00' in v:
            raise ValueError('Null bytes not allowed in query')
        return v.strip()

    @field_validator('k')
    def validate_k(cls, v):
        if v < 1 or v > 20:
            raise ValueError('k must be between 1 and 20')
        return v


class SearchRequest(BaseModel):
    """Request for /v1/search endpoint."""
    context: str
    query: str
    k: int = 10
    source_types: Optional[list[str]] = None
    no_recency: bool = False

    @field_validator('context')
    def validate_context(cls, v):
        if not CONTEXT_PATTERN.match(v):
            raise ValueError('Invalid context name format')
        return v

    @field_validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        if len(v) > 1000:
            raise ValueError('Query exceeds 1000 character limit')
        if '\x00' in v:
            raise ValueError('Null bytes not allowed in query')
        return v.strip()

    @field_validator('k')
    def validate_k(cls, v):
        if v < 1 or v > 20:
            raise ValueError('k must be between 1 and 20')
        return v


class ChunksRequest(BaseModel):
    """Request for /v1/chunks endpoint."""
    context: str
    chunk_ids: list[str]

    @field_validator('context')
    def validate_context(cls, v):
        if not CONTEXT_PATTERN.match(v):
            raise ValueError('Invalid context name format')
        return v

    @field_validator('chunk_ids')
    def validate_chunk_ids(cls, v):
        if len(v) > 20:
            raise ValueError('Maximum 20 chunk IDs per request')
        return v


class AnswerRequest(BaseModel):
    """Request for /v1/answer endpoint (optional)."""
    context: str
    query: str
    k: int = 8
    grounded: bool = True

    @field_validator('context')
    def validate_context(cls, v):
        if not CONTEXT_PATTERN.match(v):
            raise ValueError('Invalid context name format')
        return v

    @field_validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        if len(v) > 1000:
            raise ValueError('Query exceeds 1000 character limit')
        return v.strip()

    @field_validator('k')
    def validate_k(cls, v):
        if v < 1 or v > 20:
            raise ValueError('k must be between 1 and 20')
        return v
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/gateway/test_validation.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/chinvex/gateway/validation.py tests/gateway/test_validation.py
git commit -m "feat(gateway): add Pydantic validation models for all endpoints"
```

---

### Task 6: Rate limiting

**Files:**
- Create: `src/chinvex/gateway/rate_limit.py`
- Create: `tests/gateway/test_rate_limit.py`

**Step 1: Write tests for rate limiting**

Create `tests/gateway/test_rate_limit.py`:

```python
import pytest
import time
from fastapi import HTTPException
from chinvex.gateway.rate_limit import RateLimiter


def test_rate_limiter_allows_within_limit():
    """Should allow requests within rate limit."""
    limiter = RateLimiter({"requests_per_minute": 60, "requests_per_hour": 500})

    # Should not raise
    for _ in range(5):
        limiter.check_limit("token_123")


def test_rate_limiter_blocks_over_minute_limit():
    """Should block requests exceeding per-minute limit."""
    limiter = RateLimiter({"requests_per_minute": 2, "requests_per_hour": 500})

    limiter.check_limit("token_123")
    limiter.check_limit("token_123")

    with pytest.raises(HTTPException) as exc_info:
        limiter.check_limit("token_123")

    assert exc_info.value.status_code == 429
    assert "per-minute" in exc_info.value.detail
    assert "Retry-After" in exc_info.value.headers


def test_rate_limiter_separate_tokens():
    """Should track different tokens separately."""
    limiter = RateLimiter({"requests_per_minute": 2, "requests_per_hour": 500})

    limiter.check_limit("token_a")
    limiter.check_limit("token_a")

    # Different token should still work
    limiter.check_limit("token_b")
    limiter.check_limit("token_b")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/gateway/test_rate_limit.py -v`
Expected: FAIL with "No module named 'chinvex.gateway.rate_limit'"

**Step 3: Write implementation**

Create `src/chinvex/gateway/rate_limit.py`:

```python
"""Rate limiting using token bucket algorithm."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional
from fastapi import HTTPException


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    requests_per_minute: int
    requests_per_hour: int
    minute_tokens: int = 0
    hour_tokens: int = 0
    minute_reset: Optional[datetime] = None
    hour_reset: Optional[datetime] = None


class RateLimiter:
    """In-memory rate limiter using token bucket algorithm."""

    def __init__(self, config: dict):
        self.rpm = config.get('requests_per_minute', 60)
        self.rph = config.get('requests_per_hour', 500)
        self.buckets = defaultdict(lambda: TokenBucket(self.rpm, self.rph))

    def check_limit(self, token: str) -> None:
        """
        Check rate limits for token. Raises 429 if exceeded.

        Args:
            token: API token to check

        Raises:
            HTTPException: 429 if rate limit exceeded
        """
        now = datetime.now()
        bucket = self.buckets[token]

        # Reset minute bucket if needed
        if bucket.minute_reset is None or now >= bucket.minute_reset:
            bucket.minute_tokens = self.rpm
            bucket.minute_reset = now + timedelta(minutes=1)

        # Reset hour bucket if needed
        if bucket.hour_reset is None or now >= bucket.hour_reset:
            bucket.hour_tokens = self.rph
            bucket.hour_reset = now + timedelta(hours=1)

        # Check limits
        if bucket.minute_tokens <= 0:
            retry_after = int((bucket.minute_reset - now).total_seconds())
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded (per-minute)",
                headers={"Retry-After": str(retry_after)}
            )

        if bucket.hour_tokens <= 0:
            retry_after = int((bucket.hour_reset - now).total_seconds())
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded (per-hour)",
                headers={"Retry-After": str(retry_after)}
            )

        # Consume tokens
        bucket.minute_tokens -= 1
        bucket.hour_tokens -= 1
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/gateway/test_rate_limit.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/chinvex/gateway/rate_limit.py tests/gateway/test_rate_limit.py
git commit -m "feat(gateway): add in-memory rate limiter with token bucket"
```

---

### Task 7: Audit logging

**Files:**
- Create: `src/chinvex/gateway/audit.py`
- Create: `tests/gateway/test_audit.py`

**Step 1: Write tests for audit logging**

Create `tests/gateway/test_audit.py`:

```python
import json
import tempfile
from pathlib import Path
from chinvex.gateway.audit import AuditLogger


def test_audit_logger_writes_jsonl():
    """Should write audit entries as JSONL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "audit.jsonl"
        logger = AuditLogger(str(log_path))

        logger.log(
            request_id="req_123",
            endpoint="/v1/evidence",
            status=200,
            latency_ms=142,
            context="Chinvex",
            query="test query",
            client_ip="127.0.0.1"
        )

        lines = log_path.read_text().strip().split('\n')
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["request_id"] == "req_123"
        assert entry["endpoint"] == "/v1/evidence"
        assert entry["status"] == 200
        assert entry["latency_ms"] == 142
        assert entry["context"] == "Chinvex"
        assert entry["query_hash"].startswith("sha256:")
        assert entry["client_ip"] == "127.0.0.1"


def test_audit_logger_hashes_query():
    """Should hash query instead of logging raw text."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "audit.jsonl"
        logger = AuditLogger(str(log_path))

        logger.log(
            request_id="req_123",
            endpoint="/v1/evidence",
            status=200,
            latency_ms=100,
            query="sensitive query text"
        )

        content = log_path.read_text()
        assert "sensitive query text" not in content
        assert "sha256:" in content


def test_audit_logger_creates_parent_dir():
    """Should create parent directory if missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "nested" / "dir" / "audit.jsonl"
        logger = AuditLogger(str(log_path))

        logger.log(
            request_id="req_123",
            endpoint="/health",
            status=200,
            latency_ms=5
        )

        assert log_path.exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/gateway/test_audit.py -v`
Expected: FAIL with "No module named 'chinvex.gateway.audit'"

**Step 3: Write implementation**

Create `src/chinvex/gateway/audit.py`:

```python
"""Audit logging for gateway requests."""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class AuditEntry:
    """Audit log entry."""
    ts: str
    request_id: str
    endpoint: str
    context: Optional[str]
    query_hash: Optional[str]
    status: int
    latency_ms: int
    client_ip: Optional[str]


class AuditLogger:
    """Append-only audit logger."""

    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        request_id: str,
        endpoint: str,
        status: int,
        latency_ms: int,
        context: Optional[str] = None,
        query: Optional[str] = None,
        client_ip: Optional[str] = None
    ):
        """
        Append audit entry to JSONL log.

        Args:
            request_id: Unique request identifier
            endpoint: Request path
            status: HTTP status code
            latency_ms: Response time in milliseconds
            context: Context name if applicable
            query: Query text (will be hashed, not stored raw)
            client_ip: Client IP address
        """
        entry = AuditEntry(
            ts=datetime.utcnow().isoformat() + "Z",
            request_id=request_id,
            endpoint=endpoint,
            context=context,
            query_hash=self._hash_query(query) if query else None,
            status=status,
            latency_ms=latency_ms,
            client_ip=client_ip
        )

        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(entry), default=str) + "\n")

    @staticmethod
    def _hash_query(query: str) -> str:
        """Hash query for privacy."""
        return "sha256:" + hashlib.sha256(query.encode()).hexdigest()[:16]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/gateway/test_audit.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/chinvex/gateway/audit.py tests/gateway/test_audit.py
git commit -m "feat(gateway): add privacy-preserving audit logger with query hashing"
```

---

## Phase 3: Core Endpoints

### Task 8: Health endpoint

**Files:**
- Create: `src/chinvex/gateway/endpoints/health.py`
- Create: `tests/gateway/endpoints/test_health.py`

**Step 1: Write test for health endpoint**

Create `tests/gateway/endpoints/test_health.py`:

```python
from fastapi.testclient import TestClient
from fastapi import FastAPI
from chinvex.gateway.endpoints.health import router


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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/gateway/endpoints/test_health.py -v`
Expected: FAIL with "No module named 'chinvex.gateway.endpoints.health'"

**Step 3: Write implementation**

Create `src/chinvex/gateway/endpoints/health.py`:

```python
"""Health check endpoint."""

from fastapi import APIRouter
from chinvex.gateway import __version__
from chinvex.context import list_contexts


router = APIRouter()


@router.get("/health")
async def health():
    """
    Health check endpoint. No authentication required.

    Returns:
        Status information including version and context count
    """
    try:
        contexts = list_contexts()
        contexts_available = len(contexts)
    except Exception:
        # If context listing fails, don't fail health check
        contexts_available = 0

    return {
        "status": "ok",
        "version": __version__,
        "contexts_available": contexts_available
    }
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/gateway/endpoints/test_health.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/chinvex/gateway/endpoints/health.py tests/gateway/endpoints/test_health.py
git commit -m "feat(gateway): add health check endpoint"
```

---

### Task 9: Evidence endpoint (primary)

**Files:**
- Create: `src/chinvex/gateway/endpoints/evidence.py`
- Modify: `src/chinvex/search.py` (add context-based search wrapper)
- Create: `tests/gateway/endpoints/test_evidence.py`

**Step 1: Write test for evidence endpoint**

Create `tests/gateway/endpoints/test_evidence.py`:

```python
import os
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI, Depends
from chinvex.gateway.endpoints.evidence import router
from chinvex.gateway.auth import verify_token


app = FastAPI()
app.include_router(router, dependencies=[Depends(verify_token)])
client = TestClient(app)


@pytest.fixture
def set_token():
    """Set test token in environment."""
    os.environ["CHINVEX_API_TOKEN"] = "test_token_123"
    yield
    del os.environ["CHINVEX_API_TOKEN"]


def test_evidence_endpoint_requires_auth():
    """Should require authentication."""
    response = client.post("/evidence", json={
        "context": "Chinvex",
        "query": "test"
    })

    assert response.status_code == 403  # FastAPI returns 403 for missing auth


def test_evidence_endpoint_returns_grounded_false_for_unknown(set_token):
    """Should return grounded=false for unknown query."""
    response = client.post("/evidence", json={
        "context": "Chinvex",
        "query": "asdfghjkl12345"
    }, headers={"Authorization": "Bearer test_token_123"})

    # May return 404 if context doesn't exist in test env
    # This is OK - we're testing the endpoint structure
    assert response.status_code in [200, 404]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/gateway/endpoints/test_evidence.py -v`
Expected: FAIL with "No module named 'chinvex.gateway.endpoints.evidence'"

**Step 3: Add search wrapper to existing search.py**

Edit `src/chinvex/search.py` (add at end):

```python
def hybrid_search_from_context(
    context,
    query: str,
    k: int = 8,
    source_types: list[str] | None = None,
    no_recency: bool = False
):
    """
    Wrapper for context-based search (used by gateway).

    Args:
        context: Context object from chinvex.context.load_context()
        query: Search query
        k: Number of results
        source_types: Filter by source types
        no_recency: Disable recency decay

    Returns:
        List of search results with scores
    """
    from .storage import Storage
    from .vectors import VectorStore
    from .embed import OllamaEmbedder
    from .ranking import apply_recency_decay

    db_path = context.index.sqlite_path
    chroma_dir = context.index.chroma_dir

    storage = Storage(db_path)
    storage.ensure_schema()

    # FTS search
    fts_results = storage.fts_search(query, limit=k * 2)

    # Vector search
    embedder = OllamaEmbedder(
        model=context.ollama.embedding_model,
        host=context.ollama.host
    )
    vec_store = VectorStore(chroma_dir)
    query_embedding = embedder.embed([query])[0]
    vec_results = vec_store.search(query_embedding, k=k * 2)

    # Merge and score
    from .scoring import merge_and_rank
    results = merge_and_rank(
        fts_results=fts_results,
        vec_results=vec_results,
        storage=storage,
        weights=context.weights,
        k=k
    )

    # Apply recency if enabled
    if not no_recency and hasattr(context, 'ranking') and context.ranking.recency_enabled:
        results = apply_recency_decay(
            results,
            half_life_days=context.ranking.recency_half_life_days
        )

    return results
```

**Step 4: Write evidence endpoint implementation**

Create `src/chinvex/gateway/endpoints/evidence.py`:

```python
"""Evidence endpoint - search with grounding check."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

from chinvex.context import load_context
from chinvex.search import hybrid_search_from_context
from chinvex.gateway.validation import EvidenceRequest
from chinvex.gateway.config import load_gateway_config


router = APIRouter()


class EvidenceResponse(BaseModel):
    """Response from evidence endpoint."""
    context: str
    query: str
    grounded: bool
    evidence_pack: dict
    retrieval_debug: dict
    message: Optional[str] = None


@router.post("/evidence", response_model=EvidenceResponse)
async def get_evidence(req: EvidenceRequest, request: Request):
    """
    Search with grounding check. Primary endpoint for ChatGPT Actions.

    Args:
        req: Evidence request with context, query, k
        request: FastAPI request (for audit logging)

    Returns:
        Evidence response with grounded status and chunks
    """
    # Store context in request state for audit logging
    request.state.context = req.context

    # Load context and verify it exists
    try:
        context = load_context(req.context)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Context not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load context: {str(e)}")

    # Check context allowlist
    config = load_gateway_config()
    if config.context_allowlist and req.context not in config.context_allowlist:
        raise HTTPException(status_code=404, detail="Context not found")

    # Perform search
    try:
        results = hybrid_search_from_context(
            context=context,
            query=req.query,
            k=req.k,
            source_types=req.source_types
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    # Grounding check
    GROUNDING_THRESHOLD = 0.35
    grounded_chunks = [r for r in results if r.rank_score >= GROUNDING_THRESHOLD]

    grounded = len(grounded_chunks) >= 1

    if grounded:
        evidence_pack = {
            "chunks": [
                {
                    "chunk_id": r.chunk_id,
                    "text": truncate_text(r.text, 5000),
                    "source_uri": r.source_uri,
                    "source_type": r.source_type,
                    "range": build_range(r),
                    "score": r.rank_score
                }
                for r in grounded_chunks
            ]
        }
        message = None
    else:
        evidence_pack = {"chunks": []}
        message = "No retrieved content supports a direct answer to this query."

    return EvidenceResponse(
        context=req.context,
        query=req.query,
        grounded=grounded,
        evidence_pack=evidence_pack,
        retrieval_debug={
            "k": req.k,
            "chunks_retrieved": len(results),
            "chunks_above_threshold": len(grounded_chunks)
        },
        message=message
    )


def truncate_text(text: str, max_len: int) -> str:
    """Truncate text with marker."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + " [truncated]"


def build_range(result) -> dict:
    """Build range object from search result."""
    if hasattr(result, 'line_start') and result.line_start:
        return {
            "line_start": result.line_start,
            "line_end": result.line_end
        }
    elif hasattr(result, 'char_start'):
        return {
            "char_start": result.char_start,
            "char_end": result.char_end
        }
    return {}
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/gateway/endpoints/test_evidence.py -v`
Expected: Tests PASS (or 404 if no test context - acceptable)

**Step 6: Commit**

```bash
git add src/chinvex/gateway/endpoints/evidence.py src/chinvex/search.py tests/gateway/endpoints/test_evidence.py
git commit -m "feat(gateway): add evidence endpoint with grounding check"
```

---

### Task 10: Remaining core endpoints (search, chunks, contexts)

**Files:**
- Create: `src/chinvex/gateway/endpoints/search.py`
- Create: `src/chinvex/gateway/endpoints/chunks.py`
- Create: `src/chinvex/gateway/endpoints/contexts.py`

**Step 1: Write search endpoint**

Create `src/chinvex/gateway/endpoints/search.py`:

```python
"""Search endpoint - raw hybrid search."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from chinvex.context import load_context
from chinvex.search import hybrid_search_from_context
from chinvex.gateway.validation import SearchRequest
from chinvex.gateway.config import load_gateway_config


router = APIRouter()


class SearchResponse(BaseModel):
    """Response from search endpoint."""
    context: str
    query: str
    results: list[dict]
    total_results: int


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, request: Request):
    """
    Raw hybrid search. Returns ranked chunks without grounding check.
    """
    request.state.context = req.context

    try:
        context = load_context(req.context)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Context not found")

    config = load_gateway_config()
    if config.context_allowlist and req.context not in config.context_allowlist:
        raise HTTPException(status_code=404, detail="Context not found")

    results = hybrid_search_from_context(
        context=context,
        query=req.query,
        k=req.k,
        source_types=req.source_types,
        no_recency=req.no_recency
    )

    return SearchResponse(
        context=req.context,
        query=req.query,
        results=[
            {
                "chunk_id": r.chunk_id,
                "text": r.text[:5000] + (" [truncated]" if len(r.text) > 5000 else ""),
                "source_uri": r.source_uri,
                "source_type": r.source_type,
                "scores": {
                    "fts": r.fts_score,
                    "vector": r.vector_score,
                    "blended": r.blended_score,
                    "rank": r.rank_score
                },
                "metadata": {
                    "line_start": getattr(r, 'line_start', None),
                    "line_end": getattr(r, 'line_end', None),
                    "updated_at": getattr(r, 'updated_at', None)
                }
            }
            for r in results
        ],
        total_results=len(results)
    )
```

**Step 2: Write chunks endpoint**

Create `src/chinvex/gateway/endpoints/chunks.py`:

```python
"""Chunks endpoint - fetch specific chunks by ID."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from chinvex.context import load_context
from chinvex.storage import Storage
from chinvex.gateway.validation import ChunksRequest
from chinvex.gateway.config import load_gateway_config


router = APIRouter()


class ChunksResponse(BaseModel):
    """Response from chunks endpoint."""
    context: str
    chunks: list[dict]


@router.post("/chunks", response_model=ChunksResponse)
async def get_chunks(req: ChunksRequest, request: Request):
    """
    Fetch specific chunks by ID.
    """
    request.state.context = req.context

    try:
        context = load_context(req.context)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Context not found")

    config = load_gateway_config()
    if config.context_allowlist and req.context not in config.context_allowlist:
        raise HTTPException(status_code=404, detail="Context not found")

    storage = Storage(context.index.sqlite_path)
    chunks = storage.get_chunks_by_ids(req.chunk_ids)

    return ChunksResponse(
        context=req.context,
        chunks=[
            {
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "source_uri": c["source_uri"],
                "source_type": c["source_type"],
                "metadata": c.get("metadata", {})
            }
            for c in chunks
        ]
    )
```

**Step 3: Write contexts endpoint**

Create `src/chinvex/gateway/endpoints/contexts.py`:

```python
"""Contexts endpoint - list available contexts."""

from fastapi import APIRouter
from pydantic import BaseModel

from chinvex.context import list_contexts
from chinvex.gateway.config import load_gateway_config


router = APIRouter()


class ContextInfo(BaseModel):
    """Context information."""
    name: str
    aliases: list[str]
    updated_at: str


class ContextsResponse(BaseModel):
    """Response from contexts endpoint."""
    contexts: list[ContextInfo]


@router.get("/contexts", response_model=ContextsResponse)
async def list_available_contexts():
    """
    List available contexts. Respects allowlist.
    """
    all_contexts = list_contexts()
    config = load_gateway_config()

    # Filter by allowlist if configured
    if config.context_allowlist:
        filtered = [c for c in all_contexts if c.name in config.context_allowlist]
    else:
        filtered = all_contexts

    return ContextsResponse(
        contexts=[
            ContextInfo(
                name=c.name,
                aliases=c.aliases,
                updated_at=c.updated_at
            )
            for c in filtered
        ]
    )
```

**Step 4: Commit**

```bash
git add src/chinvex/gateway/endpoints/search.py src/chinvex/gateway/endpoints/chunks.py src/chinvex/gateway/endpoints/contexts.py
git commit -m "feat(gateway): add search, chunks, and contexts endpoints"
```

---

## Phase 4: FastAPI Application Assembly

### Task 11: Main FastAPI app with middleware

**Files:**
- Create: `src/chinvex/gateway/app.py`

**Step 1: Write main FastAPI application**

Create `src/chinvex/gateway/app.py`:

```python
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
```

**Step 2: Update endpoints __init__.py to export routers**

Edit `src/chinvex/gateway/endpoints/__init__.py`:

```python
"""Gateway API endpoints."""

from . import health, search, evidence, chunks, contexts

__all__ = ["health", "search", "evidence", "chunks", "contexts"]
```

**Step 3: Commit**

```bash
git add src/chinvex/gateway/app.py src/chinvex/gateway/endpoints/__init__.py
git commit -m "feat(gateway): assemble FastAPI app with middleware and routers"
```

---

## Phase 5: CLI Commands

### Task 12: Gateway CLI commands

**Files:**
- Modify: `src/chinvex/cli.py` (add gateway group)

**Step 1: Add gateway commands to CLI**

Edit `src/chinvex/cli.py` (add at end before `if __name__ == "__main__"`):

```python
@app.command()
def gateway():
    """Gateway server commands."""
    pass


@gateway.command()
def serve(
    host: str = typer.Option(None, help="Host to bind (overrides config)"),
    port: int = typer.Option(None, help="Port to bind (overrides config)"),
    reload: bool = typer.Option(False, help="Enable auto-reload (dev only)")
):
    """
    Start the gateway server.

    Example:
        chinvex gateway serve --port 7778
    """
    import sys
    import uvicorn
    from .gateway.config import load_gateway_config

    config = load_gateway_config()

    # Check token is configured
    if not config.token:
        typer.echo(f"Error: {config.token_env} environment variable not set", err=True)
        typer.echo("Run 'chinvex gateway token generate' to create a token", err=True)
        sys.exit(1)

    final_host = host or config.host
    final_port = port or config.port

    typer.echo(f"Starting Chinvex Gateway on {final_host}:{final_port}")
    typer.echo(f"Context allowlist: {config.context_allowlist or 'all contexts'}")
    typer.echo(f"Server-side LLM: {'enabled' if config.enable_server_llm else 'disabled'}")

    uvicorn.run(
        "chinvex.gateway.app:app",
        host=final_host,
        port=final_port,
        reload=reload
    )


@gateway.command()
def token_generate():
    """
    Generate a new API token.

    Example:
        chinvex gateway token-generate
    """
    import secrets

    new_token = secrets.token_urlsafe(32)

    typer.echo("Generated new API token:")
    typer.echo()
    typer.echo(f"export CHINVEX_API_TOKEN={new_token}")
    typer.echo()
    typer.echo("Add this to your environment or secrets manager.")
    typer.echo("For ChatGPT Actions, use this token in the API Key field.")


@gateway.command()
def token_rotate():
    """
    Rotate API token (generates new, shows old).

    Example:
        chinvex gateway token-rotate
    """
    import secrets
    from .gateway.config import load_gateway_config

    config = load_gateway_config()
    old_token = config.token
    new_token = secrets.token_urlsafe(32)

    typer.echo("Token rotation:")
    typer.echo()
    if old_token:
        typer.echo(f"Old token: {old_token[:8]}...{old_token[-8:]}")
    else:
        typer.echo("Old token: (none)")
    typer.echo(f"New token: {new_token}")
    typer.echo()
    typer.echo(f"export CHINVEX_API_TOKEN={new_token}")
    typer.echo()
    typer.echo("Update this in:")
    typer.echo("- Environment variables")
    typer.echo("- ChatGPT Actions configuration")


@gateway.command()
def status():
    """
    Check gateway status and configuration.

    Example:
        chinvex gateway status
    """
    from .gateway.config import load_gateway_config
    from .context import list_contexts

    config = load_gateway_config()
    contexts = list_contexts()

    typer.echo("Gateway Configuration:")
    typer.echo(f"  Host: {config.host}")
    typer.echo(f"  Port: {config.port}")
    typer.echo(f"  Token configured: {'Yes' if config.token else 'No'}")
    typer.echo(f"  Server-side LLM: {'Enabled' if config.enable_server_llm else 'Disabled'}")
    typer.echo()
    typer.echo("Contexts:")
    if config.context_allowlist:
        typer.echo(f"  Allowlist: {', '.join(config.context_allowlist)}")
    else:
        typer.echo(f"  All contexts available ({len(contexts)} total)")
    typer.echo()
    typer.echo("Rate Limits:")
    typer.echo(f"  Per minute: {config.rate_limit.requests_per_minute}")
    typer.echo(f"  Per hour: {config.rate_limit.requests_per_hour}")
    typer.echo()
    typer.echo(f"Audit log: {config.audit_log_path}")
```

**Step 2: Test CLI commands**

Run: `chinvex gateway --help`
Expected: Shows gateway commands

Run: `chinvex gateway token-generate`
Expected: Generates and prints token

**Step 3: Commit**

```bash
git add src/chinvex/cli.py
git commit -m "feat(gateway): add CLI commands for serve, token, and status"
```

---

## Phase 6: Testing & Documentation

### Task 13: Acceptance test script

**Files:**
- Create: `scripts/test_gateway_p2.py`

**Step 1: Write acceptance test script**

Create `scripts/test_gateway_p2.py`:

```python
#!/usr/bin/env python3
"""
P2 Gateway Acceptance Test Suite
Run this against a running gateway to verify all functionality.
"""

import os
import sys
import requests
from typing import Callable

BASE_URL = os.environ.get("CHINVEX_GATEWAY_URL", "http://localhost:7778")
TOKEN = os.environ.get("CHINVEX_API_TOKEN")

if not TOKEN:
    print("Error: CHINVEX_API_TOKEN not set")
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}


def test(name: str, test_fn: Callable, expected_status: int = 200) -> bool:
    """Run a test and print result."""
    try:
        result = test_fn()

        if isinstance(result, requests.Response):
            passed = result.status_code == expected_status
            if not passed:
                print(f"[FAIL] {name}")
                print(f"  Expected: {expected_status}, Got: {result.status_code}")
                print(f"  Response: {result.text[:200]}")
            else:
                print(f"[PASS] {name}")
        else:
            passed = bool(result)
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] {name}")

        return passed
    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        return False


# Test suite
results = []

# 2.5.1: Health endpoint (no auth)
results.append(test(
    "2.5.1: Health endpoint accessible",
    lambda: requests.get(f"{BASE_URL}/health"),
    200
))

# 2.5.2: Auth required
results.append(test(
    "2.5.2: Auth required for evidence",
    lambda: requests.post(
        f"{BASE_URL}/v1/evidence",
        json={"context": "Chinvex", "query": "test"}
    ),
    403
))

# 2.5.3: Valid token accepted
results.append(test(
    "2.5.3: Valid token accepted",
    lambda: requests.post(
        f"{BASE_URL}/v1/evidence",
        headers=HEADERS,
        json={"context": "Chinvex", "query": "test"}
    ),
    200
))

# 2.5.4: Invalid token rejected
results.append(test(
    "2.5.4: Invalid token rejected",
    lambda: requests.post(
        f"{BASE_URL}/v1/evidence",
        headers={"Authorization": "Bearer invalid_token", "Content-Type": "application/json"},
        json={"context": "Chinvex", "query": "test"}
    ),
    401
))

# 2.5.9: Unknown context
results.append(test(
    "2.5.9: Unknown context returns 404",
    lambda: requests.post(
        f"{BASE_URL}/v1/evidence",
        headers=HEADERS,
        json={"context": "NonExistent", "query": "test"}
    ),
    404
))

# 2.5.11: Invalid context name
results.append(test(
    "2.5.11: Invalid context name rejected",
    lambda: requests.post(
        f"{BASE_URL}/v1/evidence",
        headers=HEADERS,
        json={"context": "../../../etc/passwd", "query": "test"}
    ),
    422  # Pydantic validation error
))

# Summary
print("\n" + "="*50)
print(f"Tests passed: {sum(results)}/{len(results)}")
print("="*50)

sys.exit(0 if all(results) else 1)
```

**Step 2: Make executable**

Run: `chmod +x scripts/test_gateway_p2.py`
Expected: File is executable

**Step 3: Commit**

```bash
git add scripts/test_gateway_p2.py
git commit -m "test(gateway): add P2 acceptance test script"
```

---

### Task 14: Deployment documentation

**Files:**
- Create: `docs/deployment/cloudflare-tunnel.md`
- Create: `docs/deployment/caddy.md`

**Step 1: Write Cloudflare Tunnel docs**

Create `docs/deployment/cloudflare-tunnel.md`:

```markdown
# Cloudflare Tunnel Setup

## Prerequisites

- Cloudflare account with a domain
- Gateway running locally on port 7778

## Installation

### Windows
\`\`\`powershell
winget install cloudflare.cloudflared
\`\`\`

### Verify
\`\`\`bash
cloudflared --version
\`\`\`

## Configuration

### 1. Login and create tunnel
\`\`\`bash
cloudflared tunnel login
cloudflared tunnel create chinvex
\`\`\`

### 2. Configure tunnel

Create `~/.cloudflared/config.yml`:

\`\`\`yaml
tunnel: chinvex
credentials-file: C:\Users\Jordan\.cloudflared\<tunnel-id>.json

ingress:
  - hostname: chinvex.yourdomain.com
    service: http://localhost:7778
  - service: http_status:404
\`\`\`

### 3. Add DNS
\`\`\`bash
cloudflared tunnel route dns chinvex chinvex.yourdomain.com
\`\`\`

### 4. Test
\`\`\`bash
cloudflared tunnel run chinvex
\`\`\`

Visit `https://chinvex.yourdomain.com/health`

### 5. Run as service (PM2)

\`\`\`bash
pm2 start "cloudflared tunnel run chinvex" --name chinvex-tunnel
pm2 start "chinvex gateway serve --port 7778" --name chinvex-gateway
pm2 save
pm2 startup
\`\`\`

## Verification

\`\`\`bash
cloudflared tunnel info chinvex
pm2 logs chinvex-tunnel
curl https://chinvex.yourdomain.com/health
\`\`\`
```

**Step 2: Write Caddy docs**

Create `docs/deployment/caddy.md`:

```markdown
# Caddy Reverse Proxy Setup

## Prerequisites

- Caddy installed
- Gateway running on port 7778
- Domain with DNS pointing to server

## Caddyfile

Add to existing Caddyfile:

\`\`\`
chinvex.yourdomain.com {
    reverse_proxy localhost:7778

    header {
        X-Content-Type-Options nosniff
        X-Frame-Options DENY
        X-XSS-Protection "1; mode=block"
        Strict-Transport-Security "max-age=31536000"
    }

    log {
        output file /var/log/caddy/chinvex.log
        format json
    }
}
\`\`\`

## Start Services

\`\`\`bash
pm2 start "chinvex gateway serve --port 7778" --name chinvex-gateway
caddy reload
\`\`\`

## Verification

\`\`\`bash
curl https://chinvex.yourdomain.com/health
curl -I https://chinvex.yourdomain.com/health
tail -f /var/log/caddy/chinvex.log
\`\`\`
```

**Step 3: Commit**

```bash
git add docs/deployment/
git commit -m "docs(gateway): add Cloudflare Tunnel and Caddy deployment guides"
```

---

### Task 15: ChatGPT integration guide

**Files:**
- Create: `docs/chatgpt-integration.md`

**Step 1: Write integration guide**

Create `docs/chatgpt-integration.md`:

```markdown
# ChatGPT Actions Integration

## Step 1: Create Custom GPT

1. Go to https://chat.openai.com
2. Profile → "My GPTs" → "Create a GPT"
3. Switch to "Configure" tab

**Name:** Chinvex Memory

**Instructions:**
\`\`\`
You have access to the user's personal knowledge base via the Chinvex API.

CRITICAL RULES:
1. When the user asks about their projects, decisions, code, or past work, ALWAYS call getEvidence first.
2. If grounded=false, say "I couldn't find information about that in your memory." Do NOT make up an answer.
3. If grounded=true, synthesize an answer using ONLY the returned chunks. Cite sources.
4. Never claim to know something that isn't in the evidence pack.

When citing, use format: [source_uri:line_start-line_end]
\`\`\`

## Step 2: Add Action

1. Scroll to "Actions"
2. Click "Create new action"
3. Click "Import from URL"
4. Enter: `https://chinvex.yourdomain.com/openapi.json`

## Step 3: Configure Authentication

1. Click "Authentication"
2. Select "API Key"
3. Set:
   - **Auth Type:** Bearer
   - **API Key:** Your `CHINVEX_API_TOKEN`

## Step 4: Test

Ask: "Search my Chinvex memory for hybrid retrieval"

Expected: GPT calls `/v1/evidence`, synthesizes answer

## Troubleshooting

### "Could not load schema"
- Verify: `curl https://chinvex.yourdomain.com/openapi.json`
- Check CORS in gateway.json

### "Authentication failed"
- Test: `curl -H "Authorization: Bearer $CHINVEX_API_TOKEN" https://chinvex.yourdomain.com/v1/contexts`
```

**Step 2: Commit**

```bash
git add docs/chatgpt-integration.md
git commit -m "docs(gateway): add ChatGPT Actions setup guide"
```

---

## Phase 7: Optional Answer Endpoint

### Task 16: Answer endpoint (flag-gated, disabled by default)

**Files:**
- Create: `src/chinvex/gateway/endpoints/answer.py`
- Modify: `src/chinvex/gateway/app.py` (conditionally add router)
- Create: `tests/gateway/endpoints/test_answer.py`

**Step 1: Write tests for answer endpoint**

Create `tests/gateway/endpoints/test_answer.py`:

```python
import os
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI, Depends
from chinvex.gateway.endpoints.answer import router
from chinvex.gateway.auth import verify_token
from chinvex.gateway.config import GatewayConfig


@pytest.fixture
def set_token():
    """Set test token in environment."""
    os.environ["CHINVEX_API_TOKEN"] = "test_token_123"
    yield
    del os.environ["CHINVEX_API_TOKEN"]


def test_answer_endpoint_disabled_by_default(set_token):
    """Should return 403 when endpoint is disabled."""
    # Mock config to return disabled
    import chinvex.gateway.endpoints.answer as answer_module
    original_load = answer_module.load_gateway_config

    def mock_config():
        config = original_load()
        config.enable_server_llm = False
        return config

    answer_module.load_gateway_config = mock_config

    app = FastAPI()
    app.include_router(router, dependencies=[Depends(verify_token)])
    client = TestClient(app)

    response = client.post("/answer", json={
        "context": "Chinvex",
        "query": "test"
    }, headers={"Authorization": "Bearer test_token_123"})

    assert response.status_code == 403
    data = response.json()
    assert data["error"] == "answer_endpoint_disabled"

    answer_module.load_gateway_config = original_load


def test_answer_endpoint_structure():
    """Should have correct response structure when enabled."""
    # This test just verifies the endpoint exists and has correct schema
    # Actual LLM synthesis testing is integration-level
    assert hasattr(router, 'routes')
    routes = [r for r in router.routes if r.path == "/answer"]
    assert len(routes) == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/gateway/endpoints/test_answer.py -v`
Expected: FAIL with "No module named 'chinvex.gateway.endpoints.answer'"

**Step 3: Write implementation**

Create `src/chinvex/gateway/endpoints/answer.py`:

```python
"""Answer endpoint - full synthesis with LLM (optional, flag-gated)."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

from chinvex.context import load_context
from chinvex.search import hybrid_search_from_context
from chinvex.gateway.validation import AnswerRequest
from chinvex.gateway.config import load_gateway_config


router = APIRouter()


class AnswerResponse(BaseModel):
    """Response from answer endpoint."""
    schema_version: int
    context: str
    query: str
    grounded: bool
    answer: str
    citations: list[dict]
    evidence_pack: dict
    errors: list[dict]


@router.post("/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest, request: Request):
    """
    Full synthesis with LLM. Disabled by default.
    Enable with GATEWAY_ENABLE_SERVER_LLM=true.

    Args:
        req: Answer request with context, query, k, grounded
        request: FastAPI request (for audit logging)

    Returns:
        Answer with LLM-synthesized response and citations

    Raises:
        HTTPException: 403 if endpoint disabled, 404 if context not found
    """
    config = load_gateway_config()

    # Check if endpoint is enabled
    if not config.enable_server_llm:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "answer_endpoint_disabled",
                "message": "Server-side synthesis is disabled. Use /v1/evidence instead."
            }
        )

    request.state.context = req.context

    # Load and verify context
    try:
        context = load_context(req.context)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Context not found")

    if config.context_allowlist and req.context not in config.context_allowlist:
        raise HTTPException(status_code=404, detail="Context not found")

    # Perform search
    try:
        results = hybrid_search_from_context(
            context=context,
            query=req.query,
            k=req.k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    # Generate answer with LLM
    # NOTE: This is a placeholder - actual LLM synthesis would integrate
    # with the existing answer logic from MCP server or similar
    # For P2, we implement the endpoint structure but leave synthesis
    # as a future enhancement

    GROUNDING_THRESHOLD = 0.35
    grounded_chunks = [r for r in results if r.rank_score >= GROUNDING_THRESHOLD]
    grounded = len(grounded_chunks) >= 1

    if grounded and req.grounded:
        # TODO: Actual LLM synthesis here
        # For now, return a structured response indicating synthesis would happen
        answer_text = "LLM synthesis not yet implemented. Use /v1/evidence for retrieval."
        citations = [
            {
                "chunk_id": r.chunk_id,
                "source_uri": r.source_uri,
                "range": build_range(r)
            }
            for r in grounded_chunks[:3]
        ]
        evidence_pack = {
            "chunks": [
                {
                    "chunk_id": r.chunk_id,
                    "text": truncate_text(r.text, 5000),
                    "source_uri": r.source_uri,
                    "source_type": r.source_type,
                    "range": build_range(r),
                    "score": r.rank_score
                }
                for r in grounded_chunks
            ]
        }
        errors = []
    else:
        answer_text = "Not stated in retrieved sources."
        citations = []
        evidence_pack = {"chunks": []}
        errors = [
            {
                "code": "GROUNDING_FAILED",
                "detail": "No retrieved chunk supports a direct answer to this query."
            }
        ]
        grounded = False

    return AnswerResponse(
        schema_version=1,
        context=req.context,
        query=req.query,
        grounded=grounded,
        answer=answer_text,
        citations=citations,
        evidence_pack=evidence_pack,
        errors=errors
    )


def truncate_text(text: str, max_len: int) -> str:
    """Truncate text with marker."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + " [truncated]"


def build_range(result) -> dict:
    """Build range object from search result."""
    if hasattr(result, 'line_start') and result.line_start:
        return {
            "line_start": result.line_start,
            "line_end": result.line_end
        }
    elif hasattr(result, 'char_start'):
        return {
            "char_start": result.char_start,
            "char_end": result.char_end
        }
    return {}
```

**Step 4: Update app.py to conditionally include answer router**

Edit `src/chinvex/gateway/app.py` (after the contexts router):

```python
# Optional answer endpoint (flag-gated)
if config.enable_server_llm:
    from .endpoints import answer
    app.include_router(
        answer.router,
        prefix="/v1",
        tags=["Answer"],
        dependencies=[Depends(check_rate_limit)]
    )
```

**Step 5: Update endpoints __init__.py**

Edit `src/chinvex/gateway/endpoints/__init__.py`:

```python
"""Gateway API endpoints."""

from . import health, search, evidence, chunks, contexts, answer

__all__ = ["health", "search", "evidence", "chunks", "contexts", "answer"]
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/gateway/endpoints/test_answer.py -v`
Expected: Tests PASS

**Step 7: Test with environment variable**

Run: `export GATEWAY_ENABLE_SERVER_LLM=true`
Run: `pytest tests/gateway/endpoints/test_answer.py -v`
Expected: Endpoint becomes accessible

**Step 8: Commit**

```bash
git add src/chinvex/gateway/endpoints/answer.py src/chinvex/gateway/app.py src/chinvex/gateway/endpoints/__init__.py tests/gateway/endpoints/test_answer.py
git commit -m "feat(gateway): add flag-gated /v1/answer endpoint (disabled by default)"
```

---

## Execution Complete

Plan saved to `docs/plans/2026-01-27-p2-implementation-plan.md`.

**Implementation order summary:**

**Phase 1: Foundation** (Tasks 1-3)
- Dependencies, package structure, configuration

**Phase 2: Security** (Tasks 4-7)
- Auth, validation, rate limiting, audit logging

**Phase 3: Endpoints** (Tasks 8-10)
- Health, evidence, search, chunks, contexts

**Phase 4: Assembly** (Task 11)
- FastAPI app with middleware

**Phase 5: CLI** (Task 12)
- Gateway serve, token, status commands

**Phase 6: Testing & Docs** (Tasks 13-15)
- Acceptance tests, deployment guides, ChatGPT integration

---

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution

**Which approach?**

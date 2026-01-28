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

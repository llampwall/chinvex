"""Test Redis-backed rate limiting."""
import pytest
from unittest.mock import Mock, patch
import time


def test_redis_rate_limiter_allows_within_limit():
    """Test that requests within limit are allowed."""
    from chinvex.gateway.rate_limit_redis import RedisRateLimiter

    # Mock Redis client
    mock_redis = Mock()
    mock_redis.incr.return_value = 5  # 5 requests

    limiter = RedisRateLimiter(mock_redis, requests_per_minute=60)

    assert limiter.check_rate_limit("client_id") is True


def test_redis_rate_limiter_blocks_over_limit():
    """Test that requests over limit are blocked."""
    from chinvex.gateway.rate_limit_redis import RedisRateLimiter

    mock_redis = Mock()
    mock_redis.incr.return_value = 65  # Over 60

    limiter = RedisRateLimiter(mock_redis, requests_per_minute=60)

    assert limiter.check_rate_limit("client_id") is False


def test_redis_fallback_to_memory_on_failure():
    """Test fallback to in-memory limiter if Redis unavailable."""
    import sys

    # Remove redis from sys.modules if it exists
    redis_module = sys.modules.pop('redis', None)

    try:
        # Patch sys.modules to prevent redis import
        with patch.dict('sys.modules', {'redis': None}):
            from chinvex.gateway.rate_limit_redis import create_rate_limiter, MemoryRateLimiter
            limiter = create_rate_limiter("redis://localhost:6379")
    finally:
        # Restore redis module if it existed
        if redis_module is not None:
            sys.modules['redis'] = redis_module

    # Should fallback to memory limiter
    assert limiter is not None
    assert isinstance(limiter, MemoryRateLimiter)


def test_memory_rate_limiter_works():
    """Test that memory fallback limiter works."""
    from chinvex.gateway.rate_limit_redis import MemoryRateLimiter

    limiter = MemoryRateLimiter(requests_per_minute=5, requests_per_hour=10)

    # Should allow first 5 requests
    for i in range(5):
        assert limiter.check_rate_limit("client1") is True

    # Should block 6th request
    assert limiter.check_rate_limit("client1") is False

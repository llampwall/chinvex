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

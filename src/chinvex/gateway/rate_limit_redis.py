"""Redis-backed rate limiting."""
import time
from typing import Optional


class RedisRateLimiter:
    """Redis-backed rate limiter using sliding window."""

    def __init__(self, redis_client, requests_per_minute: int = 60, requests_per_hour: int = 500):
        self.redis = redis_client
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour

    def check_rate_limit(self, client_id: str) -> bool:
        """
        Check if client is within rate limit.

        Returns True if allowed, False if rate limited.
        """
        now = int(time.time())

        # Check minute window
        minute_key = f"ratelimit:{client_id}:minute:{now // 60}"
        try:
            count = self.redis.incr(minute_key)
            if count == 1:
                self.redis.expire(minute_key, 120)  # 2 minutes TTL

            if count > self.requests_per_minute:
                return False
        except Exception as e:
            print(f"Redis rate limit check failed: {e}")
            return True  # Fail open

        # Check hour window
        hour_key = f"ratelimit:{client_id}:hour:{now // 3600}"
        try:
            count = self.redis.incr(hour_key)
            if count == 1:
                self.redis.expire(hour_key, 7200)  # 2 hours TTL

            if count > self.requests_per_hour:
                return False
        except Exception:
            pass

        return True


class MemoryRateLimiter:
    """In-memory rate limiter fallback."""

    def __init__(self, requests_per_minute: int = 60, requests_per_hour: int = 500):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.counts = {}  # {client_id: [(timestamp, count), ...]}

    def check_rate_limit(self, client_id: str) -> bool:
        """Check rate limit using in-memory counters."""
        now = time.time()

        # Clean old entries
        if client_id in self.counts:
            self.counts[client_id] = [
                (ts, cnt) for ts, cnt in self.counts[client_id]
                if now - ts < 3600  # Keep last hour
            ]
        else:
            self.counts[client_id] = []

        # Count requests in windows
        minute_count = sum(cnt for ts, cnt in self.counts[client_id] if now - ts < 60)
        hour_count = sum(cnt for ts, cnt in self.counts[client_id] if now - ts < 3600)

        if minute_count >= self.requests_per_minute or hour_count >= self.requests_per_hour:
            return False

        # Record request
        self.counts[client_id].append((now, 1))
        return True


def create_rate_limiter(redis_url: Optional[str], requests_per_minute: int = 60, requests_per_hour: int = 500):
    """
    Create rate limiter with Redis or memory backend.

    Falls back to memory if Redis unavailable.
    """
    if redis_url:
        try:
            import redis
            client = redis.from_url(redis_url)
            client.ping()  # Test connection
            print(f"Using Redis rate limiter: {redis_url}")
            return RedisRateLimiter(client, requests_per_minute, requests_per_hour)
        except Exception as e:
            print(f"Redis connection failed, using memory fallback: {e}")

    return MemoryRateLimiter(requests_per_minute, requests_per_hour)

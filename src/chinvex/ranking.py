# src/chinvex/ranking.py
import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

def recency_factor(updated_at: datetime, half_life_days: int = 90) -> float:
    """
    Exponential decay. Score halves every `half_life_days`.

    Args:
        updated_at: When document was last updated
        half_life_days: Days for score to decay by half

    Returns:
        Decay factor in (0, 1]

    Note:
        Uses UTC everywhere to avoid DST/timezone bugs.
    """
    now = datetime.now(timezone.utc)

    # Ensure updated_at is timezone-aware
    if updated_at.tzinfo is None:
        # Assume naive timestamps are UTC (with warning)
        updated_at = updated_at.replace(tzinfo=timezone.utc)
        log.warning(f"Naive timestamp encountered, assuming UTC: {updated_at}")

    age_days = (now - updated_at).days
    if age_days <= 0:
        return 1.0

    return 0.5 ** (age_days / half_life_days)

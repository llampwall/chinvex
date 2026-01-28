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


def apply_recency_decay(results, half_life_days: int = 90):
    """
    Apply recency decay to search results.

    Multiplies each result's rank_score by a recency factor based on updated_at.
    Results without updated_at are not penalized.

    Args:
        results: List of search results with updated_at and rank_score attributes
        half_life_days: Days for score to decay by half

    Returns:
        Modified list of results with adjusted rank_score
    """
    for result in results:
        updated_at = getattr(result, 'updated_at', None)
        if updated_at and isinstance(updated_at, str):
            try:
                # Parse ISO format timestamp
                updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                factor = recency_factor(updated_at, half_life_days)
                # Multiply rank_score by recency factor
                result.rank_score = result.rank_score * factor
            except (ValueError, AttributeError) as e:
                log.warning(f"Could not parse updated_at '{updated_at}': {e}")
                # Skip recency adjustment for this result

    # Re-sort by adjusted rank_score
    results.sort(key=lambda r: r.rank_score, reverse=True)
    return results

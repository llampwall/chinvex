# tests/test_ranking.py
from datetime import datetime, timedelta, timezone
from chinvex.ranking import recency_factor

def test_recency_factor_current_document():
    """Test recency factor is 1.0 for current documents."""
    now = datetime.now(timezone.utc)
    factor = recency_factor(now, half_life_days=90)
    assert factor == 1.0

def test_recency_factor_old_document():
    """Test recency factor decays for old documents."""
    now = datetime.now(timezone.utc)
    ninety_days_ago = now - timedelta(days=90)

    factor = recency_factor(ninety_days_ago, half_life_days=90)
    assert 0.49 < factor < 0.51  # Should be ~0.5

def test_recency_factor_very_old_document():
    """Test recency factor for very old documents."""
    now = datetime.now(timezone.utc)
    one_year_ago = now - timedelta(days=365)

    factor = recency_factor(one_year_ago, half_life_days=90)
    assert 0 < factor < 0.1  # Should be very small but not zero

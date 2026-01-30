"""Status artifact generation."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


def write_status_json(
    context_dir: Path,
    stats: dict,
    sources: list[dict],
    embedding: dict,
    stale_after_hours: int = 6
) -> None:
    """
    Write STATUS.json for a context after ingest.

    Args:
        context_dir: Context directory (e.g., contexts/Chinvex)
        stats: Ingest stats dict
        sources: List of source dicts with {type, path, watching}
        embedding: Embedding config dict
        stale_after_hours: Hours before context is considered stale (default 6)
    """
    context_name = context_dir.name

    # Compute freshness
    last_sync = stats.get("last_sync", datetime.now(timezone.utc).isoformat())
    freshness = compute_freshness(last_sync, stale_after_hours=stale_after_hours)

    status = {
        "context": context_name,
        "last_sync": last_sync,
        "chunks": stats.get("chunks", 0),
        "watches_active": stats.get("watches_active", 0),
        "watches_pending_hits": stats.get("watches_pending_hits", 0),
        "freshness": freshness,
        "sources": sources,
        "embedding": embedding
    }

    status_file = context_dir / "STATUS.json"
    status_file.write_text(json.dumps(status, indent=2))
    log.info(f"Wrote {status_file}")


def compute_freshness(last_sync: str, stale_after_hours: int = 6) -> dict:
    """
    Compute freshness status.

    Args:
        last_sync: ISO timestamp of last sync
        stale_after_hours: Hours before considered stale

    Returns:
        Dict with stale_after_hours, is_stale, hours_since_sync
    """
    last_sync_dt = datetime.fromisoformat(last_sync)
    if last_sync_dt.tzinfo is None:
        last_sync_dt = last_sync_dt.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    elapsed = now - last_sync_dt
    hours_since = elapsed.total_seconds() / 3600

    return {
        "stale_after_hours": stale_after_hours,
        "is_stale": hours_since > stale_after_hours,
        "hours_since_sync": round(hours_since, 2)
    }

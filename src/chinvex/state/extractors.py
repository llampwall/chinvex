# src/chinvex/state/extractors.py
from datetime import datetime, timezone
from chinvex.state.models import RecentlyChanged


def extract_recently_changed(
    context: str,
    since: datetime,
    limit: int = 20,
    db_path: str = None
) -> list[RecentlyChanged]:
    """
    Get docs changed since last state generation.

    Args:
        context: Context name
        since: Only include docs changed after this time
        limit: Max number of results
        db_path: Override DB path (for testing)
    """
    # Import here to avoid circular dependency
    import sqlite3

    if db_path is None:
        db_path = f"P:/ai_memory/indexes/{context}/hybrid.db"

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cursor = conn.execute("""
        SELECT source_uri, source_type, doc_id, last_ingested_at_unix
        FROM source_fingerprints
        WHERE context_name = ?
          AND last_ingested_at_unix > ?
          AND last_status = 'ok'
        ORDER BY last_ingested_at_unix DESC
        LIMIT ?
    """, [context, since.timestamp(), limit])

    results = []
    for row in cursor:
        results.append(RecentlyChanged(
            doc_id=row['doc_id'],
            source_type=row['source_type'],
            source_uri=row['source_uri'],
            change_type="modified",  # TODO: detect "new" vs "modified"
            changed_at=datetime.fromtimestamp(row['last_ingested_at_unix'], tz=timezone.utc)
        ))

    conn.close()
    return results

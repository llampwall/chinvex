"""Archive tier implementation."""
from datetime import datetime, timedelta
from .storage import Storage
from .util import now_iso


def get_doc_age_timestamp(row: dict) -> datetime | None:
    """
    Get the timestamp used for archive age calculation.

    Uses updated_at for content age.
    """
    ts_str = row.get("updated_at")
    if not ts_str:
        return None

    # Parse ISO8601 timestamp
    try:
        # Remove 'Z' suffix if present
        ts_str = ts_str.rstrip("Z")
        return datetime.fromisoformat(ts_str)
    except ValueError:
        return None


def archive_old_documents(storage: Storage, age_threshold_days: int, dry_run: bool = False) -> int:
    """
    Archive documents older than threshold.

    Returns count of documents archived (or would be archived in dry-run).
    """
    threshold_date = datetime.utcnow() - timedelta(days=age_threshold_days)

    # Find candidates
    cursor = storage.conn.execute(
        """
        SELECT doc_id, updated_at
        FROM documents
        WHERE archived = 0
        """
    )

    candidates = []
    for row in cursor.fetchall():
        row_dict = dict(row)
        doc_age = get_doc_age_timestamp(row_dict)
        if doc_age and doc_age < threshold_date:
            candidates.append(row_dict["doc_id"])

    if dry_run:
        return len(candidates)

    # Execute archive
    if candidates:
        archived_at = now_iso()
        placeholders = ",".join("?" * len(candidates))
        storage._execute(
            f"""
            UPDATE documents
            SET archived = 1, archived_at = ?
            WHERE doc_id IN ({placeholders})
            """,
            (archived_at, *candidates)
        )
        storage.conn.commit()

    return len(candidates)

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
    from datetime import timezone

    threshold_date_naive = datetime.utcnow() - timedelta(days=age_threshold_days)
    threshold_date_aware = datetime.now(timezone.utc) - timedelta(days=age_threshold_days)

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
        if doc_age:
            # Use appropriate threshold based on whether doc_age is timezone-aware
            threshold = threshold_date_aware if doc_age.tzinfo is not None else threshold_date_naive
            if doc_age < threshold:
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


def list_archived_documents(storage: Storage, limit: int = 50) -> list[dict]:
    """
    List archived documents.

    Returns list of archived document metadata.
    """
    cursor = storage.conn.execute(
        """
        SELECT doc_id, source_type, title, archived_at
        FROM documents
        WHERE archived = 1
        ORDER BY archived_at DESC
        LIMIT ?
        """,
        (limit,)
    )

    return [dict(row) for row in cursor.fetchall()]


def restore_document(storage: Storage, doc_id: str) -> bool:
    """
    Restore archived document.

    Flips archived flag to 0. Does NOT re-ingest or re-embed.
    Returns True if document was found and restored.
    """
    cursor = storage.conn.execute(
        "SELECT archived FROM documents WHERE doc_id = ?",
        (doc_id,)
    )
    row = cursor.fetchone()

    if not row:
        return False

    if row["archived"] == 0:
        return False  # Already active

    storage._execute(
        "UPDATE documents SET archived = 0, archived_at = NULL WHERE doc_id = ?",
        (doc_id,)
    )
    storage.conn.commit()

    return True

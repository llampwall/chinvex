"""Watch history reading and formatting."""
import json
from datetime import datetime
from pathlib import Path


def read_watch_history(
    ctx,
    since: datetime | None = None,
    watch_id: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """
    Read watch history with filters.

    Returns list of history entries (most recent first).
    """
    # Use the index directory as the base for watch history
    history_file = Path(ctx.index.sqlite_path).parent / "watch_history.jsonl"

    if not history_file.exists():
        return []

    entries = []
    with open(history_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)

            # Apply filters
            if since:
                entry_ts = datetime.fromisoformat(entry["ts"].rstrip("Z"))
                if entry_ts < since:
                    continue

            if watch_id and entry["watch_id"] != watch_id:
                continue

            entries.append(entry)

    # Most recent first
    entries.reverse()

    # Apply limit
    return entries[:limit]


def format_history_table(entries: list[dict]) -> str:
    """Format history as ASCII table."""
    if not entries:
        return "No watch history found."

    lines = []
    lines.append(f"{'Timestamp':<20} {'Watch ID':<15} {'Hits':<5} {'Query':<30}")
    lines.append("-" * 75)

    for entry in entries:
        ts = entry["ts"][:19]  # Strip milliseconds
        watch_id = entry["watch_id"][:14]
        hits = len(entry["hits"])
        query = entry["query"][:29]

        lines.append(f"{ts:<20} {watch_id:<15} {hits:<5} {query:<30}")

    return "\n".join(lines)


def format_history_json(entries: list[dict]) -> str:
    """Format history as JSON."""
    return json.dumps(entries, indent=2)

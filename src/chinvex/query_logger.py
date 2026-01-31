from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class QueryLogEntry:
    """Single query log entry."""
    timestamp: str
    context: str
    query: str
    k: int
    num_results: int
    top_chunk_ids: list[str]
    top_scores: list[float]
    latency_ms: float


class QueryLogger:
    """Logs search queries to .chinvex/logs/queries.jsonl."""

    def __init__(self, chinvex_dir: Path):
        self.chinvex_dir = Path(chinvex_dir)
        self.log_dir = self.chinvex_dir / "logs"
        self.log_file = self.log_dir / "queries.jsonl"

        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, entry: QueryLogEntry) -> None:
        """Append query log entry to JSONL file."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry)) + "\n")


def rotate_old_logs(chinvex_dir: Path, retention_days: int = 30) -> None:
    """Remove log entries older than retention_days.

    Args:
        chinvex_dir: Path to .chinvex directory
        retention_days: Number of days to retain logs (default 30)
    """
    log_file = Path(chinvex_dir) / "logs" / "queries.jsonl"

    if not log_file.exists():
        return

    cutoff = datetime.now() - timedelta(days=retention_days)

    # Read all entries
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return  # Silently skip on read errors

    if not lines:
        return

    # Filter to recent entries only
    recent_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            entry = json.loads(line)
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if entry_time >= cutoff:
                recent_lines.append(line + "\n")
        except (json.JSONDecodeError, KeyError, ValueError):
            # Skip malformed entries
            continue

    # Write back recent entries
    with open(log_file, "w", encoding="utf-8") as f:
        f.writelines(recent_lines)


def log_search_query(
    chinvex_dir: Path,
    context: str,
    query: str,
    results: list[dict],
    k: int,
) -> None:
    """Log a search query with results metadata.

    Args:
        chinvex_dir: Path to .chinvex directory
        context: Context name
        query: Search query string
        results: List of result dicts with 'chunk_id' and 'score'
        k: Number of results requested
    """
    start = time.time()

    logger = QueryLogger(chinvex_dir)

    entry = QueryLogEntry(
        timestamp=datetime.now().isoformat(),
        context=context,
        query=query,
        k=k,
        num_results=len(results),
        top_chunk_ids=[r["chunk_id"] for r in results],
        top_scores=[r["score"] for r in results],
        latency_ms=(time.time() - start) * 1000,
    )

    logger.log(entry)

import logging
import json
from datetime import datetime
from pathlib import Path
from chinvex.state.models import WatchHit

log = logging.getLogger(__name__)

def run_watches(
    context,
    new_chunk_ids: list[str],
    watches: list,
    timeout_per_watch: int = 30
) -> list[WatchHit]:
    """
    Run all enabled watches against newly ingested chunks.

    Args:
        context: Context object with search capability
        new_chunk_ids: List of newly created chunk IDs
        watches: List of Watch objects
        timeout_per_watch: Timeout in seconds per watch

    Returns:
        List of WatchHit objects for matches

    Note:
        Timeouts and errors are logged but don't fail entire run.
    """
    hits = []

    for watch in watches:
        if not watch.enabled:
            continue

        try:
            # Import search function (avoid circular import)
            from chinvex.search import search_chunks

            # Search only new chunks
            results = search_chunks(
                context=context,
                query=watch.query,
                chunk_ids=new_chunk_ids,
                k=10
            )

            # Filter by min_score
            matching = [r for r in results if r.blended_score >= watch.min_score]

            if matching:
                hits.append(WatchHit(
                    watch_id=watch.id,
                    query=watch.query,
                    hits=[
                        {
                            "chunk_id": r.chunk_id,
                            "score": r.blended_score,
                            "snippet": r.text[:200]
                        }
                        for r in matching[:5]
                    ],
                    triggered_at=datetime.now()
                ))

        except TimeoutError:
            log.warning(f"Watch {watch.id} timed out after {timeout_per_watch}s, skipping")
            continue
        except Exception as e:
            log.error(f"Watch {watch.id} failed: {e}")
            continue

    return hits


def append_watch_history(history_file: str, entry: dict):
    """
    Append watch history entry to JSONL log.

    Creates file if it doesn't exist.
    """
    Path(history_file).parent.mkdir(parents=True, exist_ok=True)

    with open(history_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')


def create_watch_history_entry(
    watch_id: str,
    query: str,
    hits: list[dict],
    run_id: str
) -> dict:
    """
    Create watch history entry with hit capping.

    Caps hits at 10 and marks as truncated if exceeded.
    """
    truncated = len(hits) > 10
    capped_hits = hits[:10]

    # Extract snippet (first 200 chars) from each hit
    formatted_hits = []
    for hit in capped_hits:
        formatted_hits.append({
            "chunk_id": hit["chunk_id"],
            "score": hit["score"],
            "snippet": hit.get("snippet", hit.get("text", ""))[:200]
        })

    entry = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "run_id": run_id,
        "watch_id": watch_id,
        "query": query,
        "hits": formatted_hits,
    }

    if truncated:
        entry["truncated"] = True

    return entry

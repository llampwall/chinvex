"""Global status aggregation across all contexts."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


def generate_global_status_md(
    contexts_root: Path,
    output_path: Path
) -> None:
    """
    Generate GLOBAL_STATUS.md from all context STATUS.json files.

    Args:
        contexts_root: Root directory containing contexts
        output_path: Path to write GLOBAL_STATUS.md
    """
    # Collect all STATUS.json
    contexts = []

    for ctx_dir in contexts_root.iterdir():
        if not ctx_dir.is_dir():
            continue

        status_file = ctx_dir / "STATUS.json"
        if not status_file.exists():
            continue

        try:
            data = json.loads(status_file.read_text())
            contexts.append(data)
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Failed to read {status_file}: {e}")

    # Check watcher health
    watcher_status = _get_watcher_status()

    # Calculate aggregates
    total_chunks = sum(c.get("chunks", 0) for c in contexts)
    stale_contexts = [c for c in contexts if c.get("freshness", {}).get("is_stale", False)]
    pending_hits = sum(c.get("watches_pending_hits", 0) for c in contexts)

    # Generate markdown
    now = datetime.now(timezone.utc).isoformat()

    md = f"""# Dual Nature Status
Generated: {now}

## Health
- **Watcher:** {watcher_status}
- **Stale Contexts:** {len(stale_contexts)} of {len(contexts)}
- **Total Chunks:** {total_chunks:,}
- **Pending Watch Hits:** {pending_hits}

## Contexts
| Context | Last Sync | Chunks | Watch Hits | Status |
|---------|-----------|--------|------------|--------|
"""

    # Sort by last_sync (most recent first)
    contexts_sorted = sorted(
        contexts,
        key=lambda c: c.get("last_sync", ""),
        reverse=True
    )

    for ctx in contexts_sorted:
        name = ctx.get("context", "unknown")
        chunks = ctx.get("chunks", 0)
        hits = ctx.get("watches_pending_hits", 0)

        # Format last sync
        last_sync = ctx.get("last_sync")
        if last_sync:
            sync_dt = datetime.fromisoformat(last_sync)
            elapsed = datetime.now(timezone.utc) - sync_dt
            if elapsed.total_seconds() < 3600:
                sync_str = f"{int(elapsed.total_seconds() / 60)}m ago"
            elif elapsed.total_seconds() < 86400:
                sync_str = f"{int(elapsed.total_seconds() / 3600)}h ago"
            else:
                sync_str = f"{int(elapsed.total_seconds() / 86400)}d ago"
        else:
            sync_str = "never"

        # Status icon
        is_stale = ctx.get("freshness", {}).get("is_stale", False)
        status_icon = "⚠ stale" if is_stale else "✓"

        md += f"| {name} | {sync_str} | {chunks:,} | {hits} | {status_icon} |\n"

    # Pending watch hits section
    if pending_hits > 0:
        md += "\n## Pending Watch Hits\n"
        for ctx in contexts:
            hits = ctx.get("watches_pending_hits", 0)
            if hits > 0:
                md += f"- **{ctx['context']}**: {hits} hits\n"

    # Write file
    output_path.write_text(md, encoding="utf-8")
    log.info(f"Wrote {output_path}")


def _get_watcher_status() -> str:
    """Get watcher daemon status string."""
    try:
        from .sync.cli import get_state_dir
        from .sync.daemon import DaemonManager

        state_dir = get_state_dir()
        dm = DaemonManager(state_dir)

        state = dm.get_state()

        if state.value == "running":
            return "RUNNING"
        elif state.value == "stale":
            return "STALE"
        else:
            return "NOT RUNNING"

    except Exception as e:
        log.warning(f"Failed to check watcher status: {e}")
        return "UNKNOWN"

"""Status command implementation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ContextStatus:
    name: str
    chunks: int
    last_sync: str
    is_stale: bool
    hours_since_sync: float
    watcher_running: bool
    embedding_provider: str  # NEW FIELD
    embedding_model: str | None = None  # NEW FIELD


def format_status_output(contexts: list[ContextStatus], watcher_running: bool) -> str:
    """
    Format status output as table.

    Args:
        contexts: List of context statuses
        watcher_running: Whether watcher is running globally

    Returns:
        Formatted status string
    """
    lines = ["# Chinvex Global Status", ""]

    # Table header (UPDATED to include Embedding column)
    lines.append("| Context | Chunks | Last Sync | Embedding | Status |")
    lines.append("|---------|--------|-----------|-----------|--------|")

    # Rows
    for ctx in contexts:
        status_icon = "[OK]" if not ctx.is_stale else "[STALE]"
        hours_str = f"{int(ctx.hours_since_sync)}h ago"

        # Format embedding info
        if ctx.embedding_model:
            embed_str = f"{ctx.embedding_provider}/{ctx.embedding_model}"
        else:
            embed_str = ctx.embedding_provider

        # Truncate if too long
        if len(embed_str) > 18:
            embed_str = embed_str[:15] + "..."

        lines.append(
            f"| {ctx.name:<15} | {ctx.chunks:<6} | {hours_str:<9} | {embed_str:<17} | {status_icon:<6} |"
        )

    lines.append("")
    lines.append(f"Watcher: {'Running' if watcher_running else 'Stopped'}")

    return "\n".join(lines)


def read_global_status(contexts_root: Path) -> str:
    """
    Read GLOBAL_STATUS.md if it exists.

    Args:
        contexts_root: Root directory for contexts

    Returns:
        Contents of GLOBAL_STATUS.md
    """
    global_status = contexts_root / "GLOBAL_STATUS.md"
    if not global_status.exists():
        return "GLOBAL_STATUS.md not found. Run ingest to generate."

    return global_status.read_text(encoding="utf-8")


def generate_status_from_contexts(contexts_root: Path) -> str:
    """
    Generate status by reading all STATUS.json files.

    Args:
        contexts_root: Root directory for contexts

    Returns:
        Formatted status string
    """
    statuses = []

    for ctx_dir in contexts_root.iterdir():
        if not ctx_dir.is_dir():
            continue
        if ctx_dir.name.startswith("_"):
            continue

        status_json = ctx_dir / "STATUS.json"
        context_json = ctx_dir / "context.json"  # NEW: Read context.json for embedding info

        if not status_json.exists():
            continue

        try:
            data = json.loads(status_json.read_text(encoding="utf-8"))
            freshness = data.get("freshness", {})

            # NEW: Read embedding provider from context.json
            embedding_provider = "ollama"  # Default
            embedding_model = None

            if context_json.exists():
                try:
                    ctx_data = json.loads(context_json.read_text(encoding="utf-8"))
                    if "embedding" in ctx_data:
                        embedding_provider = ctx_data["embedding"].get("provider", "ollama")
                        embedding_model = ctx_data["embedding"].get("model")
                    # If no embedding field, check ollama config for model name
                    elif "ollama" in ctx_data:
                        embedding_model = ctx_data["ollama"].get("embed_model")
                except (json.JSONDecodeError, KeyError):
                    pass  # Use defaults

            statuses.append(ContextStatus(
                name=ctx_dir.name,
                chunks=data.get("chunks", 0),
                last_sync=data.get("last_sync", "unknown"),
                is_stale=freshness.get("is_stale", False),
                hours_since_sync=freshness.get("hours_since_sync", 0),
                watcher_running=False,  # Determined globally
                embedding_provider=embedding_provider,
                embedding_model=embedding_model
            ))
        except (json.JSONDecodeError, KeyError):
            continue

    # Check watcher status
    watcher_running = _check_watcher_running()

    return format_status_output(statuses, watcher_running)


def _check_watcher_running() -> bool:
    """Check if sync watcher is running via daemon state."""
    try:
        from .sync.cli import get_state_dir
        from .sync.daemon import DaemonManager, DaemonState

        state_dir = get_state_dir()
        dm = DaemonManager(state_dir)
        state = dm.get_state()

        return state == DaemonState.RUNNING
    except Exception:
        return False

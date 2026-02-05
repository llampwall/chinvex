# src/chinvex/sync/discovery.py
"""Context discovery for file watcher."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class WatchSource:
    """A source directory to watch for changes."""
    context_name: str
    source_type: str  # "repo" or "inbox"
    path: Path


def discover_watch_sources(contexts_root: Path) -> list[WatchSource]:
    """
    Discover all watchable sources from contexts.

    Reads all context.json files and extracts:
    - repos (from includes.repos)
    - inbox paths (from includes.inbox)

    Skips malformed contexts with a warning.

    Args:
        contexts_root: Root directory containing context subdirectories

    Returns:
        List of WatchSource objects
    """
    sources: list[WatchSource] = []

    if not contexts_root.exists():
        log.warning(f"Contexts root does not exist: {contexts_root}")
        return sources

    # Iterate all subdirectories
    for ctx_dir in contexts_root.iterdir():
        if not ctx_dir.is_dir():
            continue

        ctx_file = ctx_dir / "context.json"
        if not ctx_file.exists():
            continue

        try:
            # Load context config
            with open(ctx_file, "r", encoding="utf-8") as f:
                ctx_data = json.load(f)

            ctx_name = ctx_data.get("name", ctx_dir.name)
            includes = ctx_data.get("includes", {})

            # Extract repo sources
            for repo_item in includes.get("repos", []):
                # Handle both old (string) and new (RepoMetadata dict) formats
                if isinstance(repo_item, str):
                    repo_path = Path(repo_item)
                elif isinstance(repo_item, dict):
                    repo_path = Path(repo_item["path"])
                else:
                    log.warning(f"Skipping invalid repo format in {ctx_name}: {type(repo_item)}")
                    continue

                sources.append(WatchSource(
                    context_name=ctx_name,
                    source_type="repo",
                    path=repo_path
                ))

            # Extract inbox sources
            for inbox_path_str in includes.get("inbox", []):
                inbox_path = Path(inbox_path_str)
                sources.append(WatchSource(
                    context_name=ctx_name,
                    source_type="inbox",
                    path=inbox_path
                ))

        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Skipping malformed context {ctx_dir.name}: {e}")
            continue

    log.info(f"Discovered {len(sources)} watch sources from {contexts_root}")
    return sources

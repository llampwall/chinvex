"""Exclude pattern matching for file watcher."""
from __future__ import annotations

import fnmatch
import platform
from pathlib import Path


# Exclude patterns (fnmatch glob syntax)
EXCLUDE_PATTERNS = [
    # Chinvex outputs (would cause ingest storm)
    "**/STATUS.json",
    "contexts/**/ingest_runs.jsonl",
    "contexts/**/digests/**",
    "**/MORNING_BRIEF.md",
    "**/GLOBAL_STATUS.md",
    "**/*_BRIEF.md",
    "**/SESSION_BRIEF.md",
    # Per-repo chinvex artifacts
    "**/.chinvex/**",
    "**/docs/memory/**",
    # Standard ignores
    "**/.git/**",
    "**/node_modules/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.venv/**",
    "**/venv/**",
    # Chinvex internals (will be expanded with home dir)
    ".chinvex/*.log",
    ".chinvex/*.pid",
    ".chinvex/*.json",
    ".chinvex/*.jsonl",
]


def should_exclude(path: str | Path, watch_root: Path) -> bool:
    """
    Check if a path should be excluded from watching.

    Args:
        path: Absolute or relative path to check
        watch_root: Root directory being watched

    Returns:
        True if path matches any exclude pattern
    """
    path_obj = Path(path)

    # Convert to relative path if absolute
    if path_obj.is_absolute():
        try:
            rel_path = path_obj.relative_to(watch_root)
        except ValueError:
            # Path outside watch_root, check against home for .chinvex patterns
            try:
                rel_path = path_obj.relative_to(Path.home())
            except ValueError:
                # Not under watch_root or home, use as-is
                rel_path = path_obj
    else:
        rel_path = path_obj

    # Normalize to forward slashes for fnmatch
    path_str = str(rel_path).replace("\\", "/")

    # Case-insensitive on Windows
    if platform.system() == "Windows":
        path_str = path_str.lower()
        patterns = [p.lower() for p in EXCLUDE_PATTERNS]
    else:
        patterns = EXCLUDE_PATTERNS

    # Check against all patterns
    for pattern in patterns:
        if fnmatch.fnmatch(path_str, pattern):
            return True
        # For patterns starting with **/, also check without the **/ prefix
        # This allows **/foo.json to match foo.json at root
        if pattern.startswith("**/"):
            simple_pattern = pattern[3:]  # Remove **/
            if fnmatch.fnmatch(path_str, simple_pattern):
                return True
        # Also check if any parent matches pattern (for directory exclusions)
        for parent in Path(path_str).parents:
            parent_str = str(parent).replace("\\", "/")
            if parent_str and parent_str != ".":
                if fnmatch.fnmatch(parent_str, pattern):
                    return True

    return False

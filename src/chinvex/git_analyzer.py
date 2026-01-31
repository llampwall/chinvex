# src/chinvex/git_analyzer.py
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GitCommit:
    """Represents a single git commit."""
    hash: str
    date: str
    author: str
    message: str


def extract_coverage_anchor(state_file: Path) -> str | None:
    """Extract last processed commit hash from STATE.md coverage anchor.

    Looks for: <!-- chinvex:last-commit:abc123def -->

    Args:
        state_file: Path to STATE.md

    Returns:
        Commit hash if found, None otherwise
    """
    if not state_file.exists():
        return None

    content = state_file.read_text()
    match = re.search(r"<!-- chinvex:last-commit:([a-f0-9]+) -->", content)
    return match.group(1) if match else None


def parse_commits(log_output: str) -> list[GitCommit]:
    """Parse git log output into GitCommit objects.

    Expected format: hash|||date|||author|||message---
    Separator between commits: ---

    Args:
        log_output: Output from git log --format

    Returns:
        List of GitCommit objects
    """
    if not log_output.strip():
        return []

    commits = []
    # Split by --- but it appears at the end of each commit
    commit_blocks = log_output.strip().split("---")

    for block in commit_blocks:
        block = block.strip()
        if not block:
            continue

        parts = block.split("|||", 3)
        if len(parts) < 4:
            continue

        hash_val, date, author, message = parts
        # Strip the trailing separator from message if present
        message = message.rstrip("\n-")
        commits.append(GitCommit(
            hash=hash_val.strip(),
            date=date.strip(),
            author=author.strip(),
            message=message.strip()
        ))

    return commits


def get_commit_range(
    repo_root: Path,
    start_hash: str | None = None,
    max_commits: int = 50
) -> list[GitCommit]:
    """Get commits from start_hash..HEAD.

    Args:
        repo_root: Repository root directory
        start_hash: Starting commit hash (exclusive). If None, gets all commits up to max_commits.
        max_commits: Maximum number of commits to return (bounded inputs guardrail)

    Returns:
        List of GitCommit objects in reverse chronological order (newest first)
    """
    # Build git log command
    format_str = "%H|||%ai|||%an|||%B"

    if start_hash:
        commit_range = f"{start_hash}..HEAD"
    else:
        commit_range = "HEAD"

    cmd = [
        "git", "log",
        commit_range,
        f"-{max_commits}",
        f"--format={format_str}---"
    ]

    result = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True
    )

    return parse_commits(result.stdout)

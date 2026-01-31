# src/chinvex/spec_reader.py
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


# Bounded inputs guardrails (from P5b spec)
BOUNDED_INPUTS = {
    "max_commits": 50,
    "max_files": 20,
    "max_total_size_kb": 100,
}


@dataclass
class SpecContent:
    """Container for spec/plan file contents with truncation flags."""
    specs: list[dict[str, str]]  # [{"path": str, "content": str}, ...]
    total_size: int  # Total bytes read
    truncated_files: bool  # True if max_files limit reached
    truncated_size: bool  # True if max_total_size limit reached


def extract_spec_plan_files_from_commits(
    repo_root: Path,
    start_hash: str | None = None
) -> list[Path]:
    """Extract unique spec/plan files touched in commit range.

    Only returns files from /specs/ and /docs/plans/ directories.

    Args:
        repo_root: Repository root
        start_hash: Starting commit hash (exclusive). If None, uses all history.

    Returns:
        List of unique spec/plan file paths (relative to repo root)
    """
    if start_hash:
        # Get files changed between start_hash and HEAD
        cmd = ["git", "diff", "--name-only", f"{start_hash}..HEAD"]
    else:
        # Get all tracked files in the repository
        cmd = ["git", "ls-files"]

    result = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True
    )

    files = result.stdout.strip().split("\n") if result.stdout.strip() else []

    # Filter to only specs/ and docs/plans/
    spec_files = []
    for file_path in files:
        if not file_path:
            continue

        # Check if in specs/ or docs/plans/
        if file_path.startswith("specs/") or file_path.startswith("docs/plans/"):
            full_path = repo_root / file_path
            if full_path.exists() and full_path.suffix == ".md":
                spec_files.append(full_path)

    return list(set(spec_files))  # Deduplicate


def read_spec_files(spec_paths: list[Path]) -> SpecContent:
    """Read spec/plan files with bounded inputs guardrails.

    Args:
        spec_paths: List of spec/plan file paths to read

    Returns:
        SpecContent with file contents and truncation flags
    """
    max_files = BOUNDED_INPUTS["max_files"]
    max_size = BOUNDED_INPUTS["max_total_size_kb"] * 1024

    specs = []
    total_size = 0
    truncated_files = False
    truncated_size = False

    for i, path in enumerate(spec_paths):
        # Check file count limit
        if i >= max_files:
            truncated_files = True
            break

        # Check if reading this file would exceed size limit
        file_size = path.stat().st_size
        if total_size + file_size > max_size:
            truncated_size = True
            break

        # Read file
        content = path.read_text(encoding="utf-8")
        specs.append({
            "path": str(path),
            "content": content
        })
        total_size += file_size

    return SpecContent(
        specs=specs,
        total_size=total_size,
        truncated_files=truncated_files,
        truncated_size=truncated_size
    )

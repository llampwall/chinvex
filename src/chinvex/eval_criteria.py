from __future__ import annotations

from pathlib import Path
from typing import Any

from .eval_runner import QueryResult


def check_file_match(retrieved_files: list[str], expected_files: list[str]) -> bool:
    """Check if any expected file appears in retrieved files.

    Args:
        retrieved_files: List of file paths from search results
        expected_files: List of acceptable file paths

    Returns:
        True if at least one expected file is in retrieved files
    """
    return any(expected in retrieved_files for expected in expected_files)


def check_anchor_match(file_path: str, anchor: str) -> bool:
    """Check if anchor string appears in file content.

    Args:
        file_path: Path to file to check
        anchor: String to search for (case-insensitive)

    Returns:
        True if anchor found in file content
    """
    try:
        content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        return anchor.lower() in content.lower()
    except (FileNotFoundError, OSError):
        return False


def evaluate_query(query_result: QueryResult) -> dict[str, Any]:
    """Evaluate a query result against success criteria.

    Args:
        query_result: Result from executing a golden query

    Returns:
        Dictionary with:
        - passed (bool): True if at least one expected file in top K
        - file_match (bool): Same as passed
        - rank (int | None): Position of first matched file (1-indexed)
        - matched_file (str | None): Path of first matched file
        - anchor_match (bool | None): True if anchor found in matched file
    """
    file_match = check_file_match(
        query_result.retrieved_files,
        query_result.expected_files
    )

    rank = None
    matched_file = None
    anchor_match = None

    if file_match:
        # Find rank of first matching file
        for i, retrieved_file in enumerate(query_result.retrieved_files):
            if retrieved_file in query_result.expected_files:
                rank = i + 1  # 1-indexed
                matched_file = retrieved_file
                break

        # Check anchor if specified
        if query_result.anchor and matched_file:
            anchor_match = check_anchor_match(matched_file, query_result.anchor)

    return {
        "passed": file_match,
        "file_match": file_match,
        "rank": rank,
        "matched_file": matched_file,
        "anchor_match": anchor_match
    }

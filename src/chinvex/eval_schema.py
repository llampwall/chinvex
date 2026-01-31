# src/chinvex/eval_schema.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class GoldenQuery:
    """Single golden query for eval."""
    query: str
    context: str
    expected_files: list[str]
    anchor: str | None = None
    k: int = 5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldenQuery:
        """Load from dictionary with validation."""
        # Validate required fields
        required = ["query", "context", "expected_files"]
        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Validate expected_files non-empty
        if not data["expected_files"]:
            raise ValueError("expected_files must contain at least one file")

        # Validate k if present
        k = data.get("k", 5)
        if k <= 0:
            raise ValueError("k must be positive")

        return cls(
            query=data["query"],
            context=data["context"],
            expected_files=data["expected_files"],
            anchor=data.get("anchor"),
            k=k
        )


@dataclass
class GoldenQuerySet:
    """Collection of golden queries."""
    queries: list[GoldenQuery]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldenQuerySet:
        """Load from dictionary."""
        queries = [GoldenQuery.from_dict(q) for q in data.get("queries", [])]
        return cls(queries=queries)


def load_golden_queries(path: Path) -> list[GoldenQuery]:
    """Load golden queries from JSON file.

    Args:
        path: Path to golden_queries_<context>.json file

    Returns:
        List of GoldenQuery objects

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid or queries fail validation
    """
    if not path.exists():
        raise FileNotFoundError(f"Golden query file not found: {path}")

    data = json.loads(path.read_text())
    query_set = GoldenQuerySet.from_dict(data)
    return query_set.queries


def validate_golden_queries(path: Path) -> list[str]:
    """Validate golden query file format.

    Args:
        path: Path to golden_queries_<context>.json

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]
    except FileNotFoundError:
        return [f"File not found: {path}"]

    # Validate structure
    if "queries" not in data:
        return ["Missing 'queries' key in JSON"]

    if not isinstance(data["queries"], list):
        return ["'queries' must be a list"]

    # Validate each query
    for i, query_data in enumerate(data["queries"]):
        try:
            GoldenQuery.from_dict(query_data)
        except ValueError as e:
            errors.append(f"Query {i}: {e}")

    return errors

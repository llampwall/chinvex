# src/chinvex/eval_runner.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .config import AppConfig
from .eval_schema import GoldenQuery
from .search import search


@dataclass
class QueryResult:
    """Result of executing a single golden query."""
    query: str
    expected_files: list[str]
    retrieved_files: list[str]
    k: int
    anchor: str | None = None
    latency_ms: float = 0.0


class EvalRunner:
    """Executes golden queries and collects results."""

    def __init__(self, config: AppConfig, context_name: str):
        self.config = config
        self.context_name = context_name

    def execute_query(self, golden_query: GoldenQuery) -> QueryResult:
        """Execute a single golden query.

        Args:
            golden_query: Query to execute

        Returns:
            QueryResult with retrieved file paths and latency
        """
        start_time = time.perf_counter()

        # Execute search
        search_results = search(
            self.config,
            golden_query.query,
            k=golden_query.k
        )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Extract file paths from citations
        retrieved_files = []
        for result in search_results:
            # Citation format: "path/to/file.py:line" or "path/to/file.md:line"
            file_path = result.citation.split(":")[0] if ":" in result.citation else result.citation
            retrieved_files.append(file_path)

        return QueryResult(
            query=golden_query.query,
            expected_files=golden_query.expected_files,
            retrieved_files=retrieved_files,
            k=golden_query.k,
            anchor=golden_query.anchor,
            latency_ms=latency_ms
        )

    def run(self, golden_queries: list[GoldenQuery]) -> list[QueryResult]:
        """Execute all golden queries.

        Args:
            golden_queries: List of queries to execute

        Returns:
            List of QueryResult objects
        """
        results = []
        for golden_query in golden_queries:
            result = self.execute_query(golden_query)
            results.append(result)
        return results

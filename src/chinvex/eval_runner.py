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


def run_evaluation(
    context_name: str,
    golden_queries_file: Any | None = None,
    k: int | None = None,
) -> dict:
    """Run evaluation suite for a context.

    Args:
        context_name: Name of context to evaluate
        golden_queries_file: Path to golden queries JSON (default: auto-detect)
        k: Override k value for all queries (default: use per-query k)

    Returns:
        Dict with metrics:
        - hit_rate: fraction of queries with at least one expected file in top K
        - mrr: mean reciprocal rank across all queries
        - avg_latency_ms: average query latency in milliseconds
        - passed: number of queries that passed
        - failed: number of queries that failed
        - total: total number of queries
    """
    import os
    from pathlib import Path
    from .config import load_config
    from .eval_schema import load_golden_queries

    # Load golden queries
    if golden_queries_file is None:
        # Auto-detect: tests/eval/golden_queries_<context>.json
        context_lower = context_name.lower()
        golden_queries_file = Path(f"tests/eval/golden_queries_{context_lower}.json")

    if not golden_queries_file.exists():
        raise FileNotFoundError(
            f"Golden queries file not found: {golden_queries_file}\n"
            f"Create queries for {context_name} context first."
        )

    queries = load_golden_queries(golden_queries_file)

    # Filter to this context only
    context_queries = [q for q in queries if q.context == context_name]

    if not context_queries:
        raise ValueError(
            f"No golden queries found for context '{context_name}' in {golden_queries_file}"
        )

    # Load context config
    chinvex_home = Path(os.getenv("CHINVEX_HOME", Path.home() / ".chinvex"))
    context_dir = chinvex_home / "contexts" / context_name
    config_path = context_dir / "context.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Context config not found: {config_path}")

    config = load_config(config_path)

    # Run queries and collect results
    hits = 0
    reciprocal_ranks = []
    latencies = []
    passed = 0
    failed = 0

    for query in context_queries:
        query_k = k if k is not None else query.k

        start = time.time()
        results = search(
            config,
            query.query,
            k=query_k,
            min_score=0.0,  # Don't filter by score during eval
        )
        elapsed_ms = (time.time() - start) * 1000
        latencies.append(elapsed_ms)

        # Check if any expected file appears in top K
        hit = False
        first_rank = None

        for rank, result in enumerate(results, start=1):
            # Extract file path from citation or title
            result_file = _extract_file_path(result)

            for expected_file in query.expected_files:
                if result_file and expected_file in result_file:
                    hit = True
                    if first_rank is None:
                        first_rank = rank
                    break

            if hit:
                break

        if hit:
            hits += 1
            passed += 1
            reciprocal_ranks.append(1.0 / first_rank)
        else:
            failed += 1
            reciprocal_ranks.append(0.0)

    total = len(context_queries)
    hit_rate = hits / total if total > 0 else 0.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "hit_rate": hit_rate,
        "mrr": mrr,
        "avg_latency_ms": avg_latency,
        "passed": passed,
        "failed": failed,
        "total": total,
    }


def _extract_file_path(result: Any) -> str | None:
    """Extract file path from search result."""
    # Try citation first (e.g., 'src/file.py:123')
    if hasattr(result, 'citation') and result.citation:
        parts = result.citation.split(":")
        if parts:
            return parts[0]

    # Fall back to title for repo sources
    if hasattr(result, 'source_type') and result.source_type == "repo":
        if hasattr(result, 'title'):
            return result.title

    return None

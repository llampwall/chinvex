from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EvalMetrics:
    """Evaluation metrics for golden queries."""
    hit_rate: float  # Fraction of queries with at least one expected file in top K
    mrr: float  # Mean Reciprocal Rank
    mean_latency_ms: float  # Average query latency in milliseconds
    total_queries: int  # Total number of queries evaluated
    passed_queries: int  # Number of queries that passed (file match)
    anchor_match_rate: float  # Fraction of anchor checks that passed


def calculate_metrics(
    evaluations: list[dict[str, Any]],
    latencies: list[float]
) -> EvalMetrics:
    """Calculate evaluation metrics from query evaluations.

    Args:
        evaluations: List of evaluation dicts with keys:
            - passed (bool): True if file match
            - rank (int | None): Rank of first matched file (1-indexed)
            - anchor_match (bool | None): True if anchor found
        latencies: List of query latencies in milliseconds

    Returns:
        EvalMetrics with calculated metrics
    """
    if not evaluations:
        return EvalMetrics(
            hit_rate=0.0,
            mrr=0.0,
            mean_latency_ms=0.0,
            total_queries=0,
            passed_queries=0,
            anchor_match_rate=0.0
        )

    total_queries = len(evaluations)
    passed_queries = sum(1 for e in evaluations if e["passed"])

    # Calculate hit rate
    hit_rate = passed_queries / total_queries if total_queries > 0 else 0.0

    # Calculate MRR (Mean Reciprocal Rank)
    # For each query: 1/rank if hit, 0 if miss
    reciprocal_ranks = []
    for eval_result in evaluations:
        if eval_result["passed"] and eval_result["rank"] is not None:
            reciprocal_ranks.append(1.0 / eval_result["rank"])
        else:
            reciprocal_ranks.append(0.0)

    mrr = sum(reciprocal_ranks) / total_queries if total_queries > 0 else 0.0

    # Calculate mean latency
    mean_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

    # Calculate anchor match rate
    # Only count evaluations where anchor_match is not None
    anchor_checks = [e for e in evaluations if e.get("anchor_match") is not None]
    if anchor_checks:
        anchor_matches = sum(1 for e in anchor_checks if e["anchor_match"] is True)
        anchor_match_rate = anchor_matches / len(anchor_checks)
    else:
        anchor_match_rate = 0.0

    return EvalMetrics(
        hit_rate=hit_rate,
        mrr=mrr,
        mean_latency_ms=mean_latency_ms,
        total_queries=total_queries,
        passed_queries=passed_queries,
        anchor_match_rate=anchor_match_rate
    )

import pytest
from chinvex.eval_metrics import calculate_metrics, EvalMetrics


def test_eval_metrics_dataclass():
    metrics = EvalMetrics(
        hit_rate=0.85,
        mrr=0.72,
        mean_latency_ms=150.5,
        total_queries=20,
        passed_queries=17,
        anchor_match_rate=0.65
    )
    assert metrics.hit_rate == 0.85
    assert metrics.mrr == 0.72
    assert metrics.mean_latency_ms == 150.5
    assert metrics.total_queries == 20
    assert metrics.passed_queries == 17
    assert metrics.anchor_match_rate == 0.65


def test_calculate_metrics_perfect_score():
    evaluations = [
        {"passed": True, "rank": 1, "anchor_match": True},
        {"passed": True, "rank": 1, "anchor_match": True},
        {"passed": True, "rank": 1, "anchor_match": True}
    ]
    latencies = [100.0, 120.0, 110.0]

    metrics = calculate_metrics(evaluations, latencies)

    assert metrics.hit_rate == 1.0
    assert metrics.mrr == 1.0
    assert metrics.mean_latency_ms == 110.0
    assert metrics.total_queries == 3
    assert metrics.passed_queries == 3
    assert metrics.anchor_match_rate == 1.0


def test_calculate_metrics_partial_hits():
    evaluations = [
        {"passed": True, "rank": 1, "anchor_match": None},
        {"passed": False, "rank": None, "anchor_match": None},
        {"passed": True, "rank": 3, "anchor_match": None},
        {"passed": False, "rank": None, "anchor_match": None}
    ]
    latencies = [100.0, 110.0, 120.0, 130.0]

    metrics = calculate_metrics(evaluations, latencies)

    assert metrics.hit_rate == 0.5  # 2 out of 4
    assert metrics.passed_queries == 2
    assert metrics.total_queries == 4
    assert metrics.mean_latency_ms == 115.0


def test_calculate_metrics_anchor_match_rate():
    evaluations = [
        {"passed": True, "rank": 1, "anchor_match": True},
        {"passed": True, "rank": 2, "anchor_match": False},
        {"passed": True, "rank": 1, "anchor_match": True},
        {"passed": True, "rank": 1, "anchor_match": None}  # No anchor specified
    ]
    latencies = [100.0, 100.0, 100.0, 100.0]

    metrics = calculate_metrics(evaluations, latencies)

    # Only 3 queries had anchors, 2 matched
    assert abs(metrics.anchor_match_rate - (2/3)) < 0.001


def test_calculate_metrics_no_anchors():
    evaluations = [
        {"passed": True, "rank": 1, "anchor_match": None},
        {"passed": True, "rank": 2, "anchor_match": None},
        {"passed": False, "rank": None, "anchor_match": None}
    ]
    latencies = [100.0, 100.0, 100.0]

    metrics = calculate_metrics(evaluations, latencies)

    assert metrics.anchor_match_rate == 0.0


def test_calculate_metrics_all_failed():
    evaluations = [
        {"passed": False, "rank": None, "anchor_match": None},
        {"passed": False, "rank": None, "anchor_match": None}
    ]
    latencies = [100.0, 100.0]

    metrics = calculate_metrics(evaluations, latencies)

    assert metrics.hit_rate == 0.0
    assert metrics.mrr == 0.0
    assert metrics.passed_queries == 0


def test_calculate_metrics_empty_input():
    evaluations = []
    latencies = []

    metrics = calculate_metrics(evaluations, latencies)

    assert metrics.hit_rate == 0.0
    assert metrics.mrr == 0.0
    assert metrics.mean_latency_ms == 0.0
    assert metrics.total_queries == 0
    assert metrics.passed_queries == 0


def test_calculate_metrics_mrr_calculation():
    # Test MRR formula: average of (1/rank) for hits, 0 for misses
    evaluations = [
        {"passed": True, "rank": 1, "anchor_match": None},  # 1/1 = 1.0
        {"passed": True, "rank": 2, "anchor_match": None},  # 1/2 = 0.5
        {"passed": True, "rank": 5, "anchor_match": None},  # 1/5 = 0.2
        {"passed": False, "rank": None, "anchor_match": None}  # 0
    ]
    latencies = [100.0, 100.0, 100.0, 100.0]

    metrics = calculate_metrics(evaluations, latencies)

    # MRR = (1.0 + 0.5 + 0.2 + 0.0) / 4 = 1.7 / 4 = 0.425
    assert abs(metrics.mrr - 0.425) < 0.001
    assert metrics.hit_rate == 0.75

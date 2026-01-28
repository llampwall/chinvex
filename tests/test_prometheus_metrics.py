"""Test Prometheus metrics endpoint."""
import pytest
from unittest.mock import Mock


def test_metrics_endpoint_returns_prometheus_format():
    """Test that metrics are in Prometheus format."""
    from chinvex.gateway.metrics import generate_metrics, MetricsCollector

    collector = MetricsCollector()
    collector.record_request("/search", 200, 0.5)

    metrics = generate_metrics(collector)

    # Should contain metric lines
    assert "chinvex_requests_total" in metrics
    assert "chinvex_request_duration_seconds" in metrics
    assert "chinvex_grounded_ratio" in metrics


def test_metrics_track_request_counts():
    """Test that request counter increments."""
    from chinvex.gateway.metrics import MetricsCollector

    collector = MetricsCollector()

    collector.record_request(endpoint="/search", status_code=200)
    collector.record_request(endpoint="/search", status_code=200)
    collector.record_request(endpoint="/evidence", status_code=404)

    metrics = collector.get_metrics()

    # Should have counts per endpoint/status
    assert metrics["requests"]["/search"][200] == 2
    assert metrics["requests"]["/evidence"][404] == 1


def test_metrics_track_grounded_ratio():
    """Test that grounded ratio is tracked."""
    from chinvex.gateway.metrics import MetricsCollector

    collector = MetricsCollector()

    collector.record_grounded_response(True)
    collector.record_grounded_response(True)
    collector.record_grounded_response(False)

    metrics = collector.get_metrics()

    assert metrics["grounded"]["total"] == 3
    assert metrics["grounded"]["grounded"] == 2


def test_metrics_track_latency():
    """Test that latency is tracked."""
    from chinvex.gateway.metrics import MetricsCollector

    collector = MetricsCollector()

    collector.record_request("/search", 200, 0.5)
    collector.record_request("/search", 200, 0.3)

    metrics = collector.get_metrics()

    assert len(metrics["latencies"]["/search"]) == 2
    assert 0.3 in metrics["latencies"]["/search"]
    assert 0.5 in metrics["latencies"]["/search"]


def test_prometheus_format_structure():
    """Test Prometheus format has HELP and TYPE comments."""
    from chinvex.gateway.metrics import generate_metrics, MetricsCollector

    collector = MetricsCollector()
    collector.record_request("/test", 200, 0.1)

    metrics = generate_metrics(collector)

    # Should have HELP and TYPE comments
    assert "# HELP chinvex_requests_total" in metrics
    assert "# TYPE chinvex_requests_total counter" in metrics
    assert "# HELP chinvex_grounded_ratio" in metrics
    assert "# TYPE chinvex_grounded_ratio gauge" in metrics

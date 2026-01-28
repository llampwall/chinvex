"""Prometheus metrics collection."""
import time
from collections import defaultdict
from typing import Dict


class MetricsCollector:
    """In-memory metrics collector for Prometheus."""

    def __init__(self):
        self.requests = defaultdict(lambda: defaultdict(int))  # {endpoint: {status: count}}
        self.latencies = defaultdict(list)  # {endpoint: [duration, ...]}
        self.grounded_total = 0
        self.grounded_count = 0

    def record_request(self, endpoint: str, status_code: int, duration: float = 0.0):
        """Record request metrics."""
        self.requests[endpoint][status_code] += 1
        if duration > 0:
            self.latencies[endpoint].append(duration)

    def record_grounded_response(self, grounded: bool):
        """Record grounded response metric."""
        self.grounded_total += 1
        if grounded:
            self.grounded_count += 1

    def get_metrics(self) -> dict:
        """Get current metrics as dict."""
        return {
            "requests": dict(self.requests),
            "latencies": dict(self.latencies),
            "grounded": {
                "total": self.grounded_total,
                "grounded": self.grounded_count
            }
        }


def generate_metrics(collector: MetricsCollector) -> str:
    """
    Generate Prometheus metrics format.

    Returns plain text in Prometheus exposition format.
    """
    lines = []

    # Request counts
    lines.append("# HELP chinvex_requests_total Total requests by endpoint and status")
    lines.append("# TYPE chinvex_requests_total counter")
    for endpoint, statuses in collector.requests.items():
        for status, count in statuses.items():
            lines.append(f'chinvex_requests_total{{endpoint="{endpoint}",status="{status}"}} {count}')

    # Latency histograms (simplified - just avg for now)
    lines.append("# HELP chinvex_request_duration_seconds Request duration")
    lines.append("# TYPE chinvex_request_duration_seconds histogram")
    for endpoint, durations in collector.latencies.items():
        if durations:
            avg = sum(durations) / len(durations)
            lines.append(f'chinvex_request_duration_seconds{{endpoint="{endpoint}"}} {avg:.4f}')

    # Grounded ratio
    lines.append("# HELP chinvex_grounded_ratio Ratio of grounded responses")
    lines.append("# TYPE chinvex_grounded_ratio gauge")
    if collector.grounded_total > 0:
        ratio = collector.grounded_count / collector.grounded_total
        lines.append(f"chinvex_grounded_ratio {ratio:.4f}")
    else:
        lines.append("chinvex_grounded_ratio 0.0")

    return "\n".join(lines) + "\n"


# Global metrics collector (ephemeral - resets on restart)
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    return _metrics_collector

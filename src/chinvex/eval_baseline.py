from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from .eval_metrics import EvalMetrics


@dataclass
class BaselineComparison:
    """Result of comparing current metrics to baseline."""
    passed: bool  # True if current hit rate >= threshold * baseline
    current_hit_rate: float
    baseline_hit_rate: float
    hit_rate_change: float  # current - baseline
    threshold: float  # Minimum fraction of baseline required (e.g., 0.80 = 80%)


def save_baseline_metrics(
    baseline_file: Path,
    context: str,
    metrics: EvalMetrics
) -> None:
    """Save baseline metrics for a context.

    Args:
        baseline_file: Path to baseline_metrics.json
        context: Context name
        metrics: Metrics to save as baseline
    """
    # Create directory if needed
    baseline_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing baselines
    if baseline_file.exists():
        data = json.loads(baseline_file.read_text())
    else:
        data = {}

    # Update with new metrics
    data[context] = asdict(metrics)

    # Write back
    baseline_file.write_text(json.dumps(data, indent=2))


def load_baseline_metrics(
    baseline_file: Path,
    context: str
) -> EvalMetrics | None:
    """Load baseline metrics for a context.

    Args:
        baseline_file: Path to baseline_metrics.json
        context: Context name

    Returns:
        EvalMetrics if found, None otherwise
    """
    if not baseline_file.exists():
        return None

    data = json.loads(baseline_file.read_text())

    if context not in data:
        return None

    metrics_dict = data[context]
    return EvalMetrics(**metrics_dict)


def compare_to_baseline(
    current: EvalMetrics,
    baseline: EvalMetrics,
    threshold: float = 0.80
) -> BaselineComparison:
    """Compare current metrics to baseline.

    Args:
        current: Current evaluation metrics
        baseline: Baseline metrics to compare against
        threshold: Minimum fraction of baseline hit rate required (default 0.80 = 80%)

    Returns:
        BaselineComparison with pass/fail status
    """
    hit_rate_change = current.hit_rate - baseline.hit_rate

    # Special case: if baseline is 0, any positive current passes
    if baseline.hit_rate == 0.0:
        passed = current.hit_rate > 0.0
    else:
        # Pass if current >= threshold * baseline
        required_hit_rate = threshold * baseline.hit_rate
        passed = current.hit_rate >= required_hit_rate

    return BaselineComparison(
        passed=passed,
        current_hit_rate=current.hit_rate,
        baseline_hit_rate=baseline.hit_rate,
        hit_rate_change=hit_rate_change,
        threshold=threshold
    )

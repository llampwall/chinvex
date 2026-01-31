import json
import pytest
from pathlib import Path
from chinvex.eval_baseline import (
    save_baseline_metrics,
    load_baseline_metrics,
    compare_to_baseline,
    BaselineComparison
)
from chinvex.eval_metrics import EvalMetrics


@pytest.fixture
def sample_metrics():
    return EvalMetrics(
        hit_rate=0.85,
        mrr=0.72,
        mean_latency_ms=150.5,
        total_queries=20,
        passed_queries=17,
        anchor_match_rate=0.65
    )


def test_save_baseline_metrics(tmp_path, sample_metrics):
    baseline_file = tmp_path / "baseline_metrics.json"

    save_baseline_metrics(baseline_file, "Chinvex", sample_metrics)

    assert baseline_file.exists()
    data = json.loads(baseline_file.read_text())
    assert "Chinvex" in data
    assert data["Chinvex"]["hit_rate"] == 0.85
    assert data["Chinvex"]["mrr"] == 0.72
    assert data["Chinvex"]["total_queries"] == 20


def test_save_baseline_metrics_creates_directory(tmp_path, sample_metrics):
    baseline_file = tmp_path / "subdir" / "baseline_metrics.json"

    save_baseline_metrics(baseline_file, "Chinvex", sample_metrics)

    assert baseline_file.exists()


def test_save_baseline_metrics_updates_existing(tmp_path, sample_metrics):
    baseline_file = tmp_path / "baseline_metrics.json"

    # Save first context
    save_baseline_metrics(baseline_file, "Chinvex", sample_metrics)

    # Save second context
    metrics2 = EvalMetrics(
        hit_rate=0.90,
        mrr=0.80,
        mean_latency_ms=120.0,
        total_queries=15,
        passed_queries=14,
        anchor_match_rate=0.75
    )
    save_baseline_metrics(baseline_file, "OtherContext", metrics2)

    data = json.loads(baseline_file.read_text())
    assert "Chinvex" in data
    assert "OtherContext" in data
    assert data["Chinvex"]["hit_rate"] == 0.85
    assert data["OtherContext"]["hit_rate"] == 0.90


def test_save_baseline_metrics_overwrites_context(tmp_path):
    baseline_file = tmp_path / "baseline_metrics.json"

    metrics1 = EvalMetrics(0.85, 0.72, 150.0, 20, 17, 0.65)
    save_baseline_metrics(baseline_file, "Chinvex", metrics1)

    metrics2 = EvalMetrics(0.90, 0.80, 120.0, 20, 18, 0.70)
    save_baseline_metrics(baseline_file, "Chinvex", metrics2)

    data = json.loads(baseline_file.read_text())
    assert data["Chinvex"]["hit_rate"] == 0.90


def test_load_baseline_metrics(tmp_path, sample_metrics):
    baseline_file = tmp_path / "baseline_metrics.json"
    save_baseline_metrics(baseline_file, "Chinvex", sample_metrics)

    loaded = load_baseline_metrics(baseline_file, "Chinvex")

    assert loaded.hit_rate == 0.85
    assert loaded.mrr == 0.72
    assert loaded.mean_latency_ms == 150.5
    assert loaded.total_queries == 20


def test_load_baseline_metrics_not_found(tmp_path):
    baseline_file = tmp_path / "baseline_metrics.json"

    loaded = load_baseline_metrics(baseline_file, "Chinvex")

    assert loaded is None


def test_load_baseline_metrics_context_not_found(tmp_path, sample_metrics):
    baseline_file = tmp_path / "baseline_metrics.json"
    save_baseline_metrics(baseline_file, "Chinvex", sample_metrics)

    loaded = load_baseline_metrics(baseline_file, "OtherContext")

    assert loaded is None


def test_baseline_comparison_dataclass():
    comp = BaselineComparison(
        passed=True,
        current_hit_rate=0.85,
        baseline_hit_rate=0.80,
        hit_rate_change=0.05,
        threshold=0.80
    )
    assert comp.passed is True
    assert comp.current_hit_rate == 0.85
    assert comp.baseline_hit_rate == 0.80
    assert comp.hit_rate_change == 0.05


def test_compare_to_baseline_pass(sample_metrics):
    baseline = EvalMetrics(0.80, 0.70, 140.0, 20, 16, 0.60)
    current = sample_metrics  # hit_rate=0.85

    comparison = compare_to_baseline(current, baseline, threshold=0.80)

    assert comparison.passed is True
    assert comparison.current_hit_rate == 0.85
    assert comparison.baseline_hit_rate == 0.80
    assert abs(comparison.hit_rate_change - 0.05) < 0.001


def test_compare_to_baseline_fail(sample_metrics):
    baseline = EvalMetrics(0.90, 0.80, 140.0, 20, 18, 0.70)
    current = sample_metrics  # hit_rate=0.85

    # 0.85 is only 94.4% of 0.90, below 95% threshold
    comparison = compare_to_baseline(current, baseline, threshold=0.95)

    assert comparison.passed is False
    assert comparison.current_hit_rate == 0.85
    assert comparison.baseline_hit_rate == 0.90


def test_compare_to_baseline_exact_threshold():
    baseline = EvalMetrics(1.0, 0.90, 100.0, 10, 10, 1.0)
    current = EvalMetrics(0.80, 0.70, 120.0, 10, 8, 0.80)

    # current is exactly 80% of baseline (0.80 / 1.0)
    comparison = compare_to_baseline(current, baseline, threshold=0.80)

    assert comparison.passed is True


def test_compare_to_baseline_default_threshold():
    baseline = EvalMetrics(1.0, 0.90, 100.0, 10, 10, 1.0)
    current = EvalMetrics(0.79, 0.70, 120.0, 10, 8, 0.80)

    # Default threshold is 0.80, current is 79% of baseline
    comparison = compare_to_baseline(current, baseline)

    assert comparison.passed is False
    assert comparison.threshold == 0.80


def test_compare_to_baseline_improvement():
    baseline = EvalMetrics(0.80, 0.70, 140.0, 20, 16, 0.60)
    current = EvalMetrics(0.95, 0.85, 120.0, 20, 19, 0.90)

    comparison = compare_to_baseline(current, baseline, threshold=0.80)

    assert comparison.passed is True
    assert abs(comparison.hit_rate_change - 0.15) < 0.001


def test_compare_to_baseline_zero_baseline():
    baseline = EvalMetrics(0.0, 0.0, 100.0, 10, 0, 0.0)
    current = EvalMetrics(0.50, 0.40, 120.0, 10, 5, 0.50)

    # Special case: if baseline is 0, any positive current passes
    comparison = compare_to_baseline(current, baseline, threshold=0.80)

    assert comparison.passed is True

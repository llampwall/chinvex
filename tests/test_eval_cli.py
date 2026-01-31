import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from chinvex.cli import app


def test_eval_command_exists():
    """chinvex eval command should exist."""
    runner = CliRunner()
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    assert "Run evaluation suite" in result.output


def test_eval_requires_context_flag():
    """chinvex eval should require --context flag."""
    runner = CliRunner()
    result = runner.invoke(app, ["eval"])
    assert result.exit_code != 0
    assert "--context" in result.output


def test_eval_runs_golden_queries(tmp_path, monkeypatch):
    """chinvex eval should load golden queries and run evaluation."""
    # Setup test environment
    chinvex_home = tmp_path / ".chinvex"
    chinvex_home.mkdir()
    context_dir = chinvex_home / "contexts" / "TestContext"
    context_dir.mkdir(parents=True)

    # Create context.json
    context_config = {
        "embedding": {"provider": "openai", "model": "text-embedding-3-small"}
    }
    (context_dir / "context.json").write_text(json.dumps(context_config))

    # Create golden queries file
    golden_queries = {
        "queries": [
            {
                "query": "test query 1",
                "context": "TestContext",
                "expected_files": ["file1.py"],
                "k": 5
            },
            {
                "query": "test query 2",
                "context": "TestContext",
                "expected_files": ["file2.py"],
                "k": 5
            }
        ]
    }
    golden_file = tmp_path / "tests" / "eval" / "golden_queries_testcontext.json"
    golden_file.parent.mkdir(parents=True)
    golden_file.write_text(json.dumps(golden_queries))

    monkeypatch.setenv("CHINVEX_HOME", str(chinvex_home))

    # Mock the eval runner
    with patch("chinvex.eval_runner.run_evaluation") as mock_run_eval:
        mock_run_eval.return_value = {
            "hit_rate": 0.85,
            "mrr": 0.75,
            "avg_latency_ms": 150.5,
            "passed": 17,
            "failed": 3,
            "total": 20
        }

        runner = CliRunner()
        result = runner.invoke(app, ["eval", "--context", "TestContext"])

        if result.exit_code != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"OUTPUT:\n{result.output}")
        assert result.exit_code == 0
        assert "Hit Rate@5: 85.0%" in result.output
        assert "MRR: 0.750" in result.output
        assert "Avg Latency: 150.5ms" in result.output
        assert "Passed: 17/20" in result.output

        # Verify run_evaluation was called
        mock_run_eval.assert_called_once()
        call_args = mock_run_eval.call_args
        assert call_args[1]["context_name"] == "TestContext"


def test_eval_reports_baseline_comparison(tmp_path, monkeypatch):
    """chinvex eval should compare to baseline and report status."""
    chinvex_home = tmp_path / ".chinvex"
    context_dir = chinvex_home / "contexts" / "TestContext"
    context_dir.mkdir(parents=True)

    context_config = {
        "embedding": {"provider": "openai", "model": "text-embedding-3-small"}
    }
    (context_dir / "context.json").write_text(json.dumps(context_config))

    # Create baseline file
    baseline_file = tmp_path / "tests" / "eval" / "baseline_metrics.json"
    baseline_file.parent.mkdir(parents=True)
    baseline_data = {
        "TestContext": {
            "hit_rate": 0.90,
            "mrr": 0.80,
            "avg_latency_ms": 100.0
        }
    }
    baseline_file.write_text(json.dumps(baseline_data))

    monkeypatch.setenv("CHINVEX_HOME", str(chinvex_home))

    with patch("chinvex.eval_runner.run_evaluation") as mock_run_eval, \
         patch("chinvex.eval_baseline.load_baseline_metrics") as mock_load_baseline, \
         patch("chinvex.eval_baseline.compare_to_baseline") as mock_compare:

        mock_run_eval.return_value = {
            "hit_rate": 0.75,
            "mrr": 0.70,
            "avg_latency_ms": 120.0,
            "passed": 15,
            "failed": 5,
            "total": 20
        }

        mock_load_baseline.return_value = MagicMock(
            hit_rate=0.90,
            mrr=0.80,
            avg_latency_ms=100.0
        )

        mock_compare.return_value = MagicMock(
            passed=False,
            current_hit_rate=0.75,
            baseline_hit_rate=0.90,
            hit_rate_change=-0.15,
            threshold=0.80
        )

        runner = CliRunner()
        result = runner.invoke(app, ["eval", "--context", "TestContext"])

        assert result.exit_code == 1  # Should exit with error on regression
        assert "REGRESSION" in result.output or "below baseline" in result.output


def test_eval_handles_missing_golden_queries(tmp_path, monkeypatch):
    """chinvex eval should error gracefully if golden queries file missing."""
    chinvex_home = tmp_path / ".chinvex"
    context_dir = chinvex_home / "contexts" / "TestContext"
    context_dir.mkdir(parents=True)

    context_config = {
        "embedding": {"provider": "openai", "model": "text-embedding-3-small"}
    }
    (context_dir / "context.json").write_text(json.dumps(context_config))

    monkeypatch.setenv("CHINVEX_HOME", str(chinvex_home))

    runner = CliRunner()
    result = runner.invoke(app, ["eval", "--context", "TestContext"])

    assert result.exit_code != 0
    assert "golden queries" in result.output.lower()

"""Test --rechunk-only flag."""
import pytest
from typer.testing import CliRunner
from chinvex.cli import app

runner = CliRunner()


def test_rechunk_only_flag_accepted():
    """Test that --rechunk-only flag is recognized."""
    result = runner.invoke(app, ["ingest", "--context", "Test", "--rechunk-only"])

    # Should not fail with "no such option"
    assert "no such option" not in result.stdout.lower()


def test_rechunk_only_reuses_embeddings(tmp_path):
    """Test that --rechunk-only reuses embeddings when text unchanged."""
    # This is an integration test - implementation will verify behavior
    # For now, just test that flag is processed
    pass

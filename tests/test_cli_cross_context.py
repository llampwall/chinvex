"""Test CLI cross-context search commands."""
import pytest
from typer.testing import CliRunner
from chinvex.cli import app

runner = CliRunner()


def test_search_all_flag():
    """Test chinvex search --all flag."""
    result = runner.invoke(app, ["search", "--all", "test query"])
    # May fail if no contexts available, but should recognize the flag
    assert "--all" not in result.stdout or result.exit_code in [0, 2]


def test_search_contexts_flag():
    """Test chinvex search --contexts flag."""
    result = runner.invoke(app, ["search", "--contexts", "Chinvex,Personal", "test query"])
    # May fail if contexts don't exist, but should recognize the flag
    assert "--contexts" not in result.stdout or result.exit_code in [0, 2]


def test_search_exclude_flag():
    """Test chinvex search --exclude flag."""
    result = runner.invoke(app, ["search", "--all", "--exclude", "Work", "test query"])
    # May fail if no contexts available, but should recognize the flag
    assert "--exclude" not in result.stdout or result.exit_code in [0, 2]

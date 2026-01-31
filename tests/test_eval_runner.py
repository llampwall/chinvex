# tests/test_eval_runner.py
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from chinvex.eval_runner import EvalRunner, QueryResult
from chinvex.eval_schema import GoldenQuery
from chinvex.search import SearchResult


@pytest.fixture
def mock_config(tmp_path):
    config = MagicMock()
    config.index_dir = tmp_path / "index"
    config.index_dir.mkdir()
    return config


@pytest.fixture
def golden_queries():
    return [
        GoldenQuery(
            query="chromadb batch limit",
            context="Chinvex",
            expected_files=["src/chinvex/ingest.py"],
            k=5
        ),
        GoldenQuery(
            query="hook installation",
            context="Chinvex",
            expected_files=["src/chinvex/hook_installer.py", "src/chinvex/cli.py"],
            anchor="startup hook",
            k=10
        )
    ]


def test_query_result_creation():
    result = QueryResult(
        query="test query",
        expected_files=["test.py"],
        retrieved_files=["test.py", "other.py"],
        k=5,
        anchor="test",
        latency_ms=150.5
    )
    assert result.query == "test query"
    assert result.expected_files == ["test.py"]
    assert result.retrieved_files == ["test.py", "other.py"]
    assert result.k == 5
    assert result.anchor == "test"
    assert result.latency_ms == 150.5


def test_eval_runner_init(mock_config):
    runner = EvalRunner(mock_config, "Chinvex")
    assert runner.config == mock_config
    assert runner.context_name == "Chinvex"


@patch("chinvex.eval_runner.search")
def test_eval_runner_execute_single_query(mock_search, mock_config):
    mock_search.return_value = [
        SearchResult(
            chunk_id="chunk1",
            score=0.9,
            source_type="repo",
            title="ingest.py",
            citation="src/chinvex/ingest.py:50",
            snippet="batch size is 5000"
        ),
        SearchResult(
            chunk_id="chunk2",
            score=0.7,
            source_type="repo",
            title="config.py",
            citation="src/chinvex/config.py:20",
            snippet="configuration settings"
        )
    ]

    runner = EvalRunner(mock_config, "Chinvex")
    golden_query = GoldenQuery(
        query="chromadb batch limit",
        context="Chinvex",
        expected_files=["src/chinvex/ingest.py"],
        k=5
    )

    result = runner.execute_query(golden_query)

    assert result.query == "chromadb batch limit"
    assert result.expected_files == ["src/chinvex/ingest.py"]
    assert "src/chinvex/ingest.py" in result.retrieved_files
    assert "src/chinvex/config.py" in result.retrieved_files
    assert result.k == 5
    assert result.latency_ms > 0

    mock_search.assert_called_once()
    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["k"] == 5


@patch("chinvex.eval_runner.search")
def test_eval_runner_extract_file_paths(mock_search, mock_config):
    mock_search.return_value = [
        SearchResult(
            chunk_id="c1",
            score=0.9,
            source_type="repo",
            title="test",
            citation="src/chinvex/ingest.py:50",
            snippet="test"
        ),
        SearchResult(
            chunk_id="c2",
            score=0.8,
            source_type="chat",
            title="chat",
            citation="chats/2024-01.md:100",
            snippet="test"
        )
    ]

    runner = EvalRunner(mock_config, "Chinvex")
    golden_query = GoldenQuery(
        query="test",
        context="Chinvex",
        expected_files=["src/chinvex/ingest.py"],
        k=5
    )

    result = runner.execute_query(golden_query)

    assert "src/chinvex/ingest.py" in result.retrieved_files
    assert "chats/2024-01.md" in result.retrieved_files
    assert len(result.retrieved_files) == 2


@patch("chinvex.eval_runner.search")
def test_eval_runner_run_all_queries(mock_search, mock_config, golden_queries):
    mock_search.return_value = [
        SearchResult(
            chunk_id="c1",
            score=0.9,
            source_type="repo",
            title="test",
            citation="src/chinvex/ingest.py:50",
            snippet="test"
        )
    ]

    runner = EvalRunner(mock_config, "Chinvex")
    results = runner.run(golden_queries)

    assert len(results) == 2
    assert all(isinstance(r, QueryResult) for r in results)
    assert results[0].query == "chromadb batch limit"
    assert results[1].query == "hook installation"
    assert mock_search.call_count == 2


@patch("chinvex.eval_runner.search")
def test_eval_runner_handles_empty_results(mock_search, mock_config):
    mock_search.return_value = []

    runner = EvalRunner(mock_config, "Chinvex")
    golden_query = GoldenQuery(
        query="nonexistent query",
        context="Chinvex",
        expected_files=["test.py"],
        k=5
    )

    result = runner.execute_query(golden_query)

    assert result.retrieved_files == []
    assert result.latency_ms >= 0


@patch("chinvex.eval_runner.search")
def test_eval_runner_respects_k_parameter(mock_search, mock_config):
    mock_search.return_value = [
        SearchResult(f"c{i}", 0.9, "repo", "t", f"f{i}.py:1", "s")
        for i in range(20)
    ]

    runner = EvalRunner(mock_config, "Chinvex")
    golden_query = GoldenQuery(
        query="test",
        context="Chinvex",
        expected_files=["test.py"],
        k=10
    )

    result = runner.execute_query(golden_query)

    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["k"] == 10

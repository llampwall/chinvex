# tests/test_eval_schema.py
import json
import pytest
from pathlib import Path
from chinvex.eval_schema import GoldenQuery, GoldenQuerySet, load_golden_queries, validate_golden_queries


def test_golden_query_from_dict_minimal():
    data = {
        "query": "chromadb batch limit",
        "context": "Chinvex",
        "expected_files": ["src/chinvex/ingest.py"]
    }
    query = GoldenQuery.from_dict(data)
    assert query.query == "chromadb batch limit"
    assert query.context == "Chinvex"
    assert query.expected_files == ["src/chinvex/ingest.py"]
    assert query.anchor is None
    assert query.k == 5  # default


def test_golden_query_from_dict_full():
    data = {
        "query": "chromadb batch limit",
        "context": "Chinvex",
        "expected_files": ["src/chinvex/ingest.py", "docs/constraints.md"],
        "anchor": "5000 vectors",
        "k": 10
    }
    query = GoldenQuery.from_dict(data)
    assert query.query == "chromadb batch limit"
    assert query.context == "Chinvex"
    assert query.expected_files == ["src/chinvex/ingest.py", "docs/constraints.md"]
    assert query.anchor == "5000 vectors"
    assert query.k == 10


def test_golden_query_validation_missing_required_field():
    data = {
        "context": "Chinvex",
        "expected_files": ["src/chinvex/ingest.py"]
    }
    with pytest.raises(ValueError, match="Missing required field: query"):
        GoldenQuery.from_dict(data)


def test_golden_query_validation_empty_expected_files():
    data = {
        "query": "test",
        "context": "Chinvex",
        "expected_files": []
    }
    with pytest.raises(ValueError, match="expected_files must contain at least one file"):
        GoldenQuery.from_dict(data)


def test_golden_query_validation_invalid_k():
    data = {
        "query": "test",
        "context": "Chinvex",
        "expected_files": ["test.py"],
        "k": 0
    }
    with pytest.raises(ValueError, match="k must be positive"):
        GoldenQuery.from_dict(data)


def test_golden_query_set_load_from_dict():
    data = {
        "queries": [
            {
                "query": "query1",
                "context": "Chinvex",
                "expected_files": ["file1.py"]
            },
            {
                "query": "query2",
                "context": "Chinvex",
                "expected_files": ["file2.py"],
                "k": 10
            }
        ]
    }
    query_set = GoldenQuerySet.from_dict(data)
    assert len(query_set.queries) == 2
    assert query_set.queries[0].query == "query1"
    assert query_set.queries[1].k == 10


def test_load_golden_queries_from_file(tmp_path):
    query_file = tmp_path / "golden_queries_test.json"
    query_file.write_text(json.dumps({
        "queries": [
            {
                "query": "test query",
                "context": "TestContext",
                "expected_files": ["test.py"]
            }
        ]
    }))

    queries = load_golden_queries(query_file)
    assert len(queries) == 1
    assert queries[0].query == "test query"


def test_load_golden_queries_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_golden_queries(Path("nonexistent.json"))


def test_validate_golden_queries_success(tmp_path):
    query_file = tmp_path / "golden_queries_test.json"
    query_file.write_text(json.dumps({
        "queries": [
            {
                "query": "test",
                "context": "Test",
                "expected_files": ["test.py"]
            }
        ]
    }))

    errors = validate_golden_queries(query_file)
    assert len(errors) == 0


def test_validate_golden_queries_with_errors(tmp_path):
    query_file = tmp_path / "golden_queries_bad.json"
    query_file.write_text(json.dumps({
        "queries": [
            {
                "context": "Test",
                "expected_files": []
            }
        ]
    }))

    errors = validate_golden_queries(query_file)
    assert len(errors) > 0
    assert any("query" in err.lower() for err in errors)

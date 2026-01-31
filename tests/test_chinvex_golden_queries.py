import json
from pathlib import Path

from chinvex.eval_schema import load_golden_queries


def test_chinvex_golden_queries_file_exists():
    """Golden queries file for Chinvex context should exist."""
    golden_file = Path("tests/eval/golden_queries_chinvex.json")
    assert golden_file.exists(), "Create golden_queries_chinvex.json with 20+ queries"


def test_chinvex_golden_queries_has_minimum_count():
    """Chinvex golden queries should have at least 20 queries."""
    golden_file = Path("tests/eval/golden_queries_chinvex.json")
    queries = load_golden_queries(golden_file)

    chinvex_queries = [q for q in queries if q.context == "Chinvex"]
    assert len(chinvex_queries) >= 20, f"Need at least 20 queries, found {len(chinvex_queries)}"


def test_chinvex_golden_queries_all_valid():
    """All Chinvex golden queries should have valid schema."""
    golden_file = Path("tests/eval/golden_queries_chinvex.json")
    queries = load_golden_queries(golden_file)

    for query in queries:
        if query.context != "Chinvex":
            continue

        # Required fields
        assert query.query, "Query text is required"
        assert query.context == "Chinvex"
        assert query.expected_files, "At least one expected file required"
        assert query.k > 0, "k must be positive"

        # All expected files should be strings
        for expected_file in query.expected_files:
            assert isinstance(expected_file, str), "Expected files must be strings"


def test_chinvex_golden_queries_cover_key_areas():
    """Golden queries should cover key functional areas of Chinvex."""
    golden_file = Path("tests/eval/golden_queries_chinvex.json")
    queries = load_golden_queries(golden_file)

    chinvex_queries = [q for q in queries if q.context == "Chinvex"]
    query_texts = [q.query.lower() for q in chinvex_queries]

    # Check coverage of major areas
    areas = {
        "search": ["search", "query", "retrieval", "hybrid"],
        "ingest": ["ingest", "indexing", "chunking"],
        "embedding": ["embedding", "vector", "openai", "ollama"],
        "config": ["config", "context", "settings"],
        "cli": ["cli", "command"],
    }

    for area, keywords in areas.items():
        found = any(
            any(kw in qt for kw in keywords)
            for qt in query_texts
        )
        assert found, f"No queries found covering {area} area (keywords: {keywords})"


def test_chinvex_golden_queries_expected_files_exist():
    """Expected files in golden queries should exist in codebase."""
    golden_file = Path("tests/eval/golden_queries_chinvex.json")
    queries = load_golden_queries(golden_file)

    repo_root = Path(".")
    missing_files = []

    for query in queries:
        if query.context != "Chinvex":
            continue

        for expected_file in query.expected_files:
            # Normalize path
            file_path = repo_root / expected_file
            if not file_path.exists():
                missing_files.append((query.query, expected_file))

    assert not missing_files, f"Expected files not found: {missing_files}"

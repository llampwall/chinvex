import pytest
from chinvex.eval_criteria import evaluate_query, check_file_match, check_anchor_match
from chinvex.eval_runner import QueryResult


def test_check_file_match_exact():
    retrieved = ["src/chinvex/ingest.py", "src/chinvex/config.py"]
    expected = ["src/chinvex/ingest.py"]
    assert check_file_match(retrieved, expected) is True


def test_check_file_match_multiple_expected():
    retrieved = ["src/chinvex/search.py", "src/chinvex/config.py"]
    expected = ["src/chinvex/ingest.py", "src/chinvex/config.py"]
    assert check_file_match(retrieved, expected) is True


def test_check_file_match_no_match():
    retrieved = ["src/chinvex/search.py", "src/chinvex/config.py"]
    expected = ["src/chinvex/ingest.py"]
    assert check_file_match(retrieved, expected) is False


def test_check_file_match_empty_retrieved():
    retrieved = []
    expected = ["src/chinvex/ingest.py"]
    assert check_file_match(retrieved, expected) is False


def test_check_file_match_case_sensitive():
    retrieved = ["src/chinvex/Ingest.py"]
    expected = ["src/chinvex/ingest.py"]
    assert check_file_match(retrieved, expected) is False


def test_check_anchor_match_found(tmp_path):
    # Create test file with content
    test_file = tmp_path / "test.py"
    test_file.write_text("""
    # Configuration
    BATCH_SIZE = 5000  # ChromaDB batch limit is 5000 vectors
    """)

    result = check_anchor_match(
        file_path=str(test_file),
        anchor="5000 vectors"
    )
    assert result is True


def test_check_anchor_match_not_found(tmp_path):
    test_file = tmp_path / "test.py"
    test_file.write_text("""
    # Configuration
    BATCH_SIZE = 1000
    """)

    result = check_anchor_match(
        file_path=str(test_file),
        anchor="5000 vectors"
    )
    assert result is False


def test_check_anchor_match_file_not_found():
    result = check_anchor_match(
        file_path="nonexistent.py",
        anchor="test"
    )
    assert result is False


def test_check_anchor_match_case_insensitive(tmp_path):
    test_file = tmp_path / "test.py"
    test_file.write_text("BATCH SIZE IS 5000 VECTORS")

    result = check_anchor_match(
        file_path=str(test_file),
        anchor="5000 vectors"
    )
    assert result is True


def test_evaluate_query_pass_file_match():
    query_result = QueryResult(
        query="test",
        expected_files=["src/chinvex/ingest.py"],
        retrieved_files=["src/chinvex/ingest.py", "other.py"],
        k=5
    )

    evaluation = evaluate_query(query_result)

    assert evaluation["passed"] is True
    assert evaluation["file_match"] is True
    assert evaluation["rank"] == 1
    assert evaluation["anchor_match"] is None


def test_evaluate_query_pass_with_rank():
    query_result = QueryResult(
        query="test",
        expected_files=["src/chinvex/config.py"],
        retrieved_files=["a.py", "b.py", "src/chinvex/config.py"],
        k=5
    )

    evaluation = evaluate_query(query_result)

    assert evaluation["passed"] is True
    assert evaluation["file_match"] is True
    assert evaluation["rank"] == 3


def test_evaluate_query_fail_no_match():
    query_result = QueryResult(
        query="test",
        expected_files=["src/chinvex/ingest.py"],
        retrieved_files=["other.py", "another.py"],
        k=5
    )

    evaluation = evaluate_query(query_result)

    assert evaluation["passed"] is False
    assert evaluation["file_match"] is False
    assert evaluation["rank"] is None


def test_evaluate_query_with_anchor_match(tmp_path):
    # Create test file
    test_file = tmp_path / "ingest.py"
    test_file.write_text("BATCH_SIZE = 5000  # batch limit")

    query_result = QueryResult(
        query="test",
        expected_files=[str(test_file)],
        retrieved_files=[str(test_file)],
        k=5,
        anchor="batch limit"
    )

    evaluation = evaluate_query(query_result)

    assert evaluation["passed"] is True
    assert evaluation["file_match"] is True
    assert evaluation["anchor_match"] is True
    assert evaluation["rank"] == 1


def test_evaluate_query_with_anchor_no_match(tmp_path):
    test_file = tmp_path / "ingest.py"
    test_file.write_text("BATCH_SIZE = 1000")

    query_result = QueryResult(
        query="test",
        expected_files=[str(test_file)],
        retrieved_files=[str(test_file)],
        k=5,
        anchor="5000 vectors"
    )

    evaluation = evaluate_query(query_result)

    assert evaluation["passed"] is True
    assert evaluation["file_match"] is True
    assert evaluation["anchor_match"] is False


def test_evaluate_query_multiple_expected_files():
    query_result = QueryResult(
        query="test",
        expected_files=["src/a.py", "src/b.py", "src/c.py"],
        retrieved_files=["src/b.py", "other.py"],
        k=5
    )

    evaluation = evaluate_query(query_result)

    assert evaluation["passed"] is True
    assert evaluation["file_match"] is True
    assert evaluation["rank"] == 1
    assert evaluation["matched_file"] == "src/b.py"

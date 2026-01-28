import pytest
from pydantic import ValidationError
from chinvex.gateway.validation import EvidenceRequest, SearchRequest, ChunksRequest


def test_evidence_request_valid():
    """Should accept valid evidence request."""
    req = EvidenceRequest(
        context="Chinvex",
        query="test query",
        k=8
    )
    assert req.context == "Chinvex"
    assert req.query == "test query"
    assert req.k == 8


def test_context_name_invalid_chars():
    """Should reject context with path traversal."""
    with pytest.raises(ValidationError) as exc_info:
        EvidenceRequest(context="../../../etc/passwd", query="test")

    assert "Invalid context name format" in str(exc_info.value)


def test_query_too_long():
    """Should reject query exceeding 1000 chars."""
    with pytest.raises(ValidationError) as exc_info:
        EvidenceRequest(context="Test", query="a" * 1001)

    assert "exceeds 1000 character limit" in str(exc_info.value)


def test_query_empty():
    """Should reject empty query."""
    with pytest.raises(ValidationError) as exc_info:
        EvidenceRequest(context="Test", query="   ")

    assert "cannot be empty" in str(exc_info.value)


def test_query_null_bytes():
    """Should reject query with null bytes."""
    with pytest.raises(ValidationError) as exc_info:
        EvidenceRequest(context="Test", query="test\x00query")

    assert "Null bytes not allowed" in str(exc_info.value)


def test_k_out_of_range():
    """Should reject k outside 1-20."""
    with pytest.raises(ValidationError) as exc_info:
        EvidenceRequest(context="Test", query="test", k=50)

    assert "must be between 1 and 20" in str(exc_info.value)


def test_chunks_request_too_many_ids():
    """Should reject more than 20 chunk IDs."""
    with pytest.raises(ValidationError) as exc_info:
        ChunksRequest(
            context="Test",
            chunk_ids=["abc123def456"] * 21
        )

    assert "Maximum 20 chunk IDs" in str(exc_info.value)

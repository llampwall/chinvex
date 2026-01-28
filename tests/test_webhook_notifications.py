"""Test webhook notifications."""
import pytest
import json
from unittest.mock import patch, Mock


def test_webhook_url_validation_https_required():
    """Test that HTTP URLs are rejected."""
    from chinvex.notifications import validate_webhook_url

    assert validate_webhook_url("https://example.com/webhook") is True
    assert validate_webhook_url("http://example.com/webhook") is False


def test_webhook_url_validation_blocks_private_ips():
    """Test that private IPs are blocked."""
    from chinvex.notifications import validate_webhook_url

    assert validate_webhook_url("https://127.0.0.1/webhook") is False
    assert validate_webhook_url("https://localhost/webhook") is False
    assert validate_webhook_url("https://192.168.1.1/webhook") is False
    assert validate_webhook_url("https://10.0.0.1/webhook") is False


def test_send_webhook_sanitizes_source_uri():
    """Test that source_uri is sanitized to filename only."""
    from chinvex.notifications import sanitize_source_uri

    assert sanitize_source_uri(r"C:\Users\Jordan\Private\diary.md") == "diary.md"
    assert sanitize_source_uri("/home/user/secret/file.txt") == "file.txt"
    assert sanitize_source_uri("file.txt") == "file.txt"


@patch('chinvex.notifications.requests.post')
def test_send_webhook_posts_payload(mock_post):
    """Test that send_webhook posts correct payload."""
    from chinvex.notifications import send_webhook

    mock_post.return_value = Mock(status_code=200)

    payload = {
        "event": "watch_hit",
        "watch_id": "test_watch",
        "query": "test query",
        "hits": [{"chunk_id": "abc", "score": 0.85, "snippet": "test"}]
    }

    success = send_webhook("https://example.com/webhook", payload)

    assert success is True
    mock_post.assert_called_once()


@patch('chinvex.notifications.requests.post')
def test_send_webhook_retries_on_failure(mock_post):
    """Test that send_webhook retries failed requests."""
    from chinvex.notifications import send_webhook

    # First two calls fail, third succeeds
    mock_post.side_effect = [
        Mock(status_code=500),
        Mock(status_code=500),
        Mock(status_code=200)
    ]

    payload = {"event": "test"}

    success = send_webhook("https://example.com/webhook", payload, retry_count=2)

    assert success is True
    assert mock_post.call_count == 3


@patch('chinvex.notifications.requests.post')
def test_send_webhook_fails_after_all_retries(mock_post):
    """Test that send_webhook returns False after all retries exhausted."""
    from chinvex.notifications import send_webhook

    # All calls fail
    mock_post.side_effect = [
        Mock(status_code=500),
        Mock(status_code=500),
        Mock(status_code=500)
    ]

    payload = {"event": "test"}

    success = send_webhook("https://example.com/webhook", payload, retry_count=2)

    assert success is False
    assert mock_post.call_count == 3


def test_create_watch_hit_payload():
    """Test that create_watch_hit_payload formats correctly."""
    from chinvex.notifications import create_watch_hit_payload

    hits = [
        {
            "chunk_id": "chunk_1",
            "score": 0.85,
            "text": "This is a long chunk of text that should be truncated after 200 characters. " * 10,
            "source_uri": r"C:\Users\Jordan\Private\diary.md"
        },
        {
            "chunk_id": "chunk_2",
            "score": 0.75,
            "text": "Short text",
            "source_uri": "/home/user/file.txt"
        }
    ]

    payload = create_watch_hit_payload("test_watch", "test query", hits)

    assert payload["event"] == "watch_hit"
    assert payload["watch_id"] == "test_watch"
    assert payload["query"] == "test query"
    assert len(payload["hits"]) == 2

    # Verify snippet truncation (200 chars max)
    assert len(payload["hits"][0]["snippet"]) == 200

    # Verify source_uri sanitization
    assert payload["hits"][0]["source"] == "diary.md"
    assert payload["hits"][1]["source"] == "file.txt"


def test_validate_webhook_url_handles_invalid_urls():
    """Test that invalid URLs are rejected."""
    from chinvex.notifications import validate_webhook_url

    assert validate_webhook_url("not a url") is False
    assert validate_webhook_url("") is False
    assert validate_webhook_url("ftp://example.com") is False

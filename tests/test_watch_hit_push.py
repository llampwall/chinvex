# tests/test_watch_hit_push.py
import pytest
from unittest.mock import patch, Mock
from chinvex.notify import NtfyConfig


def test_search_sends_watch_hit_push():
    """Search should send push when watch query hits"""
    # This test will be implemented when watches are added in future phases
    # For now, just test the notification utility exists
    from chinvex.notify import send_ntfy_push

    config = NtfyConfig(server="https://ntfy.sh", topic="test")

    with patch('requests.post') as mock_post:
        mock_post.return_value = Mock(status_code=200)

        result = send_ntfy_push(
            config,
            "Found 3 results for watch: 'embedding metrics'",
            title="Watch Hit: embedding metrics",
            tags=["eyes", "mag"]
        )

        assert result is True

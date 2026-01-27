# tests/adapters/test_capture.py
from pathlib import Path
import json
from unittest.mock import patch
from chinvex.adapters.cx_appserver.capture import capture_sample

def test_capture_sample_writes_file(tmp_path):
    """Test that capture_sample writes JSON to correct path."""
    sample_data = {"id": "test_123", "title": "Test"}

    # Temporarily override debug path
    with patch('chinvex.adapters.cx_appserver.capture.SAMPLE_DIR', tmp_path):
        capture_sample("thread_resume", sample_data, "TestContext")

    files = list(tmp_path.glob("TestContext/thread_resume_*.json"))
    assert len(files) == 1

    with open(files[0]) as f:
        loaded = json.load(f)
    assert loaded["id"] == "test_123"

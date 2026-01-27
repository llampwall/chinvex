# tests/adapters/test_cx_schemas.py
from datetime import datetime
from chinvex.adapters.cx_appserver.schemas import ThreadSummary, ThreadDetail, ThreadMessage

def test_thread_summary_creation():
    """Test ThreadSummary model validation."""
    summary = ThreadSummary(
        id="thread_123",
        title="Test Thread",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        message_count=5
    )
    assert summary.id == "thread_123"
    assert summary.title == "Test Thread"

def test_thread_message_creation():
    """Test ThreadMessage model validation."""
    msg = ThreadMessage(
        id="msg_123",
        role="user",
        content="Hello world",
        timestamp=datetime.now()
    )
    assert msg.role == "user"
    assert msg.content == "Hello world"

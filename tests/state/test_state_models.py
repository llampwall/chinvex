# tests/state/test_state_models.py
from datetime import datetime, timezone
from chinvex.state.models import StateJson, RecentlyChanged, ActiveThread, ExtractedTodo, WatchHit


def test_state_json_creation() -> None:
    """Test StateJson model with all fields."""
    now = datetime.now(timezone.utc)
    state = StateJson(
        schema_version=1,
        context="TestContext",
        generated_at=now,
        last_ingest_run="run_abc123",
        generation_status="ok",
        generation_error=None,
        recently_changed=[],
        active_threads=[],
        extracted_todos=[],
        watch_hits=[],
        decisions=[],
        facts=[],
        annotations=[]
    )
    assert state.context == "TestContext"
    assert state.generation_status == "ok"
    assert state.schema_version == 1


def test_recently_changed_to_dict() -> None:
    """Test RecentlyChanged serialization."""
    now = datetime.now(timezone.utc)
    item = RecentlyChanged(
        doc_id="doc1",
        source_type="repo",
        source_uri="C:\\test\\file.py",
        change_type="modified",
        changed_at=now,
        summary="Test summary"
    )

    d = item.to_dict()
    assert d["doc_id"] == "doc1"
    assert d["source_type"] == "repo"
    assert d["change_type"] == "modified"
    assert isinstance(d["changed_at"], str)


def test_active_thread_to_dict() -> None:
    """Test ActiveThread serialization."""
    now = datetime.now(timezone.utc)
    thread = ActiveThread(
        id="thread_123",
        title="Test Thread",
        status="open",
        last_activity=now,
        source="codex_session"
    )

    d = thread.to_dict()
    assert d["id"] == "thread_123"
    assert d["title"] == "Test Thread"
    assert isinstance(d["last_activity"], str)


def test_extracted_todo_to_dict() -> None:
    """Test ExtractedTodo serialization."""
    now = datetime.now(timezone.utc)
    todo = ExtractedTodo(
        text="TODO: implement feature",
        source_uri="C:\\test\\file.py",
        line=42,
        extracted_at=now
    )

    d = todo.to_dict()
    assert d["text"] == "TODO: implement feature"
    assert d["line"] == 42
    assert isinstance(d["extracted_at"], str)


def test_watch_hit_to_dict() -> None:
    """Test WatchHit serialization."""
    now = datetime.now(timezone.utc)
    hit = WatchHit(
        watch_id="watch_1",
        query="test query",
        hits=[{"chunk_id": "chunk1", "score": 0.9}],
        triggered_at=now
    )

    d = hit.to_dict()
    assert d["watch_id"] == "watch_1"
    assert d["query"] == "test query"
    assert len(d["hits"]) == 1
    assert isinstance(d["triggered_at"], str)


def test_state_json_to_dict() -> None:
    """Test StateJson serialization with nested objects."""
    now = datetime.now(timezone.utc)

    recently_changed = RecentlyChanged(
        doc_id="doc1",
        source_type="repo",
        source_uri="test.py",
        change_type="new",
        changed_at=now
    )

    state = StateJson(
        schema_version=1,
        context="TestContext",
        generated_at=now,
        last_ingest_run="run_123",
        generation_status="ok",
        generation_error=None,
        recently_changed=[recently_changed],
        active_threads=[],
        extracted_todos=[],
        watch_hits=[],
        decisions=[],
        facts=[],
        annotations=[]
    )

    d = state.to_dict()
    assert d["schema_version"] == 1
    assert d["context"] == "TestContext"
    assert isinstance(d["generated_at"], str)
    assert len(d["recently_changed"]) == 1
    assert d["recently_changed"][0]["doc_id"] == "doc1"

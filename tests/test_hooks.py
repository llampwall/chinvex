# tests/test_hooks.py
from chinvex.hooks import post_ingest_hook
from chinvex.ingest import IngestRunResult
from datetime import datetime
from pathlib import Path
import json
import tempfile

def test_post_ingest_hook_success():
    """Test post-ingest hook generates state."""
    from unittest.mock import Mock

    # Create mock context with temporary path
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        mock_context = Mock()
        mock_context.name = "TestContext"
        mock_context.state_dir = tmp_path
        mock_context.db_path = None  # No DB override for test

        result = IngestRunResult(
            run_id="test_run",
            context="TestContext",
            started_at=datetime.now(),
            finished_at=datetime.now(),
            new_doc_ids=[],
            updated_doc_ids=[],
            new_chunk_ids=["chunk1"],
            skipped_doc_ids=[],
            error_doc_ids=[],
            stats={}
        )

        # Should not raise
        post_ingest_hook(mock_context, result)

        # Verify state.json created
        state_path = tmp_path / "state.json"
        assert state_path.exists()

        # Verify STATE.md created
        md_path = tmp_path / "STATE.md"
        assert md_path.exists()

        # Verify state.json is valid JSON
        state_data = json.loads(state_path.read_text())
        assert state_data["context"] == "TestContext"
        assert state_data["last_ingest_run"] == "test_run"

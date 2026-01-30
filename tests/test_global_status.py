# tests/test_global_status.py
import pytest
import json
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch
from chinvex.global_status import generate_global_status_md


def test_generate_global_status_md(tmp_path: Path):
    """Should aggregate STATUS.json files into GLOBAL_STATUS.md"""
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    # Create mock contexts with STATUS.json
    for i, name in enumerate(["Chinvex", "Streamside", "Godex"]):
        ctx_dir = contexts_root / name
        ctx_dir.mkdir()

        status = {
            "context": name,
            "last_sync": datetime.now(timezone.utc).isoformat(),
            "chunks": 10000 * (i + 1),
            "watches_pending_hits": i,
            "freshness": {
                "is_stale": i == 2,  # Godex is stale
                "hours_since_sync": i * 3
            }
        }

        (ctx_dir / "STATUS.json").write_text(json.dumps(status))

    output_path = tmp_path / "GLOBAL_STATUS.md"
    generate_global_status_md(contexts_root, output_path)

    assert output_path.exists()

    content = output_path.read_text()

    # Should include all contexts
    assert "Chinvex" in content
    assert "Streamside" in content
    assert "Godex" in content

    # Should show stale warning
    assert "⚠" in content or "stale" in content.lower()

    # Should include table
    assert "|" in content


def test_global_status_includes_watcher_health(tmp_path: Path):
    """Should include watcher daemon health in status"""
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    # Create state dir with heartbeat
    state_dir = tmp_path / ".chinvex"
    state_dir.mkdir()

    heartbeat = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pid": 12345
    }
    (state_dir / "sync_heartbeat.json").write_text(json.dumps(heartbeat))

    output_path = tmp_path / "GLOBAL_STATUS.md"

    # Mock get_state_dir to return our test dir
    with patch('chinvex.sync.cli.get_state_dir', return_value=state_dir):
        generate_global_status_md(contexts_root, output_path)

    content = output_path.read_text()

    # Should show watcher status
    assert "watcher" in content.lower() or "daemon" in content.lower()

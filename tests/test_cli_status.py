# tests/test_cli_status.py
import pytest
from pathlib import Path
from chinvex.cli_status import format_status_output, ContextStatus


def test_format_status_output_healthy():
    """Should format healthy context with green indicator"""
    contexts = [
        ContextStatus(
            name="Chinvex",
            chunks=1234,
            last_sync="2026-01-29T10:00:00Z",
            is_stale=False,
            hours_since_sync=2.5,
            watcher_running=True,
            embedding_provider="openai"
        ),
        ContextStatus(
            name="Streamside",
            chunks=567,
            last_sync="2026-01-29T11:30:00Z",
            is_stale=False,
            hours_since_sync=1.0,
            watcher_running=True,
            embedding_provider="openai"
        )
    ]

    output = format_status_output(contexts, watcher_running=True)

    assert "Chinvex" in output
    assert "1234" in output
    assert "[OK]" in output  # Healthy indicator
    assert "Watcher: Running" in output


def test_format_status_output_stale():
    """Should format stale context with warning indicator"""
    contexts = [
        ContextStatus(
            name="Godex",
            chunks=890,
            last_sync="2026-01-28T05:00:00Z",
            is_stale=True,
            hours_since_sync=31.0,
            watcher_running=False,
            embedding_provider="openai"
        )
    ]

    output = format_status_output(contexts, watcher_running=False)

    assert "Godex" in output
    assert "[STALE]" in output  # Stale indicator
    assert "31h ago" in output
    assert "Watcher: Stopped" in output


def test_format_status_reads_global_status_md(tmp_path: Path):
    """Should read GLOBAL_STATUS.md if it exists"""
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    global_status_md = contexts_root / "GLOBAL_STATUS.md"
    global_status_md.write_text("""# Chinvex Global Status

| Context | Chunks | Last Sync | Status |
|---------|--------|-----------|--------|
| Chinvex | 1234   | 2h ago    | [OK]   |
| Godex   | 890    | 31h ago   | [STALE]|

Watcher: Running
""")

    from chinvex.cli_status import read_global_status
    output = read_global_status(contexts_root)

    assert "Chinvex" in output
    assert "Godex" in output
    assert "Watcher: Running" in output

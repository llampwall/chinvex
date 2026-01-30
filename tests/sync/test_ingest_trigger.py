# tests/sync/test_ingest_trigger.py
import pytest
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch
from chinvex.sync.process import WatcherProcess
from chinvex.sync.watcher import ChangeAccumulator


def test_trigger_delta_ingest_calls_cli(tmp_path: Path):
    """Triggering delta ingest should spawn chinvex ingest process"""
    state_dir = tmp_path / "state"
    contexts_root = tmp_path / "contexts"

    watcher = WatcherProcess(
        sources=[],
        state_dir=state_dir,
        contexts_root=contexts_root,
        debounce_seconds=0.1
    )

    # Create accumulator with changes
    acc = ChangeAccumulator(debounce_seconds=0.1, max_paths=500)
    acc.add_change(Path("/repo/file1.txt"))
    acc.add_change(Path("/repo/file2.txt"))

    # Mock subprocess.Popen
    with patch('subprocess.Popen') as mock_popen:
        watcher._trigger_ingest("TestCtx", acc)

        # Should have called subprocess.Popen with chinvex ingest command
        assert mock_popen.called
        call_args = mock_popen.call_args[0][0]  # First positional arg (command list)

        # Verify command structure
        assert "chinvex" in " ".join(call_args) or "chinvex.cli" in " ".join(call_args)
        assert "ingest" in call_args
        assert "--context" in call_args
        assert "TestCtx" in call_args
        assert "--paths" in call_args


def test_trigger_full_ingest_when_over_limit(tmp_path: Path):
    """Over-limit changes should trigger full ingest (no --paths)"""
    state_dir = tmp_path / "state"
    contexts_root = tmp_path / "contexts"

    watcher = WatcherProcess(
        sources=[],
        state_dir=state_dir,
        contexts_root=contexts_root,
        debounce_seconds=0.1,
        max_paths=2  # Low limit for testing
    )

    # Create accumulator exceeding limit
    acc = ChangeAccumulator(debounce_seconds=0.1, max_paths=2)
    for i in range(5):
        acc.add_change(Path(f"/repo/file{i}.txt"))

    with patch('subprocess.Popen') as mock_popen:
        watcher._trigger_ingest("TestCtx", acc)

        call_args = mock_popen.call_args[0][0]

        # Should NOT have --paths (full ingest)
        assert "--paths" not in call_args
        assert "ingest" in call_args


def test_trigger_skips_if_lock_held(tmp_path: Path):
    """Should skip ingest if context lock already held"""
    state_dir = tmp_path / "state"
    contexts_root = tmp_path / "contexts"
    ctx_dir = contexts_root / "TestCtx"
    ctx_dir.mkdir(parents=True)

    watcher = WatcherProcess(
        sources=[],
        state_dir=state_dir,
        contexts_root=contexts_root,
        debounce_seconds=0.1
    )

    # Hold the lock
    from chinvex.sync.locks import sync_lock
    lock_file = ctx_dir / ".ingest.lock"

    with sync_lock(lock_file):
        # Try to trigger ingest while lock held
        acc = ChangeAccumulator(debounce_seconds=0.1, max_paths=500)
        acc.add_change(Path("/repo/file1.txt"))

        with patch('subprocess.Popen') as mock_popen:
            watcher._trigger_ingest("TestCtx", acc)

            # Should NOT have called subprocess
            assert not mock_popen.called

"""Kill-switch recovery test - proves this is an appliance."""
import pytest
import time
import subprocess
import psutil
import os
from pathlib import Path


@pytest.mark.slow
@pytest.mark.appliance
def test_kill_switch_recovery(tmp_path: Path):
    """
    The definitive appliance test: Kill watcher, wait for sweep to resurrect it.

    Steps:
    1. Start watcher
    2. Verify heartbeat exists
    3. Kill watcher process
    4. Delete heartbeat file
    5. Trigger sweep manually (simulates scheduled task)
    6. Verify watcher restarted automatically

    This proves the system is self-healing without human intervention.
    """
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"
    state_dir = tmp_path / "state"
    contexts_root.mkdir()
    indexes_root.mkdir()
    state_dir.mkdir()

    # Set environment variables
    env = os.environ.copy()
    env["CHINVEX_CONTEXTS_ROOT"] = str(contexts_root)
    env["CHINVEX_INDEXES_ROOT"] = str(indexes_root)
    env["CHINVEX_STATE_DIR"] = str(state_dir)

    # Create minimal context
    ctx_dir = contexts_root / "test_ctx"
    ctx_dir.mkdir()
    (ctx_dir / "context.json").write_text("""{
        "schema_version": 2,
        "name": "test_ctx",
        "includes": {
            "repos": [],
            "chat_roots": [],
            "codex_session_roots": [],
            "note_roots": []
        },
        "index": {
            "sqlite_path": "%s",
            "chroma_dir": "%s"
        },
        "ollama": {
            "base_url": "http://localhost:11434",
            "embed_model": "mxbai-embed-large"
        },
        "weights": {
            "repo": 1.0,
            "chat": 0.8,
            "codex_session": 0.9,
            "note": 0.7
        },
        "created_at": "2026-01-29T00:00:00Z",
        "updated_at": "2026-01-29T00:00:00Z"
    }""" % (
        str(indexes_root / "test_ctx" / "hybrid.db").replace("\\", "\\\\"),
        str(indexes_root / "test_ctx" / "chroma").replace("\\", "\\\\")
    ))

    # Start watcher
    result = subprocess.run(
        ["chinvex", "sync", "start"],
        capture_output=True,
        text=True,
        env=env
    )
    assert result.returncode == 0, f"Watcher start failed: {result.stderr}"

    # Verify heartbeat exists (should be written immediately on startup)
    heartbeat_file = state_dir / "sync_heartbeat.json"
    max_wait = 5  # Wait up to 5s for initial heartbeat
    waited = 0
    while waited < max_wait and not heartbeat_file.exists():
        time.sleep(0.5)
        waited += 0.5

    assert heartbeat_file.exists(), f"Heartbeat file should exist after {waited}s"

    # Find watcher process
    watcher_pid = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            if any('chinvex' in str(arg) and 'sync' in str(arg) for arg in cmdline):
                watcher_pid = proc.info['pid']
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    assert watcher_pid is not None, "Could not find watcher process"

    # Kill watcher
    psutil.Process(watcher_pid).kill()
    time.sleep(2)

    # Verify process is dead
    assert not psutil.pid_exists(watcher_pid), "Watcher process should be dead"

    # Delete heartbeat to simulate crash
    heartbeat_file.unlink()
    assert not heartbeat_file.exists(), "Heartbeat should be deleted"

    # Trigger sweep manually (instead of waiting 30 min)
    # In real usage, scheduled task would run this
    script_path = Path("C:/Code/chinvex/scripts/scheduled_sweep.ps1").resolve()
    result = subprocess.run(
        [
            "pwsh", "-NoProfile", "-File",
            str(script_path),
            "-ContextsRoot", str(contexts_root),
            "-StateDir", str(state_dir),
            "-NtfyTopic", "test-topic"
        ],
        capture_output=True,
        text=True,
        env=env
    )

    # Sweep should succeed (note: may have stderr from trying to stop already-dead process)
    assert result.returncode == 0, f"Sweep script failed with stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    # Verify watcher restarted
    # Check heartbeat exists again
    time.sleep(5)  # Give it a moment to write heartbeat

    assert heartbeat_file.exists(), "Heartbeat should be recreated by sweep"

    # Verify new watcher process exists
    watcher_running = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            if any('chinvex' in str(arg) and 'sync' in str(arg) for arg in cmdline):
                watcher_running = True
                # Kill for cleanup
                proc.kill()
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    assert watcher_running, "Sweep should have restarted watcher"

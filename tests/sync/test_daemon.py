# tests/sync/test_daemon.py
import pytest
import time
from pathlib import Path
from chinvex.sync.daemon import DaemonManager, DaemonState


def test_daemon_not_running_initially(tmp_path: Path):
    """Daemon should not be running if no PID file exists"""
    dm = DaemonManager(tmp_path)
    assert dm.get_state() == DaemonState.NOT_RUNNING


def test_daemon_writes_pid_file(tmp_path: Path):
    """Starting daemon should write PID file"""
    dm = DaemonManager(tmp_path)
    dm.write_pid(12345)

    assert (tmp_path / "sync.pid").exists()
    assert dm.read_pid() == 12345


def test_daemon_writes_heartbeat(tmp_path: Path):
    """Daemon should write heartbeat with timestamp"""
    dm = DaemonManager(tmp_path)
    dm.write_heartbeat()

    hb_file = tmp_path / "sync_heartbeat.json"
    assert hb_file.exists()

    import json
    data = json.loads(hb_file.read_text())
    assert "timestamp" in data
    assert "pid" in data


def test_daemon_heartbeat_is_stale(tmp_path: Path):
    """Heartbeat older than 5 min should be considered stale"""
    dm = DaemonManager(tmp_path)

    # Write heartbeat with old timestamp
    import json
    from datetime import datetime, timedelta, timezone

    old_time = datetime.now(timezone.utc) - timedelta(minutes=6)
    hb_data = {
        "timestamp": old_time.isoformat(),
        "pid": 12345
    }
    hb_file = tmp_path / "sync_heartbeat.json"
    hb_file.write_text(json.dumps(hb_data))

    assert dm.is_heartbeat_stale(threshold_minutes=5)


def test_daemon_heartbeat_is_fresh(tmp_path: Path):
    """Recent heartbeat should not be stale"""
    dm = DaemonManager(tmp_path)
    dm.write_heartbeat()

    assert not dm.is_heartbeat_stale(threshold_minutes=5)


def test_daemon_cleanup_removes_files(tmp_path: Path):
    """Cleanup should remove PID and heartbeat files"""
    dm = DaemonManager(tmp_path)
    dm.write_pid(12345)
    dm.write_heartbeat()

    dm.cleanup()

    assert not (tmp_path / "sync.pid").exists()
    assert not (tmp_path / "sync_heartbeat.json").exists()

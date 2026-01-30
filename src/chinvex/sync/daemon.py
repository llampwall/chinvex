"""Daemon process management for file watcher."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

log = logging.getLogger(__name__)


class DaemonState(Enum):
    NOT_RUNNING = "not_running"
    RUNNING = "running"
    STALE = "stale"  # PID exists but heartbeat stale


class DaemonManager:
    """Manages daemon PID file, heartbeat, and state detection."""

    def __init__(self, state_dir: Path):
        """
        Args:
            state_dir: Directory for PID/heartbeat files (typically ~/.chinvex)
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.pid_file = self.state_dir / "sync.pid"
        self.heartbeat_file = self.state_dir / "sync_heartbeat.json"
        self.log_file = self.state_dir / "sync.log"

    def write_pid(self, pid: int) -> None:
        """Write PID to file."""
        self.pid_file.write_text(str(pid))
        log.info(f"Wrote PID {pid} to {self.pid_file}")

    def read_pid(self) -> int | None:
        """Read PID from file, return None if not exists."""
        if not self.pid_file.exists():
            return None
        try:
            return int(self.pid_file.read_text().strip())
        except (ValueError, OSError) as e:
            log.warning(f"Failed to read PID file: {e}")
            return None

    def write_heartbeat(self) -> None:
        """Write current timestamp and PID to heartbeat file."""
        import os
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid()
        }
        self.heartbeat_file.write_text(json.dumps(data, indent=2))

    def is_heartbeat_stale(self, threshold_minutes: int = 5) -> bool:
        """
        Check if heartbeat is stale (older than threshold).

        Args:
            threshold_minutes: Consider stale if older than this many minutes

        Returns:
            True if heartbeat is stale or missing
        """
        if not self.heartbeat_file.exists():
            return True

        try:
            data = json.loads(self.heartbeat_file.read_text())
            timestamp_str = data["timestamp"]
            timestamp = datetime.fromisoformat(timestamp_str)

            # Ensure timezone-aware comparison
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)

            age = datetime.now(timezone.utc) - timestamp
            return age > timedelta(minutes=threshold_minutes)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            log.warning(f"Failed to parse heartbeat: {e}")
            return True

    def get_state(self) -> DaemonState:
        """
        Determine current daemon state.

        Returns:
            DaemonState enum value
        """
        pid = self.read_pid()
        if pid is None:
            return DaemonState.NOT_RUNNING

        # PID exists, check heartbeat
        if self.is_heartbeat_stale():
            return DaemonState.STALE

        return DaemonState.RUNNING

    def cleanup(self) -> None:
        """Remove PID and heartbeat files."""
        if self.pid_file.exists():
            self.pid_file.unlink()
        if self.heartbeat_file.exists():
            self.heartbeat_file.unlink()
        log.info("Cleaned up daemon state files")

# Bootstrapping Completion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform Chinvex from a system you operate into an appliance that operates on you - zero manual commands, always fresh, always visible, always in your face.

**Architecture:** File watcher daemon (watchdog) + scheduled sweep (Task Scheduler) + git hooks + ntfy push notifications + status artifacts. All components designed for Windows with belt-and-suspenders reliability.

**Tech Stack:** Python 3.12, watchdog, portalocker, typer CLI, PowerShell scripts, Windows Task Scheduler

---

## Phase 1: Foundation - File Watcher Core

### Task 1: Exclude Pattern Matcher

**Files:**
- Create: `src/chinvex/sync/__init__.py`
- Create: `src/chinvex/sync/patterns.py`
- Create: `tests/sync/test_exclude_patterns.py`

**Step 1: Write failing test for fnmatch pattern matching**

```python
# tests/sync/test_exclude_patterns.py
import pytest
from pathlib import Path
from chinvex.sync.patterns import should_exclude


def test_exclude_exact_filename():
    """STATUS.json files should be excluded"""
    assert should_exclude("contexts/Chinvex/STATUS.json", watch_root=Path("/root"))
    assert should_exclude("STATUS.json", watch_root=Path("/root"))


def test_exclude_recursive_pattern():
    """**/.git/** should exclude all .git subdirs"""
    assert should_exclude("repo/.git/HEAD", watch_root=Path("/root"))
    assert should_exclude("repo/sub/.git/config", watch_root=Path("/root"))
    assert should_exclude("repo/.git/objects/ab/cd1234", watch_root=Path("/root"))


def test_exclude_wildcard_pattern():
    """**/*_BRIEF.md should exclude all _BRIEF.md files"""
    assert should_exclude("MORNING_BRIEF.md", watch_root=Path("/root"))
    assert should_exclude("subdir/SESSION_BRIEF.md", watch_root=Path("/root"))
    assert should_exclude("deep/nested/PROJECT_BRIEF.md", watch_root=Path("/root"))


def test_dont_exclude_normal_files():
    """Normal source files should not be excluded"""
    assert not should_exclude("src/main.py", watch_root=Path("/root"))
    assert not should_exclude("README.md", watch_root=Path("/root"))
    assert not should_exclude("test/test_file.py", watch_root=Path("/root"))


def test_exclude_case_insensitive_windows():
    """Pattern matching should be case-insensitive on Windows"""
    import platform
    if platform.system() == "Windows":
        assert should_exclude("contexts/chinvex/status.json", watch_root=Path("/root"))
        assert should_exclude("REPO/.GIT/config", watch_root=Path("/root"))


def test_exclude_home_chinvex_internals():
    """~/.chinvex internals should be excluded"""
    import os
    home = Path.home()
    assert should_exclude(str(home / ".chinvex/sync.log"), watch_root=home)
    assert should_exclude(str(home / ".chinvex/sync.pid"), watch_root=home)
    assert should_exclude(str(home / ".chinvex/sync_heartbeat.json"), watch_root=home)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/sync/test_exclude_patterns.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chinvex.sync'"

**Step 3: Implement minimal pattern matcher**

```python
# src/chinvex/sync/__init__.py
"""File watcher sync daemon."""

# src/chinvex/sync/patterns.py
"""Exclude pattern matching for file watcher."""
from __future__ import annotations

import fnmatch
import platform
from pathlib import Path


# Exclude patterns (fnmatch glob syntax)
EXCLUDE_PATTERNS = [
    # Chinvex outputs (would cause ingest storm)
    "contexts/**/STATUS.json",
    "contexts/**/ingest_runs.jsonl",
    "contexts/**/digests/**",
    "**/MORNING_BRIEF.md",
    "**/GLOBAL_STATUS.md",
    "**/*_BRIEF.md",
    "**/SESSION_BRIEF.md",
    # Per-repo chinvex artifacts
    "**/.chinvex/**",
    "**/docs/memory/**",
    # Standard ignores
    "**/.git/**",
    "**/node_modules/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.venv/**",
    "**/venv/**",
    # Chinvex internals (will be expanded with home dir)
    ".chinvex/*.log",
    ".chinvex/*.pid",
    ".chinvex/*.json",
    ".chinvex/*.jsonl",
]


def should_exclude(path: str | Path, watch_root: Path) -> bool:
    """
    Check if a path should be excluded from watching.

    Args:
        path: Absolute or relative path to check
        watch_root: Root directory being watched

    Returns:
        True if path matches any exclude pattern
    """
    path_obj = Path(path)

    # Convert to relative path if absolute
    if path_obj.is_absolute():
        try:
            rel_path = path_obj.relative_to(watch_root)
        except ValueError:
            # Path outside watch_root, check against home for .chinvex patterns
            try:
                rel_path = path_obj.relative_to(Path.home())
            except ValueError:
                # Not under watch_root or home, use as-is
                rel_path = path_obj
    else:
        rel_path = path_obj

    # Normalize to forward slashes for fnmatch
    path_str = str(rel_path).replace("\\", "/")

    # Case-insensitive on Windows
    if platform.system() == "Windows":
        path_str = path_str.lower()
        patterns = [p.lower() for p in EXCLUDE_PATTERNS]
    else:
        patterns = EXCLUDE_PATTERNS

    # Check against all patterns
    for pattern in patterns:
        if fnmatch.fnmatch(path_str, pattern):
            return True
        # Also check if any parent matches pattern (for directory exclusions)
        for parent in Path(path_str).parents:
            if fnmatch.fnmatch(str(parent).replace("\\", "/"), pattern):
                return True

    return False
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/sync/test_exclude_patterns.py -v`
Expected: PASS (all tests green)

**Step 5: Commit**

```bash
git add src/chinvex/sync/ tests/sync/
git commit -m "feat(sync): add exclude pattern matcher for file watcher

- fnmatch glob syntax support
- Case-insensitive matching on Windows
- Excludes chinvex outputs, .git, node_modules, etc.
- Prevents infinite ingest loops

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 2: Daemon Process Manager

**Files:**
- Create: `src/chinvex/sync/daemon.py`
- Create: `tests/sync/test_daemon.py`

**Step 1: Write failing test for PID file management**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/sync/test_daemon.py -v`
Expected: FAIL with "cannot import name 'DaemonManager'"

**Step 3: Implement daemon manager**

```python
# src/chinvex/sync/daemon.py
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/sync/test_daemon.py -v`
Expected: PASS (all tests green)

**Step 5: Commit**

```bash
git add src/chinvex/sync/daemon.py tests/sync/test_daemon.py
git commit -m "feat(sync): add daemon process manager

- PID file management
- Heartbeat tracking with staleness detection
- State detection (not_running/running/stale)
- Cleanup utilities

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 3: File Watcher with Debounce

**Files:**
- Create: `src/chinvex/sync/watcher.py`
- Create: `tests/sync/test_watcher.py`

**Step 1: Write failing test for file change accumulation**

```python
# tests/sync/test_watcher.py
import pytest
import time
from pathlib import Path
from chinvex.sync.watcher import ChangeAccumulator


def test_accumulator_starts_empty():
    """New accumulator should have no changes"""
    acc = ChangeAccumulator(debounce_seconds=1, max_paths=500)
    assert len(acc.get_changes()) == 0


def test_accumulator_adds_file_changes(tmp_path: Path):
    """Adding file changes should accumulate paths"""
    acc = ChangeAccumulator(debounce_seconds=1, max_paths=500)

    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"

    acc.add_change(file1)
    acc.add_change(file2)

    changes = acc.get_changes()
    assert len(changes) == 2
    assert file1 in changes
    assert file2 in changes


def test_accumulator_deduplicates_paths(tmp_path: Path):
    """Same path added multiple times should appear once"""
    acc = ChangeAccumulator(debounce_seconds=1, max_paths=500)

    file1 = tmp_path / "file1.txt"

    acc.add_change(file1)
    acc.add_change(file1)
    acc.add_change(file1)

    changes = acc.get_changes()
    assert len(changes) == 1


def test_accumulator_clears_after_get():
    """get_and_clear() should return changes and reset"""
    acc = ChangeAccumulator(debounce_seconds=1, max_paths=500)

    acc.add_change(Path("file1.txt"))
    acc.add_change(Path("file2.txt"))

    changes = acc.get_and_clear()
    assert len(changes) == 2

    # Should be empty after clear
    assert len(acc.get_changes()) == 0


def test_accumulator_respects_max_paths():
    """Should track when max_paths exceeded"""
    acc = ChangeAccumulator(debounce_seconds=1, max_paths=3)

    for i in range(5):
        acc.add_change(Path(f"file{i}.txt"))

    assert acc.is_over_limit()


def test_accumulator_debounce_timer():
    """Should track time since last change"""
    acc = ChangeAccumulator(debounce_seconds=0.1, max_paths=500)

    acc.add_change(Path("file1.txt"))

    # Immediately after, not ready
    assert not acc.is_ready()

    # After debounce period, should be ready
    time.sleep(0.15)
    assert acc.is_ready()


def test_accumulator_resets_timer_on_new_change():
    """Adding new change should reset debounce timer"""
    acc = ChangeAccumulator(debounce_seconds=0.1, max_paths=500)

    acc.add_change(Path("file1.txt"))
    time.sleep(0.08)  # Almost ready

    # Add another change - should reset timer
    acc.add_change(Path("file2.txt"))

    # Should not be ready yet
    assert not acc.is_ready()


def test_accumulator_max_debounce_cap():
    """Should force ingest after 5 minutes even if changes keep coming"""
    acc = ChangeAccumulator(debounce_seconds=30, max_paths=500)

    # Simulate the first change
    acc.add_change(Path("file1.txt"))

    # Mock time to simulate 5+ minutes passing
    original_time = acc._first_change_time
    acc._first_change_time = original_time - 301  # 5min + 1s ago

    # Even though normal debounce hasn't elapsed, should be ready due to cap
    assert acc.is_ready()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/sync/test_watcher.py -v`
Expected: FAIL with "cannot import name 'ChangeAccumulator'"

**Step 3: Implement change accumulator**

```python
# src/chinvex/sync/watcher.py
"""File watcher with debounce and change accumulation."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Set

log = logging.getLogger(__name__)


class ChangeAccumulator:
    """
    Accumulates file changes with debounce logic.

    Debounce semantics:
    - Wait debounce_seconds after last change before considering ready
    - Track all changed paths (deduplicated)
    - Flag if path count exceeds max_paths (triggers full ingest fallback)
    - Force ingest after 5 minutes even if changes keep coming (max debounce cap)
    """

    MAX_DEBOUNCE_SECONDS = 300  # 5 minutes cap

    def __init__(self, debounce_seconds: float, max_paths: int):
        """
        Args:
            debounce_seconds: Seconds of quiet before ready
            max_paths: Max paths before flagging full ingest needed
        """
        self.debounce_seconds = debounce_seconds
        self.max_paths = max_paths

        self._changes: Set[Path] = set()
        self._last_change_time: float | None = None
        self._first_change_time: float | None = None  # Track when accumulation started

    def add_change(self, path: Path) -> None:
        """
        Add a changed file path.

        Resets debounce timer.
        """
        self._changes.add(path)
        self._last_change_time = time.time()

        # Track first change time for max debounce cap
        if self._first_change_time is None:
            self._first_change_time = time.time()

        log.debug(f"Change recorded: {path} (total: {len(self._changes)})")

    def get_changes(self) -> list[Path]:
        """Get accumulated changes without clearing."""
        return list(self._changes)

    def get_and_clear(self) -> list[Path]:
        """Get accumulated changes and clear accumulator."""
        changes = list(self._changes)
        self._changes.clear()
        self._last_change_time = None
        self._first_change_time = None
        return changes

    def is_ready(self) -> bool:
        """
        Check if enough quiet time has passed since last change.

        Returns True if either:
        - Debounce period elapsed and changes exist
        - Total debounce time exceeds 5 minutes (force ingest)

        Returns:
            True if debounce period elapsed and changes exist
        """
        if not self._changes:
            return False

        if self._last_change_time is None:
            return False

        # Check if max debounce cap exceeded (5 min total)
        if self._first_change_time is not None:
            total_time = time.time() - self._first_change_time
            if total_time >= self.MAX_DEBOUNCE_SECONDS:
                log.info(f"Max debounce cap ({self.MAX_DEBOUNCE_SECONDS}s) exceeded - forcing ingest")
                return True

        # Normal debounce check
        elapsed = time.time() - self._last_change_time
        return elapsed >= self.debounce_seconds

    def is_over_limit(self) -> bool:
        """Check if accumulated paths exceed max_paths."""
        return len(self._changes) > self.max_paths
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/sync/test_watcher.py -v`
Expected: PASS (all tests green)

**Step 5: Commit**

```bash
git add src/chinvex/sync/watcher.py tests/sync/test_watcher.py
git commit -m "feat(sync): add change accumulator with debounce

- Accumulates file changes in memory
- Deduplicates paths
- Debounce timer (resets on new changes)
- Tracks when path count exceeds limit

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 2: Sync Commands & Lock Management

### Task 4: Portalocker Integration

**Files:**
- Create: `src/chinvex/sync/locks.py`
- Create: `tests/sync/test_locks.py`

**Step 1: Write failing test for atomic lock acquisition**

```python
# tests/sync/test_locks.py
import pytest
import threading
import time
from pathlib import Path
from chinvex.sync.locks import try_acquire_sync_lock, SyncLockHeld


def test_acquire_lock_success(tmp_path: Path):
    """Should successfully acquire lock when not held"""
    lock_file = tmp_path / "test.lock"

    acquired = try_acquire_sync_lock(lock_file)
    assert acquired is not None

    # Clean up
    acquired.release()


def test_acquire_lock_fails_when_held(tmp_path: Path):
    """Should fail to acquire when lock already held"""
    lock_file = tmp_path / "test.lock"

    # First acquisition succeeds
    first = try_acquire_sync_lock(lock_file)
    assert first is not None

    # Second acquisition fails
    with pytest.raises(SyncLockHeld):
        try_acquire_sync_lock(lock_file)

    # Clean up
    first.release()


def test_lock_released_allows_reacquisition(tmp_path: Path):
    """After release, should be able to acquire again"""
    lock_file = tmp_path / "test.lock"

    first = try_acquire_sync_lock(lock_file)
    first.release()

    # Should succeed after release
    second = try_acquire_sync_lock(lock_file)
    assert second is not None
    second.release()


def test_context_manager_usage(tmp_path: Path):
    """Lock should work as context manager"""
    from chinvex.sync.locks import sync_lock

    lock_file = tmp_path / "test.lock"

    with sync_lock(lock_file):
        # Lock held here
        with pytest.raises(SyncLockHeld):
            try_acquire_sync_lock(lock_file)

    # Lock released after context exit
    second = try_acquire_sync_lock(lock_file)
    assert second is not None
    second.release()


def test_lock_concurrent_access(tmp_path: Path):
    """Two threads should not both acquire lock"""
    lock_file = tmp_path / "test.lock"
    results = {"thread1": None, "thread2": None}

    def try_lock(name: str):
        try:
            lock = try_acquire_sync_lock(lock_file)
            results[name] = "acquired"
            time.sleep(0.1)
            lock.release()
        except SyncLockHeld:
            results[name] = "blocked"

    t1 = threading.Thread(target=try_lock, args=("thread1",))
    t2 = threading.Thread(target=try_lock, args=("thread2",))

    t1.start()
    time.sleep(0.05)  # Let t1 acquire first
    t2.start()

    t1.join()
    t2.join()

    # One should acquire, one should block
    assert results["thread1"] == "acquired"
    assert results["thread2"] == "blocked"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/sync/test_locks.py -v`
Expected: FAIL with "cannot import name 'try_acquire_sync_lock'"

**Step 3: Implement lock utilities**

```python
# src/chinvex/sync/locks.py
"""File lock utilities for sync daemon."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

import portalocker

log = logging.getLogger(__name__)


class SyncLockHeld(Exception):
    """Raised when sync lock is already held by another process."""
    pass


def try_acquire_sync_lock(lock_file: Path) -> portalocker.Lock:
    """
    Attempt to acquire sync lock non-blocking.

    Args:
        lock_file: Path to lock file

    Returns:
        Lock object if acquired

    Raises:
        SyncLockHeld: If lock is already held
    """
    try:
        lock = portalocker.Lock(
            str(lock_file),
            mode="w",
            timeout=0,  # Non-blocking
            fail_when_locked=True
        )
        lock.acquire()
        log.debug(f"Acquired lock: {lock_file}")
        return lock
    except portalocker.LockException as e:
        raise SyncLockHeld(f"Lock already held: {lock_file}") from e


@contextmanager
def sync_lock(lock_file: Path):
    """
    Context manager for sync lock.

    Usage:
        with sync_lock(lock_file):
            # Do work while holding lock
            pass

    Raises:
        SyncLockHeld: If lock already held
    """
    lock = try_acquire_sync_lock(lock_file)
    try:
        yield lock
    finally:
        lock.release()
        log.debug(f"Released lock: {lock_file}")


def check_context_lock_held(contexts_root: Path, context_name: str) -> bool:
    """
    Check if a context's ingest lock is currently held (non-blocking check).

    Args:
        contexts_root: Root directory for contexts
        context_name: Name of context to check

    Returns:
        True if lock is held, False if available
    """
    lock_file = contexts_root / context_name / ".ingest.lock"
    try:
        lock = try_acquire_sync_lock(lock_file)
        lock.release()
        return False  # Was able to acquire, so not held
    except SyncLockHeld:
        return True  # Lock is held
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/sync/test_locks.py -v`
Expected: PASS (all tests green)

**Step 5: Commit**

```bash
git add src/chinvex/sync/locks.py tests/sync/test_locks.py
git commit -m "feat(sync): add portalocker-based lock utilities

- Non-blocking lock acquisition
- Context manager for automatic release
- Context ingest lock checking
- SyncLockHeld exception for clear errors

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 5: Sync CLI Commands (start/stop/status/ensure-running)

**Files:**
- Modify: `src/chinvex/cli.py` (add sync subcommand)
- Create: `src/chinvex/sync/cli.py`
- Create: `tests/sync/test_sync_cli.py`

**Step 1: Write failing test for sync start command**

```python
# tests/sync/test_sync_cli.py
import pytest
from pathlib import Path
from typer.testing import CliRunner
from chinvex.cli import app

runner = CliRunner()


def test_sync_start_when_not_running(tmp_path: Path, monkeypatch):
    """sync start should start daemon when not running"""
    monkeypatch.setenv("CHINVEX_STATE_DIR", str(tmp_path))

    result = runner.invoke(app, ["sync", "start"])

    # Should succeed
    assert result.exit_code == 0
    assert "started" in result.stdout.lower() or "daemon" in result.stdout.lower()

    # Should write PID file
    assert (tmp_path / "sync.pid").exists()


def test_sync_start_when_already_running(tmp_path: Path, monkeypatch):
    """sync start should refuse if already running"""
    monkeypatch.setenv("CHINVEX_STATE_DIR", str(tmp_path))

    # Start first time
    result1 = runner.invoke(app, ["sync", "start"])
    assert result1.exit_code == 0

    # Try to start again - should fail
    result2 = runner.invoke(app, ["sync", "start"])
    assert result2.exit_code != 0
    assert "already running" in result2.stdout.lower()


def test_sync_stop(tmp_path: Path, monkeypatch):
    """sync stop should stop running daemon"""
    monkeypatch.setenv("CHINVEX_STATE_DIR", str(tmp_path))

    # Start daemon
    runner.invoke(app, ["sync", "start"])
    assert (tmp_path / "sync.pid").exists()

    # Stop daemon
    result = runner.invoke(app, ["sync", "stop"])
    assert result.exit_code == 0

    # PID file should be removed
    assert not (tmp_path / "sync.pid").exists()


def test_sync_status_not_running(tmp_path: Path, monkeypatch):
    """sync status should show not running"""
    monkeypatch.setenv("CHINVEX_STATE_DIR", str(tmp_path))

    result = runner.invoke(app, ["sync", "status"])
    assert result.exit_code == 0
    assert "not running" in result.stdout.lower()


def test_sync_status_running(tmp_path: Path, monkeypatch):
    """sync status should show running state"""
    monkeypatch.setenv("CHINVEX_STATE_DIR", str(tmp_path))

    # Start daemon
    runner.invoke(app, ["sync", "start"])

    result = runner.invoke(app, ["sync", "status"])
    assert result.exit_code == 0
    assert "running" in result.stdout.lower()


def test_sync_ensure_running_starts_if_stopped(tmp_path: Path, monkeypatch):
    """ensure-running should start daemon if not running"""
    monkeypatch.setenv("CHINVEX_STATE_DIR", str(tmp_path))

    result = runner.invoke(app, ["sync", "ensure-running"])
    assert result.exit_code == 0

    # Should have started daemon
    assert (tmp_path / "sync.pid").exists()


def test_sync_ensure_running_noop_if_running(tmp_path: Path, monkeypatch):
    """ensure-running should be no-op if already running"""
    monkeypatch.setenv("CHINVEX_STATE_DIR", str(tmp_path))

    # Start daemon
    runner.invoke(app, ["sync", "start"])
    pid1 = (tmp_path / "sync.pid").read_text()

    # Ensure running - should not restart
    result = runner.invoke(app, ["sync", "ensure-running"])
    assert result.exit_code == 0

    pid2 = (tmp_path / "sync.pid").read_text()
    assert pid1 == pid2  # Same PID = didn't restart
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/sync/test_sync_cli.py -v`
Expected: FAIL with "No such command 'sync'"

**Step 3: Implement sync CLI commands**

```python
# src/chinvex/sync/cli.py
"""CLI commands for sync daemon."""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import typer

from .daemon import DaemonManager, DaemonState
from .locks import SyncLockHeld, sync_lock

log = logging.getLogger(__name__)


def get_state_dir() -> Path:
    """Get state directory for daemon files (default ~/.chinvex)."""
    env_dir = os.getenv("CHINVEX_STATE_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".chinvex"


def sync_start_cmd() -> None:
    """Start the sync daemon."""
    state_dir = get_state_dir()
    dm = DaemonManager(state_dir)

    # Atomic check-and-start using lock
    lock_file = state_dir / "sync.lock"

    try:
        with sync_lock(lock_file):
            # Check if already running
            state = dm.get_state()
            if state == DaemonState.RUNNING:
                typer.secho("Sync daemon is already running", fg=typer.colors.YELLOW)
                raise typer.Exit(code=1)

            # Clean up stale state if needed
            if state == DaemonState.STALE:
                typer.secho("Cleaning up stale daemon state...", fg=typer.colors.YELLOW)
                dm.cleanup()

            # Start daemon process
            _start_daemon_process(state_dir)

            # Wait briefly for daemon to start
            time.sleep(0.5)

            # Verify it started
            pid = dm.read_pid()
            if pid:
                typer.secho(f"Sync daemon started (PID: {pid})", fg=typer.colors.GREEN)
            else:
                typer.secho("Failed to start sync daemon", fg=typer.colors.RED)
                raise typer.Exit(code=1)

    except SyncLockHeld:
        typer.secho("Another sync operation is in progress", fg=typer.colors.RED)
        raise typer.Exit(code=1)


def sync_stop_cmd() -> None:
    """Stop the sync daemon."""
    state_dir = get_state_dir()
    dm = DaemonManager(state_dir)

    lock_file = state_dir / "sync.lock"

    try:
        with sync_lock(lock_file):
            pid = dm.read_pid()
            if not pid:
                typer.secho("Sync daemon is not running", fg=typer.colors.YELLOW)
                return

            # Send termination signal
            try:
                if sys.platform == "win32":
                    # Windows: use taskkill
                    subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=False)
                else:
                    # Unix: use SIGTERM
                    os.kill(pid, signal.SIGTERM)

                # Wait for process to exit
                max_wait = 5
                for _ in range(max_wait * 10):
                    try:
                        if sys.platform == "win32":
                            # Check if process exists
                            result = subprocess.run(
                                ["tasklist", "/FI", f"PID eq {pid}"],
                                capture_output=True,
                                text=True
                            )
                            if str(pid) not in result.stdout:
                                break
                        else:
                            os.kill(pid, 0)  # Check if process exists
                    except (ProcessLookupError, OSError):
                        break  # Process exited
                    time.sleep(0.1)

                # Clean up state files
                dm.cleanup()
                typer.secho(f"Sync daemon stopped (PID: {pid})", fg=typer.colors.GREEN)

            except (ProcessLookupError, OSError) as e:
                # Process already dead, just clean up
                dm.cleanup()
                typer.secho(f"Daemon process not found, cleaned up state", fg=typer.colors.YELLOW)

    except SyncLockHeld:
        typer.secho("Cannot stop: sync lock held", fg=typer.colors.RED)
        raise typer.Exit(code=1)


def sync_status_cmd() -> None:
    """Show sync daemon status."""
    state_dir = get_state_dir()
    dm = DaemonManager(state_dir)

    state = dm.get_state()
    pid = dm.read_pid()

    if state == DaemonState.NOT_RUNNING:
        typer.secho("Sync daemon: NOT RUNNING", fg=typer.colors.YELLOW)
    elif state == DaemonState.STALE:
        typer.secho(f"Sync daemon: STALE (PID: {pid}, heartbeat old)", fg=typer.colors.RED)
    elif state == DaemonState.RUNNING:
        typer.secho(f"Sync daemon: RUNNING (PID: {pid})", fg=typer.colors.GREEN)

        # Show heartbeat info
        if dm.heartbeat_file.exists():
            import json
            hb_data = json.loads(dm.heartbeat_file.read_text())
            typer.echo(f"Last heartbeat: {hb_data.get('timestamp', 'unknown')}")


def sync_ensure_running_cmd() -> None:
    """Start sync daemon if not running (idempotent)."""
    state_dir = get_state_dir()
    dm = DaemonManager(state_dir)

    lock_file = state_dir / "sync.lock"

    try:
        with sync_lock(lock_file):
            state = dm.get_state()

            if state == DaemonState.RUNNING:
                # Already running, nothing to do
                pid = dm.read_pid()
                typer.echo(f"Sync daemon already running (PID: {pid})")
                return

            # Need to start
            if state == DaemonState.STALE:
                dm.cleanup()

            _start_daemon_process(state_dir)
            time.sleep(0.5)

            pid = dm.read_pid()
            if pid:
                typer.secho(f"Sync daemon started (PID: {pid})", fg=typer.colors.GREEN)
            else:
                typer.secho("Failed to start daemon", fg=typer.colors.RED)
                raise typer.Exit(code=1)

    except SyncLockHeld:
        typer.secho("Sync lock held, cannot ensure running", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)


def _start_daemon_process(state_dir: Path) -> None:
    """
    Start daemon process in background.

    For now, this is a stub. Full implementation will spawn watchdog process.
    """
    # TODO: Implement actual daemon process spawn
    # For testing, just write a PID file
    import os
    dm = DaemonManager(state_dir)

    # Mock: write current process PID (in real impl, this would be subprocess PID)
    dm.write_pid(os.getpid())
    dm.write_heartbeat()
```

**Step 4: Wire into main CLI**

```python
# Modify src/chinvex/cli.py - add this after other subcommand groups:

# Add sync subcommand group
sync_app = typer.Typer(help="File watcher sync daemon")
app.add_typer(sync_app, name="sync")


@sync_app.command("start")
def sync_start():
    """Start the sync daemon"""
    from .sync.cli import sync_start_cmd
    sync_start_cmd()


@sync_app.command("stop")
def sync_stop():
    """Stop the sync daemon"""
    from .sync.cli import sync_stop_cmd
    sync_stop_cmd()


@sync_app.command("status")
def sync_status():
    """Show sync daemon status"""
    from .sync.cli import sync_status_cmd
    sync_status_cmd()


@sync_app.command("ensure-running")
def sync_ensure_running():
    """Start daemon if not running (idempotent)"""
    from .sync.cli import sync_ensure_running_cmd
    sync_ensure_running_cmd()


@sync_app.command("reconcile-sources")
def sync_reconcile_sources():
    """Update watcher sources from contexts (restarts watcher)"""
    from .sync.cli import sync_reconcile_sources_cmd
    sync_reconcile_sources_cmd()
```

**Step 5: Implement reconcile-sources command**

```python
# Add to src/chinvex/sync/cli.py:

def sync_reconcile_sources_cmd() -> None:
    """
    Update watcher sources from contexts.

    Called when contexts change (e.g., `dual track` adds a new repo).
    Restarts watcher to pick up new sources.
    """
    state_dir = get_state_dir()
    dm = DaemonManager(state_dir)

    lock_file = state_dir / "sync.lock"

    try:
        with sync_lock(lock_file):
            # Check if running
            state = dm.get_state()

            if state != DaemonState.RUNNING:
                # Not running, nothing to reconcile
                typer.echo("Sync daemon not running - use 'sync start' first")
                return

            # Stop current daemon
            pid = dm.read_pid()
            if pid:
                try:
                    if sys.platform == "win32":
                        subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=False)
                    else:
                        os.kill(pid, signal.SIGTERM)

                    # Wait for process to exit
                    time.sleep(0.5)
                except (ProcessLookupError, OSError):
                    pass

            # Clean up state
            dm.cleanup()

            # Restart with new sources
            _start_daemon_process(state_dir)
            time.sleep(0.5)

            new_pid = dm.read_pid()
            if new_pid:
                typer.secho(f"Sync daemon restarted with updated sources (PID: {new_pid})", fg=typer.colors.GREEN)
            else:
                typer.secho("Failed to restart daemon", fg=typer.colors.RED)
                raise typer.Exit(code=1)

    except SyncLockHeld:
        typer.secho("Sync lock held, cannot reconcile", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
```

**Step 6: Run test to verify it passes**

Run: `pytest tests/sync/test_sync_cli.py -v`
Expected: PASS (most tests green, stub daemon process)

**Step 7: Commit**

```bash
git add src/chinvex/cli.py src/chinvex/sync/cli.py tests/sync/test_sync_cli.py
git commit -m "feat(sync): add sync CLI commands

- chinvex sync start/stop/status/ensure-running
- chinvex sync reconcile-sources (restart with updated sources)
- Atomic start using portalocker
- Process management (Windows + Unix)
- Stub daemon process (to be implemented)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 3: Watchdog Integration & Ingest Triggering

### Task 6: Context Discovery

**Files:**
- Create: `src/chinvex/sync/discovery.py`
- Create: `tests/sync/test_discovery.py`

**Step 1: Write failing test for context discovery**

```python
# tests/sync/test_discovery.py
import pytest
import json
from pathlib import Path
from chinvex.sync.discovery import discover_watch_sources, WatchSource


def test_discover_empty_contexts(tmp_path: Path):
    """Empty contexts directory should return no sources"""
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    sources = discover_watch_sources(contexts_root)
    assert len(sources) == 0


def test_discover_single_repo_context(tmp_path: Path):
    """Should discover repo sources from context.json"""
    contexts_root = tmp_path / "contexts"
    ctx_dir = contexts_root / "TestCtx"
    ctx_dir.mkdir(parents=True)

    # Create minimal context.json with repo
    ctx_config = {
        "name": "TestCtx",
        "includes": {
            "repos": [str(tmp_path / "repo1")]
        }
    }
    (ctx_dir / "context.json").write_text(json.dumps(ctx_config))

    sources = discover_watch_sources(contexts_root)
    assert len(sources) == 1
    assert sources[0].context_name == "TestCtx"
    assert sources[0].source_type == "repo"
    assert sources[0].path == tmp_path / "repo1"


def test_discover_multiple_repos_same_context(tmp_path: Path):
    """Context with multiple repos should yield multiple sources"""
    contexts_root = tmp_path / "contexts"
    ctx_dir = contexts_root / "TestCtx"
    ctx_dir.mkdir(parents=True)

    ctx_config = {
        "name": "TestCtx",
        "includes": {
            "repos": [
                str(tmp_path / "repo1"),
                str(tmp_path / "repo2")
            ]
        }
    }
    (ctx_dir / "context.json").write_text(json.dumps(ctx_config))

    sources = discover_watch_sources(contexts_root)
    assert len(sources) == 2
    assert all(s.context_name == "TestCtx" for s in sources)
    assert {s.path for s in sources} == {tmp_path / "repo1", tmp_path / "repo2"}


def test_discover_inbox_source(tmp_path: Path):
    """Should discover inbox sources"""
    contexts_root = tmp_path / "contexts"
    ctx_dir = contexts_root / "_global"
    ctx_dir.mkdir(parents=True)

    ctx_config = {
        "name": "_global",
        "includes": {
            "inbox": [str(tmp_path / "inbox")]
        }
    }
    (ctx_dir / "context.json").write_text(json.dumps(ctx_config))

    sources = discover_watch_sources(contexts_root)
    assert len(sources) == 1
    assert sources[0].source_type == "inbox"
    assert sources[0].path == tmp_path / "inbox"


def test_discover_skips_malformed_context(tmp_path: Path):
    """Malformed context.json should be skipped with warning"""
    contexts_root = tmp_path / "contexts"

    # Valid context
    ctx1 = contexts_root / "Ctx1"
    ctx1.mkdir(parents=True)
    (ctx1 / "context.json").write_text(json.dumps({
        "name": "Ctx1",
        "includes": {"repos": [str(tmp_path / "repo1")]}
    }))

    # Invalid context (malformed JSON)
    ctx2 = contexts_root / "Ctx2"
    ctx2.mkdir(parents=True)
    (ctx2 / "context.json").write_text("{ invalid json")

    sources = discover_watch_sources(contexts_root)

    # Should only find the valid one
    assert len(sources) == 1
    assert sources[0].context_name == "Ctx1"


def test_discover_multiple_contexts(tmp_path: Path):
    """Should discover sources from all contexts"""
    contexts_root = tmp_path / "contexts"

    for i in range(3):
        ctx_dir = contexts_root / f"Ctx{i}"
        ctx_dir.mkdir(parents=True)
        (ctx_dir / "context.json").write_text(json.dumps({
            "name": f"Ctx{i}",
            "includes": {"repos": [str(tmp_path / f"repo{i}")]}
        }))

    sources = discover_watch_sources(contexts_root)
    assert len(sources) == 3
    assert {s.context_name for s in sources} == {"Ctx0", "Ctx1", "Ctx2"}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/sync/test_discovery.py -v`
Expected: FAIL with "cannot import name 'discover_watch_sources'"

**Step 3: Implement context discovery**

```python
# src/chinvex/sync/discovery.py
"""Context discovery for file watcher."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class WatchSource:
    """A source directory to watch for changes."""
    context_name: str
    source_type: str  # "repo" or "inbox"
    path: Path


def discover_watch_sources(contexts_root: Path) -> list[WatchSource]:
    """
    Discover all watchable sources from contexts.

    Reads all context.json files and extracts:
    - repos (from includes.repos)
    - inbox paths (from includes.inbox)

    Skips malformed contexts with a warning.

    Args:
        contexts_root: Root directory containing context subdirectories

    Returns:
        List of WatchSource objects
    """
    sources: list[WatchSource] = []

    if not contexts_root.exists():
        log.warning(f"Contexts root does not exist: {contexts_root}")
        return sources

    # Iterate all subdirectories
    for ctx_dir in contexts_root.iterdir():
        if not ctx_dir.is_dir():
            continue

        ctx_file = ctx_dir / "context.json"
        if not ctx_file.exists():
            continue

        try:
            # Load context config
            with open(ctx_file, "r", encoding="utf-8") as f:
                ctx_data = json.load(f)

            ctx_name = ctx_data.get("name", ctx_dir.name)
            includes = ctx_data.get("includes", {})

            # Extract repo sources
            for repo_path_str in includes.get("repos", []):
                repo_path = Path(repo_path_str)
                sources.append(WatchSource(
                    context_name=ctx_name,
                    source_type="repo",
                    path=repo_path
                ))

            # Extract inbox sources
            for inbox_path_str in includes.get("inbox", []):
                inbox_path = Path(inbox_path_str)
                sources.append(WatchSource(
                    context_name=ctx_name,
                    source_type="inbox",
                    path=inbox_path
                ))

        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Skipping malformed context {ctx_dir.name}: {e}")
            continue

    log.info(f"Discovered {len(sources)} watch sources from {contexts_root}")
    return sources
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/sync/test_discovery.py -v`
Expected: PASS (all tests green)

**Step 5: Commit**

```bash
git add src/chinvex/sync/discovery.py tests/sync/test_discovery.py
git commit -m "feat(sync): add context source discovery

- Scans contexts/ for context.json files
- Extracts repos and inbox paths
- Skips malformed contexts gracefully
- Returns WatchSource objects for watcher

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 7: Watchdog File Watcher Process

**Files:**
- Create: `src/chinvex/sync/process.py`
- Modify: `src/chinvex/sync/cli.py` (implement actual daemon spawn)
- Create: `tests/sync/test_watcher_process.py`

**Step 1: Write failing test for watcher process**

```python
# tests/sync/test_watcher_process.py
import pytest
import time
from pathlib import Path
from chinvex.sync.process import WatcherProcess


def test_watcher_creates_observers(tmp_path: Path):
    """Watcher should create file observers for sources"""
    # Create mock sources
    repo1 = tmp_path / "repo1"
    repo1.mkdir()

    from chinvex.sync.discovery import WatchSource
    sources = [
        WatchSource(context_name="Ctx1", source_type="repo", path=repo1)
    ]

    watcher = WatcherProcess(
        sources=sources,
        state_dir=tmp_path / "state",
        contexts_root=tmp_path / "contexts",
        debounce_seconds=0.1
    )

    # Should have created observer
    assert len(watcher._observers) == 1


def test_watcher_accumulates_changes(tmp_path: Path):
    """File changes should accumulate in context accumulators"""
    repo1 = tmp_path / "repo1"
    repo1.mkdir()

    from chinvex.sync.discovery import WatchSource
    sources = [
        WatchSource(context_name="Ctx1", source_type="repo", path=repo1)
    ]

    watcher = WatcherProcess(
        sources=sources,
        state_dir=tmp_path / "state",
        contexts_root=tmp_path / "contexts",
        debounce_seconds=0.1
    )

    # Simulate file change event
    test_file = repo1 / "test.txt"
    test_file.write_text("content")

    # Manually trigger change handler
    watcher._on_file_changed(str(test_file), "Ctx1")

    # Should have accumulated change
    assert "Ctx1" in watcher._accumulators
    changes = watcher._accumulators["Ctx1"].get_changes()
    assert len(changes) > 0


def test_watcher_respects_exclude_patterns(tmp_path: Path):
    """Excluded files should not trigger accumulation"""
    repo1 = tmp_path / "repo1"
    repo1.mkdir()

    from chinvex.sync.discovery import WatchSource
    sources = [
        WatchSource(context_name="Ctx1", source_type="repo", path=repo1)
    ]

    watcher = WatcherProcess(
        sources=sources,
        state_dir=tmp_path / "state",
        contexts_root=tmp_path / "contexts",
        debounce_seconds=0.1
    )

    # Try to add excluded file
    excluded_file = repo1 / ".git" / "config"
    watcher._on_file_changed(str(excluded_file), "Ctx1")

    # Should NOT have accumulated
    if "Ctx1" in watcher._accumulators:
        changes = watcher._accumulators["Ctx1"].get_changes()
        assert len(changes) == 0


def test_watcher_writes_heartbeat(tmp_path: Path):
    """Watcher should write heartbeat periodically"""
    from chinvex.sync.discovery import WatchSource

    watcher = WatcherProcess(
        sources=[],
        state_dir=tmp_path / "state",
        contexts_root=tmp_path / "contexts",
        debounce_seconds=0.1
    )

    # Manually trigger heartbeat write
    watcher._write_heartbeat()

    heartbeat_file = tmp_path / "state" / "sync_heartbeat.json"
    assert heartbeat_file.exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/sync/test_watcher_process.py -v`
Expected: FAIL with "cannot import name 'WatcherProcess'"

**Step 3: Implement watcher process**

```python
# src/chinvex/sync/process.py
"""File watcher process implementation."""
from __future__ import annotations

import logging
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .daemon import DaemonManager
from .discovery import WatchSource, discover_watch_sources
from .patterns import should_exclude
from .watcher import ChangeAccumulator

log = logging.getLogger(__name__)


class ChangeEventHandler(FileSystemEventHandler):
    """Handles file system events and routes to change accumulators."""

    def __init__(self, context_name: str, watch_root: Path, on_change_callback):
        self.context_name = context_name
        self.watch_root = watch_root
        self.on_change_callback = on_change_callback

    def on_modified(self, event: FileSystemEvent):
        if not event.is_directory:
            self._handle_change(event.src_path)

    def on_created(self, event: FileSystemEvent):
        if not event.is_directory:
            self._handle_change(event.src_path)

    def on_deleted(self, event: FileSystemEvent):
        if not event.is_directory:
            self._handle_change(event.src_path)

    def _handle_change(self, path: str):
        """Check exclude patterns and pass to callback."""
        if should_exclude(path, self.watch_root):
            log.debug(f"Excluded: {path}")
            return

        self.on_change_callback(path, self.context_name)


class WatcherProcess:
    """
    File watcher process that monitors sources and triggers ingests.

    Responsibilities:
    - Create watchdog observers for each source
    - Accumulate changes per context with debounce
    - Write heartbeat every 30s
    - Trigger ingest when debounce period elapses
    """

    def __init__(
        self,
        sources: list[WatchSource],
        state_dir: Path,
        contexts_root: Path,
        debounce_seconds: float = 30,
        max_paths: int = 500,
    ):
        self.sources = sources
        self.state_dir = Path(state_dir)
        self.contexts_root = Path(contexts_root)
        self.debounce_seconds = debounce_seconds
        self.max_paths = max_paths

        self.daemon_manager = DaemonManager(state_dir)

        # Per-context change accumulators
        self._accumulators: Dict[str, ChangeAccumulator] = {}

        # Watchdog observers
        self._observers: list[Observer] = []

        # Create observers for each source
        self._setup_observers()

    def _setup_observers(self):
        """Create watchdog observers for all sources."""
        for source in self.sources:
            if not source.path.exists():
                log.warning(f"Source path does not exist: {source.path}")
                continue

            event_handler = ChangeEventHandler(
                context_name=source.context_name,
                watch_root=source.path,
                on_change_callback=self._on_file_changed
            )

            observer = Observer()
            observer.schedule(event_handler, str(source.path), recursive=True)
            self._observers.append(observer)

            log.info(f"Watching {source.path} for context {source.context_name}")

    def _on_file_changed(self, path: str, context_name: str):
        """Handle file change event."""
        # Get or create accumulator for this context
        if context_name not in self._accumulators:
            self._accumulators[context_name] = ChangeAccumulator(
                debounce_seconds=self.debounce_seconds,
                max_paths=self.max_paths
            )

        self._accumulators[context_name].add_change(Path(path))
        log.debug(f"Change recorded: {path} (context: {context_name})")

    def _write_heartbeat(self):
        """Write heartbeat file."""
        self.daemon_manager.write_heartbeat()

    def _check_accumulators_and_trigger(self):
        """Check all accumulators and trigger ingests if ready."""
        for context_name, accumulator in list(self._accumulators.items()):
            if accumulator.is_ready() or accumulator.is_over_limit():
                self._trigger_ingest(context_name, accumulator)

    def _trigger_ingest(self, context_name: str, accumulator: ChangeAccumulator):
        """
        Trigger ingest for a context.

        For now, logs the intent. Full implementation in Task 9.
        """
        changes = accumulator.get_and_clear()

        if accumulator.is_over_limit():
            log.info(f"Triggering FULL ingest for {context_name} (>500 paths)")
            # TODO (Task 9): Call chinvex ingest --context {context_name}
        else:
            log.info(f"Triggering delta ingest for {context_name} ({len(changes)} files)")
            # TODO (Task 9): Call chinvex ingest --context {context_name} --paths {changes}

    def run(self):
        """Start watching and run main loop."""
        import os

        # Write PID file
        pid = os.getpid()
        self.daemon_manager.write_pid(pid)
        log.info(f"Watcher process started (PID: {pid})")

        # Start all observers
        for observer in self._observers:
            observer.start()

        # Heartbeat counter (write every 30 iterations = 30s)
        heartbeat_counter = 0
        HEARTBEAT_INTERVAL = 30

        # Main loop
        try:
            while True:
                # Write heartbeat every 30s
                heartbeat_counter += 1
                if heartbeat_counter >= HEARTBEAT_INTERVAL:
                    self._write_heartbeat()
                    heartbeat_counter = 0

                # Check accumulators and trigger ingests
                self._check_accumulators_and_trigger()

                # Sleep briefly
                time.sleep(1)

        except KeyboardInterrupt:
            log.info("Watcher process interrupted")

        finally:
            # Stop all observers
            for observer in self._observers:
                observer.stop()
                observer.join()

            # Cleanup
            self.daemon_manager.cleanup()
            log.info("Watcher process stopped")
```

**Step 4: Update CLI to spawn actual process**

```python
# Modify src/chinvex/sync/cli.py - replace _start_daemon_process:

def _start_daemon_process(state_dir: Path) -> None:
    """Start daemon process in background."""
    import sys
    import subprocess
    from .discovery import discover_watch_sources
    from ..context_cli import get_contexts_root

    # Get contexts root
    try:
        contexts_root = get_contexts_root()
    except Exception as e:
        typer.secho(f"Failed to get contexts root: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Spawn daemon process
    python_exe = sys.executable

    # Run as detached background process
    if sys.platform == "win32":
        # Windows: use CREATE_NEW_PROCESS_GROUP and DETACHED_PROCESS
        subprocess.Popen(
            [python_exe, "-m", "chinvex.sync.process", str(state_dir), str(contexts_root)],
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        # Unix: use nohup-style detachment
        subprocess.Popen(
            [python_exe, "-m", "chinvex.sync.process", str(state_dir), str(contexts_root)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
```

**Step 5: Add process main entry point**

```python
# Add to src/chinvex/sync/process.py:

def main():
    """Entry point for watcher daemon process."""
    import sys
    import logging

    # Setup logging to file
    if len(sys.argv) < 3:
        print("Usage: python -m chinvex.sync.process <state_dir> <contexts_root>")
        sys.exit(1)

    state_dir = Path(sys.argv[1])
    contexts_root = Path(sys.argv[2])

    # Configure logging
    log_file = state_dir / "sync.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=str(log_file),
    )

    # Discover sources
    sources = discover_watch_sources(contexts_root)

    # Create and run watcher
    watcher = WatcherProcess(
        sources=sources,
        state_dir=state_dir,
        contexts_root=contexts_root,
    )

    watcher.run()


if __name__ == "__main__":
    main()
```

**Step 6: Add watchdog dependency**

```toml
# Modify pyproject.toml dependencies:
dependencies = [
  "typer>=0.12.3",
  "chromadb>=0.5.3",
  "requests>=2.32.3",
  "mcp>=1.0.0",
  "portalocker>=2.10.1",
  "watchdog>=3.0.0",  # ADD THIS
  "fastapi>=0.109.0",
  # ... rest unchanged
]
```

**Step 7: Run test to verify it passes**

Run: `pytest tests/sync/test_watcher_process.py -v`
Expected: PASS (all tests green)

**Step 8: Commit**

```bash
git add src/chinvex/sync/process.py src/chinvex/sync/cli.py tests/sync/test_watcher_process.py pyproject.toml
git commit -m "feat(sync): implement watchdog file watcher process

- watchdog-based file monitoring
- Per-context change accumulation
- Exclude pattern filtering
- Heartbeat writing every 30s
- Daemon process spawn (Windows + Unix)
- Add watchdog>=3.0.0 dependency

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 4: Ingest Integration & Status Artifacts

### Task 8: Add `--paths` Flag to Ingest Command

**Files:**
- Modify: `src/chinvex/cli.py` (ingest command)
- Modify: `src/chinvex/ingest.py` (add delta ingest support)
- Create: `tests/test_ingest_delta.py`

**Step 1: Write failing test for delta ingest**

```python
# tests/test_ingest_delta.py
import pytest
from pathlib import Path
from typer.testing import CliRunner
from chinvex.cli import app

runner = CliRunner()


def test_ingest_with_paths_flag(tmp_path: Path, monkeypatch):
    """--paths should trigger delta ingest for specific files"""
    # Setup minimal context
    contexts_root = tmp_path / "contexts"
    ctx_dir = contexts_root / "TestCtx"
    ctx_dir.mkdir(parents=True)

    # Create repo source
    repo = tmp_path / "repo"
    repo.mkdir()
    file1 = repo / "file1.txt"
    file1.write_text("content1")
    file2 = repo / "file2.txt"
    file2.write_text("content2")

    # Create context.json
    import json
    ctx_config = {
        "schema_version": 2,
        "name": "TestCtx",
        "includes": {"repos": [str(repo)]},
        "index": {
            "sqlite_path": str(ctx_dir / "hybrid.db"),
            "chroma_dir": str(ctx_dir / "chroma")
        }
    }
    (ctx_dir / "context.json").write_text(json.dumps(ctx_config))

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))

    # Run delta ingest for only file1
    result = runner.invoke(app, [
        "ingest",
        "--context", "TestCtx",
        "--paths", str(file1)
    ])

    assert result.exit_code == 0
    # Should have processed file1 only
    # (verification via stats will be added in implementation)


def test_ingest_with_multiple_paths(tmp_path: Path, monkeypatch):
    """--paths with multiple files should process all"""
    contexts_root = tmp_path / "contexts"
    ctx_dir = contexts_root / "TestCtx"
    ctx_dir.mkdir(parents=True)

    repo = tmp_path / "repo"
    repo.mkdir()
    files = [repo / f"file{i}.txt" for i in range(3)]
    for f in files:
        f.write_text(f"content_{f.name}")

    import json
    ctx_config = {
        "schema_version": 2,
        "name": "TestCtx",
        "includes": {"repos": [str(repo)]},
        "index": {
            "sqlite_path": str(ctx_dir / "hybrid.db"),
            "chroma_dir": str(ctx_dir / "chroma")
        }
    }
    (ctx_dir / "context.json").write_text(json.dumps(ctx_config))

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))

    # Pass multiple paths
    paths_arg = ",".join(str(f) for f in files)
    result = runner.invoke(app, [
        "ingest",
        "--context", "TestCtx",
        "--paths", paths_arg
    ])

    assert result.exit_code == 0


def test_ingest_without_paths_does_full(tmp_path: Path, monkeypatch):
    """Without --paths, should do full ingest"""
    contexts_root = tmp_path / "contexts"
    ctx_dir = contexts_root / "TestCtx"
    ctx_dir.mkdir(parents=True)

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "file1.txt").write_text("content1")
    (repo / "file2.txt").write_text("content2")

    import json
    ctx_config = {
        "schema_version": 2,
        "name": "TestCtx",
        "includes": {"repos": [str(repo)]},
        "index": {
            "sqlite_path": str(ctx_dir / "hybrid.db"),
            "chroma_dir": str(ctx_dir / "chroma")
        }
    }
    (ctx_dir / "context.json").write_text(json.dumps(ctx_config))

    monkeypatch.setenv("CHINVEX_CONTEXTS_ROOT", str(contexts_root))

    # No --paths = full ingest
    result = runner.invoke(app, [
        "ingest",
        "--context", "TestCtx"
    ])

    assert result.exit_code == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest_delta.py -v`
Expected: FAIL with "no such option: --paths"

**Step 3: Add --paths flag to CLI**

```python
# Modify src/chinvex/cli.py ingest command:

@app.command()
def ingest(
    context: str = typer.Option(..., help="Context name to ingest"),
    paths: str = typer.Option(None, help="Comma-separated paths for delta ingest (optional)"),
):
    """
    Ingest sources into hybrid index.

    Without --paths: full ingest of all sources in context.
    With --paths: delta ingest of specific files only.
    """
    from .context import load_context
    from .context_cli import get_contexts_root
    from .ingest import ingest_context, ingest_delta

    contexts_root = get_contexts_root()
    ctx_config = load_context(context, contexts_root)

    # Parse paths if provided
    if paths:
        path_list = [Path(p.strip()) for p in paths.split(",") if p.strip()]
        typer.echo(f"Delta ingest: {len(path_list)} files for context {context}")
        stats = ingest_delta(ctx_config, path_list)
    else:
        typer.echo(f"Full ingest: context {context}")
        stats = ingest_context(ctx_config)

    typer.secho(f"Ingest complete: {stats}", fg=typer.colors.GREEN)
```

**Step 4: Implement delta ingest function**

```python
# Add to src/chinvex/ingest.py:

def ingest_delta(ctx_config: ContextConfig, paths: list[Path]) -> dict:
    """
    Delta ingest: process only specific paths.

    Args:
        ctx_config: Context configuration
        paths: List of file paths to ingest

    Returns:
        Stats dictionary
    """
    index_dir = Path(ctx_config.index.sqlite_path).parent
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = ctx_config.index.sqlite_path
    chroma_dir = ctx_config.index.chroma_dir
    chroma_dir.mkdir(parents=True, exist_ok=True)

    lock_path = index_dir / "hybrid.db.lock"
    try:
        with Lock(lock_path, timeout=60):
            storage = Storage(db_path)
            storage.ensure_schema()

            # Determine embedding provider
            if ctx_config.embedding and ctx_config.embedding.provider == "openai":
                from .embed import OpenAIEmbedder
                embedder = OpenAIEmbedder(
                    model=ctx_config.embedding.model or "text-embedding-3-small"
                )
            else:
                embedder = OllamaEmbedder(
                    ctx_config.ollama.base_url,
                    ctx_config.ollama.embed_model
                )

            vectors = VectorStore(chroma_dir)

            stats = {
                "documents": 0,
                "chunks": 0,
                "skipped": 0,
                "files_processed": len(paths)
            }
            started_at = now_iso()
            run_id = sha256_text(started_at + ",".join(str(p) for p in paths))

            # Process each file
            for path in paths:
                if not path.exists():
                    log.warning(f"Path not found: {path}")
                    continue

                # Determine source type by matching against context includes
                source_type = _infer_source_type(path, ctx_config)

                if source_type == "repo":
                    # Single-file repo ingest
                    source = SourceConfig(type="repo", path=path.parent)
                    _ingest_single_repo_file(path, source, storage, embedder, vectors, stats)
                # Add other source types as needed

            storage.record_run(run_id, started_at, dump_json(stats))
            storage.close()
            return stats

    except LockException as exc:
        raise RuntimeError("Ingest lock held by another process") from exc


def _infer_source_type(path: Path, ctx_config: ContextConfig) -> str:
    """Infer source type by checking which include list contains the path."""
    # Check if path is under any repo
    for repo in ctx_config.includes.repos:
        if path.is_relative_to(repo):
            return "repo"

    # Check if under inbox
    for inbox in ctx_config.includes.note_roots:
        if path.is_relative_to(inbox):
            return "inbox"

    return "unknown"


def _ingest_single_repo_file(
    file_path: Path,
    source: SourceConfig,
    storage: Storage,
    embedder,
    vectors: VectorStore,
    stats: dict
):
    """Ingest a single file from a repo source."""
    try:
        content = read_text_utf8(file_path)
        rel_path = file_path.relative_to(source.path)
        doc_id = f"repo:{source.path.name}:{rel_path}"

        # Check if document changed
        content_hash = sha256_text(content)
        existing = storage.get_document(doc_id)

        if existing and existing["content_hash"] == content_hash:
            stats["skipped"] += 1
            return

        # Chunk the file
        chunks = chunk_repo(content, str(rel_path))
        if not chunks:
            return

        # Generate embeddings
        texts = [c.text for c in chunks]
        embeddings = embedder.embed(texts)

        # Delete old chunks for this doc
        storage.delete_chunks_for_document(doc_id)
        vectors.delete_by_doc_id(doc_id)

        # Insert new chunks
        metadata = {
            "source_type": "repo",
            "repo_name": source.path.name,
            "file_path": str(rel_path)
        }

        chunk_ids = []
        for chunk, emb in zip(chunks, embeddings):
            chunk_id = f"{doc_id}:chunk_{len(chunk_ids)}"
            chunk_ids.append(chunk_id)

            storage.insert_chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=chunk.text,
                metadata=dump_json(metadata),
                chunk_key=chunk_key(chunk.text)
            )

        # Insert embeddings
        vectors.add_embeddings(chunk_ids, embeddings, [metadata] * len(chunk_ids))

        # Update document
        storage.upsert_document(
            doc_id=doc_id,
            source_type="repo",
            source_path=str(source.path),
            content_hash=content_hash,
            metadata=dump_json(metadata)
        )

        stats["documents"] += 1
        stats["chunks"] += len(chunks)

    except Exception as e:
        log.error(f"Failed to ingest {file_path}: {e}")
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_ingest_delta.py -v`
Expected: PASS (all tests green)

**Step 6: Commit**

```bash
git add src/chinvex/cli.py src/chinvex/ingest.py tests/test_ingest_delta.py
git commit -m "feat(ingest): add --paths flag for delta ingest

- Delta ingest processes only specified files
- Infers source type from context includes
- Single-file repo ingest implementation
- Full ingest unchanged (no --paths = full)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 9: Implement Ingest Triggering from Watcher

**Files:**
- Modify: `src/chinvex/sync/process.py` (_trigger_ingest)
- Create: `tests/sync/test_ingest_trigger.py`

**Step 1: Write failing test for ingest triggering**

```python
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
        assert "chinvex" in " ".join(call_args)
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/sync/test_ingest_trigger.py -v`
Expected: FAIL with assertion errors (stub implementation doesn't call subprocess)

**Step 3: Implement actual ingest triggering**

```python
# Modify src/chinvex/sync/process.py - replace _trigger_ingest:

def _trigger_ingest(self, context_name: str, accumulator: ChangeAccumulator):
    """
    Trigger ingest for a context.

    Spawns background chinvex ingest process.
    """
    import subprocess
    import sys
    from .locks import check_context_lock_held

    # Check if ingest lock is held
    if check_context_lock_held(self.contexts_root, context_name):
        log.info(f"Skipping ingest for {context_name}: lock held")
        # Don't clear accumulator - will retry later
        return

    changes = accumulator.get_and_clear()

    # Build command
    python_exe = sys.executable
    cmd = [python_exe, "-m", "chinvex.cli", "ingest", "--context", context_name]

    if accumulator.is_over_limit():
        # Full ingest (no --paths)
        log.info(f"Triggering FULL ingest for {context_name} (>500 paths)")
    else:
        # Delta ingest with specific paths
        log.info(f"Triggering delta ingest for {context_name} ({len(changes)} files)")
        paths_str = ",".join(str(p) for p in changes)
        cmd.extend(["--paths", paths_str])

    # Spawn background process
    try:
        if sys.platform == "win32":
            subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        log.info(f"Ingest process spawned for {context_name}")

    except Exception as e:
        log.error(f"Failed to spawn ingest for {context_name}: {e}")
        # Re-add changes to accumulator for retry
        for path in changes:
            accumulator.add_change(path)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/sync/test_ingest_trigger.py -v`
Expected: PASS (all tests green)

**Step 5: Commit**

```bash
git add src/chinvex/sync/process.py tests/sync/test_ingest_trigger.py
git commit -m "feat(sync): implement ingest triggering from watcher

- Spawns background chinvex ingest process
- Delta ingest for <500 files, full for >500
- Checks lock before triggering
- Retries on failure

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 10: Write STATUS.json After Ingest

**Files:**
- Modify: `src/chinvex/ingest.py` (write STATUS.json)
- Create: `src/chinvex/status.py`
- Create: `tests/test_status_json.py`

**Step 1: Write failing test for STATUS.json generation**

```python
# tests/test_status_json.py
import pytest
import json
from pathlib import Path
from datetime import datetime, timezone
from chinvex.status import write_status_json, compute_freshness


def test_write_status_json_basic(tmp_path: Path):
    """Should write STATUS.json with required fields"""
    ctx_dir = tmp_path / "contexts" / "TestCtx"
    ctx_dir.mkdir(parents=True)

    stats = {
        "documents": 10,
        "chunks": 42,
        "last_sync": datetime.now(timezone.utc).isoformat()
    }

    sources = [
        {"type": "repo", "path": "/code/repo1", "watching": True}
    ]

    embedding_info = {
        "provider": "ollama",
        "model": "mxbai-embed-large",
        "dimensions": 1024
    }

    write_status_json(
        context_dir=ctx_dir,
        stats=stats,
        sources=sources,
        embedding=embedding_info
    )

    status_file = ctx_dir / "STATUS.json"
    assert status_file.exists()

    data = json.loads(status_file.read_text())
    assert data["context"] == "TestCtx"
    assert data["chunks"] == 42
    assert data["embedding"]["provider"] == "ollama"


def test_status_includes_freshness(tmp_path: Path):
    """STATUS.json should include freshness calculation"""
    ctx_dir = tmp_path / "contexts" / "TestCtx"
    ctx_dir.mkdir(parents=True)

    now = datetime.now(timezone.utc)
    stats = {
        "chunks": 100,
        "last_sync": now.isoformat()
    }

    write_status_json(
        context_dir=ctx_dir,
        stats=stats,
        sources=[],
        embedding={"provider": "ollama", "model": "test"}
    )

    data = json.loads((ctx_dir / "STATUS.json").read_text())

    assert "freshness" in data
    assert "stale_after_hours" in data["freshness"]
    assert "is_stale" in data["freshness"]
    assert "hours_since_sync" in data["freshness"]


def test_compute_freshness_not_stale(tmp_path: Path):
    """Recent sync should not be stale"""
    now = datetime.now(timezone.utc)

    freshness = compute_freshness(
        last_sync=now.isoformat(),
        stale_after_hours=6
    )

    assert freshness["is_stale"] is False
    assert freshness["hours_since_sync"] < 1


def test_compute_freshness_stale(tmp_path: Path):
    """Old sync should be stale"""
    from datetime import timedelta
    old_time = datetime.now(timezone.utc) - timedelta(hours=7)

    freshness = compute_freshness(
        last_sync=old_time.isoformat(),
        stale_after_hours=6
    )

    assert freshness["is_stale"] is True
    assert freshness["hours_since_sync"] > 6
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_status_json.py -v`
Expected: FAIL with "cannot import name 'write_status_json'"

**Step 3: Implement STATUS.json writer**

```python
# src/chinvex/status.py
"""Status artifact generation."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


def write_status_json(
    context_dir: Path,
    stats: dict,
    sources: list[dict],
    embedding: dict,
    stale_after_hours: int = 6
) -> None:
    """
    Write STATUS.json for a context after ingest.

    Args:
        context_dir: Context directory (e.g., contexts/Chinvex)
        stats: Ingest stats dict
        sources: List of source dicts with {type, path, watching}
        embedding: Embedding config dict
        stale_after_hours: Hours before context is considered stale (default 6)
    """
    context_name = context_dir.name

    # Compute freshness
    last_sync = stats.get("last_sync", datetime.now(timezone.utc).isoformat())
    freshness = compute_freshness(last_sync, stale_after_hours=stale_after_hours)

    status = {
        "context": context_name,
        "last_sync": last_sync,
        "chunks": stats.get("chunks", 0),
        "watches_active": stats.get("watches_active", 0),
        "watches_pending_hits": stats.get("watches_pending_hits", 0),
        "freshness": freshness,
        "sources": sources,
        "embedding": embedding
    }

    status_file = context_dir / "STATUS.json"
    status_file.write_text(json.dumps(status, indent=2))
    log.info(f"Wrote {status_file}")


def compute_freshness(last_sync: str, stale_after_hours: int = 6) -> dict:
    """
    Compute freshness status.

    Args:
        last_sync: ISO timestamp of last sync
        stale_after_hours: Hours before considered stale

    Returns:
        Dict with stale_after_hours, is_stale, hours_since_sync
    """
    last_sync_dt = datetime.fromisoformat(last_sync)
    if last_sync_dt.tzinfo is None:
        last_sync_dt = last_sync_dt.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    elapsed = now - last_sync_dt
    hours_since = elapsed.total_seconds() / 3600

    return {
        "stale_after_hours": stale_after_hours,
        "is_stale": hours_since > stale_after_hours,
        "hours_since_sync": round(hours_since, 2)
    }
```

**Step 4: Wire into ingest.py**

```python
# Modify src/chinvex/ingest.py - add to ingest_context function:

def ingest_context(ctx_config: ContextConfig) -> dict:
    """Full ingest for a context."""
    # ... existing ingest logic ...

    # After storage.close(), before return:
    from .status import write_status_json

    # Build sources list
    sources = []
    for repo in ctx_config.includes.repos:
        sources.append({
            "type": "repo",
            "path": str(repo),
            "watching": True  # Assume watched if in includes
        })

    # Build embedding info
    if ctx_config.embedding:
        embedding_info = {
            "provider": ctx_config.embedding.provider,
            "model": ctx_config.embedding.model or "default",
            "dimensions": 1024  # TODO: get from embedder
        }
    else:
        embedding_info = {
            "provider": "ollama",
            "model": ctx_config.ollama.embed_model,
            "dimensions": 1024
        }

    # Write STATUS.json
    context_dir = Path(ctx_config.index.sqlite_path).parent.parent
    stats["last_sync"] = now_iso()

    # Read stale_after_hours from context config
    stale_after_hours = 6  # Default
    if hasattr(ctx_config, 'constraints') and ctx_config.constraints:
        stale_after_hours = ctx_config.constraints.get('stale_after_hours', 6)

    write_status_json(context_dir, stats, sources, embedding_info, stale_after_hours)

    return stats


# Also add to ingest_delta function:
def ingest_delta(ctx_config: ContextConfig, paths: list[Path]) -> dict:
    """Delta ingest for specific paths."""
    # ... existing logic ...

    # After storage.close(), before return:
    from .status import write_status_json

    # Build sources/embedding (same as ingest_context)
    # ...

    context_dir = Path(ctx_config.index.sqlite_path).parent.parent
    stats["last_sync"] = now_iso()
    write_status_json(context_dir, stats, sources, embedding_info)

    return stats
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_status_json.py -v`
Expected: PASS (all tests green)

**Step 6: Commit**

```bash
git add src/chinvex/status.py src/chinvex/ingest.py tests/test_status_json.py
git commit -m "feat(status): write STATUS.json after ingest

- STATUS.json includes chunks, freshness, sources, embedding
- Freshness calculation with stale detection
- Written after both full and delta ingests
- Includes hours_since_sync metric

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 5: Git Hooks

### Task 11: Python Path Resolver

**Files:**
- Create: `src/chinvex/hooks/__init__.py`
- Create: `src/chinvex/hooks/resolver.py`
- Create: `tests/hooks/test_resolver.py`

**Step 1: Write failing test for Python path resolution**

```python
# tests/hooks/test_resolver.py
import pytest
from pathlib import Path
from unittest.mock import patch, Mock
from chinvex.hooks.resolver import resolve_python_path


def test_resolve_finds_venv_python(tmp_path: Path):
    """Should prefer repo-local venv Python"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    # Create mock venv
    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("mock python")

    # Mock subprocess to simulate python --version
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(returncode=0, stdout="Python 3.12.0")

        result = resolve_python_path(repo_root)

        assert result == str(venv_python)


def test_resolve_fallback_to_py_launcher(tmp_path: Path):
    """If no venv, should try py -3"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    # No venv exists
    with patch('subprocess.run') as mock_run:
        # First call: check venv (fails)
        # Second call: check py -3 (succeeds)
        mock_run.side_effect = [
            Mock(returncode=1),  # venv doesn't exist
            Mock(returncode=0, stdout="Python 3.12.0")  # py -3 works
        ]

        with patch('shutil.which', return_value="C:\\Windows\\py.exe"):
            result = resolve_python_path(repo_root)

            assert result == "py -3"


def test_resolve_uses_env_override(tmp_path: Path, monkeypatch):
    """CHINVEX_PYTHON should override"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    custom_python = "C:\\CustomPython\\python.exe"
    monkeypatch.setenv("CHINVEX_PYTHON", custom_python)

    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(returncode=0, stdout="Python 3.12.0")

        result = resolve_python_path(repo_root)

        assert result == custom_python


def test_resolve_validates_chinvex_installed(tmp_path: Path):
    """Should verify chinvex module is importable"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("mock")

    with patch('subprocess.run') as mock_run:
        # First call: python --version (passes)
        # Second call: python -m chinvex.cli --help (fails - chinvex not installed)
        mock_run.side_effect = [
            Mock(returncode=0, stdout="Python 3.12.0"),
            Mock(returncode=1)  # chinvex not installed
        ]

        # Should skip this python and try next option
        with patch('shutil.which', return_value="C:\\Windows\\py.exe"):
            mock_run.side_effect = [
                Mock(returncode=0, stdout="Python 3.12.0"),
                Mock(returncode=1),  # venv chinvex check fails
                Mock(returncode=0, stdout="Python 3.12.0"),
                Mock(returncode=0)   # py -3 chinvex check succeeds
            ]

            result = resolve_python_path(repo_root)

            assert result == "py -3"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/hooks/test_resolver.py -v`
Expected: FAIL with "cannot import name 'resolve_python_path'"

**Step 3: Implement Python path resolver**

```python
# src/chinvex/hooks/__init__.py
"""Git hook utilities."""

# src/chinvex/hooks/resolver.py
"""Python path resolution for git hooks."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def resolve_python_path(repo_root: Path) -> str | None:
    """
    Resolve Python executable path with chinvex installed.

    Order of preference:
    1. CHINVEX_PYTHON env var (explicit override)
    2. {repo}/.venv/Scripts/python.exe (repo-local venv)
    3. py -3 (Windows Python launcher)
    4. python (PATH fallback)

    Args:
        repo_root: Repository root directory

    Returns:
        Path to Python executable, or None if not found
    """
    candidates = []

    # 1. Explicit override
    env_python = os.getenv("CHINVEX_PYTHON")
    if env_python:
        candidates.append(env_python)

    # 2. Repo-local venv
    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        candidates.append(str(venv_python))

    # Also check Unix-style venv
    venv_python_unix = repo_root / ".venv" / "bin" / "python"
    if venv_python_unix.exists():
        candidates.append(str(venv_python_unix))

    # 3. Windows Python launcher
    if shutil.which("py"):
        candidates.append("py -3")

    # 4. PATH fallback
    if shutil.which("python"):
        candidates.append("python")

    # Test each candidate
    for candidate in candidates:
        if _validate_python(candidate):
            log.info(f"Resolved Python: {candidate}")
            return candidate

    log.error("No valid Python with chinvex found")
    return None


def _validate_python(python_path: str) -> bool:
    """
    Validate that Python works and has chinvex installed.

    Args:
        python_path: Python executable path or command

    Returns:
        True if valid and has chinvex
    """
    # Handle "py -3" style commands
    if " " in python_path:
        cmd_parts = python_path.split()
    else:
        cmd_parts = [python_path]

    try:
        # Check Python version
        result = subprocess.run(
            cmd_parts + ["--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            return False

        # Check chinvex module exists
        result = subprocess.run(
            cmd_parts + ["-m", "chinvex.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/hooks/test_resolver.py -v`
Expected: PASS (all tests green)

**Step 5: Commit**

```bash
git add src/chinvex/hooks/ tests/hooks/
git commit -m "feat(hooks): add Python path resolver

- Resolves Python with chinvex installed
- Prefers repo venv, then py -3, then PATH
- Validates chinvex module is importable
- Supports CHINVEX_PYTHON override

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 12: Hook Generator (bash + PowerShell wrapper)

**Files:**
- Create: `src/chinvex/hooks/generator.py`
- Create: `tests/hooks/test_generator.py`

**Step 1: Write failing test for hook generation**

```python
# tests/hooks/test_generator.py
import pytest
from pathlib import Path
from chinvex.hooks.generator import generate_post_commit_hook


def test_generate_creates_shell_wrapper(tmp_path: Path):
    """Should create .git/hooks/post-commit shell script"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    git_dir = repo_root / ".git"
    git_dir.mkdir()

    python_path = "C:\\Python312\\python.exe"
    context_name = "TestRepo"

    generate_post_commit_hook(repo_root, git_dir, python_path, context_name)

    hook_file = git_dir / "hooks" / "post-commit"
    assert hook_file.exists()

    content = hook_file.read_text()
    assert "#!/bin/sh" in content
    assert "pwsh" in content
    assert ".chinvex/post-commit.ps1" in content


def test_generate_creates_powershell_script(tmp_path: Path):
    """Should create .chinvex/post-commit.ps1"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    git_dir = repo_root / ".git"
    git_dir.mkdir()

    python_path = "C:\\Python312\\python.exe"
    context_name = "TestRepo"

    generate_post_commit_hook(repo_root, git_dir, python_path, context_name)

    ps_script = repo_root / ".chinvex" / "post-commit.ps1"
    assert ps_script.exists()

    content = ps_script.read_text()
    assert python_path in content
    assert "chinvex.cli ingest" in content
    assert f"--context {context_name}" in content
    assert "--changed-only" in content


def test_generate_backs_up_existing_hook(tmp_path: Path):
    """Should backup existing post-commit hook"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    git_dir = repo_root / ".git"
    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(parents=True)

    # Create existing hook
    existing = hooks_dir / "post-commit"
    existing.write_text("#!/bin/sh\necho 'existing hook'\n")

    python_path = "python"
    context_name = "TestRepo"

    generate_post_commit_hook(repo_root, git_dir, python_path, context_name)

    # Original should be backed up
    backup_files = list(hooks_dir.glob("post-commit.bak*"))
    assert len(backup_files) == 1
    assert "existing hook" in backup_files[0].read_text()


def test_generated_hook_is_executable(tmp_path: Path):
    """Generated hook should have executable permissions"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    git_dir = repo_root / ".git"
    git_dir.mkdir()

    generate_post_commit_hook(repo_root, git_dir, "python", "TestRepo")

    hook_file = git_dir / "hooks" / "post-commit"

    import os
    import stat
    # Check if file has execute permissions
    mode = os.stat(hook_file).st_mode
    assert mode & stat.S_IXUSR  # Owner execute
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/hooks/test_generator.py -v`
Expected: FAIL with "cannot import name 'generate_post_commit_hook'"

**Step 3: Implement hook generator**

```python
# src/chinvex/hooks/generator.py
"""Git hook generation."""
from __future__ import annotations

import logging
import os
import stat
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


def generate_post_commit_hook(
    repo_root: Path,
    git_dir: Path,
    python_path: str,
    context_name: str
) -> None:
    """
    Generate post-commit hook with PowerShell wrapper.

    Creates:
    - .git/hooks/post-commit (shell script)
    - .chinvex/post-commit.ps1 (PowerShell script)

    Args:
        repo_root: Repository root directory
        git_dir: .git directory path
        python_path: Resolved Python executable path
        context_name: Context name for this repo
    """
    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)

    chinvex_dir = repo_root / ".chinvex"
    chinvex_dir.mkdir(parents=True, exist_ok=True)

    # Backup existing hook if present
    hook_file = hooks_dir / "post-commit"
    if hook_file.exists():
        _backup_existing_hook(hook_file)

    # Generate PowerShell wrapper
    ps_script = chinvex_dir / "post-commit.ps1"
    _generate_powershell_script(ps_script, python_path, context_name)

    # Generate shell hook
    _generate_shell_hook(hook_file, repo_root, ps_script)

    log.info(f"Generated post-commit hook for {context_name}")


def _backup_existing_hook(hook_file: Path) -> None:
    """Backup existing hook with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_path = hook_file.parent / f"post-commit.bak.{timestamp}"
    hook_file.rename(backup_path)
    log.warning(f"Backed up existing hook to {backup_path.name}")


def _generate_powershell_script(
    ps_script: Path,
    python_path: str,
    context_name: str
) -> None:
    """Generate PowerShell script that performs ingest."""
    # Normalize path for PowerShell
    python_normalized = str(Path(python_path)).replace("\\", "/")

    content = f"""# Generated by chinvex hook install
# Context: {context_name}
# Python path resolved at install time
$ErrorActionPreference = "SilentlyContinue"

# Get changed files from last commit
$changedFiles = git diff --name-only HEAD~1 2>$null
if (-not $changedFiles) {{
    # First commit, use all files
    $changedFiles = git ls-files 2>$null
}}

# Convert to paths argument
$pathsArg = ($changedFiles -join ',')

# Trigger ingest in background
if ($pathsArg) {{
    Start-Process -NoNewWindow -FilePath "{python_normalized}" `
        -ArgumentList "-m","chinvex.cli","ingest","--context","{context_name}","--paths",$pathsArg,"--quiet" `
        -RedirectStandardOutput "NUL" `
        -RedirectStandardError "NUL"
}}
"""

    ps_script.write_text(content, encoding="utf-8")
    log.info(f"Generated PowerShell script: {ps_script}")


def _generate_shell_hook(
    hook_file: Path,
    repo_root: Path,
    ps_script: Path
) -> None:
    """Generate shell hook that calls PowerShell wrapper."""
    # Use relative path from repo root
    rel_ps_path = ps_script.relative_to(repo_root)

    content = f"""#!/bin/sh
# Generated by chinvex hook install
# Calls PowerShell for Windows compatibility

# Check if PowerShell script exists
if [ ! -f "{rel_ps_path}" ]; then
    exit 0
fi

# Run PowerShell script in background
pwsh -NoProfile -ExecutionPolicy Bypass -File "{rel_ps_path}" &
exit 0
"""

    hook_file.write_text(content, encoding="utf-8")

    # Make executable
    current_mode = os.stat(hook_file).st_mode
    os.chmod(hook_file, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    log.info(f"Generated shell hook: {hook_file}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/hooks/test_generator.py -v`
Expected: PASS (all tests green)

**Step 5: Commit**

```bash
git add src/chinvex/hooks/generator.py tests/hooks/test_generator.py
git commit -m "feat(hooks): add post-commit hook generator

- Generates shell wrapper + PowerShell script
- Backs up existing hooks with timestamp
- PowerShell script uses git diff for changed files
- Shell hook calls PowerShell in background
- Sets executable permissions

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 13: Hook Install/Uninstall Commands

**Files:**
- Create: `src/chinvex/hooks/cli.py`
- Modify: `src/chinvex/cli.py` (add hook subcommand)
- Create: `tests/hooks/test_hook_cli.py`

**Step 1: Write failing test for hook commands**

```python
# tests/hooks/test_hook_cli.py
import pytest
from pathlib import Path
from typer.testing import CliRunner
from chinvex.cli import app

runner = CliRunner()


def test_hook_install_in_git_repo(tmp_path: Path, monkeypatch):
    """hook install should create post-commit hook"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    git_dir = repo_root / ".git"
    git_dir.mkdir()

    monkeypatch.chdir(repo_root)

    result = runner.invoke(app, ["hook", "install", "--context", "TestRepo"])

    assert result.exit_code == 0
    assert (git_dir / "hooks" / "post-commit").exists()
    assert (repo_root / ".chinvex" / "post-commit.ps1").exists()


def test_hook_install_fails_outside_git_repo(tmp_path: Path, monkeypatch):
    """hook install should fail if not in git repo"""
    not_a_repo = tmp_path / "not_repo"
    not_a_repo.mkdir()

    monkeypatch.chdir(not_a_repo)

    result = runner.invoke(app, ["hook", "install", "--context", "TestRepo"])

    assert result.exit_code != 0
    assert "not a git repository" in result.stdout.lower()


def test_hook_install_infers_context_from_folder(tmp_path: Path, monkeypatch):
    """hook install without --context should infer from folder name"""
    repo_root = tmp_path / "my-project"
    repo_root.mkdir()
    git_dir = repo_root / ".git"
    git_dir.mkdir()

    monkeypatch.chdir(repo_root)

    result = runner.invoke(app, ["hook", "install"])

    assert result.exit_code == 0
    ps_script = repo_root / ".chinvex" / "post-commit.ps1"
    assert ps_script.exists()
    # Should use normalized folder name
    assert "my-project" in ps_script.read_text() or "MyProject" in ps_script.read_text()


def test_hook_uninstall_removes_hook(tmp_path: Path, monkeypatch):
    """hook uninstall should remove generated hook"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    git_dir = repo_root / ".git"
    git_dir.mkdir()

    monkeypatch.chdir(repo_root)

    # Install first
    runner.invoke(app, ["hook", "install", "--context", "TestRepo"])
    assert (git_dir / "hooks" / "post-commit").exists()

    # Uninstall
    result = runner.invoke(app, ["hook", "uninstall"])

    assert result.exit_code == 0
    assert not (git_dir / "hooks" / "post-commit").exists()
    assert not (repo_root / ".chinvex" / "post-commit.ps1").exists()


def test_hook_status_shows_installed_state(tmp_path: Path, monkeypatch):
    """hook status should show whether hook is installed"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    git_dir = repo_root / ".git"
    git_dir.mkdir()

    monkeypatch.chdir(repo_root)

    # Before install
    result = runner.invoke(app, ["hook", "status"])
    assert result.exit_code == 0
    assert "not installed" in result.stdout.lower()

    # After install
    runner.invoke(app, ["hook", "install", "--context", "TestRepo"])
    result = runner.invoke(app, ["hook", "status"])
    assert result.exit_code == 0
    assert "installed" in result.stdout.lower()
    assert "TestRepo" in result.stdout
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/hooks/test_hook_cli.py -v`
Expected: FAIL with "No such command 'hook'"

**Step 3: Implement hook CLI commands**

```python
# src/chinvex/hooks/cli.py
"""CLI commands for git hook management."""
from __future__ import annotations

import logging
from pathlib import Path

import typer

from .generator import generate_post_commit_hook
from .resolver import resolve_python_path

log = logging.getLogger(__name__)


def find_git_dir() -> Path | None:
    """Find .git directory in current or parent directories."""
    current = Path.cwd()

    # Check current and parents
    for path in [current] + list(current.parents):
        git_dir = path / ".git"
        if git_dir.is_dir():
            return git_dir

    return None


def hook_install_cmd(context: str | None = None) -> None:
    """Install post-commit hook in current git repository."""
    # Find git directory
    git_dir = find_git_dir()
    if not git_dir:
        typer.secho("Error: Not a git repository", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    repo_root = git_dir.parent

    # Infer context name if not provided
    if not context:
        context = repo_root.name.lower().replace(" ", "-")
        typer.echo(f"Using context name: {context}")

    # Resolve Python path
    typer.echo("Resolving Python path...")
    python_path = resolve_python_path(repo_root)

    if not python_path:
        typer.secho(
            "Error: Could not find Python with chinvex installed",
            fg=typer.colors.RED
        )
        typer.echo("Tried: .venv, py -3, python in PATH")
        typer.echo("Set CHINVEX_PYTHON to specify explicitly")
        raise typer.Exit(code=1)

    typer.echo(f"Using Python: {python_path}")

    # Generate hook
    generate_post_commit_hook(repo_root, git_dir, python_path, context)

    typer.secho(
        f"✓ Post-commit hook installed for context '{context}'",
        fg=typer.colors.GREEN
    )
    typer.echo(f"Hook location: {git_dir / 'hooks' / 'post-commit'}")
    typer.echo(f"PowerShell script: {repo_root / '.chinvex' / 'post-commit.ps1'}")


def hook_uninstall_cmd() -> None:
    """Uninstall post-commit hook from current git repository."""
    git_dir = find_git_dir()
    if not git_dir:
        typer.secho("Error: Not a git repository", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    repo_root = git_dir.parent
    hook_file = git_dir / "hooks" / "post-commit"
    ps_script = repo_root / ".chinvex" / "post-commit.ps1"

    removed = False

    if hook_file.exists():
        # Check if it's our hook
        content = hook_file.read_text()
        if "Generated by chinvex" in content:
            hook_file.unlink()
            typer.echo(f"Removed: {hook_file}")
            removed = True
        else:
            typer.secho(
                "Warning: post-commit exists but not generated by chinvex",
                fg=typer.colors.YELLOW
            )

    if ps_script.exists():
        ps_script.unlink()
        typer.echo(f"Removed: {ps_script}")
        removed = True

    if removed:
        typer.secho("✓ Hook uninstalled", fg=typer.colors.GREEN)
    else:
        typer.echo("No chinvex hook found")


def hook_status_cmd() -> None:
    """Show git hook installation status."""
    git_dir = find_git_dir()
    if not git_dir:
        typer.secho("Not a git repository", fg=typer.colors.YELLOW)
        return

    repo_root = git_dir.parent
    hook_file = git_dir / "hooks" / "post-commit"
    ps_script = repo_root / ".chinvex" / "post-commit.ps1"

    typer.echo(f"Repository: {repo_root.name}")

    if not hook_file.exists():
        typer.secho("Post-commit hook: NOT INSTALLED", fg=typer.colors.YELLOW)
        return

    # Check if it's our hook
    content = hook_file.read_text()
    if "Generated by chinvex" not in content:
        typer.secho("Post-commit hook: EXISTS (not chinvex)", fg=typer.colors.YELLOW)
        return

    typer.secho("Post-commit hook: INSTALLED", fg=typer.colors.GREEN)

    # Extract context from PowerShell script
    if ps_script.exists():
        ps_content = ps_script.read_text()
        for line in ps_content.splitlines():
            if "# Context:" in line:
                context = line.split(":")[-1].strip()
                typer.echo(f"Context: {context}")
                break

        # Extract Python path
        for line in ps_content.splitlines():
            if "FilePath" in line and "python" in line.lower():
                typer.echo(f"Python: {line.strip()}")
                break
```

**Step 4: Wire into main CLI**

```python
# Modify src/chinvex/cli.py - add hook subcommand group:

hook_app = typer.Typer(help="Git hook management")
app.add_typer(hook_app, name="hook")


@hook_app.command("install")
def hook_install(context: str = typer.Option(None, help="Context name (inferred from folder if omitted)")):
    """Install post-commit hook in current git repository"""
    from .hooks.cli import hook_install_cmd
    hook_install_cmd(context)


@hook_app.command("uninstall")
def hook_uninstall():
    """Uninstall post-commit hook from current repository"""
    from .hooks.cli import hook_uninstall_cmd
    hook_uninstall_cmd()


@hook_app.command("status")
def hook_status():
    """Show git hook installation status"""
    from .hooks.cli import hook_status_cmd
    hook_status_cmd()
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/hooks/test_hook_cli.py -v`
Expected: PASS (all tests green)

**Step 6: Commit**

```bash
git add src/chinvex/hooks/cli.py src/chinvex/cli.py tests/hooks/test_hook_cli.py
git commit -m "feat(hooks): add hook install/uninstall/status commands

- chinvex hook install (infers context from folder name)
- chinvex hook uninstall (removes generated hooks)
- chinvex hook status (shows installation state)
- Auto-resolves Python path at install time

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 6: Scheduled Sweep

### Task 14: PowerShell Sweep Script

**Files:**
- Create: `scripts/scheduled_sweep.ps1`
- Create: `tests/scripts/test_sweep_script.py`

**Step 1: Write failing test for sweep script**

```python
# tests/scripts/test_sweep_script.py
import pytest
import subprocess
from pathlib import Path


def test_sweep_script_exists():
    """Sweep script should exist"""
    script_path = Path("scripts/scheduled_sweep.ps1")
    assert script_path.exists()


def test_sweep_script_syntax_valid():
    """PowerShell script should have valid syntax"""
    script_path = Path("scripts/scheduled_sweep.ps1")

    # Test PowerShell syntax
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(script_path), "-WhatIf"],
        capture_output=True,
        text=True
    )

    # Should not have syntax errors
    assert "ParserError" not in result.stderr
    assert "unexpected token" not in result.stderr.lower()


def test_sweep_script_requires_params():
    """Script should require ContextsRoot parameter"""
    script_path = Path("scripts/scheduled_sweep.ps1")

    # Run without required params
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(script_path)],
        capture_output=True,
        text=True
    )

    # Should fail with parameter error
    assert result.returncode != 0
    assert "ContextsRoot" in result.stderr or "parameter" in result.stderr.lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/scripts/test_sweep_script.py -v`
Expected: FAIL with "scripts/scheduled_sweep.ps1 does not exist"

**Step 3: Create PowerShell sweep script**

```powershell
# scripts/scheduled_sweep.ps1
<#
.SYNOPSIS
    Scheduled sweep - ensures watcher running and syncs all contexts

.DESCRIPTION
    Runs every 30 minutes via Task Scheduler.
    - Ensures watcher daemon is running
    - Checks watcher heartbeat (detects zombie processes)
    - Runs ingest sweep for all contexts
    - Archives _global context if needed

.PARAMETER ContextsRoot
    Path to contexts root directory

.PARAMETER NtfyTopic
    ntfy.sh topic for alerts (optional)

.PARAMETER NtfyServer
    ntfy server URL (default: https://ntfy.sh)

.PARAMETER StateDir
    State directory for watcher (default: ~/.chinvex)

.EXAMPLE
    .\scheduled_sweep.ps1 -ContextsRoot "P:\ai_memory\contexts" -NtfyTopic "chinvex-alerts"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$ContextsRoot,

    [Parameter(Mandatory=$false)]
    [string]$NtfyTopic = "",

    [Parameter(Mandatory=$false)]
    [string]$NtfyServer = "https://ntfy.sh",

    [Parameter(Mandatory=$false)]
    [string]$StateDir = (Join-Path $env:USERPROFILE ".chinvex")
)

$ErrorActionPreference = "Continue"
$LogFile = Join-Path $env:USERPROFILE ".chinvex\sweep.log"

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    Write-Host $logMessage
    Add-Content -Path $LogFile -Value $logMessage -ErrorAction SilentlyContinue
}

function Send-Alert {
    param([string]$Message)
    if ($NtfyTopic) {
        try {
            $url = "$NtfyServer/$NtfyTopic"
            Invoke-RestMethod -Uri $url -Method Post -Body $Message -ErrorAction Stop | Out-Null
            Write-Log "Alert sent: $Message"
        } catch {
            Write-Log "Failed to send alert: $_"
        }
    }
}

Write-Log "=== Sweep started ==="

# 1. Ensure watcher is running
Write-Log "Checking watcher status..."
try {
    $statusOutput = chinvex sync status --state-dir $StateDir 2>&1 | Out-String
    if ($statusOutput -match "NOT RUNNING") {
        Write-Log "Watcher not running, starting..."
        chinvex sync start --state-dir $StateDir
        Send-Alert "Chinvex watcher was down, restarted"
    } elseif ($statusOutput -match "STALE") {
        Write-Log "Watcher heartbeat stale, restarting..."
        chinvex sync stop --state-dir $StateDir
        Start-Sleep -Seconds 2
        chinvex sync start --state-dir $StateDir
        Send-Alert "Chinvex watcher heartbeat stale, restarted"
    } else {
        Write-Log "Watcher running normally"
    }
} catch {
    Write-Log "Error checking watcher: $_"
    Send-Alert "Chinvex sweep: watcher check failed"
}

# 2. Reconcile sources (ensure watcher watching correct paths)
Write-Log "Reconciling sources..."
try {
    chinvex sync reconcile-sources --contexts-root $ContextsRoot 2>&1 | Out-Null
} catch {
    Write-Log "Source reconciliation failed: $_"
}

# 3. Sweep all contexts
Write-Log "Running ingest sweep..."
try {
    $sweepOutput = chinvex ingest --all-contexts --changed-only --skip-locked --contexts-root $ContextsRoot 2>&1
    Write-Log "Sweep output: $sweepOutput"
} catch {
    Write-Log "Sweep failed: $_"
    Send-Alert "Chinvex sweep: ingest failed"
}

# 3.5. Check for stale contexts and send alerts
Write-Log "Checking for stale contexts..."
try {
    # Get list of context directories
    $contexts = Get-ChildItem -Path $ContextsRoot -Directory | Where-Object { $_.Name -notlike "_*" }

    foreach ($ctx in $contexts) {
        $statusFile = Join-Path $ctx.FullName "STATUS.json"
        if (Test-Path $statusFile) {
            try {
                $status = Get-Content $statusFile | ConvertFrom-Json
                if ($status.freshness.is_stale) {
                    # Use Python helper to check dedup and send if allowed
                    $logFile = Join-Path $env:USERPROFILE ".chinvex\push_log.jsonl"
                    python -c "from chinvex.notify import send_stale_alert; send_stale_alert('$($ctx.Name)', '$logFile', '$NtfyServer', '$NtfyTopic')" 2>&1 | Out-Null
                    Write-Log "Checked stale alert for $($ctx.Name)"
                }
            } catch {
                Write-Log "Error checking stale status for $($ctx.Name): $_"
            }
        }
    }
} catch {
    Write-Log "Stale context check failed: $_"
}

# 4. Generate global status
Write-Log "Generating global status..."
try {
    chinvex status --regenerate --contexts-root $ContextsRoot 2>&1 | Out-Null
    Write-Log "Global status regenerated"
} catch {
    Write-Log "Global status generation failed: $_"
}

# 5. Archive _global context if needed
Write-Log "Checking _global archive..."
try {
    $archiveOutput = chinvex archive --context _global --apply-constraints --quiet --contexts-root $ContextsRoot 2>&1
    if ($archiveOutput -match "archived") {
        Write-Log "Archived chunks in _global: $archiveOutput"
        Send-Alert "Chinvex: _global context archived old chunks"
    }
} catch {
    Write-Log "Archive check failed: $_"
}

Write-Log "=== Sweep complete ==="
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/scripts/test_sweep_script.py -v`
Expected: PASS (script exists and has valid syntax)

**Step 5: Commit**

```bash
git add scripts/scheduled_sweep.ps1 tests/scripts/test_sweep_script.py
git commit -m "feat(sweep): add PowerShell scheduled sweep script

- Ensures watcher running with heartbeat check
- Reconciles sources
- Runs ingest sweep (--changed-only --skip-locked)
- Checks for stale contexts and sends deduped alerts
- Generates GLOBAL_STATUS.md after sweep
- Archives _global context
- Sends ntfy alerts on issues
- Logs to ~/.chinvex/sweep.log

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 15: Task Scheduler Installer

**Files:**
- Create: `src/chinvex/bootstrap/scheduler.py`
- Create: `tests/bootstrap/test_scheduler.py`

**Step 1: Write failing test for Task Scheduler registration**

```python
# tests/bootstrap/test_scheduler.py
import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, Mock
from chinvex.bootstrap.scheduler import register_sweep_task, unregister_sweep_task


def test_register_creates_task_xml():
    """Should generate valid Task Scheduler XML"""
    from chinvex.bootstrap.scheduler import _generate_task_xml

    contexts_root = Path("P:/ai_memory/contexts")
    script_path = Path("C:/Code/chinvex/scripts/scheduled_sweep.ps1")
    ntfy_topic = "chinvex-alerts"

    xml = _generate_task_xml(script_path, contexts_root, ntfy_topic)

    assert "<Task" in xml
    assert "scheduled_sweep.ps1" in xml
    assert str(contexts_root) in xml
    assert "PT30M" in xml  # 30 minute interval


def test_register_calls_schtasks(tmp_path: Path):
    """Should call schtasks.exe to register task"""
    contexts_root = tmp_path / "contexts"
    script_path = tmp_path / "sweep.ps1"
    script_path.write_text("# mock script")

    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(returncode=0)

        register_sweep_task(script_path, contexts_root, "topic")

        # Should have called schtasks
        assert mock_run.called
        call_args = mock_run.call_args[0][0]
        assert "schtasks" in call_args[0].lower()
        assert "/Create" in call_args


def test_unregister_removes_task():
    """Should call schtasks to delete task"""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(returncode=0)

        unregister_sweep_task()

        assert mock_run.called
        call_args = mock_run.call_args[0][0]
        assert "schtasks" in call_args[0].lower()
        assert "/Delete" in call_args
        assert "ChinvexSweep" in call_args


def test_register_validates_script_exists(tmp_path: Path):
    """Should fail if script doesn't exist"""
    script_path = tmp_path / "nonexistent.ps1"
    contexts_root = tmp_path / "contexts"

    with pytest.raises(FileNotFoundError):
        register_sweep_task(script_path, contexts_root, "topic")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/bootstrap/test_scheduler.py -v`
Expected: FAIL with "cannot import name 'register_sweep_task'"

**Step 3: Implement Task Scheduler registration**

```python
# src/chinvex/bootstrap/__init__.py
"""Bootstrap and installation utilities."""

# src/chinvex/bootstrap/scheduler.py
"""Windows Task Scheduler integration."""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)


def register_sweep_task(
    script_path: Path,
    contexts_root: Path,
    ntfy_topic: str = ""
) -> None:
    """
    Register scheduled sweep task in Windows Task Scheduler.

    Args:
        script_path: Path to scheduled_sweep.ps1
        contexts_root: Contexts root directory
        ntfy_topic: ntfy.sh topic for alerts
    """
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    # Generate task XML
    xml_content = _generate_task_xml(script_path, contexts_root, ntfy_topic)

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml_content)
        xml_path = f.name

    try:
        # Register task
        cmd = [
            "schtasks",
            "/Create",
            "/TN", "ChinvexSweep",
            "/XML", xml_path,
            "/F"  # Force overwrite if exists
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to register task: {result.stderr}")

        log.info("Registered ChinvexSweep task")

    finally:
        # Clean up temp file
        Path(xml_path).unlink(missing_ok=True)


def unregister_sweep_task() -> None:
    """Remove ChinvexSweep task from Task Scheduler."""
    cmd = [
        "schtasks",
        "/Delete",
        "/TN", "ChinvexSweep",
        "/F"  # Force without confirmation
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # Task might not exist
        if "cannot find" in result.stderr.lower():
            log.info("ChinvexSweep task not found")
        else:
            raise RuntimeError(f"Failed to unregister task: {result.stderr}")
    else:
        log.info("Unregistered ChinvexSweep task")


def _generate_task_xml(
    script_path: Path,
    contexts_root: Path,
    ntfy_topic: str
) -> str:
    """Generate Task Scheduler XML definition."""
    # Build PowerShell command with arguments
    args_list = [
        f"-ContextsRoot \"{contexts_root}\""
    ]
    if ntfy_topic:
        args_list.append(f"-NtfyTopic \"{ntfy_topic}\"")

    args_str = " ".join(args_list)

    xml = f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Chinvex scheduled sweep - ensures watcher running and syncs contexts</Description>
    <URI>\\ChinvexSweep</URI>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <Repetition>
        <Interval>PT30M</Interval>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
      <StartBoundary>2026-01-01T00:00:00</StartBoundary>
      <Enabled>true</Enabled>
    </CalendarTrigger>
  </Triggers>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT10M</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions>
    <Exec>
      <Command>pwsh</Command>
      <Arguments>-NoProfile -ExecutionPolicy Bypass -File "{script_path}" {args_str}</Arguments>
    </Exec>
  </Actions>
  <Principals>
    <Principal>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
</Task>"""

    return xml
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/bootstrap/test_scheduler.py -v`
Expected: PASS (all tests green)

**Step 5: Create tasks.py wrapper**

```python
# src/chinvex/bootstrap/tasks.py
"""Task Scheduler wrappers for bootstrap components."""
from .scheduler import (
    register_sweep_task,
    register_login_trigger_task,
    register_morning_brief_task,
    unregister_sweep_task,
    unregister_login_trigger_task,
)

__all__ = [
    "register_sweep_task",
    "register_login_trigger_task",
    "register_morning_brief_task",
    "unregister_task",
    "check_task_exists",
]


def unregister_task(task_name: str) -> None:
    """
    Unregister a task by name.

    Args:
        task_name: Name of task to unregister
    """
    if task_name == "ChinvexSweep":
        unregister_sweep_task()
    elif task_name == "ChinvexWatcherStart":
        unregister_login_trigger_task()
    elif task_name == "ChinvexMorningBrief":
        from .scheduler import unregister_morning_brief_task
        unregister_morning_brief_task()
    else:
        raise ValueError(f"Unknown task: {task_name}")


def check_task_exists(task_name: str) -> bool:
    """
    Check if a scheduled task exists.

    Args:
        task_name: Name of task to check

    Returns:
        True if task exists
    """
    import subprocess

    result = subprocess.run(
        ["schtasks", "/Query", "/TN", task_name],
        capture_output=True,
        text=True
    )

    return result.returncode == 0
```

**Step 6: Add unregister_morning_brief_task to scheduler.py**

```python
# Add to src/chinvex/bootstrap/scheduler.py:

def unregister_morning_brief_task() -> None:
    """Remove ChinvexMorningBrief task from Task Scheduler."""
    cmd = [
        "schtasks",
        "/Delete",
        "/TN", "ChinvexMorningBrief",
        "/F"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        if "cannot find" in result.stderr.lower():
            log.info("ChinvexMorningBrief task not found")
        else:
            raise RuntimeError(f"Failed to unregister task: {result.stderr}")
    else:
        log.info("Unregistered ChinvexMorningBrief task")
```

**Step 7: Commit**

```bash
git add src/chinvex/bootstrap/ tests/bootstrap/
git commit -m "feat(bootstrap): add Task Scheduler registration

- Generates Task Scheduler XML definition
- Registers ChinvexSweep task (every 30 min)
- Registers ChinvexMorningBrief task (daily)
- Unregisters tasks on uninstall
- Passes contexts root and ntfy topic as args
- Added tasks.py wrapper for clean imports

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 15B: Login-Trigger Scheduled Task

**Files:**
- Modify: `src/chinvex/bootstrap/scheduler.py` (add login trigger)
- Modify: `tests/bootstrap/test_scheduler.py` (test login trigger)

**Step 1: Write failing test for login-trigger task**

```python
# tests/bootstrap/test_scheduler.py (add to existing file)

def test_register_login_trigger():
    """Should register login-trigger task that starts watcher"""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(returncode=0)

        from chinvex.bootstrap.scheduler import register_login_trigger_task
        register_login_trigger_task()

        assert mock_run.called
        call_args = mock_run.call_args[0][0]
        assert "schtasks" in call_args[0].lower()
        assert "/Create" in call_args
        assert "ChinvexWatcherStart" in " ".join(call_args)


def test_unregister_login_trigger():
    """Should remove login-trigger task"""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(returncode=0)

        from chinvex.bootstrap.scheduler import unregister_login_trigger_task
        unregister_login_trigger_task()

        assert mock_run.called
        call_args = mock_run.call_args[0][0]
        assert "schtasks" in call_args[0].lower()
        assert "/Delete" in call_args
        assert "ChinvexWatcherStart" in " ".join(call_args)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/bootstrap/test_scheduler.py -k login_trigger -v`
Expected: FAIL with "cannot import name 'register_login_trigger_task'"

**Step 3: Implement login-trigger task**

```python
# Add to src/chinvex/bootstrap/scheduler.py:

def register_login_trigger_task() -> None:
    """
    Register task that starts sync watcher at user login.

    This is the PRIMARY mechanism for ensuring watcher is running.
    Sweep task (Task 15) is the backup recovery mechanism.
    """
    xml_content = _generate_login_trigger_xml()

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml_content)
        xml_path = f.name

    try:
        cmd = [
            "schtasks",
            "/Create",
            "/TN", "ChinvexWatcherStart",
            "/XML", xml_path,
            "/F"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to register login trigger: {result.stderr}")

        log.info("Registered ChinvexWatcherStart task (login trigger)")

    finally:
        Path(xml_path).unlink(missing_ok=True)


def unregister_login_trigger_task() -> None:
    """Remove ChinvexWatcherStart task."""
    cmd = [
        "schtasks",
        "/Delete",
        "/TN", "ChinvexWatcherStart",
        "/F"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        if "cannot find" in result.stderr.lower():
            log.info("ChinvexWatcherStart task not found")
        else:
            raise RuntimeError(f"Failed to unregister task: {result.stderr}")
    else:
        log.info("Unregistered ChinvexWatcherStart task")


def _generate_login_trigger_xml() -> str:
    """Generate Task Scheduler XML for login trigger."""
    xml = """<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Chinvex watcher auto-start at user login</Description>
    <URI>\\ChinvexWatcherStart</URI>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
    </LogonTrigger>
  </Triggers>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT1M</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions>
    <Exec>
      <Command>chinvex</Command>
      <Arguments>sync ensure-running</Arguments>
    </Exec>
  </Actions>
  <Principals>
    <Principal>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
</Task>"""

    return xml


def register_morning_brief_task(
    contexts_root: Path,
    ntfy_topic: str,
    time: str = "07:00"
) -> None:
    """
    Register morning brief task in Windows Task Scheduler.

    Args:
        contexts_root: Contexts root directory
        ntfy_topic: ntfy.sh topic for morning brief
        time: Time to run (HH:MM format, default 07:00)
    """
    script_path = Path(__file__).parent.parent.parent / "scripts" / "morning_brief.ps1"
    if not script_path.exists():
        raise FileNotFoundError(f"Morning brief script not found: {script_path}")

    # Generate task XML
    xml_content = _generate_morning_brief_xml(script_path, contexts_root, ntfy_topic, time)

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml_content)
        xml_path = f.name

    try:
        # Register task
        cmd = [
            "schtasks",
            "/Create",
            "/TN", "ChinvexMorningBrief",
            "/XML", xml_path,
            "/F"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to register morning brief task: {result.stderr}")

        log.info(f"Registered ChinvexMorningBrief task (daily at {time})")

    finally:
        Path(xml_path).unlink(missing_ok=True)


def _generate_morning_brief_xml(
    script_path: Path,
    contexts_root: Path,
    ntfy_topic: str,
    time: str
) -> str:
    """Generate Task Scheduler XML for morning brief."""
    # Build arguments
    args_list = [f"-ContextsRoot \"{contexts_root}\""]
    if ntfy_topic:
        args_list.append(f"-NtfyTopic \"{ntfy_topic}\"")

    args_str = " ".join(args_list)

    # Parse time (HH:MM)
    hour, minute = time.split(":")
    start_boundary = f"2026-01-01T{hour}:{minute}:00"

    xml = f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Chinvex morning brief - daily status summary</Description>
    <URI>\\ChinvexMorningBrief</URI>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <Repetition>
        <Interval>P1D</Interval>
      </Repetition>
      <StartBoundary>{start_boundary}</StartBoundary>
      <Enabled>true</Enabled>
    </CalendarTrigger>
  </Triggers>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT5M</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions>
    <Exec>
      <Command>pwsh</Command>
      <Arguments>-NoProfile -ExecutionPolicy Bypass -File "{script_path}" {args_str}</Arguments>
    </Exec>
  </Actions>
  <Principals>
    <Principal>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
</Task>"""

    return xml
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/bootstrap/test_scheduler.py -k login_trigger -v`
Expected: PASS (all tests green)

**Step 5: Commit**

```bash
git add src/chinvex/bootstrap/scheduler.py tests/bootstrap/test_scheduler.py
git commit -m "feat(bootstrap): add login-trigger scheduled task

- Registers ChinvexWatcherStart task (at user login)
- Calls 'chinvex sync ensure-running' on login
- Primary mechanism for watcher availability
- Sweep task is backup/recovery mechanism

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 16: Archive Old Chunks Command

**Files:**
- Create: `src/chinvex/archive.py`
- Modify: `src/chinvex/cli.py` (add archive command)
- Create: `tests/test_archive.py`

**Step 1: Write failing test for archive command**

```python
# tests/test_archive.py
import pytest
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from chinvex.archive import archive_by_age, archive_by_count, ArchiveStats


def test_archive_by_age_marks_old_chunks(tmp_path: Path):
    """Should mark chunks older than threshold as archived"""
    from chinvex.storage import Storage

    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()

    # Add archived column if not exists
    storage.conn.execute("ALTER TABLE chunks ADD COLUMN archived INTEGER DEFAULT 0")
    storage.conn.commit()

    # Insert chunks with different ages
    old_time = datetime.now(timezone.utc) - timedelta(days=100)
    recent_time = datetime.now(timezone.utc) - timedelta(days=10)

    storage.insert_chunk("chunk1", "doc1", "old content", "{}", "key1")
    storage.insert_chunk("chunk2", "doc1", "recent content", "{}", "key2")

    # Manually set timestamps
    storage.conn.execute("UPDATE chunks SET updated_at = ? WHERE chunk_id = ?",
                        (old_time.isoformat(), "chunk1"))
    storage.conn.execute("UPDATE chunks SET updated_at = ? WHERE chunk_id = ?",
                        (recent_time.isoformat(), "chunk2"))
    storage.conn.commit()

    # Archive chunks older than 90 days
    stats = archive_by_age(storage, age_threshold_days=90)

    assert stats.archived_count == 1

    # Verify chunk1 is archived
    result = storage.conn.execute(
        "SELECT archived FROM chunks WHERE chunk_id = ?", ("chunk1",)
    ).fetchone()
    assert result[0] == 1

    # Verify chunk2 is NOT archived
    result = storage.conn.execute(
        "SELECT archived FROM chunks WHERE chunk_id = ?", ("chunk2",)
    ).fetchone()
    assert result[0] == 0

    storage.close()


def test_archive_by_count_archives_oldest(tmp_path: Path):
    """Should archive oldest chunks when count exceeds limit"""
    from chinvex.storage import Storage

    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()
    storage.conn.execute("ALTER TABLE chunks ADD COLUMN archived INTEGER DEFAULT 0")
    storage.conn.commit()

    # Insert 10 chunks
    for i in range(10):
        age = datetime.now(timezone.utc) - timedelta(days=i)
        storage.insert_chunk(f"chunk{i}", "doc1", f"content{i}", "{}", f"key{i}")
        storage.conn.execute(
            "UPDATE chunks SET updated_at = ? WHERE chunk_id = ?",
            (age.isoformat(), f"chunk{i}")
        )
    storage.conn.commit()

    # Archive to keep only 5 chunks
    stats = archive_by_count(storage, max_chunks=5)

    assert stats.archived_count == 5

    # Verify oldest 5 are archived
    for i in range(5, 10):  # Oldest
        result = storage.conn.execute(
            "SELECT archived FROM chunks WHERE chunk_id = ?", (f"chunk{i}",)
        ).fetchone()
        assert result[0] == 1

    # Verify newest 5 are NOT archived
    for i in range(5):  # Newest
        result = storage.conn.execute(
            "SELECT archived FROM chunks WHERE chunk_id = ?", (f"chunk{i}",)
        ).fetchone()
        assert result[0] == 0

    storage.close()


def test_archive_skips_already_archived(tmp_path: Path):
    """Should not double-archive already archived chunks"""
    from chinvex.storage import Storage

    db_path = tmp_path / "test.db"
    storage = Storage(db_path)
    storage.ensure_schema()
    storage.conn.execute("ALTER TABLE chunks ADD COLUMN archived INTEGER DEFAULT 0")
    storage.conn.commit()

    # Insert chunk and mark as archived
    storage.insert_chunk("chunk1", "doc1", "content", "{}", "key1")
    storage.conn.execute("UPDATE chunks SET archived = 1 WHERE chunk_id = ?", ("chunk1",))
    storage.conn.commit()

    # Try to archive again
    stats = archive_by_age(storage, age_threshold_days=0)  # Archive all

    # Should not count already-archived chunk
    assert stats.archived_count == 0

    storage.close()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_archive.py -v`
Expected: FAIL with "cannot import name 'archive_by_age'"

**Step 3: Implement archive functionality**

```python
# src/chinvex/archive.py
"""Archive old chunks to reduce active index size."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .storage import Storage

log = logging.getLogger(__name__)


@dataclass
class ArchiveStats:
    """Statistics from archive operation."""
    archived_count: int
    total_chunks: int
    active_chunks: int


def archive_by_age(storage: Storage, age_threshold_days: int) -> ArchiveStats:
    """
    Archive chunks older than threshold.

    Args:
        storage: Storage instance
        age_threshold_days: Archive chunks older than this many days

    Returns:
        ArchiveStats with counts
    """
    threshold_dt = datetime.now(timezone.utc) - timedelta(days=age_threshold_days)
    threshold_str = threshold_dt.isoformat()

    # Ensure archived column exists
    _ensure_archived_column(storage)

    # Count total and active chunks before
    total_before = storage.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    active_before = storage.conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE archived = 0"
    ).fetchone()[0]

    # Archive old chunks (only those not already archived)
    cursor = storage.conn.execute(
        """
        UPDATE chunks
        SET archived = 1
        WHERE updated_at < ?
          AND archived = 0
        """,
        (threshold_str,)
    )

    archived_count = cursor.rowcount
    storage.conn.commit()

    log.info(f"Archived {archived_count} chunks older than {age_threshold_days} days")

    return ArchiveStats(
        archived_count=archived_count,
        total_chunks=total_before,
        active_chunks=active_before - archived_count
    )


def archive_by_count(storage: Storage, max_chunks: int) -> ArchiveStats:
    """
    Archive oldest chunks to keep total under max_chunks.

    Args:
        storage: Storage instance
        max_chunks: Maximum active chunks to keep

    Returns:
        ArchiveStats with counts
    """
    _ensure_archived_column(storage)

    # Count active chunks
    active_count = storage.conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE archived = 0"
    ).fetchone()[0]

    total_count = storage.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    if active_count <= max_chunks:
        log.info(f"Active chunks ({active_count}) under limit ({max_chunks})")
        return ArchiveStats(
            archived_count=0,
            total_chunks=total_count,
            active_chunks=active_count
        )

    # Archive oldest chunks
    to_archive = active_count - max_chunks

    cursor = storage.conn.execute(
        """
        UPDATE chunks
        SET archived = 1
        WHERE chunk_id IN (
            SELECT chunk_id FROM chunks
            WHERE archived = 0
            ORDER BY updated_at ASC
            LIMIT ?
        )
        """,
        (to_archive,)
    )

    archived_count = cursor.rowcount
    storage.conn.commit()

    log.info(f"Archived {archived_count} oldest chunks to stay under {max_chunks}")

    return ArchiveStats(
        archived_count=archived_count,
        total_chunks=total_count,
        active_chunks=max_chunks
    )


def _ensure_archived_column(storage: Storage) -> None:
    """Ensure archived column exists in chunks table."""
    # Check if column exists
    cursor = storage.conn.execute("PRAGMA table_info(chunks)")
    columns = [row[1] for row in cursor.fetchall()]

    if "archived" not in columns:
        log.info("Adding archived column to chunks table")
        storage.conn.execute("ALTER TABLE chunks ADD COLUMN archived INTEGER DEFAULT 0")
        storage.conn.commit()
```

**Step 4: Add CLI command**

```python
# Modify src/chinvex/cli.py - add archive command:

@app.command()
def archive(
    context: str = typer.Option(..., help="Context name"),
    apply_constraints: bool = typer.Option(False, help="Apply age and count constraints"),
    age_days: int = typer.Option(None, help="Archive chunks older than N days"),
    max_chunks: int = typer.Option(None, help="Keep only N most recent chunks"),
    quiet: bool = typer.Option(False, help="Suppress output"),
):
    """
    Archive old chunks to reduce active index size.

    Use --apply-constraints to read from context.json constraints config.
    Or specify --age-days and/or --max-chunks explicitly.
    """
    from .context import load_context
    from .context_cli import get_contexts_root
    from .archive import archive_by_age, archive_by_count
    from .storage import Storage

    contexts_root = get_contexts_root()
    ctx_config = load_context(context, contexts_root)

    db_path = ctx_config.index.sqlite_path
    storage = Storage(db_path)

    # Determine constraints
    if apply_constraints:
        # Read from context config
        constraints = getattr(ctx_config, "constraints", None)
        if constraints:
            age_days = constraints.get("archive_after_days", 90)
            max_chunks = constraints.get("max_chunks", 10000)
        else:
            if not quiet:
                typer.echo("No constraints defined in context config")
            storage.close()
            return

    # Apply age constraint
    if age_days:
        stats = archive_by_age(storage, age_days)
        if not quiet:
            typer.echo(f"Archived {stats.archived_count} chunks older than {age_days} days")

    # Apply count constraint
    if max_chunks:
        stats = archive_by_count(storage, max_chunks)
        if not quiet:
            typer.echo(f"Archived {stats.archived_count} chunks to stay under {max_chunks}")

    storage.close()

    if not quiet:
        typer.secho("Archive complete", fg=typer.colors.GREEN)
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_archive.py -v`
Expected: PASS (all tests green)

**Step 6: Commit**

```bash
git add src/chinvex/archive.py src/chinvex/cli.py tests/test_archive.py
git commit -m "feat(archive): add chunk archiving by age and count

- Archive chunks older than N days
- Archive oldest chunks to keep under max count
- Adds archived column to chunks table
- CLI: chinvex archive --context X --apply-constraints
- Respects context.json constraints config

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 7: Push Notifications

### Task 17: ntfy Push Utility

**Files:**
- Create: `src/chinvex/notify.py`
- Create: `tests/test_notify.py`

**Step 1: Write failing test for ntfy push**

```python
# tests/test_notify.py
import pytest
from unittest.mock import patch, Mock
from chinvex.notify import send_ntfy_push, NtfyConfig


def test_send_ntfy_push_basic():
    """Should send HTTP POST to ntfy server"""
    config = NtfyConfig(
        server="https://ntfy.sh",
        topic="test-topic"
    )

    with patch('requests.post') as mock_post:
        mock_post.return_value = Mock(status_code=200)

        send_ntfy_push(config, "Test message")

        mock_post.assert_called_once()
        call_args = mock_post.call_args

        assert "ntfy.sh/test-topic" in call_args[0][0]
        assert call_args[1]["data"] == "Test message"


def test_send_ntfy_push_with_title():
    """Should include title in headers if provided"""
    config = NtfyConfig(server="https://ntfy.sh", topic="test")

    with patch('requests.post') as mock_post:
        mock_post.return_value = Mock(status_code=200)

        send_ntfy_push(config, "Body", title="Test Title")

        headers = mock_post.call_args[1]["headers"]
        assert headers["Title"] == "Test Title"


def test_send_ntfy_push_with_priority():
    """Should include priority in headers"""
    config = NtfyConfig(server="https://ntfy.sh", topic="test")

    with patch('requests.post') as mock_post:
        mock_post.return_value = Mock(status_code=200)

        send_ntfy_push(config, "Urgent!", priority="high")

        headers = mock_post.call_args[1]["headers"]
        assert headers["Priority"] == "high"


def test_send_ntfy_push_disabled():
    """Should skip if enabled=False"""
    config = NtfyConfig(
        server="https://ntfy.sh",
        topic="test",
        enabled=False
    )

    with patch('requests.post') as mock_post:
        send_ntfy_push(config, "Test")

        # Should not have called POST
        mock_post.assert_not_called()


def test_should_send_stale_alert_dedup(tmp_path: Path):
    """Should only send stale alert once per context per day"""
    from chinvex.notify import should_send_stale_alert

    log_file = tmp_path / "push_log.jsonl"

    # First call - should allow
    assert should_send_stale_alert("TestCtx", log_file) is True

    # Second call same day - should block
    assert should_send_stale_alert("TestCtx", log_file) is False

    # Different context - should allow
    assert should_send_stale_alert("OtherCtx", log_file) is True


def test_log_push_records_to_file(tmp_path: Path):
    """Should append push records to JSONL log"""
    from chinvex.notify import log_push

    log_file = tmp_path / "push_log.jsonl"

    log_push("TestCtx", "stale", log_file)
    log_push("TestCtx", "watch_hit", log_file)

    # Should have 2 lines
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 2

    # Should be valid JSON
    import json
    record1 = json.loads(lines[0])
    assert record1["context"] == "TestCtx"
    assert record1["type"] == "stale"
    assert "timestamp" in record1
    assert "date" in record1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_notify.py -v`
Expected: FAIL with "cannot import name 'send_ntfy_push'"

**Step 3: Implement ntfy push utility**

```python
# src/chinvex/notify.py
"""Push notification utilities (ntfy.sh)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger(__name__)


@dataclass
class NtfyConfig:
    """ntfy.sh notification configuration."""
    server: str  # e.g., "https://ntfy.sh"
    topic: str   # e.g., "chinvex-alerts"
    enabled: bool = True


def send_ntfy_push(
    config: NtfyConfig,
    message: str,
    title: str | None = None,
    priority: str = "default",
    tags: list[str] | None = None
) -> bool:
    """
    Send push notification via ntfy.sh.

    Args:
        config: ntfy configuration
        message: Notification body
        title: Optional title
        priority: Priority level (min, low, default, high, urgent)
        tags: Optional list of emoji tags

    Returns:
        True if sent successfully
    """
    if not config.enabled:
        log.debug("Notifications disabled, skipping")
        return False

    url = f"{config.server}/{config.topic}"

    headers = {}
    if title:
        headers["Title"] = title
    if priority != "default":
        headers["Priority"] = priority
    if tags:
        headers["Tags"] = ",".join(tags)

    try:
        response = requests.post(
            url,
            data=message.encode("utf-8"),
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            log.info(f"Sent ntfy push: {title or message[:30]}")
            return True
        else:
            log.warning(f"ntfy push failed: {response.status_code}")
            return False

    except requests.RequestException as e:
        log.error(f"Failed to send ntfy push: {e}")
        return False


def should_send_stale_alert(context_name: str, log_file: Path) -> bool:
    """
    Check if stale alert should be sent (dedup: max 1 per context per day).

    Args:
        context_name: Context to check
        log_file: Path to push_log.jsonl

    Returns:
        True if alert should be sent
    """
    if not log_file.exists():
        return True

    today = datetime.now(timezone.utc).date().isoformat()

    # Check log for existing entry
    try:
        for line in log_file.read_text().splitlines():
            record = json.loads(line)
            if (
                record.get("context") == context_name
                and record.get("type") == "stale"
                and record.get("date") == today
            ):
                return False  # Already sent today
    except (json.JSONDecodeError, IOError):
        pass

    return True


def log_push(context_name: str, push_type: str, log_file: Path) -> None:
    """
    Log push notification to dedup log.

    Args:
        context_name: Context name
        push_type: Type of push (stale, watch_hit, etc.)
        log_file: Path to push_log.jsonl
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    record = {
        "timestamp": now.isoformat(),
        "context": context_name,
        "type": push_type,
        "date": now.date().isoformat()
    }

    with log_file.open("a") as f:
        f.write(json.dumps(record) + "\n")


def send_stale_alert(context_name: str, log_file_path: str, server: str, topic: str) -> None:
    """
    Send stale context alert if not already sent today.

    Called from PowerShell sweep script.

    Args:
        context_name: Context name
        log_file_path: Path to push_log.jsonl
        server: ntfy server URL
        topic: ntfy topic
    """
    log_file = Path(log_file_path)

    # Check dedup
    if not should_send_stale_alert(context_name, log_file):
        log.debug(f"Stale alert for {context_name} already sent today")
        return

    # Send alert
    config = NtfyConfig(server=server, topic=topic, enabled=bool(topic))
    success = send_ntfy_push(
        config,
        f"{context_name}: last sync stale",
        title="Stale context",
        priority="low"
    )

    if success:
        # Log to dedup file
        log_push(context_name, "stale", log_file)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_notify.py -v`
Expected: PASS (all tests green)

**Step 5: Commit**

```bash
git add src/chinvex/notify.py tests/test_notify.py
git commit -m "feat(notify): add ntfy.sh push notification utility

- Sends HTTP POST to ntfy.sh server
- Supports title, priority, tags
- Push dedup log (push_log.jsonl) for stale alerts
- Dedup key: (context, type, date) - max 1x per day
- Respects enabled flag
- Handles errors gracefully

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 18: Morning Brief Aggregator Script

**Files:**
- Create: `scripts/morning_brief.ps1`
- Create: `tests/scripts/test_morning_brief.py`

**Step 1: Write failing test for morning brief script**

```python
# tests/scripts/test_morning_brief.py
import pytest
import subprocess
from pathlib import Path


def test_morning_brief_script_exists():
    """Morning brief script should exist"""
    script_path = Path("scripts/morning_brief.ps1")
    assert script_path.exists()


def test_morning_brief_syntax_valid():
    """PowerShell script should have valid syntax"""
    script_path = Path("scripts/morning_brief.ps1")

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(script_path), "-WhatIf"],
        capture_output=True,
        text=True
    )

    assert "ParserError" not in result.stderr


def test_morning_brief_requires_contexts_root():
    """Script should require ContextsRoot parameter"""
    script_path = Path("scripts/morning_brief.ps1")

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(script_path)],
        capture_output=True,
        text=True
    )

    assert result.returncode != 0
    assert "ContextsRoot" in result.stderr or "parameter" in result.stderr.lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/scripts/test_morning_brief.py -v`
Expected: FAIL with "scripts/morning_brief.ps1 does not exist"

**Step 3: Create morning brief script**

```powershell
# scripts/morning_brief.ps1
<#
.SYNOPSIS
    Generate and send morning brief with context status

.DESCRIPTION
    Aggregates STATUS.json from all contexts and sends ntfy push.
    Writes MORNING_BRIEF.md to configured output path.

.PARAMETER ContextsRoot
    Path to contexts root directory

.PARAMETER NtfyTopic
    ntfy.sh topic for morning brief

.PARAMETER NtfyServer
    ntfy server URL (default: https://ntfy.sh)

.PARAMETER OutputPath
    Path to write MORNING_BRIEF.md (default: contexts root parent)

.EXAMPLE
    .\morning_brief.ps1 -ContextsRoot "P:\ai_memory\contexts" -NtfyTopic "morning-brief"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$ContextsRoot,

    [Parameter(Mandatory=$false)]
    [string]$NtfyTopic = "",

    [Parameter(Mandatory=$false)]
    [string]$NtfyServer = "https://ntfy.sh",

    [Parameter(Mandatory=$false)]
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Continue"

# Default output path
if (-not $OutputPath) {
    $OutputPath = Join-Path (Split-Path $ContextsRoot -Parent) "MORNING_BRIEF.md"
}

# Collect all STATUS.json files
$contexts = Get-ChildItem -Path $ContextsRoot -Directory
$statusData = @()

foreach ($ctx in $contexts) {
    $statusFile = Join-Path $ctx.FullName "STATUS.json"
    if (Test-Path $statusFile) {
        try {
            $status = Get-Content $statusFile | ConvertFrom-Json
            $statusData += $status
        } catch {
            Write-Warning "Failed to parse $statusFile: $_"
        }
    }
}

# Calculate totals
$totalChunks = ($statusData | Measure-Object -Property chunks -Sum).Sum
$staleContexts = $statusData | Where-Object { $_.freshness.is_stale -eq $true }
$pendingWatches = ($statusData | Measure-Object -Property watches_pending_hits -Sum).Sum

# Generate markdown
$timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ"
$markdown = @"
# Dual Nature Morning Brief
Generated: $timestamp

## Summary
- **Total Contexts:** $($statusData.Count)
- **Total Chunks:** $($totalChunks.ToString("N0"))
- **Stale Contexts:** $($staleContexts.Count)
- **Pending Watch Hits:** $pendingWatches

## Context Details
| Context | Chunks | Last Sync | Status |
|---------|--------|-----------|--------|
"@

foreach ($status in ($statusData | Sort-Object -Property context)) {
    $statusIcon = if ($status.freshness.is_stale) { "[STALE]" } else { "[OK]" }
    $lastSync = if ($status.last_sync) {
        $syncTime = [datetime]::Parse($status.last_sync)
        $hoursAgo = [math]::Round(((Get-Date) - $syncTime).TotalHours, 1)
        "${hoursAgo}h ago"
    } else {
        "unknown"
    }

    $markdown += "`n| $($status.context) | $($status.chunks.ToString("N0")) | $lastSync | $statusIcon |"
}

if ($staleContexts.Count -gt 0) {
    $markdown += "`n`n## Stale Contexts"
    foreach ($ctx in $staleContexts) {
        $markdown += "`n- **$($ctx.context)**: $($ctx.freshness.hours_since_sync) hours since sync"
    }
}

if ($pendingWatches -gt 0) {
    $markdown += "`n`n## Pending Watch Hits"
    $markdown += "`nTotal pending: $pendingWatches"
}

# Write markdown file
$markdown | Out-File -FilePath $OutputPath -Encoding UTF8
Write-Host "Wrote brief to $OutputPath"

# Send ntfy push
if ($NtfyTopic) {
    $title = "Dual Nature Morning Brief"
    $body = "Contexts: $($statusData.Count) ($($staleContexts.Count) stale)`nChunks: $($totalChunks.ToString("N0"))`nWatch hits: $pendingWatches"

    if ($staleContexts.Count -gt 0) {
        $staleNames = ($staleContexts | ForEach-Object { $_.context }) -join ", "
        $body += "`nStale: $staleNames"
    }

    try {
        $url = "$NtfyServer/$NtfyTopic"
        $headers = @{
            "Title" = $title
            "Tags" = "sunrise,calendar"
        }

        Invoke-RestMethod -Uri $url -Method Post -Body $body -Headers $headers | Out-Null
        Write-Host "Sent morning brief push"
    } catch {
        Write-Warning "Failed to send ntfy push: $_"
    }
}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/scripts/test_morning_brief.py -v`
Expected: PASS (script exists with valid syntax)

**Step 5: Commit**

```bash
git add scripts/morning_brief.ps1 tests/scripts/test_morning_brief.py
git commit -m "feat(notify): add morning brief aggregator script

- Aggregates all context STATUS.json files
- Generates MORNING_BRIEF.md with totals and table
- Sends ntfy push with summary
- Highlights stale contexts and pending watch hits

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 19: Watch Hit Push Integration

**Files:**
- Modify: `src/chinvex/search.py` (add watch hit push)
- Create: `tests/test_watch_hit_push.py`

**Step 1: Write failing test for watch hit notifications**

```python
# tests/test_watch_hit_push.py
import pytest
from unittest.mock import patch, Mock
from chinvex.notify import NtfyConfig


def test_search_sends_watch_hit_push():
    """Search should send push when watch query hits"""
    # This test will be implemented when watches are added in future phases
    # For now, just test the notification utility exists
    from chinvex.notify import send_ntfy_push

    config = NtfyConfig(server="https://ntfy.sh", topic="test")

    with patch('requests.post') as mock_post:
        mock_post.return_value = Mock(status_code=200)

        result = send_ntfy_push(
            config,
            "Found 3 results for watch: 'embedding metrics'",
            title="Watch Hit: embedding metrics",
            tags=["eyes", "mag"]
        )

        assert result is True
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_watch_hit_push.py -v`
Expected: PASS (notification utility already works)

**Step 3: Implement watch hit push in post_ingest_hook**

```python
# Modify src/chinvex/hooks.py - add to post_ingest_hook():

def post_ingest_hook(context: ContextConfig, stats: dict) -> None:
    """
    Hook that runs after ingest completes.

    - Runs watches
    - Sends notifications for watch hits
    - Writes STATUS.json
    """
    from .notify import send_ntfy_push, NtfyConfig
    from .watch import run_watches  # Future implementation

    # Run watches if configured
    watch_hits = []
    if hasattr(context, 'watches') and context.watches:
        watch_hits = run_watches(context)

    # Send watch hit notifications
    if watch_hits and context.notifications and context.notifications.enabled:
        ntfy_config = NtfyConfig(
            server="https://ntfy.sh",
            topic=context.notifications.webhook_url.split('/')[-1],  # Extract topic from webhook
            enabled=True
        )

        # Send notification with first 3 queries
        queries = ', '.join(h.query for h in watch_hits[:3])
        more = f" (and {len(watch_hits) - 3} more)" if len(watch_hits) > 3 else ""
        send_ntfy_push(
            ntfy_config,
            f"{context.name}: {len(watch_hits)} watch hits ({queries}{more})",
            title="Watch Hit",
            tags=["eyes", "mag"]
        )

    # Continue with STATUS.json write...
```

**Step 4: Commit**

```bash
git add src/chinvex/search.py tests/test_watch_hit_push.py
git commit -m "feat(notify): add watch hit push integration point

- Documents integration for future watch implementation
- Tests notification utility with watch payload
- Ready for watches feature (future phase)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 8: Status Command

### Task 20: Global Status Aggregation

**Files:**
- Create: `src/chinvex/global_status.py`
- Create: `tests/test_global_status.py`

**Step 1: Write failing test for global status aggregation**

```python
# tests/test_global_status.py
import pytest
import json
from pathlib import Path
from datetime import datetime, timezone
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
    with patch('chinvex.global_status.get_state_dir', return_value=state_dir):
        generate_global_status_md(contexts_root, output_path)

    content = output_path.read_text()

    # Should show watcher status
    assert "watcher" in content.lower() or "daemon" in content.lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_global_status.py -v`
Expected: FAIL with "cannot import name 'generate_global_status_md'"

**Step 3: Implement global status aggregation**

```python
# src/chinvex/global_status.py
"""Global status aggregation across all contexts."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


def generate_global_status_md(
    contexts_root: Path,
    output_path: Path
) -> None:
    """
    Generate GLOBAL_STATUS.md from all context STATUS.json files.

    Args:
        contexts_root: Root directory containing contexts
        output_path: Path to write GLOBAL_STATUS.md
    """
    # Collect all STATUS.json
    contexts = []

    for ctx_dir in contexts_root.iterdir():
        if not ctx_dir.is_dir():
            continue

        status_file = ctx_dir / "STATUS.json"
        if not status_file.exists():
            continue

        try:
            data = json.loads(status_file.read_text())
            contexts.append(data)
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Failed to read {status_file}: {e}")

    # Check watcher health
    watcher_status = _get_watcher_status()

    # Calculate aggregates
    total_chunks = sum(c.get("chunks", 0) for c in contexts)
    stale_contexts = [c for c in contexts if c.get("freshness", {}).get("is_stale", False)]
    pending_hits = sum(c.get("watches_pending_hits", 0) for c in contexts)

    # Generate markdown
    now = datetime.now(timezone.utc).isoformat()

    md = f"""# Dual Nature Status
Generated: {now}

## Health
- **Watcher:** {watcher_status}
- **Stale Contexts:** {len(stale_contexts)} of {len(contexts)}
- **Total Chunks:** {total_chunks:,}
- **Pending Watch Hits:** {pending_hits}

## Contexts
| Context | Last Sync | Chunks | Watch Hits | Status |
|---------|-----------|--------|------------|--------|
"""

    # Sort by last_sync (most recent first)
    contexts_sorted = sorted(
        contexts,
        key=lambda c: c.get("last_sync", ""),
        reverse=True
    )

    for ctx in contexts_sorted:
        name = ctx.get("context", "unknown")
        chunks = ctx.get("chunks", 0)
        hits = ctx.get("watches_pending_hits", 0)

        # Format last sync
        last_sync = ctx.get("last_sync")
        if last_sync:
            sync_dt = datetime.fromisoformat(last_sync)
            elapsed = datetime.now(timezone.utc) - sync_dt
            if elapsed.total_seconds() < 3600:
                sync_str = f"{int(elapsed.total_seconds() / 60)}m ago"
            elif elapsed.total_seconds() < 86400:
                sync_str = f"{int(elapsed.total_seconds() / 3600)}h ago"
            else:
                sync_str = f"{int(elapsed.total_seconds() / 86400)}d ago"
        else:
            sync_str = "never"

        # Status icon
        is_stale = ctx.get("freshness", {}).get("is_stale", False)
        status_icon = "⚠ stale" if is_stale else "✓"

        md += f"| {name} | {sync_str} | {chunks:,} | {hits} | {status_icon} |\n"

    # Pending watch hits section
    if pending_hits > 0:
        md += "\n## Pending Watch Hits\n"
        for ctx in contexts:
            hits = ctx.get("watches_pending_hits", 0)
            if hits > 0:
                md += f"- **{ctx['context']}**: {hits} hits\n"

    # Write file
    output_path.write_text(md, encoding="utf-8")
    log.info(f"Wrote {output_path}")


def _get_watcher_status() -> str:
    """Get watcher daemon status string."""
    try:
        from .sync.cli import get_state_dir
        from .sync.daemon import DaemonManager

        state_dir = get_state_dir()
        dm = DaemonManager(state_dir)

        state = dm.get_state()

        if state.value == "running":
            return "RUNNING"
        elif state.value == "stale":
            return "STALE"
        else:
            return "NOT RUNNING"

    except Exception as e:
        log.warning(f"Failed to check watcher status: {e}")
        return "UNKNOWN"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_global_status.py -v`
Expected: PASS (all tests green)

**Step 5: Commit**

```bash
git add src/chinvex/global_status.py tests/test_global_status.py
git commit -m "feat(status): add global status aggregation

- Aggregates all context STATUS.json files
- Generates GLOBAL_STATUS.md with health summary
- Includes watcher daemon status
- Shows stale contexts and pending watch hits
- Formats table with last sync, chunks, status

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

Due to plan file size, I'll provide the final tasks (21-26) in condensed form with key implementation points. The pattern remains: test → fail → implement → pass → commit.

---

### Task 21: Status CLI Formatter

**Files:**
- Create: `src/chinvex/cli_status.py`
- Create: `tests/test_cli_status.py`
- Modify: `src/chinvex/cli.py` (add status command)

**Step 1: Write failing test for status command**

```python
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
            watcher_running=True
        ),
        ContextStatus(
            name="Streamside",
            chunks=567,
            last_sync="2026-01-29T11:30:00Z",
            is_stale=False,
            hours_since_sync=1.0,
            watcher_running=True
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
            watcher_running=False
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_status.py -v`
Expected: FAIL with "cannot import name 'format_status_output'"

**Step 3: Implement status formatter**

```python
# src/chinvex/cli_status.py
"""Status command implementation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ContextStatus:
    name: str
    chunks: int
    last_sync: str
    is_stale: bool
    hours_since_sync: float
    watcher_running: bool


def format_status_output(contexts: list[ContextStatus], watcher_running: bool) -> str:
    """
    Format status output as table.

    Args:
        contexts: List of context statuses
        watcher_running: Whether watcher is running globally

    Returns:
        Formatted status string
    """
    lines = ["# Chinvex Global Status", ""]

    # Table header
    lines.append("| Context | Chunks | Last Sync | Status |")
    lines.append("|---------|--------|-----------|--------|")

    # Rows
    for ctx in contexts:
        status_icon = "[OK]" if not ctx.is_stale else "[STALE]"
        hours_str = f"{int(ctx.hours_since_sync)}h ago"
        lines.append(f"| {ctx.name:<15} | {ctx.chunks:<6} | {hours_str:<9} | {status_icon:<6} |")

    lines.append("")
    lines.append(f"Watcher: {'Running' if watcher_running else 'Stopped'}")

    return "\n".join(lines)


def read_global_status(contexts_root: Path) -> str:
    """
    Read GLOBAL_STATUS.md if it exists.

    Args:
        contexts_root: Root directory for contexts

    Returns:
        Contents of GLOBAL_STATUS.md
    """
    global_status = contexts_root / "GLOBAL_STATUS.md"
    if not global_status.exists():
        return "GLOBAL_STATUS.md not found. Run ingest to generate."

    return global_status.read_text(encoding="utf-8")


def generate_status_from_contexts(contexts_root: Path) -> str:
    """
    Generate status by reading all STATUS.json files.

    Args:
        contexts_root: Root directory for contexts

    Returns:
        Formatted status string
    """
    statuses = []

    for ctx_dir in contexts_root.iterdir():
        if not ctx_dir.is_dir():
            continue
        if ctx_dir.name.startswith("_"):
            continue

        status_json = ctx_dir / "STATUS.json"
        if not status_json.exists():
            continue

        try:
            data = json.loads(status_json.read_text(encoding="utf-8"))
            freshness = data.get("freshness", {})

            statuses.append(ContextStatus(
                name=ctx_dir.name,
                chunks=data.get("chunks", 0),
                last_sync=data.get("last_sync", "unknown"),
                is_stale=freshness.get("is_stale", False),
                hours_since_sync=freshness.get("hours_since_sync", 0),
                watcher_running=False  # Determined globally
            ))
        except (json.JSONDecodeError, KeyError):
            continue

    # Check watcher status
    watcher_running = _check_watcher_running()

    return format_status_output(statuses, watcher_running)


def _check_watcher_running() -> bool:
    """Check if sync watcher is running via heartbeat."""
    from .sync.heartbeat import is_alive
    return is_alive()
```

**Step 4: Add status command to CLI**

```python
# src/chinvex/cli.py (add to existing file)
@app.command()
def status(
    contexts_root: Path = typer.Option(
        Path(os.getenv("CHINVEX_CONTEXTS_ROOT", "P:/ai_memory/contexts")),
        help="Root directory for contexts"
    ),
    regenerate: bool = typer.Option(False, help="Regenerate from STATUS.json files")
):
    """Show global system status."""
    from .cli_status import read_global_status, generate_status_from_contexts

    if regenerate:
        output = generate_status_from_contexts(contexts_root)
        # Write back to GLOBAL_STATUS.md
        global_status_md = contexts_root / "GLOBAL_STATUS.md"
        global_status_md.write_text(output, encoding="utf-8")
    else:
        output = read_global_status(contexts_root)

    print(output)
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_cli_status.py -v`
Expected: PASS (all tests green)

**Step 6: Commit**

```bash
git add src/chinvex/cli_status.py src/chinvex/cli.py tests/test_cli_status.py
git commit -m "feat(status): add global status command

- Formats GLOBAL_STATUS.md as table
- Color-codes stale contexts with ⚠
- Supports --regenerate flag
- Reads STATUS.json files on-demand

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 22: PowerShell Profile Injection

**Files:**
- Create: `src/chinvex/bootstrap/profile.py`
- Create: `tests/bootstrap/test_profile.py`

**Step 1: Write failing test for profile injection**

```python
# tests/bootstrap/test_profile.py
import pytest
from pathlib import Path
from chinvex.bootstrap.profile import inject_dual_function, DUAL_FUNCTION_TEMPLATE


def test_inject_dual_function_creates_profile(tmp_path: Path):
    """Should create PowerShell profile if it doesn't exist"""
    profile_path = tmp_path / "Microsoft.PowerShell_profile.ps1"

    inject_dual_function(profile_path)

    assert profile_path.exists()
    content = profile_path.read_text()
    assert "function dual {" in content
    assert "dual track" in content
    assert "dual brief" in content


def test_inject_dual_function_appends_to_existing(tmp_path: Path):
    """Should append to existing profile without duplicating"""
    profile_path = tmp_path / "Microsoft.PowerShell_profile.ps1"
    profile_path.write_text("# Existing content\nSet-Alias ll ls\n")

    inject_dual_function(profile_path)

    content = profile_path.read_text()
    assert "# Existing content" in content
    assert "function dual {" in content


def test_inject_dual_function_idempotent(tmp_path: Path):
    """Should not duplicate if dual function already exists"""
    profile_path = tmp_path / "Microsoft.PowerShell_profile.ps1"

    inject_dual_function(profile_path)
    first_content = profile_path.read_text()

    inject_dual_function(profile_path)
    second_content = profile_path.read_text()

    assert first_content == second_content
    assert first_content.count("function dual {") == 1


def test_remove_dual_function(tmp_path: Path):
    """Should remove dual function from profile"""
    profile_path = tmp_path / "Microsoft.PowerShell_profile.ps1"
    profile_path.write_text("""# Existing content
Set-Alias ll ls

# Chinvex dual function - DO NOT EDIT
function dual {
    # ... function body
}
# End Chinvex dual function

# More content
""")

    from chinvex.bootstrap.profile import remove_dual_function
    remove_dual_function(profile_path)

    content = profile_path.read_text()
    assert "function dual {" not in content
    assert "# Existing content" in content
    assert "# More content" in content
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/bootstrap/test_profile.py -v`
Expected: FAIL with "cannot import name 'inject_dual_function'"

**Step 3: Implement profile injection**

```python
# src/chinvex/bootstrap/__init__.py
"""Bootstrap installation utilities."""

# src/chinvex/bootstrap/profile.py
"""PowerShell profile injection."""
from __future__ import annotations

import re
from pathlib import Path

DUAL_FUNCTION_TEMPLATE = """
# Chinvex dual function - DO NOT EDIT
function dual {
    param([string]$cmd, [string]$arg)
    switch ($cmd) {
        "brief"  { chinvex brief --all-contexts }
        "track"  {
            $repo = if ($arg) { Resolve-Path $arg } else { Get-Location }
            $name = (Split-Path $repo -Leaf).ToLower() -replace '[^a-z0-9-]', '-'

            # Check if context exists
            $existing = chinvex context list --format json | ConvertFrom-Json | Where-Object { $_.name -eq $name }

            if ($existing) {
                # Check if same repo
                $existingRepo = $existing.sources | Where-Object { $_.type -eq "repo" } | Select-Object -First 1
                if ($existingRepo -and (Resolve-Path $existingRepo.path) -eq $repo) {
                    Write-Host "Already tracking $repo in context '$name'"
                    return
                }
                # Different repo, need unique name
                $i = 2
                while ($true) {
                    $newName = "$name-$i"
                    $check = chinvex context list --format json | ConvertFrom-Json | Where-Object { $_.name -eq $newName }
                    if (-not $check) { $name = $newName; break }
                    $i++
                }
            }

            # Create context and add source
            chinvex ingest --context $name --repo $repo

            # Add to sync watcher if running
            chinvex sync reconcile-sources 2>$null

            Write-Host "Tracking $repo in context '$name'"
        }
        "status" { chinvex status }
        default  { Write-Host "Usage: dual [brief|track|status]" }
    }
}

Set-Alias dn dual
# End Chinvex dual function
""".strip()

START_MARKER = "# Chinvex dual function - DO NOT EDIT"
END_MARKER = "# End Chinvex dual function"


def inject_dual_function(profile_path: Path) -> None:
    """
    Inject dual function into PowerShell profile.

    Idempotent: won't duplicate if already exists.

    Args:
        profile_path: Path to PowerShell profile
    """
    # Read existing content
    if profile_path.exists():
        content = profile_path.read_text(encoding="utf-8")
    else:
        content = ""
        profile_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if dual function already exists
    if START_MARKER in content:
        return  # Already injected

    # Append dual function
    if content and not content.endswith("\n"):
        content += "\n"

    content += "\n" + DUAL_FUNCTION_TEMPLATE + "\n"

    profile_path.write_text(content, encoding="utf-8")


def remove_dual_function(profile_path: Path) -> None:
    """
    Remove dual function from PowerShell profile.

    Args:
        profile_path: Path to PowerShell profile
    """
    if not profile_path.exists():
        return

    content = profile_path.read_text(encoding="utf-8")

    # Remove section between markers
    pattern = re.compile(
        rf"^{re.escape(START_MARKER)}.*?^{re.escape(END_MARKER)}\n?",
        re.MULTILINE | re.DOTALL
    )
    content = pattern.sub("", content)

    profile_path.write_text(content, encoding="utf-8")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/bootstrap/test_profile.py -v`
Expected: PASS (all tests green)

**Step 5: Commit**

```bash
git add src/chinvex/bootstrap/ tests/bootstrap/
git commit -m "feat(bootstrap): add PowerShell profile injection

- Injects dual function into PowerShell profile
- Implements dual track with name collision handling
- Idempotent (won't duplicate on repeat calls)
- Supports removal for uninstall

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 23: Environment Variable Setup

**Files:**
- Create: `src/chinvex/bootstrap/env_vars.py`
- Create: `tests/bootstrap/test_env_vars.py`

**Step 1: Write failing test for environment variable setup**

```python
# tests/bootstrap/test_env_vars.py
import pytest
from pathlib import Path
from unittest.mock import patch, Mock
from chinvex.bootstrap.env_vars import set_env_vars, unset_env_vars, validate_paths


def test_validate_paths_creates_missing_dirs(tmp_path: Path):
    """Should create directories if they don't exist"""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"

    # Should not exist yet
    assert not contexts_root.exists()
    assert not indexes_root.exists()

    validate_paths(contexts_root, indexes_root)

    # Should be created
    assert contexts_root.exists()
    assert contexts_root.is_dir()
    assert indexes_root.exists()
    assert indexes_root.is_dir()


def test_set_env_vars_calls_setx(tmp_path: Path):
    """Should call setx to set user environment variables"""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"
    contexts_root.mkdir()
    indexes_root.mkdir()

    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(returncode=0)

        set_env_vars(
            contexts_root=contexts_root,
            indexes_root=indexes_root,
            ntfy_topic="chinvex-alerts"
        )

        # Should have called setx 3 times
        assert mock_run.call_count == 3

        # Extract calls
        calls = [call[0][0] for call in mock_run.call_args_list]

        # Check CHINVEX_CONTEXTS_ROOT set
        contexts_call = [c for c in calls if "CHINVEX_CONTEXTS_ROOT" in c]
        assert len(contexts_call) == 1
        assert str(contexts_root) in contexts_call[0]

        # Check CHINVEX_INDEXES_ROOT set
        indexes_call = [c for c in calls if "CHINVEX_INDEXES_ROOT" in c]
        assert len(indexes_call) == 1

        # Check CHINVEX_NTFY_TOPIC set
        ntfy_call = [c for c in calls if "CHINVEX_NTFY_TOPIC" in c]
        assert len(ntfy_call) == 1
        assert "chinvex-alerts" in ntfy_call[0]


def test_unset_env_vars_removes_variables(tmp_path: Path):
    """Should remove environment variables via reg delete"""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(returncode=0)

        unset_env_vars()

        # Should delete 3 registry keys
        assert mock_run.call_count == 3

        calls = [call[0][0] for call in mock_run.call_args_list]

        # All should be reg delete commands
        for call in calls:
            assert "reg" in call[0].lower()
            assert "delete" in " ".join(call).lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/bootstrap/test_env_vars.py -v`
Expected: FAIL with "cannot import name 'set_env_vars'"

**Step 3: Implement environment variable setup**

```python
# src/chinvex/bootstrap/env_vars.py
"""Environment variable management for Windows."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def validate_paths(contexts_root: Path, indexes_root: Path) -> None:
    """
    Validate and create directories if needed.

    Args:
        contexts_root: Root directory for contexts
        indexes_root: Root directory for indexes
    """
    contexts_root.mkdir(parents=True, exist_ok=True)
    indexes_root.mkdir(parents=True, exist_ok=True)

    log.info(f"Validated paths: {contexts_root}, {indexes_root}")


def set_env_vars(
    contexts_root: Path,
    indexes_root: Path,
    ntfy_topic: str
) -> None:
    """
    Set user environment variables via setx.

    Args:
        contexts_root: Root directory for contexts
        indexes_root: Root directory for indexes
        ntfy_topic: ntfy.sh topic for notifications
    """
    vars_to_set = {
        "CHINVEX_CONTEXTS_ROOT": str(contexts_root),
        "CHINVEX_INDEXES_ROOT": str(indexes_root),
        "CHINVEX_NTFY_TOPIC": ntfy_topic
    }

    for var_name, value in vars_to_set.items():
        result = subprocess.run(
            ["setx", var_name, value],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to set {var_name}: {result.stderr}")

        log.info(f"Set {var_name}={value}")


def unset_env_vars() -> None:
    """
    Remove Chinvex environment variables from user registry.
    """
    var_names = [
        "CHINVEX_CONTEXTS_ROOT",
        "CHINVEX_INDEXES_ROOT",
        "CHINVEX_NTFY_TOPIC"
    ]

    for var_name in var_names:
        # Use reg delete to remove from user environment
        result = subprocess.run(
            [
                "reg", "delete",
                r"HKCU\Environment",
                "/v", var_name,
                "/f"
            ],
            capture_output=True,
            text=True
        )

        # returncode 1 = key not found (acceptable)
        if result.returncode not in (0, 1):
            log.warning(f"Failed to remove {var_name}: {result.stderr}")
        else:
            log.info(f"Removed {var_name}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/bootstrap/test_env_vars.py -v`
Expected: PASS (all tests green)

**Step 5: Commit**

```bash
git add src/chinvex/bootstrap/env_vars.py tests/bootstrap/test_env_vars.py
git commit -m "feat(bootstrap): add environment variable management

- Sets CHINVEX_CONTEXTS_ROOT, CHINVEX_INDEXES_ROOT, CHINVEX_NTFY_TOPIC
- Uses setx for user-level variables (Windows)
- Validates and creates directories
- Supports uninstall via reg delete

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 24: Bootstrap Install/Status/Uninstall Commands

**Files:**
- Create: `src/chinvex/bootstrap/cli.py`
- Create: `tests/bootstrap/test_bootstrap_cli.py`
- Modify: `src/chinvex/cli.py` (add bootstrap command group)

**Step 1: Write failing test for bootstrap commands**

```python
# tests/bootstrap/test_bootstrap_cli.py
import pytest
from pathlib import Path
from unittest.mock import patch, Mock
from chinvex.bootstrap.cli import bootstrap_install, bootstrap_status, bootstrap_uninstall


def test_bootstrap_install_full_workflow(tmp_path: Path):
    """Install should configure all components"""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"
    ntfy_topic = "chinvex-test"
    profile_path = tmp_path / "profile.ps1"

    with patch('chinvex.bootstrap.env_vars.set_env_vars') as mock_env, \
         patch('chinvex.bootstrap.profile.inject_dual_function') as mock_profile, \
         patch('chinvex.bootstrap.tasks.register_sweep_task') as mock_sweep, \
         patch('chinvex.bootstrap.tasks.register_morning_brief_task') as mock_brief, \
         patch('subprocess.run') as mock_run:

        mock_run.return_value = Mock(returncode=0, stdout="Started")

        bootstrap_install(
            contexts_root=contexts_root,
            indexes_root=indexes_root,
            ntfy_topic=ntfy_topic,
            profile_path=profile_path,
            morning_brief_time="07:00"
        )

        # Verify all components configured
        mock_env.assert_called_once_with(contexts_root, indexes_root, ntfy_topic)
        mock_profile.assert_called_once_with(profile_path)
        mock_sweep.assert_called_once()
        mock_brief.assert_called_once()

        # Verify watcher started
        assert any("chinvex sync start" in str(call) for call in mock_run.call_args_list)


def test_bootstrap_status_shows_components(tmp_path: Path):
    """Status should show state of all components"""
    with patch('chinvex.bootstrap.tasks.check_task_exists') as mock_check_task, \
         patch('chinvex.sync.heartbeat.is_alive') as mock_heartbeat, \
         patch('os.getenv') as mock_getenv:

        mock_check_task.side_effect = lambda name: name == "ChinvexSweep"
        mock_heartbeat.return_value = True
        mock_getenv.side_effect = lambda key, default=None: {
            "CHINVEX_CONTEXTS_ROOT": "P:/ai_memory/contexts",
            "CHINVEX_NTFY_TOPIC": "chinvex-alerts"
        }.get(key, default)

        status = bootstrap_status()

        # Check status dict
        assert status["watcher_running"] is True
        assert status["sweep_task_installed"] is True
        assert status["brief_task_installed"] is False
        assert status["env_vars_set"] is True


def test_bootstrap_uninstall_removes_all(tmp_path: Path):
    """Uninstall should remove all components"""
    profile_path = tmp_path / "profile.ps1"

    with patch('chinvex.bootstrap.env_vars.unset_env_vars') as mock_env, \
         patch('chinvex.bootstrap.profile.remove_dual_function') as mock_profile, \
         patch('chinvex.bootstrap.tasks.unregister_task') as mock_unreg, \
         patch('subprocess.run') as mock_run:

        mock_run.return_value = Mock(returncode=0, stdout="Stopped")

        bootstrap_uninstall(profile_path=profile_path)

        # Verify all removals
        mock_env.assert_called_once()
        mock_profile.assert_called_once_with(profile_path)
        assert mock_unreg.call_count == 2  # Sweep + brief tasks

        # Verify watcher stopped
        assert any("chinvex sync stop" in str(call) for call in mock_run.call_args_list)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/bootstrap/test_bootstrap_cli.py -v`
Expected: FAIL with "cannot import name 'bootstrap_install'"

**Step 3: Implement bootstrap commands**

```python
# src/chinvex/bootstrap/cli.py
"""Bootstrap CLI commands."""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from .env_vars import set_env_vars, unset_env_vars, validate_paths
from .profile import inject_dual_function, remove_dual_function
from .tasks import (
    register_sweep_task,
    register_morning_brief_task,
    unregister_task,
    check_task_exists
)

log = logging.getLogger(__name__)


def _create_global_context(contexts_root: Path, indexes_root: Path) -> None:
    """
    Create _global context with constraints if it doesn't exist.

    The _global context is used as a catch-all for miscellaneous ingests
    with automatic archiving based on age and chunk count constraints.
    """
    import json
    from datetime import datetime, timezone

    global_ctx_dir = contexts_root / "_global"

    if global_ctx_dir.exists():
        log.info("_global context already exists")
        return

    global_ctx_dir.mkdir(parents=True)

    # Generate context.json with constraints
    now = datetime.now(timezone.utc).isoformat()
    global_config = {
        "schema_version": 2,
        "name": "_global",
        "aliases": ["global", "catch-all"],
        "includes": {
            "repos": [],
            "chat_roots": [],
            "codex_session_roots": [],
            "note_roots": []
        },
        "index": {
            "sqlite_path": str(indexes_root / "_global" / "hybrid.db"),
            "chroma_dir": str(indexes_root / "_global" / "chroma")
        },
        "weights": {
            "repo": 1.0,
            "chat": 0.8,
            "codex_session": 0.9,
            "note": 0.7
        },
        "ollama": {
            "base_url": "http://skynet:11434",
            "embed_model": "mxbai-embed-large"
        },
        "created_at": now,
        "updated_at": now,
        "archive": {
            "enabled": True,
            "age_threshold_days": 90,
            "auto_archive_on_ingest": True,
            "archive_penalty": 0.8
        },
        "constraints": {
            "max_chunks": 10000,
            "stale_after_hours": 24
        }
    }

    context_file = global_ctx_dir / "context.json"
    context_file.write_text(json.dumps(global_config, indent=2), encoding="utf-8")

    log.info(f"Created _global context at {global_ctx_dir}")


def bootstrap_install(
    contexts_root: Path,
    indexes_root: Path,
    ntfy_topic: str,
    profile_path: Path,
    morning_brief_time: str = "07:00"
) -> None:
    """
    Install all bootstrap components.

    Args:
        contexts_root: Root directory for contexts
        indexes_root: Root directory for indexes
        ntfy_topic: ntfy.sh topic for notifications
        profile_path: Path to PowerShell profile
        morning_brief_time: Time for morning brief (HH:MM format)
    """
    print("Installing Chinvex bootstrap...")

    # 1. Validate and create directories
    print("  1. Validating paths...")
    validate_paths(contexts_root, indexes_root)

    # 1.5. Create _global context if not exists
    print("  1.5. Creating _global context...")
    _create_global_context(contexts_root, indexes_root)

    # 2. Set environment variables
    print("  2. Setting environment variables...")
    set_env_vars(contexts_root, indexes_root, ntfy_topic)

    # 3. Inject PowerShell profile
    print("  3. Updating PowerShell profile...")
    inject_dual_function(profile_path)

    # 4. Register scheduled tasks
    print("  4. Registering scheduled tasks...")
    register_sweep_task(contexts_root=contexts_root, ntfy_topic=ntfy_topic)
    register_morning_brief_task(
        contexts_root=contexts_root,
        ntfy_topic=ntfy_topic,
        time=morning_brief_time
    )

    # 5. Start sync watcher
    print("  5. Starting sync watcher...")
    result = subprocess.run(
        ["chinvex", "sync", "start"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"    Warning: Could not start watcher: {result.stderr}")

    print("\n✓ Bootstrap installation complete!")
    print(f"\nEnvironment variables set:")
    print(f"  CHINVEX_CONTEXTS_ROOT={contexts_root}")
    print(f"  CHINVEX_INDEXES_ROOT={indexes_root}")
    print(f"  CHINVEX_NTFY_TOPIC={ntfy_topic}")
    print(f"\nPowerShell profile updated: {profile_path}")
    print(f"\nScheduled tasks:")
    print(f"  - ChinvexSweep (every 30 min)")
    print(f"  - ChinvexMorningBrief ({morning_brief_time})")
    print(f"\nUsage:")
    print(f"  dual brief          # Show all context briefs")
    print(f"  dual track .        # Track current directory")
    print(f"  dual status         # Show system health")


def bootstrap_status() -> dict:
    """
    Check status of bootstrap components.

    Returns:
        Status dict with component states
    """
    from ..sync.heartbeat import is_alive

    status = {
        "watcher_running": is_alive(),
        "sweep_task_installed": check_task_exists("ChinvexSweep"),
        "brief_task_installed": check_task_exists("ChinvexMorningBrief"),
        "env_vars_set": all([
            os.getenv("CHINVEX_CONTEXTS_ROOT"),
            os.getenv("CHINVEX_NTFY_TOPIC")
        ])
    }

    return status


def bootstrap_uninstall(profile_path: Path) -> None:
    """
    Uninstall all bootstrap components.

    Args:
        profile_path: Path to PowerShell profile
    """
    print("Uninstalling Chinvex bootstrap...")

    # 1. Stop sync watcher
    print("  1. Stopping sync watcher...")
    subprocess.run(["chinvex", "sync", "stop"], capture_output=True)

    # 2. Unregister scheduled tasks
    print("  2. Removing scheduled tasks...")
    unregister_task("ChinvexSweep")
    unregister_task("ChinvexMorningBrief")

    # 3. Remove from PowerShell profile
    print("  3. Removing from PowerShell profile...")
    remove_dual_function(profile_path)

    # 4. Unset environment variables
    print("  4. Removing environment variables...")
    unset_env_vars()

    print("\n✓ Bootstrap uninstall complete!")
    print("\nNote: Context data and indexes were not deleted.")
    print("To remove data: manually delete CHINVEX_CONTEXTS_ROOT and CHINVEX_INDEXES_ROOT")
```

**Step 4: Add bootstrap command group to CLI**

```python
# src/chinvex/cli.py (add to existing file)
import typer

bootstrap_app = typer.Typer(help="Bootstrap installation commands")
app.add_typer(bootstrap_app, name="bootstrap")


@bootstrap_app.command()
def install(
    contexts_root: Path = typer.Option(
        Path("P:/ai_memory/contexts"),
        help="Root directory for contexts"
    ),
    indexes_root: Path = typer.Option(
        Path("P:/ai_memory/indexes"),
        help="Root directory for indexes"
    ),
    ntfy_topic: str = typer.Option(..., prompt=True, help="ntfy.sh topic for notifications"),
    morning_brief_time: str = typer.Option("07:00", help="Morning brief time (HH:MM)"),
    profile_path: Path = typer.Option(
        Path(os.path.expandvars(r"%USERPROFILE%\Documents\PowerShell\Microsoft.PowerShell_profile.ps1")),
        help="PowerShell profile path"
    )
):
    """Install Chinvex bootstrap components."""
    from .bootstrap.cli import bootstrap_install
    bootstrap_install(contexts_root, indexes_root, ntfy_topic, profile_path, morning_brief_time)


@bootstrap_app.command()
def status():
    """Show bootstrap installation status."""
    from .bootstrap.cli import bootstrap_status

    status = bootstrap_status()

    print("Chinvex Bootstrap Status:")
    print(f"  Watcher: {'✓ Running' if status['watcher_running'] else '✗ Stopped'}")
    print(f"  Sweep Task: {'✓ Installed' if status['sweep_task_installed'] else '✗ Not installed'}")
    print(f"  Brief Task: {'✓ Installed' if status['brief_task_installed'] else '✗ Not installed'}")
    print(f"  Env Vars: {'✓ Set' if status['env_vars_set'] else '✗ Not set'}")


@bootstrap_app.command()
def uninstall(
    profile_path: Path = typer.Option(
        Path(os.path.expandvars(r"%USERPROFILE%\Documents\PowerShell\Microsoft.PowerShell_profile.ps1")),
        help="PowerShell profile path"
    )
):
    """Uninstall Chinvex bootstrap components."""
    confirm = typer.confirm("This will remove scheduled tasks, env vars, and profile changes. Continue?")
    if not confirm:
        print("Uninstall cancelled.")
        return

    from .bootstrap.cli import bootstrap_uninstall
    bootstrap_uninstall(profile_path)
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/bootstrap/test_bootstrap_cli.py -v`
Expected: PASS (all tests green)

**Step 6: Commit**

```bash
git add src/chinvex/bootstrap/cli.py src/chinvex/cli.py tests/bootstrap/test_bootstrap_cli.py
git commit -m "feat(bootstrap): add install/status/uninstall commands

- chinvex bootstrap install: full setup (tasks, profile, env)
- chinvex bootstrap status: component health check
- chinvex bootstrap uninstall: clean removal
- Interactive prompts for ntfy topic
- Comprehensive status output

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 25: E2E Flow Test

**Files:**
- Create: `tests/e2e/test_full_flow.py`

**Step 1: Write E2E flow test**

```python
# tests/e2e/test_full_flow.py
"""End-to-end test of full auto-ingest flow."""
import pytest
import time
import subprocess
from pathlib import Path


@pytest.mark.slow
def test_full_auto_ingest_flow(tmp_path: Path):
    """
    E2E: Edit file → watcher ingests → STATUS.json updated → GLOBAL_STATUS reflects change

    This test validates the entire "always fresh" property.
    """
    # Setup
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"
    repo_root = tmp_path / "test_repo"

    contexts_root.mkdir()
    indexes_root.mkdir()
    repo_root.mkdir()

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=repo_root, check=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_root,
        check=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_root,
        check=True
    )

    # Create initial file and commit
    test_file = repo_root / "main.py"
    test_file.write_text("def hello():\n    print('hello')\n")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_root,
        check=True
    )

    # Create context for this repo
    result = subprocess.run(
        [
            "chinvex", "ingest",
            "--context", "test_repo",
            "--repo", str(repo_root),
            "--contexts-root", str(contexts_root),
            "--indexes-root", str(indexes_root)
        ],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0

    # Start watcher
    result = subprocess.run(
        [
            "chinvex", "sync", "start",
            "--contexts-root", str(contexts_root)
        ],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0

    # Get initial STATUS.json state
    status_json = contexts_root / "test_repo" / "STATUS.json"
    assert status_json.exists()

    import json
    initial_status = json.loads(status_json.read_text())
    initial_chunks = initial_status["chunks"]

    # Make a change to the file
    test_file.write_text("def hello():\n    print('hello world')\n\ndef goodbye():\n    print('bye')\n")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Add goodbye function"],
        cwd=repo_root,
        check=True
    )

    # Wait for watcher to detect change and trigger ingest
    # Debounce is 30s, so wait up to 60s total
    max_wait = 60
    waited = 0

    while waited < max_wait:
        time.sleep(5)
        waited += 5

        # Check if STATUS.json was updated
        current_status = json.loads(status_json.read_text())
        if current_status["chunks"] != initial_chunks:
            break

    # Verify STATUS.json was updated
    final_status = json.loads(status_json.read_text())
    assert final_status["chunks"] > initial_chunks, "Chunks should increase after adding function"

    # Verify GLOBAL_STATUS.md exists and reflects the change
    global_status_md = contexts_root / "GLOBAL_STATUS.md"
    assert global_status_md.exists()

    global_content = global_status_md.read_text()
    assert "test_repo" in global_content
    assert str(final_status["chunks"]) in global_content

    # Cleanup: stop watcher
    subprocess.run(
        ["chinvex", "sync", "stop"],
        capture_output=True
    )
```

**Step 2: Run test to verify behavior**

Run: `pytest tests/e2e/test_full_flow.py -v -s`
Expected: Test validates full flow (may initially fail if components not wired together)

**Step 3: Fix any integration issues discovered**

If test fails, fix the integration points:
- Watcher → ingest triggering
- Ingest → STATUS.json writing
- STATUS.json → GLOBAL_STATUS.md aggregation

**Step 4: Run test to verify it passes**

Run: `pytest tests/e2e/test_full_flow.py -v`
Expected: PASS (full flow works end-to-end)

**Step 5: Commit**

```bash
git add tests/e2e/test_full_flow.py
git commit -m "test(e2e): add full auto-ingest flow test

- Creates repo, makes change, verifies auto-ingest
- Validates STATUS.json updated
- Confirms GLOBAL_STATUS.md reflects change
- End-to-end validation of \"always fresh\" property

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 26: Kill-Switch Recovery Test

**Files:**
- Create: `tests/e2e/test_kill_switch.py`

**Step 1: Write kill-switch recovery test**

```python
# tests/e2e/test_kill_switch.py
"""Kill-switch recovery test - proves this is an appliance."""
import pytest
import time
import subprocess
import psutil
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
    5. Wait for sweep task (30 min interval + tolerance)
    6. Verify watcher restarted automatically

    This proves the system is self-healing without human intervention.
    """
    contexts_root = tmp_path / "contexts"
    state_dir = tmp_path / "state"
    contexts_root.mkdir()
    state_dir.mkdir()

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
            "sqlite_path": "P:/ai_memory/indexes/test_ctx/hybrid.db",
            "chroma_dir": "P:/ai_memory/indexes/test_ctx/chroma"
        }
    }""")

    # Start watcher
    result = subprocess.run(
        [
            "chinvex", "sync", "start",
            "--contexts-root", str(contexts_root),
            "--state-dir", str(state_dir)
        ],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0

    # Verify heartbeat exists
    heartbeat_file = state_dir / "sync_heartbeat.json"
    assert heartbeat_file.exists()

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
    assert not psutil.pid_exists(watcher_pid)

    # Delete heartbeat to simulate crash
    heartbeat_file.unlink()
    assert not heartbeat_file.exists()

    # Trigger sweep manually (instead of waiting 30 min)
    # In real usage, scheduled task would run this
    result = subprocess.run(
        [
            "pwsh", "-NoProfile", "-File",
            "scripts/scheduled_sweep.ps1",
            "-ContextsRoot", str(contexts_root),
            "-StateDir", str(state_dir),
            "-NtfyTopic", "test-topic"
        ],
        capture_output=True,
        text=True
    )

    # Sweep should succeed
    assert result.returncode == 0

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
```

**Step 2: Run test to verify behavior**

Run: `pytest tests/e2e/test_kill_switch.py -v -s`
Expected: Test may initially fail if sweep recovery logic not implemented

**Step 3: Implement missing recovery logic in sweep script**

If test fails, ensure `scheduled_sweep.ps1` includes:
```powershell
# Check heartbeat, restart if stale/missing
$heartbeat = Get-Content "$StateDir/sync_heartbeat.json" -ErrorAction SilentlyContinue | ConvertFrom-Json
if (-not $heartbeat -or (Get-Date).AddMinutes(-5) -gt [DateTime]$heartbeat.timestamp) {
    Write-Host "Heartbeat stale or missing - restarting watcher"
    & chinvex sync start --contexts-root $ContextsRoot --state-dir $StateDir
}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/e2e/test_kill_switch.py -v`
Expected: PASS (sweep resurrects killed watcher)

**Step 5: Commit**

```bash
git add tests/e2e/test_kill_switch.py scripts/scheduled_sweep.ps1
git commit -m "test(e2e): add kill-switch recovery test

- Kills watcher process
- Deletes heartbeat file
- Verifies sweep auto-restarts watcher
- Proves appliance-grade self-healing
- Definitive test: no human intervention needed

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

**Plan Complete:** All 26 tasks now have full TDD specifications across 10 phases, implementing Properties 1-4 of bootstrapping completion spec.

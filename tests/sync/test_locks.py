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

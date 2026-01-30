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

    # Ensure directory exists
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        lock = try_acquire_sync_lock(lock_file)
        lock.release()
        return False  # Was able to acquire, so not held
    except SyncLockHeld:
        return True  # Lock is held

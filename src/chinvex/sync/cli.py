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
                    # Windows: use taskkill (hidden window)
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(pid)],
                        check=False,
                        capture_output=True,
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
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
                        subprocess.run(
                            ["taskkill", "/F", "/PID", str(pid)],
                            check=False,
                            capture_output=True,
                            creationflags=subprocess.CREATE_NO_WINDOW
                        )
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


def _start_daemon_process(state_dir: Path) -> None:
    """Start daemon process in background."""
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

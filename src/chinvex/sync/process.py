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

        # Map context_name -> watch_root for exclusion checks
        self._context_roots: Dict[str, Path] = {
            source.context_name: source.path for source in sources
        }

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
        # Double-check exclusion (defensive, handler should have already checked)
        watch_root = self._context_roots.get(context_name)
        if watch_root and should_exclude(path, watch_root):
            log.debug(f"Excluded: {path}")
            return

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

        Spawns background chinvex ingest process.
        """
        import sys
        from .locks import check_context_lock_held

        # Check if ingest lock is held
        if check_context_lock_held(self.contexts_root, context_name):
            log.info(f"Skipping ingest for {context_name}: lock held")
            # Don't clear accumulator - will retry later
            return

        # Check if over limit BEFORE clearing
        is_over_limit = accumulator.is_over_limit()
        changes = accumulator.get_and_clear()

        # Build command
        python_exe = sys.executable
        cmd = [python_exe, "-m", "chinvex.cli", "ingest", "--context", context_name]

        if is_over_limit:
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

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

    print("\n[OK] Bootstrap installation complete!")
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
    try:
        from ..sync.heartbeat import is_alive
        watcher_running = is_alive()
    except ImportError:
        watcher_running = False

    status = {
        "watcher_running": watcher_running,
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

    print("\n[OK] Bootstrap uninstall complete!")
    print("\nNote: Context data and indexes were not deleted.")
    print("To remove data: manually delete CHINVEX_CONTEXTS_ROOT and CHINVEX_INDEXES_ROOT")

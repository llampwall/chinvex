"""Task scheduler wrapper for bootstrap commands."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .scheduler import (
    register_sweep_task as _register_sweep_task,
    register_morning_brief_task as _register_morning_brief_task,
    unregister_sweep_task,
    unregister_morning_brief_task,
)


def register_sweep_task(contexts_root: Path, ntfy_topic: str = "") -> None:
    """
    Register scheduled sweep task.

    Args:
        contexts_root: Contexts root directory
        ntfy_topic: ntfy.sh topic for alerts
    """
    # Find the scheduled_sweep.ps1 script
    script_path = Path(__file__).parent.parent.parent / "scripts" / "scheduled_sweep.ps1"
    if not script_path.exists():
        raise FileNotFoundError(f"Sweep script not found: {script_path}")

    _register_sweep_task(script_path, contexts_root, ntfy_topic)


def register_morning_brief_task(
    contexts_root: Path,
    ntfy_topic: str,
    time: str = "07:00"
) -> None:
    """
    Register morning brief task.

    Args:
        contexts_root: Contexts root directory
        ntfy_topic: ntfy.sh topic for morning brief
        time: Time to run (HH:MM format)
    """
    _register_morning_brief_task(contexts_root, ntfy_topic, time)


def unregister_task(task_name: str) -> None:
    """
    Unregister a scheduled task by name.

    Args:
        task_name: Task name (e.g., "ChinvexSweep", "ChinvexMorningBrief")
    """
    if task_name == "ChinvexSweep":
        unregister_sweep_task()
    elif task_name == "ChinvexMorningBrief":
        unregister_morning_brief_task()
    else:
        raise ValueError(f"Unknown task: {task_name}")


def check_task_exists(task_name: str) -> bool:
    """
    Check if a scheduled task exists.

    Args:
        task_name: Task name to check

    Returns:
        True if task exists, False otherwise
    """
    result = subprocess.run(
        ["schtasks", "/Query", "/TN", task_name],
        capture_output=True,
        text=True
    )
    return result.returncode == 0

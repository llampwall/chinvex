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

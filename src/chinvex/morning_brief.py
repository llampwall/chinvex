from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path


def detect_active_stale_contexts(
    contexts_root: Path,
    max_active: int = 5,
    active_threshold_days: int = 7
) -> tuple[list[dict], list[dict]]:
    """
    Detect active and stale contexts based on last_sync timestamp.

    Args:
        contexts_root: Path to contexts directory
        max_active: Maximum number of active contexts to return (sorted by recency)
        active_threshold_days: Days threshold for active vs stale

    Returns:
        (active_contexts, stale_contexts) where each is a list of status dicts
    """
    cutoff = datetime.now() - timedelta(days=active_threshold_days)

    all_contexts = []

    # Collect all STATUS.json files
    for ctx_dir in contexts_root.iterdir():
        if not ctx_dir.is_dir():
            continue

        status_file = ctx_dir / "STATUS.json"
        if not status_file.exists():
            continue

        try:
            status = json.loads(status_file.read_text())
            all_contexts.append(status)
        except (json.JSONDecodeError, OSError):
            continue

    # Separate into active and stale
    active = []
    stale = []

    for status in all_contexts:
        last_sync_str = status.get("last_sync")

        if not last_sync_str:
            stale.append(status)
            continue

        try:
            last_sync = datetime.fromisoformat(last_sync_str.replace("Z", "+00:00"))
            # Remove timezone info for comparison
            if last_sync.tzinfo:
                last_sync = last_sync.replace(tzinfo=None)

            if last_sync >= cutoff:
                active.append(status)
            else:
                stale.append(status)
        except (ValueError, AttributeError):
            stale.append(status)

    # Sort active by last_sync (most recent first)
    active.sort(key=lambda s: s.get("last_sync", ""), reverse=True)

    # Cap at max_active
    active = active[:max_active]

    return active, stale


def parse_state_md(
    state_md_path: Path,
    max_actions: int = 5
) -> tuple[str | None, list[str]]:
    """
    Parse STATE.md to extract Current Objective and Next Actions.

    Args:
        state_md_path: Path to STATE.md file
        max_actions: Maximum number of actions to return

    Returns:
        (objective, actions) where objective is a string or None,
        and actions is a list of action strings (without checkbox prefix)
    """
    if not state_md_path.exists():
        return None, []

    try:
        content = state_md_path.read_text()
    except OSError:
        return None, []

    lines = content.split("\n")

    objective = None
    actions = []

    in_objective_section = False
    in_actions_section = False

    for line in lines:
        # Detect sections
        if line.strip() == "## Current Objective":
            in_objective_section = True
            in_actions_section = False
            continue

        if line.strip() == "## Next Actions":
            in_objective_section = False
            in_actions_section = True
            continue

        # Stop at next section
        if line.startswith("## "):
            in_objective_section = False
            in_actions_section = False
            continue

        # Extract objective (first non-empty line after header)
        if in_objective_section and not objective:
            stripped = line.strip()
            if stripped:
                objective = stripped
                in_objective_section = False  # Only take first line

        # Extract actions (checkbox items)
        if in_actions_section:
            # Match "- [ ] Action text" or "- [x] Action text"
            match = re.match(r"^-\s*\[[ xX]\]\s*(.+)", line.strip())
            if match:
                action_text = match.group(1).strip()
                actions.append(action_text)

                if len(actions) >= max_actions:
                    break

    return objective, actions

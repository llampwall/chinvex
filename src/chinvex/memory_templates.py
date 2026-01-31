# src/chinvex/memory_templates.py
from __future__ import annotations

import datetime
from pathlib import Path


def get_state_template(commit_hash: str = "unknown") -> str:
    """Return STATE.md template with coverage anchor."""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    return f"""# State

## Current Objective
Unknown (needs human)

## Active Work
- None

## Blockers
None

## Next Actions
- [ ] Run chinvex update-memory to populate this file

## Out of Scope (for now)
- TBD

---
Last memory update: {today}
Commits covered through: {commit_hash}

<!-- chinvex:last-commit:{commit_hash} -->
"""


def get_constraints_template() -> str:
    """Return CONSTRAINTS.md template with core sections."""
    return """# Constraints

## Infrastructure
- TBD

## Rules
- TBD

## Key Facts
- TBD

## Hazards
- TBD

## Superseded
(None yet)
"""


def get_decisions_template() -> str:
    """Return DECISIONS.md template with current month section."""
    current_month = datetime.datetime.now().strftime("%Y-%m")
    return f"""# Decisions

## Recent (last 30 days)
- TBD

## {current_month}
(No decisions recorded yet)
"""


def bootstrap_memory_files(repo_root: Path, initial_commit_hash: str = "unknown") -> None:
    """Create docs/memory/ with STATE.md, CONSTRAINTS.md, DECISIONS.md if they don't exist.

    Args:
        repo_root: Root of the git repository
        initial_commit_hash: Starting commit hash for coverage anchor
    """
    memory_dir = repo_root / "docs" / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    # Create STATE.md if missing
    state_file = memory_dir / "STATE.md"
    if not state_file.exists():
        state_file.write_text(get_state_template(initial_commit_hash))

    # Create CONSTRAINTS.md if missing
    constraints_file = memory_dir / "CONSTRAINTS.md"
    if not constraints_file.exists():
        constraints_file.write_text(get_constraints_template())

    # Create DECISIONS.md if missing
    decisions_file = memory_dir / "DECISIONS.md"
    if not decisions_file.exists():
        decisions_file.write_text(get_decisions_template())

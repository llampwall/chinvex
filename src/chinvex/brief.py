from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from prometheus_client import Counter

BRIEF_GENERATED = Counter(
    "chinvex_brief_generated_total",
    "Total briefs generated",
    ["context"]
)


def generate_brief(
    context_name: str,
    state_md: Path | None,
    constraints_md: Path | None,
    decisions_md: Path | None,
    latest_digest: Path | None,
    watch_history_log: Path | None,
    output: Path
) -> None:
    """
    Generate session brief from memory files and recent activity.
    Missing files are silently skipped (graceful degradation).
    """
    BRIEF_GENERATED.labels(context=context_name).inc()

    lines = [
        f"# Session Brief: {context_name}",
        f"Generated: {datetime.utcnow().isoformat()}",
        "",
    ]

    # Check if memory files are uninitialized
    uninitialized = _is_uninitialized(state_md, constraints_md, decisions_md)

    # STATE.md: full content
    if state_md and state_md.exists():
        state_content = _extract_state_sections(state_md)
        if state_content:
            lines.extend(state_content)
            lines.append("")

    # CONSTRAINTS.md: top section only (until first ##)
    if constraints_md and constraints_md.exists():
        constraints_content = _extract_constraints_top(constraints_md)
        if constraints_content:
            lines.append("## Constraints (highlights)")
            lines.extend(constraints_content)
            lines.append("")

    # DECISIONS.md: last 7 days
    if decisions_md and decisions_md.exists():
        recent_decisions = _extract_recent_decisions(decisions_md, days=7)
        if recent_decisions:
            lines.append("## Recent Decisions (7d)")
            lines.extend(recent_decisions)
            lines.append("")

    # Latest digest
    if latest_digest and latest_digest.exists():
        digest_summary = _extract_digest_summary(latest_digest)
        if digest_summary:
            lines.append("## Recent Activity")
            lines.extend(digest_summary)
            lines.append("")

    # Watch history: last 5 hits or 24h
    if watch_history_log and watch_history_log.exists():
        watch_summary = _extract_watch_summary(watch_history_log)
        if watch_summary:
            lines.append("## Recent Watch Hits")
            lines.extend(watch_summary)
            lines.append("")

    # Warning if memory files are uninitialized
    if uninitialized:
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("**[!] Memory files for this context are empty or uninitialized.**")
        lines.append("")
        lines.append("Run `/update-memory` to generate them from git history.")
        lines.append("")

    # Context files reference
    lines.append("## Context Files")
    if state_md:
        lines.append(f"- State: `{state_md}`")
    if latest_digest:
        lines.append(f"- Digest: `{latest_digest}`")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def _is_uninitialized(
    state_md: Path | None,
    constraints_md: Path | None,
    decisions_md: Path | None
) -> bool:
    """
    Check if memory files are uninitialized (missing or still using bootstrap template).

    Returns True if:
    - STATE.md is missing or has bootstrap markers ("Unknown (needs human)", "Needs triage")
    - CONSTRAINTS.md is missing or has placeholder text
    - DECISIONS.md is missing or has placeholder text
    """
    # If STATE.md is missing, definitely uninitialized
    if not state_md or not state_md.exists():
        return True

    # Read STATE.md and check for bootstrap markers
    content = state_md.read_text()

    # Count bootstrap markers (both old and new templates)
    markers = [
        "Unknown (needs human)",
        "Needs triage",
        "Review this file and update with current project state",
        "Fill in Quick Reference with actual commands",
        "Run chinvex update-memory to populate this file",  # Old template
        "- TBD",  # Generic placeholder
    ]

    marker_count = sum(1 for marker in markers if marker in content)

    # If 2+ bootstrap markers are present, it's still uninitialized
    if marker_count >= 2:
        return True

    # Check CONSTRAINTS.md for placeholder text
    if constraints_md and constraints_md.exists():
        constraints_content = constraints_md.read_text()
        if "(Technical limits, batch sizes, ports, paths)" in constraints_content:
            return True

    # Check DECISIONS.md for placeholder text
    if decisions_md and decisions_md.exists():
        decisions_content = decisions_md.read_text()
        if "(5-10 bullet summary of recent decisions — rewritable)" in decisions_content:
            return True

    return False


def _extract_state_sections(state_md: Path) -> list[str]:
    """Extract all sections from STATE.md."""
    content = state_md.read_text()
    lines = content.split("\n")

    result = []
    for line in lines:
        if line.startswith("# State"):
            continue  # Skip title
        result.append(line)

    return result


def _extract_constraints_top(constraints_md: Path) -> list[str]:
    """Extract only Infrastructure, Rules, and Hazards sections from CONSTRAINTS.md."""
    content = constraints_md.read_text()
    lines = content.split("\n")

    # Exact headers to extract
    target_headers = {"## Infrastructure", "## Rules", "## Hazards"}

    result = []
    in_target_section = False
    current_section_lines = []

    for line in lines:
        if line.startswith("# Constraints"):
            continue  # Skip title

        if line.startswith("## "):
            # Save previous section if it was a target
            if in_target_section and current_section_lines:
                result.extend(current_section_lines)
                result.append("")  # Blank line between sections

            # Check if this is a target section
            if line in target_headers:
                in_target_section = True
                current_section_lines = [line]
            else:
                in_target_section = False
                current_section_lines = []
        elif in_target_section:
            current_section_lines.append(line)

    # Save last section if it was a target
    if in_target_section and current_section_lines:
        result.extend(current_section_lines)

    return result


def _extract_recent_decisions(decisions_md: Path, days: int) -> list[str]:
    """Extract Recent rollup section + dated entries from last N days."""
    content = decisions_md.read_text()
    lines = content.split("\n")

    cutoff_date = datetime.now() - timedelta(days=days)
    result = []

    # Extract Recent rollup section first
    in_recent_rollup = False
    rollup_lines = []

    for i, line in enumerate(lines):
        if line.strip() == "## Recent (last 30 days)":
            in_recent_rollup = True
            rollup_lines.append(line)
            continue

        if in_recent_rollup:
            # Stop at next ## heading or ### entry
            if line.startswith("## ") or line.startswith("### "):
                break
            rollup_lines.append(line)

    # Add rollup section if found
    if rollup_lines:
        result.extend(rollup_lines)
        result.append("")

    # Extract dated entries from last N days
    current_decision = []
    current_date = None

    for line in lines:
        # Match decision heading: ### YYYY-MM-DD — Title
        match = re.match(r"^### (\d{4}-\d{2}-\d{2}) — (.+)", line)
        if match:
            date_str, title = match.groups()
            decision_date = datetime.strptime(date_str, "%Y-%m-%d")

            # Save previous decision if within window
            if current_decision and current_date and current_date >= cutoff_date:
                result.extend(current_decision)
                result.append("")

            # Start new decision
            current_date = decision_date
            current_decision = [line]
        elif current_decision:
            current_decision.append(line)

    # Save last decision
    if current_decision and current_date and current_date >= cutoff_date:
        result.extend(current_decision)

    return result


def _extract_digest_summary(digest_md: Path) -> list[str]:
    """Extract summary from latest digest."""
    content = digest_md.read_text()
    lines = content.split("\n")

    # Extract "Recent Changes" section
    result = []
    in_changes = False

    for line in lines:
        if line.startswith("## Recent Changes"):
            in_changes = True
            continue
        if in_changes:
            if line.startswith("##"):
                break
            result.append(line)

    return result


def _extract_watch_summary(watch_log: Path) -> list[str]:
    """Extract last 5 watch hits or 24h."""
    import json
    from datetime import datetime, timedelta

    cutoff = datetime.utcnow() - timedelta(hours=24)
    hits = []

    with watch_log.open("r") as f:
        for line in f:
            entry = json.loads(line.strip())
            ts = datetime.fromisoformat(entry["ts"].replace("Z", "+00:00"))
            if ts >= cutoff:
                hits.append(entry)

    # Take last 5
    hits = hits[-5:]

    result = []
    for hit in hits:
        result.append(f"- **\"{hit['query']}\"** ({len(hit['hits'])} hits)")

    return result

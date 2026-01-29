from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


def generate_digest(
    context_name: str,
    ingest_runs_log: Path | None,
    watch_history_log: Path | None,
    state_md: Path | None,
    output_md: Path,
    output_json: Path | None,
    since_hours: int = 24
) -> None:
    """
    Generate digest markdown and JSON from ingest runs and watch history.
    Deterministic except for timestamps in the data.
    """
    # Calculate since timestamp
    since_ts = datetime.now(timezone.utc) - timedelta(hours=since_hours)

    # Gather data
    ingest_stats = _gather_ingest_stats(ingest_runs_log, since_ts)
    watch_hits = _gather_watch_hits(watch_history_log, since_ts)
    state_summary = _extract_state_summary(state_md) if state_md else None

    # Generate markdown
    md_lines = [
        f"# Digest: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "",
    ]

    if watch_hits:
        md_lines.append(f"## Watch Hits ({len(watch_hits)})")
        for hit in watch_hits:
            md_lines.append(f"- **\"{hit['query']}\"** hit {hit['count']}x")
        md_lines.append("")

    if ingest_stats:
        md_lines.append("## Recent Changes (since last digest)")
        md_lines.append(f"- {ingest_stats['docs_changed']} files ingested, {ingest_stats['chunks_total']} chunks updated")
        md_lines.append("")

    if state_summary:
        md_lines.append("## State Summary")
        md_lines.append(state_summary)
        md_lines.append("")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(md_lines))

    # Generate JSON
    if output_json:
        data = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "context": context_name,
            "since_hours": since_hours,
            "watch_hits": watch_hits,
            "ingest_stats": ingest_stats,
            "state_summary": state_summary
        }
        output_json.write_text(json.dumps(data, indent=2))


def _gather_ingest_stats(log_path: Path | None, since: datetime) -> dict | None:
    """Gather ingest stats since timestamp."""
    if not log_path or not log_path.exists():
        return None

    from .ingest_log import read_ingest_runs
    runs = read_ingest_runs(log_path, completed_only=True)

    # Filter by timestamp
    recent_runs = [
        r for r in runs
        if r["status"] == "succeeded" and
           datetime.fromisoformat(r["ended_at"].replace("Z", "+00:00")) >= since
    ]

    if not recent_runs:
        return None

    # Aggregate
    total_docs_changed = sum(r.get("docs_changed", 0) for r in recent_runs)
    total_chunks_new = sum(r.get("chunks_new", 0) for r in recent_runs)
    total_chunks_updated = sum(r.get("chunks_updated", 0) for r in recent_runs)

    return {
        "docs_changed": total_docs_changed,
        "chunks_total": total_chunks_new + total_chunks_updated,
        "runs": len(recent_runs)
    }


def _gather_watch_hits(log_path: Path | None, since: datetime) -> list[dict]:
    """Gather watch hits since timestamp."""
    if not log_path or not log_path.exists():
        return []

    hits = []
    with log_path.open("r") as f:
        for line in f:
            entry = json.loads(line.strip())
            ts = datetime.fromisoformat(entry["ts"].replace("Z", "+00:00"))
            if ts >= since:
                hits.append({
                    "query": entry["query"],
                    "count": len(entry["hits"])
                })

    return hits


def _extract_state_summary(state_md: Path) -> str | None:
    """Extract state summary from STATE.md."""
    if not state_md.exists():
        return None

    content = state_md.read_text()
    # Simple extraction: first paragraph under "Current Objective"
    lines = content.split("\n")
    in_objective = False
    summary_lines = []

    for line in lines:
        if line.startswith("## Current Objective"):
            in_objective = True
            continue
        if in_objective:
            if line.startswith("##"):
                break
            if line.strip():
                summary_lines.append(line.strip())

    return " ".join(summary_lines) if summary_lines else None

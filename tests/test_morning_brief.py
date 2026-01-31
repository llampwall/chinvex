import json
import pytest
from pathlib import Path
from datetime import datetime, timedelta


def test_detect_active_contexts_7_day_threshold(tmp_path):
    """Test active context detection using 7-day threshold."""
    from chinvex.morning_brief import detect_active_stale_contexts

    # Create contexts with different last_sync timestamps
    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    # Active: 3 days ago
    active_ctx = contexts_root / "ActiveContext"
    active_ctx.mkdir()
    (active_ctx / "STATUS.json").write_text(json.dumps({
        "context": "ActiveContext",
        "chunks": 100,
        "last_sync": (datetime.now() - timedelta(days=3)).isoformat()
    }))

    # Stale: 10 days ago
    stale_ctx = contexts_root / "StaleContext"
    stale_ctx.mkdir()
    (stale_ctx / "STATUS.json").write_text(json.dumps({
        "context": "StaleContext",
        "chunks": 50,
        "last_sync": (datetime.now() - timedelta(days=10)).isoformat()
    }))

    # Edge case: exactly 7 days ago (should be active)
    edge_ctx = contexts_root / "EdgeContext"
    edge_ctx.mkdir()
    (edge_ctx / "STATUS.json").write_text(json.dumps({
        "context": "EdgeContext",
        "chunks": 75,
        "last_sync": (datetime.now() - timedelta(days=7, seconds=-60)).isoformat()
    }))

    active, stale = detect_active_stale_contexts(contexts_root)

    assert len(active) == 2
    assert len(stale) == 1

    active_names = {ctx["context"] for ctx in active}
    assert "ActiveContext" in active_names
    assert "EdgeContext" in active_names

    stale_names = {ctx["context"] for ctx in stale}
    assert "StaleContext" in stale_names


def test_detect_contexts_sorted_by_last_sync(tmp_path):
    """Test active contexts are sorted by last_sync (most recent first)."""
    from chinvex.morning_brief import detect_active_stale_contexts

    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    # Create 3 active contexts with different timestamps
    for i, days_ago in [(1, 1), (2, 5), (3, 2)]:
        ctx_dir = contexts_root / f"Context{i}"
        ctx_dir.mkdir()
        (ctx_dir / "STATUS.json").write_text(json.dumps({
            "context": f"Context{i}",
            "chunks": 100,
            "last_sync": (datetime.now() - timedelta(days=days_ago)).isoformat()
        }))

    active, stale = detect_active_stale_contexts(contexts_root)

    assert len(active) == 3
    assert active[0]["context"] == "Context1"  # 1 day ago (most recent)
    assert active[1]["context"] == "Context3"  # 2 days ago
    assert active[2]["context"] == "Context2"  # 5 days ago


def test_detect_contexts_missing_last_sync(tmp_path):
    """Test contexts without last_sync are treated as stale."""
    from chinvex.morning_brief import detect_active_stale_contexts

    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    # Context with no last_sync field
    no_sync_ctx = contexts_root / "NoSyncContext"
    no_sync_ctx.mkdir()
    (no_sync_ctx / "STATUS.json").write_text(json.dumps({
        "context": "NoSyncContext",
        "chunks": 100
    }))

    active, stale = detect_active_stale_contexts(contexts_root)

    assert len(active) == 0
    assert len(stale) == 1
    assert stale[0]["context"] == "NoSyncContext"


def test_detect_contexts_cap_at_top_5(tmp_path):
    """Test active contexts are capped at top 5 by recent activity."""
    from chinvex.morning_brief import detect_active_stale_contexts

    contexts_root = tmp_path / "contexts"
    contexts_root.mkdir()

    # Create 6 active contexts (all within 7 days)
    for i in range(6):
        ctx_dir = contexts_root / f"Context{i}"
        ctx_dir.mkdir()
        (ctx_dir / "STATUS.json").write_text(json.dumps({
            "context": f"Context{i}",
            "chunks": 100,
            "last_sync": (datetime.now() - timedelta(days=i+1)).isoformat()
        }))

    active, stale = detect_active_stale_contexts(contexts_root, max_active=5)

    # Should return only top 5 most recent
    assert len(active) == 5
    assert active[0]["context"] == "Context0"  # 1 day ago
    assert active[4]["context"] == "Context4"  # 5 days ago

    # Context5 is still within 7 days but not in the top 5, not returned as stale
    assert len(stale) == 0


def test_parse_state_md_objective_and_actions(tmp_path):
    """Test extracting Current Objective and Next Actions from STATE.md."""
    from chinvex.morning_brief import parse_state_md

    state_md = tmp_path / "STATE.md"
    state_md.write_text("""# State

## Current Objective
P5 implementation - reliability and retrieval quality

## Active Work
- Brief generation updates
- Morning brief overhaul

## Blockers
None

## Next Actions
- [ ] Update CONSTRAINTS.md extraction
- [ ] Add Recent rollup to DECISIONS.md
- [ ] Implement active/stale detection
- [ ] Parse STATE.md for objectives
- [ ] Format morning brief with ntfy

## Out of Scope (for now)
- Multi-user auth
- Smart scheduling agent
""")

    objective, actions = parse_state_md(state_md)

    assert objective == "P5 implementation - reliability and retrieval quality"
    assert len(actions) == 5
    assert "Update CONSTRAINTS.md extraction" in actions
    assert "Format morning brief with ntfy" in actions


def test_parse_state_md_max_5_actions(tmp_path):
    """Test Next Actions are capped at 5 bullets."""
    from chinvex.morning_brief import parse_state_md

    state_md = tmp_path / "STATE.md"
    state_md.write_text("""# State

## Current Objective
Test objective

## Next Actions
- [ ] Action 1
- [ ] Action 2
- [ ] Action 3
- [ ] Action 4
- [ ] Action 5
- [ ] Action 6
- [ ] Action 7
- [ ] Action 8
""")

    objective, actions = parse_state_md(state_md, max_actions=5)

    assert objective == "Test objective"
    assert len(actions) == 5
    assert "Action 1" in actions
    assert "Action 5" in actions
    assert "Action 6" not in actions


def test_parse_state_md_missing_sections(tmp_path):
    """Test graceful handling when sections are missing."""
    from chinvex.morning_brief import parse_state_md

    state_md = tmp_path / "STATE.md"
    state_md.write_text("""# State

## Active Work
- Some work

## Blockers
None
""")

    objective, actions = parse_state_md(state_md)

    assert objective is None
    assert actions == []


def test_parse_state_md_file_not_exists(tmp_path):
    """Test handling when STATE.md doesn't exist."""
    from chinvex.morning_brief import parse_state_md

    state_md = tmp_path / "nonexistent.md"

    objective, actions = parse_state_md(state_md)

    assert objective is None
    assert actions == []


def test_parse_state_md_multiline_objective(tmp_path):
    """Test objective extraction takes only first line."""
    from chinvex.morning_brief import parse_state_md

    state_md = tmp_path / "STATE.md"
    state_md.write_text("""# State

## Current Objective
First line is the objective
Second line should be ignored
Third line too

## Next Actions
- [ ] Action 1
""")

    objective, actions = parse_state_md(state_md)

    assert objective == "First line is the objective"
    assert len(actions) == 1

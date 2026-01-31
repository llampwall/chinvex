# tests/test_memory_templates.py
from pathlib import Path
from chinvex.memory_templates import (
    get_state_template,
    get_constraints_template,
    get_decisions_template,
    bootstrap_memory_files,
)


def test_state_template_has_required_sections():
    """STATE.md template must include all required sections."""
    template = get_state_template()
    assert "# State" in template
    assert "## Current Objective" in template
    assert "## Active Work" in template
    assert "## Blockers" in template
    assert "## Next Actions" in template
    assert "## Out of Scope (for now)" in template
    assert "Last memory update:" in template
    assert "Commits covered through:" in template


def test_constraints_template_has_core_sections():
    """CONSTRAINTS.md template must include core sections."""
    template = get_constraints_template()
    assert "# Constraints" in template
    assert "## Infrastructure" in template
    assert "## Rules" in template
    assert "## Key Facts" in template
    assert "## Hazards" in template
    assert "## Superseded" in template


def test_decisions_template_has_structure():
    """DECISIONS.md template must include rollup and monthly sections."""
    template = get_decisions_template()
    assert "# Decisions" in template
    assert "## Recent (last 30 days)" in template
    # Should have current month section (e.g., "## 2026-01")
    import datetime
    current_month = datetime.datetime.now().strftime("%Y-%m")
    assert f"## {current_month}" in template


def test_bootstrap_creates_memory_files(tmp_path):
    """bootstrap_memory_files should create docs/memory/ with all three files."""
    repo_root = tmp_path / "test_repo"
    repo_root.mkdir()

    bootstrap_memory_files(repo_root, initial_commit_hash="abc123def")

    memory_dir = repo_root / "docs" / "memory"
    assert memory_dir.exists()
    assert (memory_dir / "STATE.md").exists()
    assert (memory_dir / "CONSTRAINTS.md").exists()
    assert (memory_dir / "DECISIONS.md").exists()

    # Verify STATE.md has commit hash
    state_content = (memory_dir / "STATE.md").read_text()
    assert "abc123def" in state_content
    assert "<!-- chinvex:last-commit:abc123def -->" in state_content


def test_bootstrap_skips_if_files_exist(tmp_path):
    """bootstrap_memory_files should not overwrite existing files."""
    repo_root = tmp_path / "test_repo"
    memory_dir = repo_root / "docs" / "memory"
    memory_dir.mkdir(parents=True)

    # Create existing STATE.md with custom content
    state_file = memory_dir / "STATE.md"
    state_file.write_text("# Custom State\n\nDo not overwrite")

    bootstrap_memory_files(repo_root, initial_commit_hash="abc123")

    # Should not overwrite
    assert state_file.read_text() == "# Custom State\n\nDo not overwrite"

    # But should create missing files
    assert (memory_dir / "CONSTRAINTS.md").exists()
    assert (memory_dir / "DECISIONS.md").exists()

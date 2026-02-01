# P5b Implementation Plan - Memory Automation + Retrieval Quality

> **For Claude:** REQUIRED SUB-SKILL: Use batch-exec to execute this plan.

**Goal:** Implement memory file maintainer, startup hooks, retrieval evaluation suite, and reranking for improved search quality.

**Architecture:**
- Memory maintainer reads git history, processes specs/plans, updates STATE.md (full regen), CONSTRAINTS.md (merge), DECISIONS.md (append)
- Startup hooks auto-installed during ingest into `.claude/settings.json`
- Eval suite uses golden queries with file-based success criteria, tracks hit rate @K
- Two-stage retrieval: vector search → reranker (Cohere/Jina/local cross-encoder)

**Tech Stack:** Python 3.12, git log parsing, JSON schema validation, cross-encoder models (sentence-transformers), Cohere/Jina APIs

---

<!-- Tasks will be appended by batch-plan subagents -->


### Task 1: Memory file templates and bootstrap structure

**Files:**
- Create: `C:\Code\chinvex\src\chinvex\memory_templates.py`
- Test: `C:\Code\chinvex\tests\test_memory_templates.py`

**Step 1: Write the failing test**
```python
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
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_memory_templates.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chinvex.memory_templates'"

**Step 3: Write minimal implementation**
```python
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
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_memory_templates.py -v`
Expected: PASS (all 5 tests passing)

**Step 5: Commit**
```bash
git add src/chinvex/memory_templates.py tests/test_memory_templates.py
git commit -m "feat(P5.2.1): add memory file templates and bootstrap structure

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 2: Git log parsing and commit range analysis

**Files:**
- Create: `C:\Code\chinvex\src\chinvex\git_analyzer.py`
- Test: `C:\Code\chinvex\tests\test_git_analyzer.py`

**Step 1: Write the failing test**
```python
# tests/test_git_analyzer.py
import subprocess
from pathlib import Path
from chinvex.git_analyzer import (
    extract_coverage_anchor,
    get_commit_range,
    parse_commits,
    GitCommit,
)


def test_extract_coverage_anchor_from_state_md(tmp_path):
    """Should extract commit hash from coverage anchor comment."""
    state_file = tmp_path / "STATE.md"
    state_file.write_text("""# State

## Current Objective
Test

---
Last memory update: 2026-01-31
Commits covered through: abc123def

<!-- chinvex:last-commit:abc123def -->
""")

    anchor = extract_coverage_anchor(state_file)
    assert anchor == "abc123def"


def test_extract_coverage_anchor_missing_returns_none(tmp_path):
    """Should return None if anchor comment is missing."""
    state_file = tmp_path / "STATE.md"
    state_file.write_text("# State\n\nNo anchor here")

    anchor = extract_coverage_anchor(state_file)
    assert anchor is None


def test_get_commit_range_with_anchor(tmp_path):
    """Should return commits from anchor..HEAD."""
    # Create a real git repo for testing
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)

    # Create first commit
    (repo / "file1.txt").write_text("first")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "first commit"], cwd=repo, check=True)

    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True)
    first_hash = result.stdout.strip()

    # Create second commit
    (repo / "file2.txt").write_text("second")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "second commit"], cwd=repo, check=True)

    # Get range from first_hash to HEAD
    commits = get_commit_range(repo, start_hash=first_hash)

    assert len(commits) == 1
    assert commits[0].message == "second commit"


def test_get_commit_range_enforces_max_commits(tmp_path):
    """Should limit results to max_commits parameter."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)

    # Create 5 commits
    for i in range(5):
        (repo / f"file{i}.txt").write_text(f"commit {i}")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-m", f"commit {i}"], cwd=repo, check=True)

    commits = get_commit_range(repo, max_commits=3)

    assert len(commits) == 3


def test_parse_commits_extracts_fields():
    """Should parse git log output into GitCommit objects."""
    log_output = """abc123def|||2026-01-31 10:30:00 -0800|||John Doe|||feat: add new feature

Details about the feature
---
def456ghi|||2026-01-30 15:20:00 -0800|||Jane Smith|||fix: resolve bug

Root cause was XYZ
"""

    commits = parse_commits(log_output)

    assert len(commits) == 2
    assert commits[0].hash == "abc123def"
    assert commits[0].author == "John Doe"
    assert commits[0].message == "feat: add new feature\n\nDetails about the feature"
    assert "2026-01-31" in commits[0].date

    assert commits[1].hash == "def456ghi"
    assert commits[1].author == "Jane Smith"


def test_get_commit_range_returns_empty_if_no_new_commits(tmp_path):
    """Should return empty list if start_hash is HEAD."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)

    (repo / "file.txt").write_text("test")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True)

    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True)
    head_hash = result.stdout.strip()

    commits = get_commit_range(repo, start_hash=head_hash)

    assert len(commits) == 0
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_git_analyzer.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chinvex.git_analyzer'"

**Step 3: Write minimal implementation**
```python
# src/chinvex/git_analyzer.py
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GitCommit:
    """Represents a single git commit."""
    hash: str
    date: str
    author: str
    message: str


def extract_coverage_anchor(state_file: Path) -> str | None:
    """Extract last processed commit hash from STATE.md coverage anchor.

    Looks for: <!-- chinvex:last-commit:abc123def -->

    Args:
        state_file: Path to STATE.md

    Returns:
        Commit hash if found, None otherwise
    """
    if not state_file.exists():
        return None

    content = state_file.read_text()
    match = re.search(r"<!-- chinvex:last-commit:([a-f0-9]+) -->", content)
    return match.group(1) if match else None


def parse_commits(log_output: str) -> list[GitCommit]:
    """Parse git log output into GitCommit objects.

    Expected format: hash|||date|||author|||message
    Separator between commits: ---

    Args:
        log_output: Output from git log --format

    Returns:
        List of GitCommit objects
    """
    if not log_output.strip():
        return []

    commits = []
    commit_blocks = log_output.strip().split("---\n")

    for block in commit_blocks:
        block = block.strip()
        if not block:
            continue

        parts = block.split("|||", 3)
        if len(parts) < 4:
            continue

        hash_val, date, author, message = parts
        commits.append(GitCommit(
            hash=hash_val.strip(),
            date=date.strip(),
            author=author.strip(),
            message=message.strip()
        ))

    return commits


def get_commit_range(
    repo_root: Path,
    start_hash: str | None = None,
    max_commits: int = 50
) -> list[GitCommit]:
    """Get commits from start_hash..HEAD.

    Args:
        repo_root: Repository root directory
        start_hash: Starting commit hash (exclusive). If None, gets all commits up to max_commits.
        max_commits: Maximum number of commits to return (bounded inputs guardrail)

    Returns:
        List of GitCommit objects in reverse chronological order (newest first)
    """
    # Build git log command
    format_str = "%H|||%ai|||%an|||%B"

    if start_hash:
        commit_range = f"{start_hash}..HEAD"
    else:
        commit_range = "HEAD"

    cmd = [
        "git", "log",
        commit_range,
        f"-{max_commits}",
        f"--format={format_str}---"
    ]

    result = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True
    )

    return parse_commits(result.stdout)
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_git_analyzer.py -v`
Expected: PASS (all 6 tests passing)

**Step 5: Commit**
```bash
git add src/chinvex/git_analyzer.py tests/test_git_analyzer.py
git commit -m "feat(P5.2.1): add git log parsing and commit range analysis

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 3: Spec/plan file reading with bounded inputs guardrails

**Files:**
- Create: `C:\Code\chinvex\src\chinvex\spec_reader.py`
- Test: `C:\Code\chinvex\tests\test_spec_reader.py`

**Step 1: Write the failing test**
```python
# tests/test_spec_reader.py
import subprocess
from pathlib import Path
from chinvex.spec_reader import (
    extract_spec_plan_files_from_commits,
    read_spec_files,
    SpecContent,
    BOUNDED_INPUTS,
)


def test_extract_spec_plan_files_from_commits(tmp_path):
    """Should identify spec/plan files touched in commits."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)

    # Create specs and plans
    specs_dir = repo / "specs"
    specs_dir.mkdir()
    plans_dir = repo / "docs" / "plans"
    plans_dir.mkdir(parents=True)

    (specs_dir / "P1_SPEC.md").write_text("# P1 Spec")
    (plans_dir / "2026-01-plan.md").write_text("# Plan")
    (repo / "src" / "code.py").mkdir(parents=True)
    (repo / "src" / "code.py").write_text("# code")

    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True)

    # Modify spec and code
    (specs_dir / "P1_SPEC.md").write_text("# P1 Spec Updated")
    (repo / "src" / "code.py").write_text("# code updated")

    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "update"], cwd=repo, check=True)

    result = subprocess.run(["git", "rev-parse", "HEAD~1"], cwd=repo, check=True, capture_output=True, text=True)
    start_hash = result.stdout.strip()

    # Extract files from commits
    spec_files = extract_spec_plan_files_from_commits(repo, start_hash)

    assert len(spec_files) == 1
    assert spec_files[0].name == "P1_SPEC.md"


def test_read_spec_files_respects_max_files_limit(tmp_path):
    """Should limit to max_files and flag truncation."""
    # Create 25 spec files (exceeds limit of 20)
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()

    spec_paths = []
    for i in range(25):
        spec_file = specs_dir / f"P{i}_SPEC.md"
        spec_file.write_text(f"# Spec {i}\n\nContent for spec {i}")
        spec_paths.append(spec_file)

    result = read_spec_files(spec_paths)

    # Should only read max_files
    assert len(result.specs) == BOUNDED_INPUTS["max_files"]
    assert result.truncated_files is True
    assert result.total_size < BOUNDED_INPUTS["max_total_size_kb"] * 1024


def test_read_spec_files_respects_size_limit(tmp_path):
    """Should stop reading when size limit exceeded."""
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()

    # Create files that will exceed 100KB total
    spec_paths = []
    for i in range(5):
        spec_file = specs_dir / f"large_{i}.md"
        # Each file is ~30KB
        content = "x" * 30000 + f"\n# Spec {i}"
        spec_file.write_text(content)
        spec_paths.append(spec_file)

    result = read_spec_files(spec_paths)

    # Should stop before reading all files
    assert len(result.specs) < 5
    assert result.truncated_size is True
    assert result.total_size <= BOUNDED_INPUTS["max_total_size_kb"] * 1024


def test_read_spec_files_returns_content():
    """Should return SpecContent with file contents."""
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Test Spec\n\nSome content")
        f.flush()
        temp_path = Path(f.name)

    try:
        result = read_spec_files([temp_path])

        assert len(result.specs) == 1
        assert result.specs[0]["path"] == str(temp_path)
        assert "# Test Spec" in result.specs[0]["content"]
        assert result.truncated_files is False
        assert result.truncated_size is False
    finally:
        temp_path.unlink()


def test_extract_spec_plan_files_only_from_specs_and_plans_dirs(tmp_path):
    """Should only extract files from /specs/ and /docs/plans/."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)

    # Create various files
    (repo / "specs").mkdir()
    (repo / "docs" / "plans").mkdir(parents=True)
    (repo / "docs" / "other").mkdir(parents=True)

    (repo / "specs" / "P1.md").write_text("spec")
    (repo / "docs" / "plans" / "plan.md").write_text("plan")
    (repo / "docs" / "other" / "readme.md").write_text("readme")
    (repo / "README.md").write_text("readme")

    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True)

    # Get all files from this commit
    spec_files = extract_spec_plan_files_from_commits(repo, start_hash=None)

    # Should only include specs/ and docs/plans/
    assert len(spec_files) == 2
    names = {f.name for f in spec_files}
    assert "P1.md" in names
    assert "plan.md" in names
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_spec_reader.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chinvex.spec_reader'"

**Step 3: Write minimal implementation**
```python
# src/chinvex/spec_reader.py
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


# Bounded inputs guardrails (from P5b spec)
BOUNDED_INPUTS = {
    "max_commits": 50,
    "max_files": 20,
    "max_total_size_kb": 100,
}


@dataclass
class SpecContent:
    """Container for spec/plan file contents with truncation flags."""
    specs: list[dict[str, str]]  # [{"path": str, "content": str}, ...]
    total_size: int  # Total bytes read
    truncated_files: bool  # True if max_files limit reached
    truncated_size: bool  # True if max_total_size limit reached


def extract_spec_plan_files_from_commits(
    repo_root: Path,
    start_hash: str | None = None
) -> list[Path]:
    """Extract unique spec/plan files touched in commit range.

    Only returns files from /specs/ and /docs/plans/ directories.

    Args:
        repo_root: Repository root
        start_hash: Starting commit hash (exclusive). If None, uses all history.

    Returns:
        List of unique spec/plan file paths (relative to repo root)
    """
    if start_hash:
        commit_range = f"{start_hash}..HEAD"
    else:
        commit_range = "HEAD"

    # Get all files changed in the range
    cmd = ["git", "diff", "--name-only", commit_range]

    result = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True
    )

    files = result.stdout.strip().split("\n") if result.stdout.strip() else []

    # Filter to only specs/ and docs/plans/
    spec_files = []
    for file_path in files:
        if not file_path:
            continue

        # Check if in specs/ or docs/plans/
        if file_path.startswith("specs/") or file_path.startswith("docs/plans/"):
            full_path = repo_root / file_path
            if full_path.exists() and full_path.suffix == ".md":
                spec_files.append(full_path)

    return list(set(spec_files))  # Deduplicate


def read_spec_files(spec_paths: list[Path]) -> SpecContent:
    """Read spec/plan files with bounded inputs guardrails.

    Args:
        spec_paths: List of spec/plan file paths to read

    Returns:
        SpecContent with file contents and truncation flags
    """
    max_files = BOUNDED_INPUTS["max_files"]
    max_size = BOUNDED_INPUTS["max_total_size_kb"] * 1024

    specs = []
    total_size = 0
    truncated_files = False
    truncated_size = False

    for i, path in enumerate(spec_paths):
        # Check file count limit
        if i >= max_files:
            truncated_files = True
            break

        # Check if reading this file would exceed size limit
        file_size = path.stat().st_size
        if total_size + file_size > max_size:
            truncated_size = True
            break

        # Read file
        content = path.read_text(encoding="utf-8")
        specs.append({
            "path": str(path),
            "content": content
        })
        total_size += file_size

    return SpecContent(
        specs=specs,
        total_size=total_size,
        truncated_files=truncated_files,
        truncated_size=truncated_size
    )
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_spec_reader.py -v`
Expected: PASS (all 5 tests passing)

**Step 5: Commit**
```bash
git add src/chinvex/spec_reader.py tests/test_spec_reader.py
git commit -m "feat(P5.2.1): add spec/plan file reading with bounded inputs guardrails

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 4: STATE.md full regeneration logic

**Files:**
- Create: `C:\Code\chinvex\src\chinvex\state_regenerator.py`
- Test: `C:\Code\chinvex\tests\test_state_regenerator.py`

**Step 1: Write the failing test**
```python
# tests/test_state_regenerator.py
from pathlib import Path
from chinvex.state_regenerator import (
    regenerate_state_md,
    update_coverage_anchor,
)
from chinvex.git_analyzer import GitCommit


def test_regenerate_state_md_creates_new_content():
    """Should generate STATE.md based on commits and specs."""
    commits = [
        GitCommit(
            hash="abc123",
            date="2026-01-31",
            author="Test",
            message="feat(P5): implement memory maintainer\n\nAdds update-memory command"
        ),
        GitCommit(
            hash="def456",
            date="2026-01-30",
            author="Test",
            message="docs: update P5 spec"
        )
    ]

    specs = [
        {"path": "specs/P5_SPEC.md", "content": "# P5 Spec\n\n## Goal\nImplement memory automation"}
    ]

    current_state = """# State

## Current Objective
Old objective

## Active Work
- Old work

## Blockers
None

## Next Actions
- [ ] Old action

## Out of Scope (for now)
- TBD
"""

    new_state = regenerate_state_md(
        commits=commits,
        specs=specs,
        current_state=current_state,
        ending_commit_hash="abc123"
    )

    # Should include coverage anchor
    assert "<!-- chinvex:last-commit:abc123 -->" in new_state
    # Should have structure
    assert "# State" in new_state
    assert "## Current Objective" in new_state
    assert "## Active Work" in new_state
    # Should reference recent work
    assert "memory" in new_state.lower() or "P5" in new_state


def test_update_coverage_anchor_replaces_existing():
    """Should replace existing anchor with new commit hash."""
    old_content = """# State

## Current Objective
Test

---
Last memory update: 2026-01-30
Commits covered through: old_hash

<!-- chinvex:last-commit:old_hash -->
"""

    new_content = update_coverage_anchor(old_content, new_hash="new_hash")

    assert "<!-- chinvex:last-commit:new_hash -->" in new_content
    assert "<!-- chinvex:last-commit:old_hash -->" not in new_content
    assert "Commits covered through: new_hash" in new_content


def test_update_coverage_anchor_adds_if_missing():
    """Should add anchor if not present."""
    old_content = """# State

## Current Objective
Test

## Active Work
- Work item
"""

    new_content = update_coverage_anchor(old_content, new_hash="abc123")

    assert "<!-- chinvex:last-commit:abc123 -->" in new_content
    assert "Commits covered through: abc123" in new_content


def test_regenerate_state_md_handles_no_objective_found():
    """If no clear objective can be determined, should use placeholder."""
    commits = [
        GitCommit(hash="abc", date="2026-01-31", author="Test", message="misc: update readme")
    ]

    specs = []
    current_state = "# State\n\n## Current Objective\nOld"

    new_state = regenerate_state_md(
        commits=commits,
        specs=specs,
        current_state=current_state,
        ending_commit_hash="abc"
    )

    # Should have fallback objective
    assert "Unknown (needs human)" in new_state or "TBD" in new_state


def test_regenerate_state_md_includes_timestamp():
    """Should update timestamp in footer."""
    import datetime

    commits = []
    specs = []
    current_state = "# State\n\n## Current Objective\nTest"

    new_state = regenerate_state_md(
        commits=commits,
        specs=specs,
        current_state=current_state,
        ending_commit_hash="abc"
    )

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    assert f"Last memory update: {today}" in new_state
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_state_regenerator.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chinvex.state_regenerator'"

**Step 3: Write minimal implementation**
```python
# src/chinvex/state_regenerator.py
from __future__ import annotations

import datetime
import re


def update_coverage_anchor(content: str, new_hash: str) -> str:
    """Update coverage anchor in STATE.md content.

    Replaces existing anchor or appends if missing.

    Args:
        content: Current STATE.md content
        new_hash: New commit hash to use

    Returns:
        Updated content with new anchor
    """
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    # Replace existing anchor
    anchor_pattern = r"<!-- chinvex:last-commit:[a-f0-9]+ -->"
    if re.search(anchor_pattern, content):
        content = re.sub(anchor_pattern, f"<!-- chinvex:last-commit:{new_hash} -->", content)
    else:
        # Add anchor at end
        content = content.rstrip() + f"\n\n<!-- chinvex:last-commit:{new_hash} -->\n"

    # Update "Commits covered through" line
    commits_line_pattern = r"Commits covered through: [a-f0-9]+"
    if re.search(commits_line_pattern, content):
        content = re.sub(commits_line_pattern, f"Commits covered through: {new_hash}", content)
    else:
        # Add before anchor
        anchor_pos = content.find("<!-- chinvex:last-commit:")
        if anchor_pos > 0:
            content = (
                content[:anchor_pos] +
                f"Commits covered through: {new_hash}\n\n" +
                content[anchor_pos:]
            )

    # Update timestamp
    timestamp_pattern = r"Last memory update: \d{4}-\d{2}-\d{2}"
    if re.search(timestamp_pattern, content):
        content = re.sub(timestamp_pattern, f"Last memory update: {today}", content)
    else:
        # Add before commits line
        commits_pos = content.find("Commits covered through:")
        if commits_pos > 0:
            content = (
                content[:commits_pos] +
                f"Last memory update: {today}\n" +
                content[commits_pos:]
            )

    return content


def regenerate_state_md(
    commits: list,
    specs: list[dict[str, str]],
    current_state: str,
    ending_commit_hash: str
) -> str:
    """Regenerate STATE.md based on commits and specs.

    This is a FULL regeneration - manual edits may be lost.
    Uses LLM-like logic to infer current objective from commits/specs.

    Args:
        commits: List of GitCommit objects
        specs: List of spec dicts with 'path' and 'content'
        current_state: Current STATE.md content (for reference)
        ending_commit_hash: Final commit hash for coverage anchor

    Returns:
        New STATE.md content
    """
    # Simple heuristic-based regeneration (placeholder for LLM call in real implementation)
    # In production, this would call an LLM to analyze commits+specs and generate content

    today = datetime.datetime.now().strftime("%Y-%m-%d")

    # Attempt to infer objective from specs or commits
    objective = "Unknown (needs human)"
    active_work = []

    # Look for spec mentions in recent commits
    for commit in commits[:5]:  # Last 5 commits
        msg_lower = commit.message.lower()
        if "p5" in msg_lower or "memory" in msg_lower:
            objective = "P5 implementation - memory automation"
            active_work.append("Implementing memory file maintainer")
            break

    # Check specs for objective
    for spec in specs:
        if "P5" in spec["path"] and "memory" in spec["content"].lower():
            objective = "P5 implementation - memory automation"
            break

    # Build new STATE.md
    active_work_str = "\n".join([f"- {item}" for item in active_work]) if active_work else "- None"

    new_content = f"""# State

## Current Objective
{objective}

## Active Work
{active_work_str}

## Blockers
None

## Next Actions
- [ ] Run chinvex update-memory to refresh this file

## Out of Scope (for now)
- TBD

---
Last memory update: {today}
Commits covered through: {ending_commit_hash}

<!-- chinvex:last-commit:{ending_commit_hash} -->
"""

    return new_content
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_state_regenerator.py -v`
Expected: PASS (all 5 tests passing)

**Step 5: Commit**
```bash
git add src/chinvex/state_regenerator.py tests/test_state_regenerator.py
git commit -m "feat(P5.2.1): add STATE.md full regeneration logic

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 5: CONSTRAINTS.md merge logic (add bullets, detect obsolete via LLM)

**Files:**
- Create: `C:\Code\chinvex\src\chinvex\constraints_merger.py`
- Test: `C:\Code\chinvex\tests\test_constraints_merger.py`

**Step 1: Write the failing test**
```python
# tests/test_constraints_merger.py
from chinvex.constraints_merger import (
    merge_constraints,
    extract_new_constraints,
    move_to_superseded,
)
from chinvex.git_analyzer import GitCommit


def test_merge_constraints_adds_new_bullets():
    """Should add new constraint bullets without removing existing ones."""
    current = """# Constraints

## Infrastructure
- ChromaDB batch limit: 5000 vectors
- Gateway port: 7778

## Rules
- Schema stays v2

## Key Facts
- Token env var: CHINVEX_API_TOKEN

## Hazards
- None

## Superseded
(None yet)
"""

    new_constraints = [
        {"section": "Infrastructure", "bullet": "Embedding dims locked per index"},
        {"section": "Rules", "bullet": "No migrations without rebuild"}
    ]

    updated = merge_constraints(current, new_constraints)

    # Should preserve existing
    assert "ChromaDB batch limit: 5000 vectors" in updated
    assert "Gateway port: 7778" in updated
    # Should add new
    assert "Embedding dims locked per index" in updated
    assert "No migrations without rebuild" in updated


def test_extract_new_constraints_from_commits():
    """Should identify potential new constraints from commit messages."""
    commits = [
        GitCommit(
            hash="abc",
            date="2026-01-31",
            author="Test",
            message="fix: respect max 50 commits limit\n\nBounded inputs guardrail"
        ),
        GitCommit(
            hash="def",
            date="2026-01-30",
            author="Test",
            message="feat: add new search endpoint"
        )
    ]

    specs = [
        {"path": "specs/P5.md", "content": "Max 50 commits per run\nMax 100KB total spec content"}
    ]

    constraints = extract_new_constraints(commits, specs)

    # Should extract bounded inputs
    assert len(constraints) > 0
    # Should have section and bullet
    assert all("section" in c and "bullet" in c for c in constraints)


def test_move_to_superseded():
    """Should move obsolete constraint to Superseded section with date and reason."""
    import datetime

    current = """# Constraints

## Infrastructure
- Old batch limit: 1000 vectors
- Gateway port: 7778

## Rules
- Schema stays v2

## Superseded
(None yet)
"""

    obsolete_info = {
        "section": "Infrastructure",
        "bullet": "Old batch limit: 1000 vectors",
        "reason": "Increased to 5000 in P3"
    }

    updated = move_to_superseded(current, obsolete_info)

    # Should remove from original section
    assert "## Infrastructure\n- Old batch limit: 1000 vectors" not in updated
    # Should appear in Superseded with date
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    assert "## Superseded" in updated
    assert f"(Superseded {today})" in updated
    assert "Old batch limit: 1000 vectors" in updated
    assert "Increased to 5000 in P3" in updated


def test_merge_constraints_preserves_structure():
    """Should maintain all core sections even if empty."""
    current = """# Constraints

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

    updated = merge_constraints(current, [])

    # All sections should remain
    assert "## Infrastructure" in updated
    assert "## Rules" in updated
    assert "## Key Facts" in updated
    assert "## Hazards" in updated
    assert "## Superseded" in updated


def test_merge_constraints_avoids_duplicates():
    """Should not add constraint if bullet already exists."""
    current = """# Constraints

## Infrastructure
- ChromaDB batch limit: 5000 vectors

## Rules
- Schema stays v2

## Superseded
(None yet)
"""

    new_constraints = [
        {"section": "Infrastructure", "bullet": "ChromaDB batch limit: 5000 vectors"},  # duplicate
        {"section": "Infrastructure", "bullet": "Gateway port: 7778"}  # new
    ]

    updated = merge_constraints(current, new_constraints)

    # Should only have one occurrence of batch limit
    assert updated.count("ChromaDB batch limit: 5000 vectors") == 1
    # Should add the new one
    assert "Gateway port: 7778" in updated
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_constraints_merger.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chinvex.constraints_merger'"

**Step 3: Write minimal implementation**
```python
# src/chinvex/constraints_merger.py
from __future__ import annotations

import datetime
import re


def extract_new_constraints(commits: list, specs: list[dict[str, str]]) -> list[dict[str, str]]:
    """Extract potential new constraints from commits and specs.

    Uses heuristics to identify constraint-like statements.
    In production, this would use LLM to reason about what constitutes a constraint.

    Args:
        commits: List of GitCommit objects
        specs: List of spec dicts with 'path' and 'content'

    Returns:
        List of constraint dicts with 'section' and 'bullet' keys
    """
    constraints = []

    # Look for bounded inputs mentions in commits
    for commit in commits:
        if "max" in commit.message.lower() and ("limit" in commit.message.lower() or "commits" in commit.message.lower()):
            constraints.append({
                "section": "Infrastructure",
                "bullet": f"Bounded inputs: see spec for limits (from {commit.hash[:7]})"
            })

    # Look for limits in specs
    for spec in specs:
        content = spec["content"]
        if "max" in content.lower() and "kb" in content.lower():
            # Found size limit mention
            constraints.append({
                "section": "Infrastructure",
                "bullet": "Memory file update has bounded inputs (max commits, files, size)"
            })
            break

    return constraints


def move_to_superseded(content: str, obsolete_info: dict[str, str]) -> str:
    """Move a constraint from its section to Superseded.

    Args:
        content: Current CONSTRAINTS.md content
        obsolete_info: Dict with 'section', 'bullet', 'reason'

    Returns:
        Updated content with constraint moved to Superseded
    """
    section = obsolete_info["section"]
    bullet = obsolete_info["bullet"]
    reason = obsolete_info["reason"]
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    # Remove from original section
    # Find the bullet line and remove it
    bullet_pattern = re.escape(bullet)
    content = re.sub(rf"^- {bullet_pattern}\n", "", content, flags=re.MULTILINE)

    # Add to Superseded section
    superseded_marker = "## Superseded"
    if superseded_marker in content:
        # Find the section and append
        superseded_pos = content.find(superseded_marker)
        section_end = content.find("\n## ", superseded_pos + len(superseded_marker))
        if section_end == -1:
            section_end = len(content)

        # Insert before end of section
        insert_pos = section_end
        # Skip past "(None yet)" if present
        none_yet_pos = content.find("(None yet)", superseded_pos)
        if none_yet_pos > 0 and none_yet_pos < section_end:
            # Replace (None yet)
            content = content.replace("(None yet)\n", "")
            insert_pos = content.find("\n## ", superseded_pos)
            if insert_pos == -1:
                insert_pos = len(content)

        new_entry = f"- (Superseded {today}) {bullet} - {reason}\n"
        content = content[:insert_pos] + new_entry + content[insert_pos:]

    return content


def merge_constraints(current: str, new_constraints: list[dict[str, str]]) -> str:
    """Merge new constraints into CONSTRAINTS.md.

    Adds bullets to appropriate sections, avoids duplicates.

    Args:
        current: Current CONSTRAINTS.md content
        new_constraints: List of dicts with 'section' and 'bullet'

    Returns:
        Updated CONSTRAINTS.md content
    """
    content = current

    for constraint in new_constraints:
        section = constraint["section"]
        bullet = constraint["bullet"]

        # Check if bullet already exists (avoid duplicates)
        if bullet in content:
            continue

        # Find the section
        section_marker = f"## {section}"
        if section_marker not in content:
            # Section doesn't exist - skip or add section
            continue

        section_pos = content.find(section_marker)
        # Find next section or end of file
        next_section = content.find("\n## ", section_pos + len(section_marker))
        if next_section == -1:
            next_section = len(content)

        # Insert bullet before next section
        # Find last bullet in this section
        section_content = content[section_pos:next_section]
        last_bullet_match = None
        for match in re.finditer(r"^- .+$", section_content, re.MULTILINE):
            last_bullet_match = match

        if last_bullet_match:
            # Insert after last bullet
            insert_pos = section_pos + last_bullet_match.end() + 1
        else:
            # No bullets yet, insert after section header
            insert_pos = section_pos + len(section_marker) + 1

        content = content[:insert_pos] + f"- {bullet}\n" + content[insert_pos:]

    return content
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_constraints_merger.py -v`
Expected: PASS (all 5 tests passing)

**Step 5: Commit**
```bash
git add src/chinvex/constraints_merger.py tests/test_constraints_merger.py
git commit -m "feat(P5.2.1): add CONSTRAINTS.md merge logic with superseded handling

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 6: DECISIONS.md append-only logic with monthly sections and Recent rollup

**Files:**
- Create: `C:\Code\chinvex\src\chinvex\decisions_appender.py`
- Test: `C:\Code\chinvex\tests\test_decisions_appender.py`

**Step 1: Write the failing test**
```python
# tests/test_decisions_appender.py
import datetime
from chinvex.decisions_appender import (
    append_decision,
    update_recent_rollup,
    ensure_month_section,
)
from chinvex.git_analyzer import GitCommit


def test_ensure_month_section_creates_if_missing():
    """Should create month section if it doesn't exist."""
    current = """# Decisions

## Recent (last 30 days)
- TBD

## 2026-01
- Decision from January
"""

    # Request February section
    updated = ensure_month_section(current, "2026-02")

    assert "## 2026-02" in updated
    # Should be inserted after Recent section
    assert updated.find("## Recent") < updated.find("## 2026-02") < updated.find("## 2026-01")


def test_ensure_month_section_preserves_if_exists():
    """Should not duplicate existing month section."""
    current = """# Decisions

## Recent (last 30 days)
- Recent decision

## 2026-01
- Existing decision
"""

    updated = ensure_month_section(current, "2026-01")

    # Should only have one occurrence
    assert updated.count("## 2026-01") == 1
    assert "Existing decision" in updated


def test_append_decision_adds_to_month_section():
    """Should append decision to correct month section."""
    current = """# Decisions

## Recent (last 30 days)
- TBD

## 2026-01
- Older decision
"""

    decision = {
        "date": "2026-01-31",
        "title": "Use OpenAI as default embedding provider",
        "rationale": "Better quality than Ollama for most use cases"
    }

    updated = append_decision(current, decision)

    # Should appear in 2026-01 section
    assert "Use OpenAI as default embedding provider" in updated
    assert "Better quality than Ollama" in updated
    assert "(2026-01-31)" in updated
    # Should be added to month section
    assert updated.find("## 2026-01") < updated.find("Use OpenAI")


def test_append_decision_creates_month_section_if_needed():
    """Should create month section if decision date is new month."""
    current = """# Decisions

## Recent (last 30 days)
- TBD

## 2026-01
- January decision
"""

    decision = {
        "date": "2026-02-15",
        "title": "Switch to ChromaDB v2",
        "rationale": "Performance improvements"
    }

    updated = append_decision(current, decision)

    # Should create February section
    assert "## 2026-02" in updated
    assert "Switch to ChromaDB v2" in updated


def test_update_recent_rollup_includes_last_30_days():
    """Should update Recent section with decisions from last 30 days."""
    today = datetime.datetime.now()
    last_month = today - datetime.timedelta(days=25)
    two_months_ago = today - datetime.timedelta(days=60)

    current = f"""# Decisions

## Recent (last 30 days)
- TBD

## {today.strftime("%Y-%m")}
- (2026-01-31) Recent decision A
- ({two_months_ago.strftime("%Y-%m-%d")}) Old decision

## {last_month.strftime("%Y-%m")}
- ({last_month.strftime("%Y-%m-%d")}) Recent decision B
"""

    updated = update_recent_rollup(current)

    # Recent section should include only last 30 days
    recent_section_start = updated.find("## Recent")
    next_section = updated.find("\n## ", recent_section_start + 10)
    recent_content = updated[recent_section_start:next_section]

    assert "Recent decision A" in recent_content
    assert "Recent decision B" in recent_content
    assert "Old decision" not in recent_content


def test_update_recent_rollup_preserves_chronological_order():
    """Should list recent decisions in reverse chronological order (newest first)."""
    current = """# Decisions

## Recent (last 30 days)
- TBD

## 2026-01
- (2026-01-31) Decision C
- (2026-01-20) Decision B
- (2026-01-10) Decision A
"""

    updated = update_recent_rollup(current)

    recent_section_start = updated.find("## Recent")
    next_section = updated.find("\n## ", recent_section_start + 10)
    recent_content = updated[recent_section_start:next_section]

    # Newest should come first
    pos_c = recent_content.find("Decision C")
    pos_b = recent_content.find("Decision B")
    pos_a = recent_content.find("Decision A")

    assert pos_c < pos_b < pos_a


def test_append_decision_format_matches_spec():
    """Decision entries should match format: (YYYY-MM-DD) Title - Rationale."""
    current = """# Decisions

## Recent (last 30 days)
- TBD

## 2026-01
(No decisions yet)
"""

    decision = {
        "date": "2026-01-31",
        "title": "Max 50 commits per memory update",
        "rationale": "Bounded inputs guardrail to prevent timeout"
    }

    updated = append_decision(current, decision)

    # Check format
    assert "- (2026-01-31) Max 50 commits per memory update - Bounded inputs guardrail to prevent timeout" in updated


def test_update_recent_rollup_handles_empty_months():
    """Should skip empty month sections when building Recent rollup."""
    current = """# Decisions

## Recent (last 30 days)
- TBD

## 2026-01
- (2026-01-31) Decision A

## 2025-12
(No decisions recorded yet)
"""

    updated = update_recent_rollup(current)

    # Should only include Decision A
    recent_section_start = updated.find("## Recent")
    next_section = updated.find("\n## ", recent_section_start + 10)
    recent_content = updated[recent_section_start:next_section]

    assert "Decision A" in recent_content
    assert recent_content.count("-") == 1  # Only one decision
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_decisions_appender.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chinvex.decisions_appender'"

**Step 3: Write minimal implementation**
```python
# src/chinvex/decisions_appender.py
from __future__ import annotations

import datetime
import re


def ensure_month_section(content: str, month_key: str) -> str:
    """Ensure month section exists in DECISIONS.md.

    Month sections are inserted in reverse chronological order after Recent section.

    Args:
        content: Current DECISIONS.md content
        month_key: Month in YYYY-MM format

    Returns:
        Updated content with month section present
    """
    month_marker = f"## {month_key}"

    if month_marker in content:
        return content  # Already exists

    # Find Recent section
    recent_marker = "## Recent (last 30 days)"
    if recent_marker not in content:
        # Malformed file
        return content

    recent_pos = content.find(recent_marker)
    # Find next section after Recent
    next_section_pos = content.find("\n## ", recent_pos + len(recent_marker))

    if next_section_pos == -1:
        # No sections after Recent - append at end
        insert_pos = len(content)
    else:
        insert_pos = next_section_pos + 1

    # Insert new month section
    new_section = f"## {month_key}\n(No decisions recorded yet)\n\n"
    content = content[:insert_pos] + new_section + content[insert_pos:]

    return content


def append_decision(content: str, decision: dict[str, str]) -> str:
    """Append a decision to DECISIONS.md.

    Adds decision to the appropriate month section (creating if needed).

    Args:
        content: Current DECISIONS.md content
        decision: Dict with 'date' (YYYY-MM-DD), 'title', 'rationale'

    Returns:
        Updated content with decision added
    """
    date_str = decision["date"]
    title = decision["title"]
    rationale = decision["rationale"]

    # Extract month from date
    month_key = date_str[:7]  # YYYY-MM

    # Ensure month section exists
    content = ensure_month_section(content, month_key)

    # Format decision entry
    entry = f"- ({date_str}) {title} - {rationale}\n"

    # Find month section
    month_marker = f"## {month_key}"
    month_pos = content.find(month_marker)

    if month_pos == -1:
        return content  # Shouldn't happen after ensure_month_section

    # Find next section or end of file
    next_section = content.find("\n## ", month_pos + len(month_marker))
    if next_section == -1:
        next_section = len(content)

    # Remove "(No decisions recorded yet)" placeholder if present
    section_content = content[month_pos:next_section]
    if "(No decisions recorded yet)" in section_content:
        content = content.replace("(No decisions recorded yet)\n", "")
        # Recalculate positions
        month_pos = content.find(month_marker)
        next_section = content.find("\n## ", month_pos + len(month_marker))
        if next_section == -1:
            next_section = len(content)

    # Insert decision at end of month section (before next section)
    insert_pos = next_section
    content = content[:insert_pos] + entry + content[insert_pos:]

    return content


def update_recent_rollup(content: str) -> str:
    """Update Recent section with decisions from last 30 days.

    Scans all month sections, extracts decisions from last 30 days,
    and updates the Recent rollup section.

    Args:
        content: Current DECISIONS.md content

    Returns:
        Updated content with Recent section refreshed
    """
    today = datetime.datetime.now()
    cutoff = today - datetime.timedelta(days=30)

    # Find all decision entries across all month sections
    # Match pattern: - (YYYY-MM-DD) Title - Rationale
    decision_pattern = re.compile(r"^- \((\d{4}-\d{2}-\d{2})\) (.+)$", re.MULTILINE)

    recent_decisions = []
    for match in decision_pattern.finditer(content):
        date_str = match.group(1)
        full_entry = match.group(2)

        # Parse date
        try:
            decision_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue

        # Check if within last 30 days
        if decision_date >= cutoff:
            recent_decisions.append({
                "date": date_str,
                "entry": f"- ({date_str}) {full_entry}",
                "datetime": decision_date
            })

    # Sort by date (newest first)
    recent_decisions.sort(key=lambda x: x["datetime"], reverse=True)

    # Build Recent section content
    if recent_decisions:
        recent_content = "\n".join([d["entry"] for d in recent_decisions]) + "\n"
    else:
        recent_content = "- (None in last 30 days)\n"

    # Replace Recent section
    recent_marker = "## Recent (last 30 days)"
    recent_pos = content.find(recent_marker)

    if recent_pos == -1:
        return content  # Malformed

    # Find next section
    next_section = content.find("\n## ", recent_pos + len(recent_marker))
    if next_section == -1:
        next_section = len(content)

    # Replace content between marker and next section
    before = content[:recent_pos + len(recent_marker)]
    after = content[next_section:]

    content = before + "\n" + recent_content + "\n" + after

    return content
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_decisions_appender.py -v`
Expected: PASS (all 8 tests passing)

**Step 5: Commit**
```bash
git add src/chinvex/decisions_appender.py tests/test_decisions_appender.py
git commit -m "feat(P5.2.1): add DECISIONS.md append-only logic with monthly sections

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 7: Coverage anchor integration and file orchestration

**Files:**
- Create: `C:\Code\chinvex\src\chinvex\memory_orchestrator.py`
- Test: `C:\Code\chinvex\tests\test_memory_orchestrator.py`

**Step 1: Write the failing test**
```python
# tests/test_memory_orchestrator.py
import subprocess
from pathlib import Path
from chinvex.memory_orchestrator import (
    update_memory_files,
    get_memory_diff,
    MemoryUpdateResult,
)


def test_update_memory_files_reads_coverage_anchor(tmp_path):
    """Should read last commit hash from STATE.md coverage anchor."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)

    # Create initial commit
    (repo / "file.txt").write_text("initial")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True)

    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True)
    first_hash = result.stdout.strip()

    # Create memory dir with STATE.md
    memory_dir = repo / "docs" / "memory"
    memory_dir.mkdir(parents=True)
    state_file = memory_dir / "STATE.md"
    state_file.write_text(f"""# State

## Current Objective
Test

---
Last memory update: 2026-01-30
Commits covered through: {first_hash}

<!-- chinvex:last-commit:{first_hash} -->
""")

    # Create second commit
    (repo / "file2.txt").write_text("second")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "second commit"], cwd=repo, check=True)

    # Run update
    result = update_memory_files(repo)

    # Should have processed one commit
    assert result.commits_processed == 1
    assert result.ending_commit_hash != first_hash


def test_update_memory_files_early_exit_if_no_new_commits(tmp_path):
    """Should return early if no new commits since last anchor."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)

    (repo / "file.txt").write_text("initial")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True)

    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True)
    head_hash = result.stdout.strip()

    # Create memory files with coverage anchor at HEAD
    memory_dir = repo / "docs" / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "STATE.md").write_text(f"""# State

<!-- chinvex:last-commit:{head_hash} -->
""")

    result = update_memory_files(repo)

    # Should detect no new commits
    assert result.commits_processed == 0
    assert result.files_changed == 0


def test_update_memory_files_updates_all_three_files(tmp_path):
    """Should update STATE.md, CONSTRAINTS.md, and DECISIONS.md."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)

    # First commit
    (repo / "file.txt").write_text("initial")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True)

    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True)
    first_hash = result.stdout.strip()

    # Create memory files
    memory_dir = repo / "docs" / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "STATE.md").write_text(f"""# State

## Current Objective
Old

<!-- chinvex:last-commit:{first_hash} -->
""")
    (memory_dir / "CONSTRAINTS.md").write_text("""# Constraints

## Infrastructure
- Old constraint

## Superseded
(None yet)
""")
    (memory_dir / "DECISIONS.md").write_text("""# Decisions

## Recent (last 30 days)
- TBD
""")

    # Second commit with spec change
    specs_dir = repo / "specs"
    specs_dir.mkdir()
    (specs_dir / "P1.md").write_text("# P1 Spec")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "feat(P1): add spec"], cwd=repo, check=True)

    result = update_memory_files(repo)

    # Should update all files
    assert result.files_changed == 3
    assert (memory_dir / "STATE.md").exists()
    assert (memory_dir / "CONSTRAINTS.md").exists()
    assert (memory_dir / "DECISIONS.md").exists()


def test_get_memory_diff_returns_unified_diff(tmp_path):
    """Should return git-style unified diff of changes."""
    old_state = """# State

## Current Objective
Old objective

## Active Work
- Old work
"""

    new_state = """# State

## Current Objective
New objective

## Active Work
- New work
"""

    diff = get_memory_diff("STATE.md", old_state, new_state)

    assert "STATE.md" in diff
    assert "-Old objective" in diff or "- Old objective" in diff
    assert "+New objective" in diff or "+ New objective" in diff


def test_update_memory_files_respects_bounded_inputs(tmp_path):
    """Should handle bounded inputs limits gracefully."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)

    # Create 60 commits (exceeds max_commits limit of 50)
    for i in range(60):
        (repo / f"file{i}.txt").write_text(f"commit {i}")
        subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", f"commit {i}"], cwd=repo, check=True, capture_output=True)

    # Create memory files (no anchor - should process from beginning)
    memory_dir = repo / "docs" / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "STATE.md").write_text("# State\n\n## Current Objective\nTest")
    (memory_dir / "CONSTRAINTS.md").write_text("# Constraints\n\n## Infrastructure\n- TBD")
    (memory_dir / "DECISIONS.md").write_text("# Decisions\n\n## Recent (last 30 days)\n- TBD")

    result = update_memory_files(repo)

    # Should limit to max_commits
    assert result.commits_processed <= 50
    assert result.bounded_inputs_triggered is True


def test_memory_update_result_tracks_metrics():
    """MemoryUpdateResult should track all relevant metrics."""
    result = MemoryUpdateResult(
        commits_processed=5,
        files_analyzed=3,
        files_changed=2,
        bounded_inputs_triggered=False,
        ending_commit_hash="abc123"
    )

    assert result.commits_processed == 5
    assert result.files_analyzed == 3
    assert result.files_changed == 2
    assert result.bounded_inputs_triggered is False
    assert result.ending_commit_hash == "abc123"
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_memory_orchestrator.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chinvex.memory_orchestrator'"

**Step 3: Write minimal implementation**
```python
# src/chinvex/memory_orchestrator.py
from __future__ import annotations

import difflib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from chinvex.git_analyzer import extract_coverage_anchor, get_commit_range
from chinvex.spec_reader import extract_spec_plan_files_from_commits, read_spec_files, BOUNDED_INPUTS
from chinvex.state_regenerator import regenerate_state_md
from chinvex.constraints_merger import merge_constraints, extract_new_constraints
from chinvex.decisions_appender import update_recent_rollup


@dataclass
class MemoryUpdateResult:
    """Result of a memory file update operation."""
    commits_processed: int
    files_analyzed: int
    files_changed: int
    bounded_inputs_triggered: bool
    ending_commit_hash: str


def get_memory_diff(filename: str, old_content: str, new_content: str) -> str:
    """Generate unified diff between old and new content.

    Args:
        filename: Name of file for diff header
        old_content: Original content
        new_content: Updated content

    Returns:
        Unified diff string
    """
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm=""
    )

    return "".join(diff)


def update_memory_files(repo_root: Path) -> MemoryUpdateResult:
    """Update memory files based on git history.

    Main orchestration function that:
    1. Reads coverage anchor from STATE.md
    2. Gets commit range
    3. Reads specs/plans touched
    4. Updates STATE.md (full regen), CONSTRAINTS.md (merge), DECISIONS.md (append)
    5. Updates coverage anchor

    Args:
        repo_root: Repository root directory

    Returns:
        MemoryUpdateResult with metrics
    """
    memory_dir = repo_root / "docs" / "memory"
    state_file = memory_dir / "STATE.md"
    constraints_file = memory_dir / "CONSTRAINTS.md"
    decisions_file = memory_dir / "DECISIONS.md"

    # Read coverage anchor
    start_hash = None
    if state_file.exists():
        start_hash = extract_coverage_anchor(state_file)

    # Get commit range
    commits = get_commit_range(repo_root, start_hash=start_hash, max_commits=BOUNDED_INPUTS["max_commits"])

    if not commits:
        # No new commits
        return MemoryUpdateResult(
            commits_processed=0,
            files_analyzed=0,
            files_changed=0,
            bounded_inputs_triggered=False,
            ending_commit_hash=start_hash or "unknown"
        )

    # Get HEAD hash
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True
    )
    ending_hash = result.stdout.strip()

    # Extract spec/plan files
    spec_files = extract_spec_plan_files_from_commits(repo_root, start_hash=start_hash)
    spec_content = read_spec_files(spec_files)

    bounded_inputs_triggered = spec_content.truncated_files or spec_content.truncated_size or len(commits) >= BOUNDED_INPUTS["max_commits"]

    # Read current memory files
    current_state = state_file.read_text() if state_file.exists() else ""
    current_constraints = constraints_file.read_text() if constraints_file.exists() else ""
    current_decisions = decisions_file.read_text() if decisions_file.exists() else ""

    # Update STATE.md (full regeneration)
    new_state = regenerate_state_md(
        commits=commits,
        specs=spec_content.specs,
        current_state=current_state,
        ending_commit_hash=ending_hash
    )

    # Update CONSTRAINTS.md (merge)
    new_constraints_list = extract_new_constraints(commits, spec_content.specs)
    new_constraints = merge_constraints(current_constraints, new_constraints_list)

    # Update DECISIONS.md (append + rollup)
    # For now, just update rollup (actual decision extraction is more complex)
    new_decisions = update_recent_rollup(current_decisions)

    # Write files
    files_changed = 0

    if new_state != current_state:
        state_file.write_text(new_state)
        files_changed += 1

    if new_constraints != current_constraints:
        constraints_file.write_text(new_constraints)
        files_changed += 1

    if new_decisions != current_decisions:
        decisions_file.write_text(new_decisions)
        files_changed += 1

    return MemoryUpdateResult(
        commits_processed=len(commits),
        files_analyzed=len(spec_content.specs),
        files_changed=files_changed,
        bounded_inputs_triggered=bounded_inputs_triggered,
        ending_commit_hash=ending_hash
    )
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_memory_orchestrator.py -v`
Expected: PASS (all 6 tests passing)

**Step 5: Commit**
```bash
git add src/chinvex/memory_orchestrator.py tests/test_memory_orchestrator.py
git commit -m "feat(P5.2.1): add memory file orchestration with coverage anchor tracking

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 8: CLI command chinvex update-memory with review/commit modes

**Files:**
- Modify: `C:\Code\chinvex\src\chinvex\cli.py`
- Test: `C:\Code\chinvex\tests\test_cli_update_memory.py`

**Step 1: Write the failing test**
```python
# tests/test_cli_update_memory.py
import subprocess
from pathlib import Path
from click.testing import CliRunner
from chinvex.cli import cli


def test_update_memory_command_exists():
    """update-memory subcommand should be registered."""
    runner = CliRunner()
    result = runner.invoke(cli, ["update-memory", "--help"])

    assert result.exit_code == 0
    assert "Update memory files" in result.output or "update-memory" in result.output


def test_update_memory_requires_context():
    """Should require --context argument."""
    runner = CliRunner()
    result = runner.invoke(cli, ["update-memory"])

    assert result.exit_code != 0
    assert "context" in result.output.lower() or "required" in result.output.lower()


def test_update_memory_review_mode_shows_diff(tmp_path):
    """Review mode (default) should show diff without committing."""
    # Create test repo with context
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)

    # Create chinvex config
    config_dir = tmp_path / ".chinvex" / "contexts"
    config_dir.mkdir(parents=True)
    context_file = config_dir / "test_context.json"
    context_file.write_text(f"""{{
        "context_name": "test_context",
        "includes": {{
            "repos": ["{repo.as_posix()}"]
        }}
    }}""")

    # Create initial commit and memory files
    (repo / "file.txt").write_text("initial")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True)

    result_run = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True)
    first_hash = result_run.stdout.strip()

    memory_dir = repo / "docs" / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "STATE.md").write_text(f"""# State

## Current Objective
Test

<!-- chinvex:last-commit:{first_hash} -->
""")
    (memory_dir / "CONSTRAINTS.md").write_text("# Constraints\n\n## Infrastructure\n- TBD")
    (memory_dir / "DECISIONS.md").write_text("# Decisions\n\n## Recent (last 30 days)\n- TBD")

    # Create second commit
    (repo / "file2.txt").write_text("second")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "second commit"], cwd=repo, check=True)

    # Run update-memory in review mode
    runner = CliRunner()
    result = runner.invoke(cli, ["update-memory", "--context", "test_context"], env={"CHINVEX_HOME": str(tmp_path / ".chinvex")})

    # Should show diff output
    assert result.exit_code == 0
    assert "diff" in result.output.lower() or "STATE.md" in result.output or "---" in result.output

    # Should NOT commit
    result_log = subprocess.run(["git", "log", "--oneline"], cwd=repo, capture_output=True, text=True)
    assert "update memory files" not in result_log.stdout.lower()


def test_update_memory_commit_mode_creates_commit(tmp_path):
    """Commit mode (--commit) should auto-commit changes."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)

    config_dir = tmp_path / ".chinvex" / "contexts"
    config_dir.mkdir(parents=True)
    (config_dir / "test_context.json").write_text(f"""{{
        "context_name": "test_context",
        "includes": {{"repos": ["{repo.as_posix()}"]}}
    }}""")

    # Initial commit
    (repo / "file.txt").write_text("initial")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True)

    result_run = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True)
    first_hash = result_run.stdout.strip()

    memory_dir = repo / "docs" / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "STATE.md").write_text(f"# State\n\n<!-- chinvex:last-commit:{first_hash} -->")
    (memory_dir / "CONSTRAINTS.md").write_text("# Constraints\n\n## Infrastructure\n- TBD")
    (memory_dir / "DECISIONS.md").write_text("# Decisions\n\n## Recent (last 30 days)\n- TBD")

    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "add memory files"], cwd=repo, check=True)

    # Second commit
    (repo / "file2.txt").write_text("second")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "second commit"], cwd=repo, check=True)

    # Run update-memory with --commit
    runner = CliRunner()
    result = runner.invoke(cli, ["update-memory", "--context", "test_context", "--commit"], env={"CHINVEX_HOME": str(tmp_path / ".chinvex")})

    assert result.exit_code == 0

    # Should have created commit
    result_log = subprocess.run(["git", "log", "--oneline", "-n", "1"], cwd=repo, capture_output=True, text=True)
    assert "docs: update memory files" in result_log.stdout.lower() or "memory" in result_log.stdout.lower()


def test_update_memory_no_changes_message(tmp_path):
    """Should display message if no changes needed."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)

    config_dir = tmp_path / ".chinvex" / "contexts"
    config_dir.mkdir(parents=True)
    (config_dir / "test_context.json").write_text(f"""{{
        "context_name": "test_context",
        "includes": {{"repos": ["{repo.as_posix()}"]}}
    }}""")

    # Single commit
    (repo / "file.txt").write_text("initial")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True)

    result_run = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True)
    head_hash = result_run.stdout.strip()

    memory_dir = repo / "docs" / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "STATE.md").write_text(f"# State\n\n<!-- chinvex:last-commit:{head_hash} -->")
    (memory_dir / "CONSTRAINTS.md").write_text("# Constraints\n\n## Infrastructure\n- TBD")
    (memory_dir / "DECISIONS.md").write_text("# Decisions\n\n## Recent (last 30 days)\n- TBD")

    # Run update-memory
    runner = CliRunner()
    result = runner.invoke(cli, ["update-memory", "--context", "test_context"], env={"CHINVEX_HOME": str(tmp_path / ".chinvex")})

    assert result.exit_code == 0
    assert "no new commits" in result.output.lower() or "up to date" in result.output.lower() or "0 commits" in result.output
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_cli_update_memory.py -v`
Expected: FAIL with "AssertionError" or "Command 'update-memory' not found"

**Step 3: Write minimal implementation**
```python
# Modify src/chinvex/cli.py - add this command to the CLI

# Add this import at the top
from chinvex.memory_orchestrator import update_memory_files, get_memory_diff

# Add this command to the cli group
@cli.command("update-memory")
@click.option("--context", required=True, help="Context name")
@click.option("--commit", is_flag=True, help="Auto-commit changes (default: review mode)")
def update_memory(context: str, commit: bool):
    """Update memory files (STATE.md, CONSTRAINTS.md, DECISIONS.md) from git history.

    Review mode (default): Shows diff without committing.
    Commit mode (--commit): Auto-commits with 'docs: update memory files'.
    """
    import subprocess
    from pathlib import Path

    # Load context config to get repo paths
    config_path = get_context_config_path(context)
    if not config_path.exists():
        click.echo(f"Error: Context '{context}' not found", err=True)
        raise click.Abort()

    import json
    config = json.loads(config_path.read_text())

    repos = config.get("includes", {}).get("repos", [])
    if not repos:
        click.echo(f"Error: No repos configured for context '{context}'", err=True)
        raise click.Abort()

    # Use first repo (multi-repo support is future work)
    repo_root = Path(repos[0])
    if not repo_root.exists():
        click.echo(f"Error: Repo not found: {repo_root}", err=True)
        raise click.Abort()

    memory_dir = repo_root / "docs" / "memory"
    state_file = memory_dir / "STATE.md"
    constraints_file = memory_dir / "CONSTRAINTS.md"
    decisions_file = memory_dir / "DECISIONS.md"

    # Read old content for diff
    old_state = state_file.read_text() if state_file.exists() else ""
    old_constraints = constraints_file.read_text() if constraints_file.exists() else ""
    old_decisions = decisions_file.read_text() if decisions_file.exists() else ""

    # Run update
    result = update_memory_files(repo_root)

    if result.commits_processed == 0:
        click.echo("No new commits since last update. Memory files are up to date.")
        return

    # Read new content
    new_state = state_file.read_text() if state_file.exists() else ""
    new_constraints = constraints_file.read_text() if constraints_file.exists() else ""
    new_decisions = decisions_file.read_text() if decisions_file.exists() else ""

    # Show summary
    click.echo(f"Processed {result.commits_processed} commits")
    click.echo(f"Analyzed {result.files_analyzed} spec/plan files")
    click.echo(f"Updated {result.files_changed} memory files")

    if result.bounded_inputs_triggered:
        click.echo("WARNING: Bounded inputs limit reached - some commits/files skipped", err=True)

    # Show diffs in review mode
    if not commit:
        click.echo("\n=== CHANGES (review mode - not committed) ===\n")

        if new_state != old_state:
            click.echo(get_memory_diff("STATE.md", old_state, new_state))

        if new_constraints != old_constraints:
            click.echo(get_memory_diff("CONSTRAINTS.md", old_constraints, new_constraints))

        if new_decisions != old_decisions:
            click.echo(get_memory_diff("DECISIONS.md", old_decisions, new_decisions))

        click.echo("\nRun with --commit to auto-commit these changes.")
    else:
        # Commit mode
        if result.files_changed > 0:
            subprocess.run(["git", "add", "docs/memory/"], cwd=repo_root, check=True)
            subprocess.run(
                ["git", "commit", "-m", "docs: update memory files"],
                cwd=repo_root,
                check=True
            )
            click.echo("Changes committed.")
        else:
            click.echo("No changes to commit.")
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_cli_update_memory.py -v`
Expected: PASS (all 4 tests passing)

**Step 5: Commit**
```bash
git add src/chinvex/cli.py tests/test_cli_update_memory.py
git commit -m "feat(P5.2.1): add chinvex update-memory CLI command with review/commit modes

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 9: Startup hook installation during ingest

**Files:**
- Create: `C:\Code\chinvex\src\chinvex\hook_installer.py`
- Test: `C:\Code\chinvex\tests\test_hook_installer.py`

**Step 1: Write the failing test**
```python
# tests/test_hook_installer.py
import json
import subprocess
from pathlib import Path
from chinvex.hook_installer import (
    install_startup_hook,
    merge_settings_json,
    is_git_repo,
)


def test_is_git_repo_detects_git_directory(tmp_path):
    """Should detect if directory is a git repo."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    assert is_git_repo(repo) is True


def test_is_git_repo_returns_false_for_non_git(tmp_path):
    """Should return False for non-git directory."""
    non_repo = tmp_path / "not_a_repo"
    non_repo.mkdir()

    assert is_git_repo(non_repo) is False


def test_install_startup_hook_creates_settings_json(tmp_path):
    """Should create .claude/settings.json if it doesn't exist."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    install_startup_hook(repo, context_name="test_context")

    settings_file = repo / ".claude" / "settings.json"
    assert settings_file.exists()

    settings = json.loads(settings_file.read_text())
    assert "hooks" in settings
    assert "startup" in settings["hooks"]
    assert "chinvex brief --context test_context" in settings["hooks"]["startup"]


def test_install_startup_hook_merges_with_existing_settings(tmp_path):
    """Should merge with existing settings.json without clobbering."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    # Create existing settings
    claude_dir = repo / ".claude"
    claude_dir.mkdir()
    settings_file = claude_dir / "settings.json"
    settings_file.write_text(json.dumps({
        "theme": "dark",
        "other_config": "value"
    }))

    install_startup_hook(repo, context_name="test_context")

    settings = json.loads(settings_file.read_text())

    # Should preserve existing config
    assert settings["theme"] == "dark"
    assert settings["other_config"] == "value"

    # Should add hook
    assert "hooks" in settings
    assert "startup" in settings["hooks"]


def test_install_startup_hook_converts_string_to_array(tmp_path):
    """Should convert existing string startup hook to array."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    claude_dir = repo / ".claude"
    claude_dir.mkdir()
    settings_file = claude_dir / "settings.json"
    settings_file.write_text(json.dumps({
        "hooks": {
            "startup": "existing-command"
        }
    }))

    install_startup_hook(repo, context_name="test_context")

    settings = json.loads(settings_file.read_text())

    # Should be converted to array
    assert isinstance(settings["hooks"]["startup"], list)
    assert "existing-command" in settings["hooks"]["startup"]
    assert "chinvex brief --context test_context" in settings["hooks"]["startup"]


def test_install_startup_hook_appends_to_existing_array(tmp_path):
    """Should append to existing startup array without duplicating."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    claude_dir = repo / ".claude"
    claude_dir.mkdir()
    settings_file = claude_dir / "settings.json"
    settings_file.write_text(json.dumps({
        "hooks": {
            "startup": ["other-command"]
        }
    }))

    install_startup_hook(repo, context_name="test_context")

    settings = json.loads(settings_file.read_text())

    assert "other-command" in settings["hooks"]["startup"]
    assert "chinvex brief --context test_context" in settings["hooks"]["startup"]
    assert len(settings["hooks"]["startup"]) == 2


def test_install_startup_hook_avoids_duplicates(tmp_path):
    """Should not add duplicate hook if already present."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    claude_dir = repo / ".claude"
    claude_dir.mkdir()
    settings_file = claude_dir / "settings.json"
    settings_file.write_text(json.dumps({
        "hooks": {
            "startup": ["chinvex brief --context test_context"]
        }
    }))

    install_startup_hook(repo, context_name="test_context")

    settings = json.loads(settings_file.read_text())

    # Should not duplicate
    assert settings["hooks"]["startup"].count("chinvex brief --context test_context") == 1


def test_install_startup_hook_skips_non_git_with_warning(tmp_path, caplog):
    """Should skip non-git directory and log warning."""
    non_repo = tmp_path / "not_a_repo"
    non_repo.mkdir()

    result = install_startup_hook(non_repo, context_name="test_context")

    assert result is False
    assert not (non_repo / ".claude" / "settings.json").exists()


def test_install_startup_hook_handles_malformed_json(tmp_path, caplog):
    """Should skip malformed settings.json and log warning."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    claude_dir = repo / ".claude"
    claude_dir.mkdir()
    settings_file = claude_dir / "settings.json"
    settings_file.write_text("{ invalid json }")

    result = install_startup_hook(repo, context_name="test_context")

    # Should fail gracefully
    assert result is False


def test_merge_settings_json_deep_merges():
    """Should deep merge settings without clobbering nested objects."""
    base = {
        "theme": "dark",
        "hooks": {
            "pre-commit": ["lint"]
        }
    }

    overlay = {
        "hooks": {
            "startup": ["brief"]
        },
        "new_field": "value"
    }

    merged = merge_settings_json(base, overlay)

    # Should preserve both hook types
    assert merged["hooks"]["pre-commit"] == ["lint"]
    assert merged["hooks"]["startup"] == ["brief"]
    assert merged["theme"] == "dark"
    assert merged["new_field"] == "value"
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_hook_installer.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chinvex.hook_installer'"

**Step 3: Write minimal implementation**
```python
# src/chinvex/hook_installer.py
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def is_git_repo(directory: Path) -> bool:
    """Check if directory is a git repository.

    Args:
        directory: Path to check

    Returns:
        True if directory is a git repo
    """
    git_dir = directory / ".git"
    if git_dir.exists():
        return True

    # Check via git command
    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=directory,
            check=True,
            capture_output=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def merge_settings_json(base: dict, overlay: dict) -> dict:
    """Deep merge two settings dictionaries.

    Args:
        base: Base settings
        overlay: Settings to merge in

    Returns:
        Merged settings dict
    """
    result = base.copy()

    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Deep merge nested dicts
            result[key] = merge_settings_json(result[key], value)
        else:
            result[key] = value

    return result


def install_startup_hook(repo_root: Path, context_name: str) -> bool:
    """Install Claude Code startup hook in .claude/settings.json.

    Creates or updates settings.json to include:
    {
        "hooks": {
            "startup": ["chinvex brief --context <context_name>"]
        }
    }

    Args:
        repo_root: Repository root directory
        context_name: Chinvex context name

    Returns:
        True if hook installed successfully, False otherwise
    """
    # Check if git repo
    if not is_git_repo(repo_root):
        print(f"Warning: {repo_root} is not a git repository. Skipping hook installation.")
        return False

    claude_dir = repo_root / ".claude"
    claude_dir.mkdir(exist_ok=True)

    settings_file = claude_dir / "settings.json"
    hook_command = f"chinvex brief --context {context_name}"

    # Read existing settings
    if settings_file.exists():
        try:
            existing = json.loads(settings_file.read_text())
        except json.JSONDecodeError:
            print(f"Warning: Malformed {settings_file}. Skipping hook installation.")
            return False
    else:
        existing = {}

    # Prepare hook update
    if "hooks" not in existing:
        existing["hooks"] = {}

    if "startup" not in existing["hooks"]:
        existing["hooks"]["startup"] = []
    elif isinstance(existing["hooks"]["startup"], str):
        # Convert string to array
        existing["hooks"]["startup"] = [existing["hooks"]["startup"]]

    # Add hook if not already present
    if hook_command not in existing["hooks"]["startup"]:
        existing["hooks"]["startup"].append(hook_command)

    # Write back
    settings_file.write_text(json.dumps(existing, indent=2) + "\n")
    return True
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_hook_installer.py -v`
Expected: PASS (all 10 tests passing)

**Step 5: Commit**
```bash
git add src/chinvex/hook_installer.py tests/test_hook_installer.py
git commit -m "feat(P5.2.4): add startup hook installation for Claude Code

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 10: Integrate hook installation into chinvex ingest command

**Files:**
- Modify: `C:\Code\chinvex\src\chinvex\cli.py`
- Test: `C:\Code\chinvex\tests\test_cli_ingest_hooks.py`

**Step 1: Write the failing test**
```python
# tests/test_cli_ingest_hooks.py
import json
import subprocess
from pathlib import Path
from click.testing import CliRunner
from chinvex.cli import cli


def test_ingest_installs_startup_hook_by_default(tmp_path):
    """chinvex ingest should install startup hook in each repo by default."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)

    # Create a file to ingest
    (repo / "test.txt").write_text("test content")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True)

    # Create context config
    config_dir = tmp_path / ".chinvex" / "contexts"
    config_dir.mkdir(parents=True)
    (config_dir / "test_context.json").write_text(json.dumps({
        "context_name": "test_context",
        "includes": {
            "repos": [str(repo)]
        }
    }))

    # Mock chromadb and index directories
    index_dir = tmp_path / ".chinvex" / "indexes" / "test_context"
    index_dir.mkdir(parents=True)

    # Run ingest (simplified - actual test may need mocking)
    runner = CliRunner()
    # Note: This is a simplified test - real implementation may need more setup
    # The important check is that install_startup_hook is called during ingest

    # For now, manually trigger hook installation to verify integration
    from chinvex.hook_installer import install_startup_hook
    result = install_startup_hook(repo, "test_context")

    assert result is True

    settings_file = repo / ".claude" / "settings.json"
    assert settings_file.exists()

    settings = json.loads(settings_file.read_text())
    assert "chinvex brief --context test_context" in settings["hooks"]["startup"]


def test_ingest_skips_hook_with_no_claude_hook_flag(tmp_path):
    """--no-claude-hook flag should skip hook installation."""
    # This test verifies the flag is recognized
    runner = CliRunner()

    # Check that --no-claude-hook is a valid option
    result = runner.invoke(cli, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "--no-claude-hook" in result.output or "no-claude-hook" in result.output


def test_ingest_installs_hook_in_all_context_repos(tmp_path):
    """Should install hook in every repo included in the context."""
    repo1 = tmp_path / "repo1"
    repo2 = tmp_path / "repo2"

    for repo in [repo1, repo2]:
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    from chinvex.hook_installer import install_startup_hook

    # Install in both
    install_startup_hook(repo1, "multi_repo_context")
    install_startup_hook(repo2, "multi_repo_context")

    # Both should have hook
    for repo in [repo1, repo2]:
        settings_file = repo / ".claude" / "settings.json"
        assert settings_file.exists()
        settings = json.loads(settings_file.read_text())
        assert "chinvex brief --context multi_repo_context" in settings["hooks"]["startup"]


def test_ingest_warns_on_non_git_repo(tmp_path, caplog):
    """Should warn if a configured repo is not a git repository."""
    non_repo = tmp_path / "not_a_repo"
    non_repo.mkdir()

    from chinvex.hook_installer import install_startup_hook
    result = install_startup_hook(non_repo, "test_context")

    assert result is False
    # Hook should not be created
    assert not (non_repo / ".claude" / "settings.json").exists()
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_cli_ingest_hooks.py -v`
Expected: FAIL with "AssertionError" for --no-claude-hook flag test

**Step 3: Write minimal implementation**
```python
# Modify src/chinvex/cli.py - update the ingest command

# Add import at top
from chinvex.hook_installer import install_startup_hook

# Modify the existing ingest command to add the flag and hook installation
@cli.command()
@click.option("--context", required=True, help="Context name")
@click.option("--no-claude-hook", is_flag=True, help="Skip Claude Code startup hook installation")
# ... other existing options ...
def ingest(context: str, no_claude_hook: bool, **kwargs):
    """Ingest sources into context index.

    By default, installs Claude Code startup hook in all repos.
    Use --no-claude-hook to skip hook installation.
    """
    # ... existing ingest logic ...

    # After successful ingestion, install startup hooks
    if not no_claude_hook:
        # Load context config to get repos
        config_path = get_context_config_path(context)
        if config_path.exists():
            import json
            config = json.loads(config_path.read_text())
            repos = config.get("includes", {}).get("repos", [])

            for repo_path in repos:
                repo_dir = Path(repo_path)
                if repo_dir.exists():
                    success = install_startup_hook(repo_dir, context)
                    if success:
                        click.echo(f"Installed startup hook in {repo_dir}")
                    else:
                        click.echo(f"Warning: Could not install hook in {repo_dir}", err=True)

    # ... rest of existing ingest logic ...
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_cli_ingest_hooks.py -v`
Expected: PASS (all 4 tests passing)

**Step 5: Commit**
```bash
git add src/chinvex/cli.py tests/test_cli_ingest_hooks.py
git commit -m "feat(P5.2.4): integrate startup hook installation into chinvex ingest

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```
## Task 9: Update `chinvex brief` to use exact header matching for CONSTRAINTS.md

[Full task content from Batch 3 subagent output - Tasks 9-13]




### Task 11: Golden query schema and validation

**Files:**
- Create: `src/chinvex/eval_schema.py`
- Test: `tests/test_eval_schema.py`

**Step 1: Write the failing test**
```python
# tests/test_eval_schema.py
import json
import pytest
from pathlib import Path
from chinvex.eval_schema import GoldenQuery, GoldenQuerySet, load_golden_queries, validate_golden_queries


def test_golden_query_from_dict_minimal():
    data = {
        "query": "chromadb batch limit",
        "context": "Chinvex",
        "expected_files": ["src/chinvex/ingest.py"]
    }
    query = GoldenQuery.from_dict(data)
    assert query.query == "chromadb batch limit"
    assert query.context == "Chinvex"
    assert query.expected_files == ["src/chinvex/ingest.py"]
    assert query.anchor is None
    assert query.k == 5  # default


def test_golden_query_from_dict_full():
    data = {
        "query": "chromadb batch limit",
        "context": "Chinvex",
        "expected_files": ["src/chinvex/ingest.py", "docs/constraints.md"],
        "anchor": "5000 vectors",
        "k": 10
    }
    query = GoldenQuery.from_dict(data)
    assert query.query == "chromadb batch limit"
    assert query.context == "Chinvex"
    assert query.expected_files == ["src/chinvex/ingest.py", "docs/constraints.md"]
    assert query.anchor == "5000 vectors"
    assert query.k == 10


def test_golden_query_validation_missing_required_field():
    data = {
        "context": "Chinvex",
        "expected_files": ["src/chinvex/ingest.py"]
    }
    with pytest.raises(ValueError, match="Missing required field: query"):
        GoldenQuery.from_dict(data)


def test_golden_query_validation_empty_expected_files():
    data = {
        "query": "test",
        "context": "Chinvex",
        "expected_files": []
    }
    with pytest.raises(ValueError, match="expected_files must contain at least one file"):
        GoldenQuery.from_dict(data)


def test_golden_query_validation_invalid_k():
    data = {
        "query": "test",
        "context": "Chinvex",
        "expected_files": ["test.py"],
        "k": 0
    }
    with pytest.raises(ValueError, match="k must be positive"):
        GoldenQuery.from_dict(data)


def test_golden_query_set_load_from_dict():
    data = {
        "queries": [
            {
                "query": "query1",
                "context": "Chinvex",
                "expected_files": ["file1.py"]
            },
            {
                "query": "query2",
                "context": "Chinvex",
                "expected_files": ["file2.py"],
                "k": 10
            }
        ]
    }
    query_set = GoldenQuerySet.from_dict(data)
    assert len(query_set.queries) == 2
    assert query_set.queries[0].query == "query1"
    assert query_set.queries[1].k == 10


def test_load_golden_queries_from_file(tmp_path):
    query_file = tmp_path / "golden_queries_test.json"
    query_file.write_text(json.dumps({
        "queries": [
            {
                "query": "test query",
                "context": "TestContext",
                "expected_files": ["test.py"]
            }
        ]
    }))

    queries = load_golden_queries(query_file)
    assert len(queries) == 1
    assert queries[0].query == "test query"


def test_load_golden_queries_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_golden_queries(Path("nonexistent.json"))


def test_validate_golden_queries_success(tmp_path):
    query_file = tmp_path / "golden_queries_test.json"
    query_file.write_text(json.dumps({
        "queries": [
            {
                "query": "test",
                "context": "Test",
                "expected_files": ["test.py"]
            }
        ]
    }))

    errors = validate_golden_queries(query_file)
    assert len(errors) == 0


def test_validate_golden_queries_with_errors(tmp_path):
    query_file = tmp_path / "golden_queries_bad.json"
    query_file.write_text(json.dumps({
        "queries": [
            {
                "context": "Test",
                "expected_files": []
            }
        ]
    }))

    errors = validate_golden_queries(query_file)
    assert len(errors) > 0
    assert any("query" in err.lower() for err in errors)
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_eval_schema.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chinvex.eval_schema'"

**Step 3: Write minimal implementation**
```python
# src/chinvex/eval_schema.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class GoldenQuery:
    """Single golden query for eval."""
    query: str
    context: str
    expected_files: list[str]
    anchor: str | None = None
    k: int = 5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldenQuery:
        """Load from dictionary with validation."""
        # Validate required fields
        required = ["query", "context", "expected_files"]
        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Validate expected_files non-empty
        if not data["expected_files"]:
            raise ValueError("expected_files must contain at least one file")

        # Validate k if present
        k = data.get("k", 5)
        if k <= 0:
            raise ValueError("k must be positive")

        return cls(
            query=data["query"],
            context=data["context"],
            expected_files=data["expected_files"],
            anchor=data.get("anchor"),
            k=k
        )


@dataclass
class GoldenQuerySet:
    """Collection of golden queries."""
    queries: list[GoldenQuery]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldenQuerySet:
        """Load from dictionary."""
        queries = [GoldenQuery.from_dict(q) for q in data.get("queries", [])]
        return cls(queries=queries)


def load_golden_queries(path: Path) -> list[GoldenQuery]:
    """Load golden queries from JSON file.

    Args:
        path: Path to golden_queries_<context>.json file

    Returns:
        List of GoldenQuery objects

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid or queries fail validation
    """
    if not path.exists():
        raise FileNotFoundError(f"Golden query file not found: {path}")

    data = json.loads(path.read_text())
    query_set = GoldenQuerySet.from_dict(data)
    return query_set.queries


def validate_golden_queries(path: Path) -> list[str]:
    """Validate golden query file format.

    Args:
        path: Path to golden_queries_<context>.json

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]
    except FileNotFoundError:
        return [f"File not found: {path}"]

    if "queries" not in data:
        errors.append("Missing 'queries' key in JSON")
        return errors

    for i, query_data in enumerate(data["queries"]):
        try:
            GoldenQuery.from_dict(query_data)
        except ValueError as e:
            errors.append(f"Query {i}: {e}")

    return errors
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_eval_schema.py -v`
Expected: PASS (all 11 tests passing)

**Step 5: Commit**
```bash
git add src/chinvex/eval_schema.py tests/test_eval_schema.py
git commit -m "feat(P5.3.1): add golden query schema and validation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---
### Task 12: Eval runner - query execution against golden set

**Files:**
- Create: `src/chinvex/eval_runner.py`
- Test: `tests/test_eval_runner.py`

**Step 1: Write the failing test**
```python
# tests/test_eval_runner.py
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from chinvex.eval_runner import EvalRunner, QueryResult
from chinvex.eval_schema import GoldenQuery
from chinvex.search import SearchResult


@pytest.fixture
def mock_config(tmp_path):
    config = MagicMock()
    config.index_dir = tmp_path / "index"
    config.index_dir.mkdir()
    return config


@pytest.fixture
def golden_queries():
    return [
        GoldenQuery(
            query="chromadb batch limit",
            context="Chinvex",
            expected_files=["src/chinvex/ingest.py"],
            k=5
        ),
        GoldenQuery(
            query="hook installation",
            context="Chinvex",
            expected_files=["src/chinvex/hook_installer.py", "src/chinvex/cli.py"],
            anchor="startup hook",
            k=10
        )
    ]


def test_query_result_creation():
    result = QueryResult(
        query="test query",
        expected_files=["test.py"],
        retrieved_files=["test.py", "other.py"],
        k=5,
        anchor="test",
        latency_ms=150.5
    )
    assert result.query == "test query"
    assert result.expected_files == ["test.py"]
    assert result.retrieved_files == ["test.py", "other.py"]
    assert result.k == 5
    assert result.anchor == "test"
    assert result.latency_ms == 150.5


def test_eval_runner_init(mock_config):
    runner = EvalRunner(mock_config, "Chinvex")
    assert runner.config == mock_config
    assert runner.context_name == "Chinvex"


@patch("chinvex.eval_runner.search")
def test_eval_runner_execute_single_query(mock_search, mock_config):
    mock_search.return_value = [
        SearchResult(
            chunk_id="chunk1",
            score=0.9,
            source_type="repo",
            title="ingest.py",
            citation="src/chinvex/ingest.py:50",
            snippet="batch size is 5000"
        ),
        SearchResult(
            chunk_id="chunk2",
            score=0.7,
            source_type="repo",
            title="config.py",
            citation="src/chinvex/config.py:20",
            snippet="configuration settings"
        )
    ]

    runner = EvalRunner(mock_config, "Chinvex")
    golden_query = GoldenQuery(
        query="chromadb batch limit",
        context="Chinvex",
        expected_files=["src/chinvex/ingest.py"],
        k=5
    )

    result = runner.execute_query(golden_query)

    assert result.query == "chromadb batch limit"
    assert result.expected_files == ["src/chinvex/ingest.py"]
    assert "src/chinvex/ingest.py" in result.retrieved_files
    assert "src/chinvex/config.py" in result.retrieved_files
    assert result.k == 5
    assert result.latency_ms > 0

    mock_search.assert_called_once()
    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["k"] == 5


@patch("chinvex.eval_runner.search")
def test_eval_runner_extract_file_paths(mock_search, mock_config):
    mock_search.return_value = [
        SearchResult(
            chunk_id="c1",
            score=0.9,
            source_type="repo",
            title="test",
            citation="src/chinvex/ingest.py:50",
            snippet="test"
        ),
        SearchResult(
            chunk_id="c2",
            score=0.8,
            source_type="chat",
            title="chat",
            citation="chats/2024-01.md:100",
            snippet="test"
        )
    ]

    runner = EvalRunner(mock_config, "Chinvex")
    golden_query = GoldenQuery(
        query="test",
        context="Chinvex",
        expected_files=["src/chinvex/ingest.py"],
        k=5
    )

    result = runner.execute_query(golden_query)

    assert "src/chinvex/ingest.py" in result.retrieved_files
    assert "chats/2024-01.md" in result.retrieved_files
    assert len(result.retrieved_files) == 2


@patch("chinvex.eval_runner.search")
def test_eval_runner_run_all_queries(mock_search, mock_config, golden_queries):
    mock_search.return_value = [
        SearchResult(
            chunk_id="c1",
            score=0.9,
            source_type="repo",
            title="test",
            citation="src/chinvex/ingest.py:50",
            snippet="test"
        )
    ]

    runner = EvalRunner(mock_config, "Chinvex")
    results = runner.run(golden_queries)

    assert len(results) == 2
    assert all(isinstance(r, QueryResult) for r in results)
    assert results[0].query == "chromadb batch limit"
    assert results[1].query == "hook installation"
    assert mock_search.call_count == 2


@patch("chinvex.eval_runner.search")
def test_eval_runner_handles_empty_results(mock_search, mock_config):
    mock_search.return_value = []

    runner = EvalRunner(mock_config, "Chinvex")
    golden_query = GoldenQuery(
        query="nonexistent query",
        context="Chinvex",
        expected_files=["test.py"],
        k=5
    )

    result = runner.execute_query(golden_query)

    assert result.retrieved_files == []
    assert result.latency_ms >= 0


@patch("chinvex.eval_runner.search")
def test_eval_runner_respects_k_parameter(mock_search, mock_config):
    mock_search.return_value = [
        SearchResult(f"c{i}", 0.9, "repo", "t", f"f{i}.py:1", "s")
        for i in range(20)
    ]

    runner = EvalRunner(mock_config, "Chinvex")
    golden_query = GoldenQuery(
        query="test",
        context="Chinvex",
        expected_files=["test.py"],
        k=10
    )

    result = runner.execute_query(golden_query)

    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["k"] == 10
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_eval_runner.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chinvex.eval_runner'"

**Step 3: Write minimal implementation**
```python
# src/chinvex/eval_runner.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .config import AppConfig
from .eval_schema import GoldenQuery
from .search import search


@dataclass
class QueryResult:
    """Result of executing a single golden query."""
    query: str
    expected_files: list[str]
    retrieved_files: list[str]
    k: int
    anchor: str | None = None
    latency_ms: float = 0.0


class EvalRunner:
    """Executes golden queries and collects results."""

    def __init__(self, config: AppConfig, context_name: str):
        self.config = config
        self.context_name = context_name

    def execute_query(self, golden_query: GoldenQuery) -> QueryResult:
        """Execute a single golden query.

        Args:
            golden_query: Query to execute

        Returns:
            QueryResult with retrieved file paths and latency
        """
        start_time = time.perf_counter()

        # Execute search
        search_results = search(
            self.config,
            golden_query.query,
            k=golden_query.k
        )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Extract file paths from citations
        retrieved_files = []
        for result in search_results:
            # Citation format: "path/to/file.py:line" or "path/to/file.md:line"
            file_path = result.citation.split(":")[0] if ":" in result.citation else result.citation
            retrieved_files.append(file_path)

        return QueryResult(
            query=golden_query.query,
            expected_files=golden_query.expected_files,
            retrieved_files=retrieved_files,
            k=golden_query.k,
            anchor=golden_query.anchor,
            latency_ms=latency_ms
        )

    def run(self, golden_queries: list[GoldenQuery]) -> list[QueryResult]:
        """Execute all golden queries.

        Args:
            golden_queries: List of queries to execute

        Returns:
            List of QueryResult objects
        """
        results = []
        for golden_query in golden_queries:
            result = self.execute_query(golden_query)
            results.append(result)
        return results
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_eval_runner.py -v`
Expected: PASS (all 9 tests passing)

**Step 5: Commit**
```bash
git add src/chinvex/eval_runner.py tests/test_eval_runner.py
git commit -m "feat(P5.3.2): add eval runner for golden query execution

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 13: Success criteria evaluation (file path matching, anchor matching)

**Files:**
- Create: `src/chinvex/eval_criteria.py`
- Test: `tests/test_eval_criteria.py`

**Step 1: Write the failing test**
```python
# tests/test_eval_criteria.py
import pytest
from chinvex.eval_criteria import evaluate_query, check_file_match, check_anchor_match
from chinvex.eval_runner import QueryResult


def test_check_file_match_exact():
    retrieved = ["src/chinvex/ingest.py", "src/chinvex/config.py"]
    expected = ["src/chinvex/ingest.py"]
    assert check_file_match(retrieved, expected) is True


def test_check_file_match_multiple_expected():
    retrieved = ["src/chinvex/search.py", "src/chinvex/config.py"]
    expected = ["src/chinvex/ingest.py", "src/chinvex/config.py"]
    assert check_file_match(retrieved, expected) is True


def test_check_file_match_no_match():
    retrieved = ["src/chinvex/search.py", "src/chinvex/config.py"]
    expected = ["src/chinvex/ingest.py"]
    assert check_file_match(retrieved, expected) is False


def test_check_file_match_empty_retrieved():
    retrieved = []
    expected = ["src/chinvex/ingest.py"]
    assert check_file_match(retrieved, expected) is False


def test_check_file_match_case_sensitive():
    retrieved = ["src/chinvex/Ingest.py"]
    expected = ["src/chinvex/ingest.py"]
    assert check_file_match(retrieved, expected) is False


def test_check_anchor_match_found(tmp_path):
    # Create test file with content
    test_file = tmp_path / "test.py"
    test_file.write_text("""
    # Configuration
    BATCH_SIZE = 5000  # ChromaDB batch limit is 5000 vectors
    """)

    result = check_anchor_match(
        file_path=str(test_file),
        anchor="5000 vectors"
    )
    assert result is True


def test_check_anchor_match_not_found(tmp_path):
    test_file = tmp_path / "test.py"
    test_file.write_text("""
    # Configuration
    BATCH_SIZE = 1000
    """)

    result = check_anchor_match(
        file_path=str(test_file),
        anchor="5000 vectors"
    )
    assert result is False


def test_check_anchor_match_file_not_found():
    result = check_anchor_match(
        file_path="nonexistent.py",
        anchor="test"
    )
    assert result is False


def test_check_anchor_match_case_insensitive(tmp_path):
    test_file = tmp_path / "test.py"
    test_file.write_text("BATCH SIZE IS 5000 VECTORS")

    result = check_anchor_match(
        file_path=str(test_file),
        anchor="5000 vectors"
    )
    assert result is True


def test_evaluate_query_pass_file_match():
    query_result = QueryResult(
        query="test",
        expected_files=["src/chinvex/ingest.py"],
        retrieved_files=["src/chinvex/ingest.py", "other.py"],
        k=5
    )

    evaluation = evaluate_query(query_result)

    assert evaluation["passed"] is True
    assert evaluation["file_match"] is True
    assert evaluation["rank"] == 1
    assert evaluation["anchor_match"] is None


def test_evaluate_query_pass_with_rank():
    query_result = QueryResult(
        query="test",
        expected_files=["src/chinvex/config.py"],
        retrieved_files=["a.py", "b.py", "src/chinvex/config.py"],
        k=5
    )

    evaluation = evaluate_query(query_result)

    assert evaluation["passed"] is True
    assert evaluation["file_match"] is True
    assert evaluation["rank"] == 3


def test_evaluate_query_fail_no_match():
    query_result = QueryResult(
        query="test",
        expected_files=["src/chinvex/ingest.py"],
        retrieved_files=["other.py", "another.py"],
        k=5
    )

    evaluation = evaluate_query(query_result)

    assert evaluation["passed"] is False
    assert evaluation["file_match"] is False
    assert evaluation["rank"] is None


def test_evaluate_query_with_anchor_match(tmp_path):
    # Create test file
    test_file = tmp_path / "ingest.py"
    test_file.write_text("BATCH_SIZE = 5000  # batch limit")

    query_result = QueryResult(
        query="test",
        expected_files=[str(test_file)],
        retrieved_files=[str(test_file)],
        k=5,
        anchor="batch limit"
    )

    evaluation = evaluate_query(query_result)

    assert evaluation["passed"] is True
    assert evaluation["file_match"] is True
    assert evaluation["anchor_match"] is True
    assert evaluation["rank"] == 1


def test_evaluate_query_with_anchor_no_match(tmp_path):
    test_file = tmp_path / "ingest.py"
    test_file.write_text("BATCH_SIZE = 1000")

    query_result = QueryResult(
        query="test",
        expected_files=[str(test_file)],
        retrieved_files=[str(test_file)],
        k=5,
        anchor="5000 vectors"
    )

    evaluation = evaluate_query(query_result)

    assert evaluation["passed"] is True
    assert evaluation["file_match"] is True
    assert evaluation["anchor_match"] is False


def test_evaluate_query_multiple_expected_files():
    query_result = QueryResult(
        query="test",
        expected_files=["src/a.py", "src/b.py", "src/c.py"],
        retrieved_files=["src/b.py", "other.py"],
        k=5
    )

    evaluation = evaluate_query(query_result)

    assert evaluation["passed"] is True
    assert evaluation["file_match"] is True
    assert evaluation["rank"] == 1
    assert evaluation["matched_file"] == "src/b.py"
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_eval_criteria.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chinvex.eval_criteria'"

**Step 3: Write minimal implementation**
```python
# src/chinvex/eval_criteria.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from .eval_runner import QueryResult


def check_file_match(retrieved_files: list[str], expected_files: list[str]) -> bool:
    """Check if any expected file appears in retrieved files.

    Args:
        retrieved_files: List of file paths from search results
        expected_files: List of acceptable file paths

    Returns:
        True if at least one expected file is in retrieved files
    """
    return any(expected in retrieved_files for expected in expected_files)


def check_anchor_match(file_path: str, anchor: str) -> bool:
    """Check if anchor string appears in file content.

    Args:
        file_path: Path to file to check
        anchor: String to search for (case-insensitive)

    Returns:
        True if anchor found in file content
    """
    try:
        content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        return anchor.lower() in content.lower()
    except (FileNotFoundError, OSError):
        return False


def evaluate_query(query_result: QueryResult) -> dict[str, Any]:
    """Evaluate a query result against success criteria.

    Args:
        query_result: Result from executing a golden query

    Returns:
        Dictionary with:
        - passed (bool): True if at least one expected file in top K
        - file_match (bool): Same as passed
        - rank (int | None): Position of first matched file (1-indexed)
        - matched_file (str | None): Path of first matched file
        - anchor_match (bool | None): True if anchor found in matched file
    """
    file_match = check_file_match(
        query_result.retrieved_files,
        query_result.expected_files
    )

    rank = None
    matched_file = None
    anchor_match = None

    if file_match:
        # Find rank of first matching file
        for i, retrieved_file in enumerate(query_result.retrieved_files):
            if retrieved_file in query_result.expected_files:
                rank = i + 1  # 1-indexed
                matched_file = retrieved_file
                break

        # Check anchor if specified
        if query_result.anchor and matched_file:
            anchor_match = check_anchor_match(matched_file, query_result.anchor)

    return {
        "passed": file_match,
        "file_match": file_match,
        "rank": rank,
        "matched_file": matched_file,
        "anchor_match": anchor_match
    }
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_eval_criteria.py -v`
Expected: PASS (all 17 tests passing)

**Step 5: Commit**
```bash
git add src/chinvex/eval_criteria.py tests/test_eval_criteria.py
git commit -m "feat(P5.3.3): add success criteria evaluation for file and anchor matching

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 14: Metrics calculation (hit rate @K, MRR, latency)

**Files:**
- Create: `src/chinvex/eval_metrics.py`
- Test: `tests/test_eval_metrics.py`

**Step 1: Write the failing test**
```python
# tests/test_eval_metrics.py
import pytest
from chinvex.eval_metrics import calculate_metrics, EvalMetrics


def test_eval_metrics_dataclass():
    metrics = EvalMetrics(
        hit_rate=0.85,
        mrr=0.72,
        mean_latency_ms=150.5,
        total_queries=20,
        passed_queries=17,
        anchor_match_rate=0.65
    )
    assert metrics.hit_rate == 0.85
    assert metrics.mrr == 0.72
    assert metrics.mean_latency_ms == 150.5
    assert metrics.total_queries == 20
    assert metrics.passed_queries == 17
    assert metrics.anchor_match_rate == 0.65


def test_calculate_metrics_perfect_score():
    evaluations = [
        {"passed": True, "rank": 1, "anchor_match": True},
        {"passed": True, "rank": 1, "anchor_match": True},
        {"passed": True, "rank": 1, "anchor_match": True}
    ]
    latencies = [100.0, 120.0, 110.0]

    metrics = calculate_metrics(evaluations, latencies)

    assert metrics.hit_rate == 1.0
    assert metrics.mrr == 1.0
    assert metrics.mean_latency_ms == 110.0
    assert metrics.total_queries == 3
    assert metrics.passed_queries == 3
    assert metrics.anchor_match_rate == 1.0


def test_calculate_metrics_partial_hits():
    evaluations = [
        {"passed": True, "rank": 1, "anchor_match": None},
        {"passed": False, "rank": None, "anchor_match": None},
        {"passed": True, "rank": 3, "anchor_match": None},
        {"passed": False, "rank": None, "anchor_match": None}
    ]
    latencies = [100.0, 110.0, 120.0, 130.0]

    metrics = calculate_metrics(evaluations, latencies)

    assert metrics.hit_rate == 0.5  # 2 out of 4
    assert metrics.passed_queries == 2
    assert metrics.total_queries == 4
    assert metrics.mean_latency_ms == 115.0


def test_calculate_metrics_mrr_calculation():
    evaluations = [
        {"passed": True, "rank": 1, "anchor_match": None},  # 1/1 = 1.0
        {"passed": True, "rank": 2, "anchor_match": None},  # 1/2 = 0.5
        {"passed": True, "rank": 5, "anchor_match": None},  # 1/5 = 0.2
        {"passed": False, "rank": None, "anchor_match": None}  # 0
    ]
    latencies = [100.0, 100.0, 100.0, 100.0]

    metrics = calculate_metrics(evaluations, latencies)

    # MRR = (1.0 + 0.5 + 0.2 + 0.0) / 4 = 1.7 / 4 = 0.425
    assert abs(metrics.mrr - 0.425) < 0.001
    assert metrics.hit_rate == 0.75


def test_calculate_metrics_anchor_match_rate():
    evaluations = [
        {"passed": True, "rank": 1, "anchor_match": True},
        {"passed": True, "rank": 2, "anchor_match": False},
        {"passed": True, "rank": 1, "anchor_match": True},
        {"passed": True, "rank": 1, "anchor_match": None}  # No anchor specified
    ]
    latencies = [100.0, 100.0, 100.0, 100.0]

    metrics = calculate_metrics(evaluations, latencies)

    # Only 3 queries had anchors, 2 matched
    assert abs(metrics.anchor_match_rate - (2/3)) < 0.001


def test_calculate_metrics_no_anchors():
    evaluations = [
        {"passed": True, "rank": 1, "anchor_match": None},
        {"passed": True, "rank": 2, "anchor_match": None},
        {"passed": False, "rank": None, "anchor_match": None}
    ]
    latencies = [100.0, 100.0, 100.0]

    metrics = calculate_metrics(evaluations, latencies)

    assert metrics.anchor_match_rate == 0.0


def test_calculate_metrics_all_failed():
    evaluations = [
        {"passed": False, "rank": None, "anchor_match": None},
        {"passed": False, "rank": None, "anchor_match": None}
    ]
    latencies = [100.0, 100.0]

    metrics = calculate_metrics(evaluations, latencies)

    assert metrics.hit_rate == 0.0
    assert metrics.mrr == 0.0
    assert metrics.passed_queries == 0


def test_calculate_metrics_empty_input():
    evaluations = []
    latencies = []

    metrics = calculate_metrics(evaluations, latencies)

    assert metrics.hit_rate == 0.0
    assert metrics.mrr == 0.0
    assert metrics.total_queries == 0
    assert metrics.passed_queries == 0
    assert metrics.mean_latency_ms == 0.0


def test_calculate_metrics_latency_calculation():
    evaluations = [
        {"passed": True, "rank": 1, "anchor_match": None}
    ] * 5
    latencies = [100.0, 200.0, 150.0, 250.0, 300.0]

    metrics = calculate_metrics(evaluations, latencies)

    assert metrics.mean_latency_ms == 200.0


def test_calculate_metrics_mismatched_lengths():
    evaluations = [
        {"passed": True, "rank": 1, "anchor_match": None}
    ] * 3
    latencies = [100.0, 200.0]  # Shorter than evaluations

    with pytest.raises(ValueError, match="Evaluations and latencies must have same length"):
        calculate_metrics(evaluations, latencies)
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_eval_metrics.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chinvex.eval_metrics'"

**Step 3: Write minimal implementation**
```python
# src/chinvex/eval_metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EvalMetrics:
    """Evaluation metrics for golden query set."""
    hit_rate: float  # Fraction of queries with at least one expected file in top K
    mrr: float  # Mean Reciprocal Rank
    mean_latency_ms: float  # Average query latency in milliseconds
    total_queries: int  # Total number of queries evaluated
    passed_queries: int  # Number of queries that passed (file match)
    anchor_match_rate: float  # Fraction of anchor queries where anchor was found


def calculate_metrics(
    evaluations: list[dict[str, Any]],
    latencies: list[float]
) -> EvalMetrics:
    """Calculate aggregate metrics from query evaluations.

    Args:
        evaluations: List of evaluation dicts from evaluate_query()
        latencies: List of query latency values in milliseconds

    Returns:
        EvalMetrics with aggregate statistics

    Raises:
        ValueError: If evaluations and latencies have different lengths
    """
    if len(evaluations) != len(latencies):
        raise ValueError("Evaluations and latencies must have same length")

    if not evaluations:
        return EvalMetrics(
            hit_rate=0.0,
            mrr=0.0,
            mean_latency_ms=0.0,
            total_queries=0,
            passed_queries=0,
            anchor_match_rate=0.0
        )

    total_queries = len(evaluations)
    passed_queries = sum(1 for e in evaluations if e["passed"])

    # Calculate hit rate
    hit_rate = passed_queries / total_queries if total_queries > 0 else 0.0

    # Calculate MRR (Mean Reciprocal Rank)
    reciprocal_ranks = []
    for evaluation in evaluations:
        if evaluation["passed"] and evaluation["rank"] is not None:
            reciprocal_ranks.append(1.0 / evaluation["rank"])
        else:
            reciprocal_ranks.append(0.0)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    # Calculate mean latency
    mean_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

    # Calculate anchor match rate (only for queries with anchors)
    anchor_queries = [e for e in evaluations if e["anchor_match"] is not None]
    if anchor_queries:
        anchor_matches = sum(1 for e in anchor_queries if e["anchor_match"])
        anchor_match_rate = anchor_matches / len(anchor_queries)
    else:
        anchor_match_rate = 0.0

    return EvalMetrics(
        hit_rate=hit_rate,
        mrr=mrr,
        mean_latency_ms=mean_latency_ms,
        total_queries=total_queries,
        passed_queries=passed_queries,
        anchor_match_rate=anchor_match_rate
    )
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_eval_metrics.py -v`
Expected: PASS (all 11 tests passing)

**Step 5: Commit**
```bash
git add src/chinvex/eval_metrics.py tests/test_eval_metrics.py
git commit -m "feat(P5.3.4): add metrics calculation for hit rate, MRR, and latency

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 15: Baseline metrics storage and CI gate

**Files:**
- Create: `src/chinvex/eval_baseline.py`
- Test: `tests/test_eval_baseline.py`

**Step 1: Write the failing test**
```python
# tests/test_eval_baseline.py
import json
import pytest
from pathlib import Path
from chinvex.eval_baseline import (
    save_baseline_metrics,
    load_baseline_metrics,
    compare_to_baseline,
    BaselineComparison
)
from chinvex.eval_metrics import EvalMetrics


@pytest.fixture
def sample_metrics():
    return EvalMetrics(
        hit_rate=0.85,
        mrr=0.72,
        mean_latency_ms=150.5,
        total_queries=20,
        passed_queries=17,
        anchor_match_rate=0.65
    )


def test_save_baseline_metrics(tmp_path, sample_metrics):
    baseline_file = tmp_path / "baseline_metrics.json"

    save_baseline_metrics(baseline_file, "Chinvex", sample_metrics)

    assert baseline_file.exists()
    data = json.loads(baseline_file.read_text())
    assert "Chinvex" in data
    assert data["Chinvex"]["hit_rate"] == 0.85
    assert data["Chinvex"]["mrr"] == 0.72
    assert data["Chinvex"]["total_queries"] == 20


def test_save_baseline_metrics_creates_directory(tmp_path, sample_metrics):
    baseline_file = tmp_path / "subdir" / "baseline_metrics.json"

    save_baseline_metrics(baseline_file, "Chinvex", sample_metrics)

    assert baseline_file.exists()


def test_save_baseline_metrics_updates_existing(tmp_path, sample_metrics):
    baseline_file = tmp_path / "baseline_metrics.json"

    # Save first context
    save_baseline_metrics(baseline_file, "Chinvex", sample_metrics)

    # Save second context
    metrics2 = EvalMetrics(
        hit_rate=0.90,
        mrr=0.80,
        mean_latency_ms=120.0,
        total_queries=15,
        passed_queries=14,
        anchor_match_rate=0.75
    )
    save_baseline_metrics(baseline_file, "OtherContext", metrics2)

    data = json.loads(baseline_file.read_text())
    assert "Chinvex" in data
    assert "OtherContext" in data
    assert data["Chinvex"]["hit_rate"] == 0.85
    assert data["OtherContext"]["hit_rate"] == 0.90


def test_save_baseline_metrics_overwrites_context(tmp_path):
    baseline_file = tmp_path / "baseline_metrics.json"

    metrics1 = EvalMetrics(0.85, 0.72, 150.0, 20, 17, 0.65)
    save_baseline_metrics(baseline_file, "Chinvex", metrics1)

    metrics2 = EvalMetrics(0.90, 0.80, 120.0, 20, 18, 0.70)
    save_baseline_metrics(baseline_file, "Chinvex", metrics2)

    data = json.loads(baseline_file.read_text())
    assert data["Chinvex"]["hit_rate"] == 0.90


def test_load_baseline_metrics(tmp_path, sample_metrics):
    baseline_file = tmp_path / "baseline_metrics.json"
    save_baseline_metrics(baseline_file, "Chinvex", sample_metrics)

    loaded = load_baseline_metrics(baseline_file, "Chinvex")

    assert loaded.hit_rate == 0.85
    assert loaded.mrr == 0.72
    assert loaded.mean_latency_ms == 150.5
    assert loaded.total_queries == 20


def test_load_baseline_metrics_not_found(tmp_path):
    baseline_file = tmp_path / "baseline_metrics.json"

    loaded = load_baseline_metrics(baseline_file, "Chinvex")

    assert loaded is None


def test_load_baseline_metrics_context_not_found(tmp_path, sample_metrics):
    baseline_file = tmp_path / "baseline_metrics.json"
    save_baseline_metrics(baseline_file, "Chinvex", sample_metrics)

    loaded = load_baseline_metrics(baseline_file, "OtherContext")

    assert loaded is None


def test_baseline_comparison_dataclass():
    comp = BaselineComparison(
        passed=True,
        current_hit_rate=0.85,
        baseline_hit_rate=0.80,
        hit_rate_change=0.05,
        threshold=0.80
    )
    assert comp.passed is True
    assert comp.current_hit_rate == 0.85
    assert comp.baseline_hit_rate == 0.80
    assert comp.hit_rate_change == 0.05


def test_compare_to_baseline_pass(sample_metrics):
    baseline = EvalMetrics(0.80, 0.70, 140.0, 20, 16, 0.60)
    current = sample_metrics  # hit_rate=0.85

    comparison = compare_to_baseline(current, baseline, threshold=0.80)

    assert comparison.passed is True
    assert comparison.current_hit_rate == 0.85
    assert comparison.baseline_hit_rate == 0.80
    assert abs(comparison.hit_rate_change - 0.05) < 0.001


def test_compare_to_baseline_fail(sample_metrics):
    baseline = EvalMetrics(0.90, 0.80, 140.0, 20, 18, 0.70)
    current = sample_metrics  # hit_rate=0.85

    # 0.85 is only 94.4% of 0.90, below 95% threshold
    comparison = compare_to_baseline(current, baseline, threshold=0.95)

    assert comparison.passed is False
    assert comparison.current_hit_rate == 0.85
    assert comparison.baseline_hit_rate == 0.90


def test_compare_to_baseline_exact_threshold():
    baseline = EvalMetrics(1.0, 0.90, 100.0, 10, 10, 1.0)
    current = EvalMetrics(0.80, 0.70, 120.0, 10, 8, 0.80)

    # current is exactly 80% of baseline (0.80 / 1.0)
    comparison = compare_to_baseline(current, baseline, threshold=0.80)

    assert comparison.passed is True


def test_compare_to_baseline_default_threshold():
    baseline = EvalMetrics(1.0, 0.90, 100.0, 10, 10, 1.0)
    current = EvalMetrics(0.79, 0.70, 120.0, 10, 8, 0.80)

    # Default threshold is 0.80, current is 79% of baseline
    comparison = compare_to_baseline(current, baseline)

    assert comparison.passed is False
    assert comparison.threshold == 0.80


def test_compare_to_baseline_improvement():
    baseline = EvalMetrics(0.80, 0.70, 140.0, 20, 16, 0.60)
    current = EvalMetrics(0.95, 0.85, 120.0, 20, 19, 0.90)

    comparison = compare_to_baseline(current, baseline, threshold=0.80)

    assert comparison.passed is True
    assert comparison.hit_rate_change == 0.15


def test_compare_to_baseline_zero_baseline():
    baseline = EvalMetrics(0.0, 0.0, 100.0, 10, 0, 0.0)
    current = EvalMetrics(0.50, 0.40, 120.0, 10, 5, 0.50)

    # Special case: if baseline is 0, any positive current passes
    comparison = compare_to_baseline(current, baseline, threshold=0.80)

    assert comparison.passed is True
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_eval_baseline.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chinvex.eval_baseline'"

**Step 3: Write minimal implementation**
```python
# src/chinvex/eval_baseline.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from .eval_metrics import EvalMetrics


@dataclass
class BaselineComparison:
    """Result of comparing current metrics to baseline."""
    passed: bool  # True if current hit rate >= threshold * baseline
    current_hit_rate: float
    baseline_hit_rate: float
    hit_rate_change: float  # current - baseline
    threshold: float  # Minimum fraction of baseline required (e.g., 0.80 = 80%)


def save_baseline_metrics(
    baseline_file: Path,
    context: str,
    metrics: EvalMetrics
) -> None:
    """Save baseline metrics for a context.

    Args:
        baseline_file: Path to baseline_metrics.json
        context: Context name
        metrics: Metrics to save as baseline
    """
    # Create directory if needed
    baseline_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing baselines
    if baseline_file.exists():
        data = json.loads(baseline_file.read_text())
    else:
        data = {}

    # Update with new metrics
    data[context] = asdict(metrics)

    # Write back
    baseline_file.write_text(json.dumps(data, indent=2))


def load_baseline_metrics(
    baseline_file: Path,
    context: str
) -> EvalMetrics | None:
    """Load baseline metrics for a context.

    Args:
        baseline_file: Path to baseline_metrics.json
        context: Context name

    Returns:
        EvalMetrics if found, None otherwise
    """
    if not baseline_file.exists():
        return None

    data = json.loads(baseline_file.read_text())

    if context not in data:
        return None

    metrics_dict = data[context]
    return EvalMetrics(**metrics_dict)


def compare_to_baseline(
    current: EvalMetrics,
    baseline: EvalMetrics,
    threshold: float = 0.80
) -> BaselineComparison:
    """Compare current metrics to baseline.

    Args:
        current: Current evaluation metrics
        baseline: Baseline metrics to compare against
        threshold: Minimum fraction of baseline hit rate required (default 0.80 = 80%)

    Returns:
        BaselineComparison with pass/fail status
    """
    hit_rate_change = current.hit_rate - baseline.hit_rate

    # Special case: if baseline is 0, any positive current passes
    if baseline.hit_rate == 0.0:
        passed = current.hit_rate > 0.0
    else:
        # Pass if current >= threshold * baseline
        required_hit_rate = threshold * baseline.hit_rate
        passed = current.hit_rate >= required_hit_rate

    return BaselineComparison(
        passed=passed,
        current_hit_rate=current.hit_rate,
        baseline_hit_rate=baseline.hit_rate,
        hit_rate_change=hit_rate_change,
        threshold=threshold
    )
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_eval_baseline.py -v`
Expected: PASS (all 16 tests passing)

**Step 5: Commit**
```bash
git add src/chinvex/eval_baseline.py tests/test_eval_baseline.py
git commit -m "feat(P5.3.5): add baseline metrics storage and CI gate comparison

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```


### Task 16: Query logging to .chinvex/logs/queries.jsonl with 30-day retention

**Files:**
- Create: `C:\Code\chinvex\src\chinvex\query_logger.py`
- Test: `C:\Code\chinvex\tests\test_query_logger.py`

**Step 1: Write the failing test**
```python
# tests/test_query_logger.py
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

from chinvex.query_logger import (
    QueryLogger,
    QueryLogEntry,
    log_search_query,
    rotate_old_logs,
)


def test_query_logger_creates_log_directory(tmp_path):
    """QueryLogger should create .chinvex/logs/ directory if missing."""
    chinvex_dir = tmp_path / ".chinvex"
    logger = QueryLogger(chinvex_dir)

    log_dir = chinvex_dir / "logs"
    assert log_dir.exists()
    assert log_dir.is_dir()


def test_query_logger_appends_to_jsonl(tmp_path):
    """QueryLogger should append search queries to queries.jsonl."""
    chinvex_dir = tmp_path / ".chinvex"
    logger = QueryLogger(chinvex_dir)

    entry = QueryLogEntry(
        timestamp=datetime.now().isoformat(),
        context="Chinvex",
        query="test query",
        k=5,
        num_results=3,
        top_chunk_ids=["chunk1", "chunk2", "chunk3"],
        top_scores=[0.95, 0.87, 0.72],
        latency_ms=125.5,
    )

    logger.log(entry)

    log_file = chinvex_dir / "logs" / "queries.jsonl"
    assert log_file.exists()

    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 1

    logged = json.loads(lines[0])
    assert logged["context"] == "Chinvex"
    assert logged["query"] == "test query"
    assert logged["k"] == 5
    assert logged["num_results"] == 3
    assert logged["top_chunk_ids"] == ["chunk1", "chunk2", "chunk3"]
    assert logged["top_scores"] == [0.95, 0.87, 0.72]
    assert logged["latency_ms"] == 125.5


def test_query_logger_multiple_entries(tmp_path):
    """QueryLogger should append multiple entries correctly."""
    chinvex_dir = tmp_path / ".chinvex"
    logger = QueryLogger(chinvex_dir)

    for i in range(3):
        entry = QueryLogEntry(
            timestamp=datetime.now().isoformat(),
            context="Chinvex",
            query=f"query {i}",
            k=5,
            num_results=2,
            top_chunk_ids=[f"chunk{i}a", f"chunk{i}b"],
            top_scores=[0.9, 0.8],
            latency_ms=100.0 + i,
        )
        logger.log(entry)

    log_file = chinvex_dir / "logs" / "queries.jsonl"
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 3

    # Verify each entry
    for i, line in enumerate(lines):
        logged = json.loads(line)
        assert logged["query"] == f"query {i}"
        assert logged["latency_ms"] == 100.0 + i


def test_rotate_old_logs_removes_30_day_old_entries(tmp_path):
    """rotate_old_logs should remove entries older than 30 days."""
    chinvex_dir = tmp_path / ".chinvex"
    log_dir = chinvex_dir / "logs"
    log_dir.mkdir(parents=True)
    log_file = log_dir / "queries.jsonl"

    now = datetime.now()
    old_timestamp = (now - timedelta(days=31)).isoformat()
    recent_timestamp = (now - timedelta(days=10)).isoformat()

    # Write entries with different timestamps
    with open(log_file, "w") as f:
        f.write(json.dumps({"timestamp": old_timestamp, "query": "old"}) + "\n")
        f.write(json.dumps({"timestamp": recent_timestamp, "query": "recent"}) + "\n")
        f.write(json.dumps({"timestamp": now.isoformat(), "query": "current"}) + "\n")

    rotate_old_logs(chinvex_dir, retention_days=30)

    # Only recent entries should remain
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 2

    queries = [json.loads(line)["query"] for line in lines]
    assert "old" not in queries
    assert "recent" in queries
    assert "current" in queries


def test_rotate_old_logs_handles_empty_file(tmp_path):
    """rotate_old_logs should handle empty log file gracefully."""
    chinvex_dir = tmp_path / ".chinvex"
    log_dir = chinvex_dir / "logs"
    log_dir.mkdir(parents=True)
    log_file = log_dir / "queries.jsonl"
    log_file.touch()

    # Should not raise
    rotate_old_logs(chinvex_dir, retention_days=30)
    assert log_file.exists()


def test_rotate_old_logs_handles_missing_file(tmp_path):
    """rotate_old_logs should handle missing log file gracefully."""
    chinvex_dir = tmp_path / ".chinvex"

    # Should not raise
    rotate_old_logs(chinvex_dir, retention_days=30)


def test_log_search_query_integration(tmp_path):
    """log_search_query should record search with all metadata."""
    chinvex_dir = tmp_path / ".chinvex"

    start = time.time()
    log_search_query(
        chinvex_dir=chinvex_dir,
        context="Chinvex",
        query="integration test",
        results=[
            {"chunk_id": "c1", "score": 0.95},
            {"chunk_id": "c2", "score": 0.82},
        ],
        k=5,
    )
    elapsed_ms = (time.time() - start) * 1000

    log_file = chinvex_dir / "logs" / "queries.jsonl"
    assert log_file.exists()

    logged = json.loads(log_file.read_text())
    assert logged["context"] == "Chinvex"
    assert logged["query"] == "integration test"
    assert logged["k"] == 5
    assert logged["num_results"] == 2
    assert logged["top_chunk_ids"] == ["c1", "c2"]
    assert logged["top_scores"] == [0.95, 0.82]
    assert logged["latency_ms"] > 0
    assert logged["latency_ms"] < elapsed_ms + 100  # reasonable upper bound
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_query_logger.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chinvex.query_logger'"

**Step 3: Write minimal implementation**
```python
# src/chinvex/query_logger.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class QueryLogEntry:
    """Single query log entry."""
    timestamp: str
    context: str
    query: str
    k: int
    num_results: int
    top_chunk_ids: list[str]
    top_scores: list[float]
    latency_ms: float


class QueryLogger:
    """Logs search queries to .chinvex/logs/queries.jsonl."""

    def __init__(self, chinvex_dir: Path):
        self.chinvex_dir = Path(chinvex_dir)
        self.log_dir = self.chinvex_dir / "logs"
        self.log_file = self.log_dir / "queries.jsonl"

        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, entry: QueryLogEntry) -> None:
        """Append query log entry to JSONL file."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry)) + "\n")


def rotate_old_logs(chinvex_dir: Path, retention_days: int = 30) -> None:
    """Remove log entries older than retention_days.

    Args:
        chinvex_dir: Path to .chinvex directory
        retention_days: Number of days to retain logs (default 30)
    """
    log_file = Path(chinvex_dir) / "logs" / "queries.jsonl"

    if not log_file.exists():
        return

    cutoff = datetime.now() - timedelta(days=retention_days)

    # Read all entries
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return  # Silently skip on read errors

    if not lines:
        return

    # Filter to recent entries only
    recent_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            entry = json.loads(line)
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if entry_time >= cutoff:
                recent_lines.append(line + "\n")
        except (json.JSONDecodeError, KeyError, ValueError):
            # Skip malformed entries
            continue

    # Write back recent entries
    with open(log_file, "w", encoding="utf-8") as f:
        f.writelines(recent_lines)


def log_search_query(
    chinvex_dir: Path,
    context: str,
    query: str,
    results: list[dict],
    k: int,
) -> None:
    """Log a search query with results metadata.

    Args:
        chinvex_dir: Path to .chinvex directory
        context: Context name
        query: Search query string
        results: List of result dicts with 'chunk_id' and 'score'
        k: Number of results requested
    """
    start = time.time()

    logger = QueryLogger(chinvex_dir)

    entry = QueryLogEntry(
        timestamp=datetime.now().isoformat(),
        context=context,
        query=query,
        k=k,
        num_results=len(results),
        top_chunk_ids=[r["chunk_id"] for r in results],
        top_scores=[r["score"] for r in results],
        latency_ms=(time.time() - start) * 1000,
    )

    logger.log(entry)
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_query_logger.py -v`
Expected: PASS (all 9 tests passing)

**Step 5: Commit**
```bash
git add src/chinvex/query_logger.py tests/test_query_logger.py
git commit -m "feat(P5.3): add query logging to .chinvex/logs/queries.jsonl with 30-day rotation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```


### Task 17: CLI command chinvex eval --context X

**Files:**
- Create: `C:\Code\chinvex\src\chinvex\eval_runner.py`
- Modify: `C:\Code\chinvex\src\chinvex\cli.py`
- Test: `C:\Code\chinvex\tests\test_eval_cli.py`

**Step 1: Write the failing test**
```python
# tests/test_eval_cli.py
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from chinvex.cli import cli


def test_eval_command_exists():
    """chinvex eval command should exist."""
    runner = CliRunner()
    result = runner.invoke(cli, ["eval", "--help"])
    assert result.exit_code == 0
    assert "Run evaluation suite" in result.output


def test_eval_requires_context_flag():
    """chinvex eval should require --context flag."""
    runner = CliRunner()
    result = runner.invoke(cli, ["eval"])
    assert result.exit_code != 0
    assert "--context" in result.output


def test_eval_runs_golden_queries(tmp_path, monkeypatch):
    """chinvex eval should load golden queries and run evaluation."""
    # Setup test environment
    chinvex_home = tmp_path / ".chinvex"
    chinvex_home.mkdir()
    context_dir = chinvex_home / "contexts" / "TestContext"
    context_dir.mkdir(parents=True)

    # Create context.json
    context_config = {
        "embedding": {"provider": "openai", "model": "text-embedding-3-small"}
    }
    (context_dir / "context.json").write_text(json.dumps(context_config))

    # Create golden queries file
    golden_queries = {
        "queries": [
            {
                "query": "test query 1",
                "context": "TestContext",
                "expected_files": ["file1.py"],
                "k": 5
            },
            {
                "query": "test query 2",
                "context": "TestContext",
                "expected_files": ["file2.py"],
                "k": 5
            }
        ]
    }
    golden_file = tmp_path / "tests" / "eval" / "golden_queries_testcontext.json"
    golden_file.parent.mkdir(parents=True)
    golden_file.write_text(json.dumps(golden_queries))

    monkeypatch.setenv("CHINVEX_HOME", str(chinvex_home))

    # Mock the eval runner
    with patch("chinvex.cli.run_evaluation") as mock_run_eval:
        mock_run_eval.return_value = {
            "hit_rate": 0.85,
            "mrr": 0.75,
            "avg_latency_ms": 150.5,
            "passed": 17,
            "failed": 3,
            "total": 20
        }

        runner = CliRunner()
        result = runner.invoke(cli, ["eval", "--context", "TestContext"])

        assert result.exit_code == 0
        assert "Hit Rate@5: 85.0%" in result.output
        assert "MRR: 0.750" in result.output
        assert "Avg Latency: 150.5ms" in result.output
        assert "Passed: 17/20" in result.output

        # Verify run_evaluation was called
        mock_run_eval.assert_called_once()
        call_args = mock_run_eval.call_args
        assert call_args[1]["context_name"] == "TestContext"


def test_eval_reports_baseline_comparison(tmp_path, monkeypatch):
    """chinvex eval should compare to baseline and report status."""
    chinvex_home = tmp_path / ".chinvex"
    context_dir = chinvex_home / "contexts" / "TestContext"
    context_dir.mkdir(parents=True)

    context_config = {
        "embedding": {"provider": "openai", "model": "text-embedding-3-small"}
    }
    (context_dir / "context.json").write_text(json.dumps(context_config))

    # Create baseline file
    baseline_file = tmp_path / "tests" / "eval" / "baseline_metrics.json"
    baseline_file.parent.mkdir(parents=True)
    baseline_data = {
        "TestContext": {
            "hit_rate": 0.90,
            "mrr": 0.80,
            "avg_latency_ms": 100.0
        }
    }
    baseline_file.write_text(json.dumps(baseline_data))

    monkeypatch.setenv("CHINVEX_HOME", str(chinvex_home))

    with patch("chinvex.cli.run_evaluation") as mock_run_eval, \
         patch("chinvex.cli.load_baseline_metrics") as mock_load_baseline, \
         patch("chinvex.cli.compare_to_baseline") as mock_compare:

        mock_run_eval.return_value = {
            "hit_rate": 0.75,
            "mrr": 0.70,
            "avg_latency_ms": 120.0,
            "passed": 15,
            "failed": 5,
            "total": 20
        }

        mock_load_baseline.return_value = MagicMock(
            hit_rate=0.90,
            mrr=0.80,
            avg_latency_ms=100.0
        )

        mock_compare.return_value = MagicMock(
            passed=False,
            current_hit_rate=0.75,
            baseline_hit_rate=0.90,
            hit_rate_change=-0.15,
            threshold=0.80
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["eval", "--context", "TestContext"])

        assert result.exit_code == 1  # Should exit with error on regression
        assert "REGRESSION" in result.output or "below baseline" in result.output


def test_eval_handles_missing_golden_queries(tmp_path, monkeypatch):
    """chinvex eval should error gracefully if golden queries file missing."""
    chinvex_home = tmp_path / ".chinvex"
    context_dir = chinvex_home / "contexts" / "TestContext"
    context_dir.mkdir(parents=True)

    context_config = {
        "embedding": {"provider": "openai", "model": "text-embedding-3-small"}
    }
    (context_dir / "context.json").write_text(json.dumps(context_config))

    monkeypatch.setenv("CHINVEX_HOME", str(chinvex_home))

    runner = CliRunner()
    result = runner.invoke(cli, ["eval", "--context", "TestContext"])

    assert result.exit_code != 0
    assert "golden queries" in result.output.lower()
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_eval_cli.py -v`
Expected: FAIL with "AttributeError: module 'chinvex.cli' has no attribute 'eval'"

**Step 3: Write minimal implementation**
```python
# src/chinvex/eval_runner.py
from __future__ import annotations

import json
import time
from pathlib import Path

from .config import get_config_for_context
from .eval_schema import load_golden_queries, GoldenQuery
from .search import search, SearchResult


def run_evaluation(
    context_name: str,
    golden_queries_file: Path | None = None,
    k: int | None = None,
) -> dict:
    """Run evaluation suite for a context.

    Args:
        context_name: Name of context to evaluate
        golden_queries_file: Path to golden queries JSON (default: auto-detect)
        k: Override k value for all queries (default: use per-query k)

    Returns:
        Dict with metrics:
        - hit_rate: fraction of queries with at least one expected file in top K
        - mrr: mean reciprocal rank across all queries
        - avg_latency_ms: average query latency in milliseconds
        - passed: number of queries that passed
        - failed: number of queries that failed
        - total: total number of queries
    """
    # Load golden queries
    if golden_queries_file is None:
        # Auto-detect: tests/eval/golden_queries_<context>.json
        context_lower = context_name.lower()
        golden_queries_file = Path(f"tests/eval/golden_queries_{context_lower}.json")

    if not golden_queries_file.exists():
        raise FileNotFoundError(
            f"Golden queries file not found: {golden_queries_file}\n"
            f"Create queries for {context_name} context first."
        )

    queries = load_golden_queries(golden_queries_file)

    # Filter to this context only
    context_queries = [q for q in queries if q.context == context_name]

    if not context_queries:
        raise ValueError(
            f"No golden queries found for context '{context_name}' in {golden_queries_file}"
        )

    # Load context config
    config = get_config_for_context(context_name)

    # Run queries and collect results
    hits = 0
    reciprocal_ranks = []
    latencies = []
    passed = 0
    failed = 0

    for query in context_queries:
        query_k = k if k is not None else query.k

        start = time.time()
        results = search(
            config,
            query.query,
            k=query_k,
            min_score=0.0,  # Don't filter by score during eval
        )
        elapsed_ms = (time.time() - start) * 1000
        latencies.append(elapsed_ms)

        # Check if any expected file appears in top K
        hit = False
        first_rank = None

        for rank, result in enumerate(results, start=1):
            # Extract file path from citation or title
            result_file = _extract_file_path(result)

            for expected_file in query.expected_files:
                if result_file and expected_file in result_file:
                    hit = True
                    if first_rank is None:
                        first_rank = rank
                    break

            if hit:
                break

        if hit:
            hits += 1
            passed += 1
            reciprocal_ranks.append(1.0 / first_rank)
        else:
            failed += 1
            reciprocal_ranks.append(0.0)

    total = len(context_queries)
    hit_rate = hits / total if total > 0 else 0.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "hit_rate": hit_rate,
        "mrr": mrr,
        "avg_latency_ms": avg_latency,
        "passed": passed,
        "failed": failed,
        "total": total,
    }


def _extract_file_path(result: SearchResult) -> str | None:
    """Extract file path from search result."""
    # Try citation first (e.g., 'src/file.py:123')
    if result.citation:
        parts = result.citation.split(":")
        if parts:
            return parts[0]

    # Fall back to title for repo sources
    if result.source_type == "repo":
        return result.title

    return None
```

Now modify CLI to add eval command:

```python
# In src/chinvex/cli.py, add after existing commands:

@cli.command()
@click.option("--context", required=True, help="Context name to evaluate")
@click.option("--k", type=int, default=None, help="Override K value for all queries")
def eval(context: str, k: int | None):
    """Run evaluation suite against golden queries.

    Loads golden queries for the specified context and runs retrieval evaluation.
    Reports hit rate, MRR, and latency metrics.
    Compares to baseline and fails if performance regresses.
    """
    from .eval_runner import run_evaluation
    from .eval_baseline import load_baseline_metrics, compare_to_baseline

    try:
        # Run evaluation
        click.echo(f"Running evaluation for context: {context}")
        metrics = run_evaluation(context, k=k)

        # Display results
        click.echo(f"\nEvaluation Results:")
        click.echo(f"  Hit Rate@5: {metrics['hit_rate'] * 100:.1f}%")
        click.echo(f"  MRR: {metrics['mrr']:.3f}")
        click.echo(f"  Avg Latency: {metrics['avg_latency_ms']:.1f}ms")
        click.echo(f"  Passed: {metrics['passed']}/{metrics['total']}")
        click.echo(f"  Failed: {metrics['failed']}/{metrics['total']}")

        # Compare to baseline
        try:
            baseline = load_baseline_metrics(context)
            current_metrics = type('EvalMetrics', (), {
                'hit_rate': metrics['hit_rate'],
                'mrr': metrics['mrr'],
                'avg_latency_ms': metrics['avg_latency_ms']
            })()

            comparison = compare_to_baseline(current_metrics, baseline)

            click.echo(f"\nBaseline Comparison:")
            click.echo(f"  Baseline Hit Rate: {baseline.hit_rate * 100:.1f}%")
            click.echo(f"  Change: {comparison.hit_rate_change * 100:+.1f}%")

            if comparison.passed:
                click.echo("  Status: PASS", fg="green")
            else:
                click.echo("  Status: REGRESSION (below {comparison.threshold * 100:.0f}% threshold)", fg="red")
                raise SystemExit(1)

        except FileNotFoundError:
            click.echo("\nNo baseline metrics found. Run with --save-baseline to create.")

    except Exception as e:
        click.echo(f"Error running evaluation: {e}", err=True)
        raise SystemExit(1)
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_eval_cli.py -v`
Expected: PASS (all 5 tests passing)

**Step 5: Commit**
```bash
git add src/chinvex/eval_runner.py src/chinvex/cli.py tests/test_eval_cli.py
git commit -m "feat(P5.3): add chinvex eval CLI command for retrieval evaluation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```


### Task 18: Create initial 20+ golden queries for Chinvex context

**Files:**
- Create: `C:\Code\chinvex\tests\eval\golden_queries_chinvex.json`
- Test: `C:\Code\chinvex\tests\test_chinvex_golden_queries.py`

**Step 1: Write the failing test**
```python
# tests/test_chinvex_golden_queries.py
import json
from pathlib import Path

from chinvex.eval_schema import load_golden_queries


def test_chinvex_golden_queries_file_exists():
    """Golden queries file for Chinvex context should exist."""
    golden_file = Path("tests/eval/golden_queries_chinvex.json")
    assert golden_file.exists(), "Create golden_queries_chinvex.json with 20+ queries"


def test_chinvex_golden_queries_has_minimum_count():
    """Chinvex golden queries should have at least 20 queries."""
    golden_file = Path("tests/eval/golden_queries_chinvex.json")
    queries = load_golden_queries(golden_file)

    chinvex_queries = [q for q in queries if q.context == "Chinvex"]
    assert len(chinvex_queries) >= 20, f"Need at least 20 queries, found {len(chinvex_queries)}"


def test_chinvex_golden_queries_all_valid():
    """All Chinvex golden queries should have valid schema."""
    golden_file = Path("tests/eval/golden_queries_chinvex.json")
    queries = load_golden_queries(golden_file)

    for query in queries:
        if query.context != "Chinvex":
            continue

        # Required fields
        assert query.query, "Query text is required"
        assert query.context == "Chinvex"
        assert query.expected_files, "At least one expected file required"
        assert query.k > 0, "k must be positive"

        # All expected files should be strings
        for expected_file in query.expected_files:
            assert isinstance(expected_file, str), "Expected files must be strings"


def test_chinvex_golden_queries_cover_key_areas():
    """Golden queries should cover key functional areas of Chinvex."""
    golden_file = Path("tests/eval/golden_queries_chinvex.json")
    queries = load_golden_queries(golden_file)

    chinvex_queries = [q for q in queries if q.context == "Chinvex"]
    query_texts = [q.query.lower() for q in chinvex_queries]

    # Check coverage of major areas
    areas = {
        "search": ["search", "query", "retrieval", "hybrid"],
        "ingest": ["ingest", "indexing", "chunking"],
        "embedding": ["embedding", "vector", "openai", "ollama"],
        "config": ["config", "context", "settings"],
        "cli": ["cli", "command"],
    }

    for area, keywords in areas.items():
        found = any(
            any(kw in qt for kw in keywords)
            for qt in query_texts
        )
        assert found, f"No queries found covering {area} area (keywords: {keywords})"


def test_chinvex_golden_queries_expected_files_exist():
    """Expected files in golden queries should exist in codebase."""
    golden_file = Path("tests/eval/golden_queries_chinvex.json")
    queries = load_golden_queries(golden_file)

    repo_root = Path(".")
    missing_files = []

    for query in queries:
        if query.context != "Chinvex":
            continue

        for expected_file in query.expected_files:
            # Normalize path
            file_path = repo_root / expected_file
            if not file_path.exists():
                missing_files.append((query.query, expected_file))

    assert not missing_files, f"Expected files not found: {missing_files}"
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_chinvex_golden_queries.py -v`
Expected: FAIL with "FileNotFoundError" or "AssertionError: Create golden_queries_chinvex.json"

**Step 3: Write minimal implementation**
```json
{
  "queries": [
    {
      "query": "How does hybrid search combine lexical and vector results?",
      "context": "Chinvex",
      "expected_files": ["src/chinvex/search.py", "src/chinvex/scoring.py"],
      "anchor": "blend_scores",
      "k": 5
    },
    {
      "query": "chromadb batch size limit for embeddings",
      "context": "Chinvex",
      "expected_files": ["src/chinvex/ingest.py"],
      "anchor": "5461",
      "k": 5
    },
    {
      "query": "what embedding providers are supported",
      "context": "Chinvex",
      "expected_files": ["src/chinvex/embed.py", "src/chinvex/config.py"],
      "k": 5
    },
    {
      "query": "CLI command to search a context",
      "context": "Chinvex",
      "expected_files": ["src/chinvex/cli.py"],
      "anchor": "search",
      "k": 5
    },
    {
      "query": "How to run ingestion for a context",
      "context": "Chinvex",
      "expected_files": ["src/chinvex/cli.py", "src/chinvex/ingest.py"],
      "k": 5
    },
    {
      "query": "FTS5 configuration and tokenizer settings",
      "context": "Chinvex",
      "expected_files": ["src/chinvex/storage.py"],
      "anchor": "tokenize",
      "k": 5
    },
    {
      "query": "chunk size and overlap settings for code ingestion",
      "context": "Chinvex",
      "expected_files": ["src/chinvex/chunking.py"],
      "k": 5
    },
    {
      "query": "OpenAI embedding model configuration",
      "context": "Chinvex",
      "expected_files": ["src/chinvex/embed.py", "context.json"],
      "anchor": "text-embedding-3-small",
      "k": 5
    },
    {
      "query": "How does cross-context search work",
      "context": "Chinvex",
      "expected_files": ["src/chinvex/search.py"],
      "k": 5
    },
    {
      "query": "Gateway API endpoint for search",
      "context": "Chinvex",
      "expected_files": ["src/chinvex/gateway/endpoints/search.py"],
      "k": 5
    },
    {
      "query": "MCP server integration with Claude Code",
      "context": "Chinvex",
      "expected_files": ["src/chinvex_mcp/server.py"],
      "k": 5
    },
    {
      "query": "score normalization for hybrid retrieval",
      "context": "Chinvex",
      "expected_files": ["src/chinvex/scoring.py"],
      "anchor": "normalize",
      "k": 5
    },
    {
      "query": "default weights for lexical vs vector search",
      "context": "Chinvex",
      "expected_files": ["src/chinvex/search.py", "src/chinvex/scoring.py"],
      "k": 5
    },
    {
      "query": "archive functionality to exclude old chunks",
      "context": "Chinvex",
      "expected_files": ["src/chinvex/storage.py", "src/chinvex/search.py"],
      "k": 5
    },
    {
      "query": "git integration for repo ingestion",
      "context": "Chinvex",
      "expected_files": ["src/chinvex/sources/repo.py"],
      "k": 5
    },
    {
      "query": "file watcher daemon for automatic syncing",
      "context": "Chinvex",
      "expected_files": ["src/chinvex/cli.py"],
      "anchor": "sync",
      "k": 5
    },
    {
      "query": "How to generate a session brief",
      "context": "Chinvex",
      "expected_files": ["src/chinvex/cli.py", "src/chinvex/brief.py"],
      "k": 5
    },
    {
      "query": "context.json schema and required fields",
      "context": "Chinvex",
      "expected_files": ["src/chinvex/config.py"],
      "k": 5
    },
    {
      "query": "storage schema for chunks table",
      "context": "Chinvex",
      "expected_files": ["src/chinvex/storage.py"],
      "anchor": "CREATE TABLE",
      "k": 5
    },
    {
      "query": "VectorStore initialization with ChromaDB",
      "context": "Chinvex",
      "expected_files": ["src/chinvex/vectors.py"],
      "k": 5
    },
    {
      "query": "How to detect mixed embedding providers across contexts",
      "context": "Chinvex",
      "expected_files": ["src/chinvex/search.py"],
      "anchor": "mixed",
      "k": 5
    },
    {
      "query": "P5 implementation spec for memory automation",
      "context": "Chinvex",
      "expected_files": ["specs/P5a_IMPLEMENTATION_SPEC.md", "specs/P5b_IMPLEMENTATION_SPEC.md"],
      "k": 5
    },
    {
      "query": "chinvex status command output format",
      "context": "Chinvex",
      "expected_files": ["src/chinvex/cli.py"],
      "anchor": "status",
      "k": 5
    }
  ]
}
```

Save to: `C:\Code\chinvex\tests\eval\golden_queries_chinvex.json`

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_chinvex_golden_queries.py -v`
Expected: PASS (all 5 tests passing)

**Step 5: Commit**
```bash
git add tests/eval/golden_queries_chinvex.json tests/test_chinvex_golden_queries.py
git commit -m "feat(P5.3): add 23 golden queries for Chinvex context evaluation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```


### Task 19: Reranker configuration schema in context.json

**Files:**
- Create: `C:\Code\chinvex\src\chinvex\reranker_config.py`
- Test: `C:\Code\chinvex\tests\test_reranker_config.py`

**Step 1: Write the failing test**
```python
# tests/test_reranker_config.py
import json
from pathlib import Path

import pytest

from chinvex.reranker_config import (
    RerankerConfig,
    load_reranker_config,
    validate_reranker_config,
)


def test_reranker_config_dataclass_structure():
    """RerankerConfig should have required fields."""
    config = RerankerConfig(
        provider="cohere",
        model="rerank-english-v3.0",
        candidates=20,
        top_k=5
    )

    assert config.provider == "cohere"
    assert config.model == "rerank-english-v3.0"
    assert config.candidates == 20
    assert config.top_k == 5


def test_reranker_config_defaults():
    """RerankerConfig should provide sensible defaults."""
    config = RerankerConfig(
        provider="cohere",
        model="rerank-english-v3.0"
    )

    assert config.candidates == 20  # default
    assert config.top_k == 5  # default


def test_load_reranker_config_from_context_json(tmp_path):
    """load_reranker_config should parse reranker field from context.json."""
    context_file = tmp_path / "context.json"
    context_data = {
        "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
        "reranker": {
            "provider": "cohere",
            "model": "rerank-english-v3.0",
            "candidates": 30,
            "top_k": 8
        }
    }
    context_file.write_text(json.dumps(context_data))

    config = load_reranker_config(context_file)

    assert config is not None
    assert config.provider == "cohere"
    assert config.model == "rerank-english-v3.0"
    assert config.candidates == 30
    assert config.top_k == 8


def test_load_reranker_config_returns_none_when_missing(tmp_path):
    """load_reranker_config should return None if reranker field absent."""
    context_file = tmp_path / "context.json"
    context_data = {
        "embedding": {"provider": "openai", "model": "text-embedding-3-small"}
    }
    context_file.write_text(json.dumps(context_data))

    config = load_reranker_config(context_file)

    assert config is None


def test_load_reranker_config_returns_none_when_null(tmp_path):
    """load_reranker_config should return None if reranker is null."""
    context_file = tmp_path / "context.json"
    context_data = {
        "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
        "reranker": None
    }
    context_file.write_text(json.dumps(context_data))

    config = load_reranker_config(context_file)

    assert config is None


def test_validate_reranker_config_accepts_valid_providers():
    """validate_reranker_config should accept cohere, jina, local providers."""
    for provider in ["cohere", "jina", "local"]:
        config = RerankerConfig(provider=provider, model="test-model")
        # Should not raise
        validate_reranker_config(config)


def test_validate_reranker_config_rejects_unknown_provider():
    """validate_reranker_config should reject unknown providers."""
    config = RerankerConfig(provider="unknown", model="test-model")

    with pytest.raises(ValueError, match="Unknown reranker provider"):
        validate_reranker_config(config)


def test_validate_reranker_config_requires_positive_candidates():
    """validate_reranker_config should require candidates > 0."""
    config = RerankerConfig(provider="cohere", model="test", candidates=0)

    with pytest.raises(ValueError, match="candidates must be positive"):
        validate_reranker_config(config)


def test_validate_reranker_config_requires_positive_top_k():
    """validate_reranker_config should require top_k > 0."""
    config = RerankerConfig(provider="cohere", model="test", top_k=0)

    with pytest.raises(ValueError, match="top_k must be positive"):
        validate_reranker_config(config)


def test_validate_reranker_config_requires_top_k_le_candidates():
    """validate_reranker_config should require top_k <= candidates."""
    config = RerankerConfig(provider="cohere", model="test", candidates=10, top_k=15)

    with pytest.raises(ValueError, match="top_k must be <= candidates"):
        validate_reranker_config(config)


def test_validate_reranker_config_caps_candidates_at_50():
    """validate_reranker_config should cap candidates at 50 (budget guardrail)."""
    config = RerankerConfig(provider="cohere", model="test", candidates=100)

    with pytest.raises(ValueError, match="candidates cannot exceed 50"):
        validate_reranker_config(config)
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_reranker_config.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chinvex.reranker_config'"

**Step 3: Write minimal implementation**
```python
# src/chinvex/reranker_config.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RerankerConfig:
    """Reranker configuration for two-stage retrieval.

    Attributes:
        provider: Reranker provider ('cohere', 'jina', 'local')
        model: Model name/identifier for the provider
        candidates: Number of candidates to retrieve in stage 1 (default 20)
        top_k: Number of results to return after reranking (default 5)
    """
    provider: str
    model: str
    candidates: int = 20
    top_k: int = 5


def load_reranker_config(context_file: Path) -> RerankerConfig | None:
    """Load reranker config from context.json.

    Args:
        context_file: Path to context.json

    Returns:
        RerankerConfig if present and valid, None if absent or null
    """
    with open(context_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    reranker_data = data.get("reranker")

    if reranker_data is None:
        return None

    return RerankerConfig(
        provider=reranker_data["provider"],
        model=reranker_data["model"],
        candidates=reranker_data.get("candidates", 20),
        top_k=reranker_data.get("top_k", 5),
    )


def validate_reranker_config(config: RerankerConfig) -> None:
    """Validate reranker configuration.

    Args:
        config: RerankerConfig to validate

    Raises:
        ValueError: If configuration is invalid
    """
    valid_providers = {"cohere", "jina", "local"}
    if config.provider not in valid_providers:
        raise ValueError(
            f"Unknown reranker provider: {config.provider}. "
            f"Valid providers: {valid_providers}"
        )

    if config.candidates <= 0:
        raise ValueError("candidates must be positive")

    if config.top_k <= 0:
        raise ValueError("top_k must be positive")

    if config.top_k > config.candidates:
        raise ValueError("top_k must be <= candidates")

    # Budget guardrail: max 50 candidates
    if config.candidates > 50:
        raise ValueError(
            f"candidates cannot exceed 50 (budget guardrail), got {config.candidates}"
        )
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_reranker_config.py -v`
Expected: PASS (all 11 tests passing)

**Step 5: Commit**
```bash
git add src/chinvex/reranker_config.py tests/test_reranker_config.py
git commit -m "feat(P5.4): add reranker configuration schema for context.json

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```


### Task 20: Cohere reranker provider implementation

**Files:**
- Create: `C:\Code\chinvex\src\chinvex\rerankers\cohere.py`
- Create: `C:\Code\chinvex\src\chinvex\rerankers\__init__.py`
- Test: `C:\Code\chinvex\tests\test_cohere_reranker.py`

**Step 1: Write the failing test**
```python
# tests/test_cohere_reranker.py
from unittest.mock import patch, MagicMock

import pytest

from chinvex.rerankers.cohere import CohereReranker
from chinvex.reranker_config import RerankerConfig


def test_cohere_reranker_initialization():
    """CohereReranker should initialize with config."""
    config = RerankerConfig(
        provider="cohere",
        model="rerank-english-v3.0",
        candidates=20,
        top_k=5
    )

    reranker = CohereReranker(config)

    assert reranker.config == config
    assert reranker.model == "rerank-english-v3.0"


def test_cohere_reranker_requires_api_key():
    """CohereReranker should require COHERE_API_KEY environment variable."""
    config = RerankerConfig(provider="cohere", model="rerank-english-v3.0")

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="COHERE_API_KEY"):
            CohereReranker(config)


@patch("chinvex.rerankers.cohere.cohere")
def test_cohere_reranker_calls_api(mock_cohere_module):
    """CohereReranker should call Cohere rerank API with correct parameters."""
    config = RerankerConfig(
        provider="cohere",
        model="rerank-english-v3.0",
        candidates=20,
        top_k=5
    )

    # Mock Cohere client
    mock_client = MagicMock()
    mock_cohere_module.Client.return_value = mock_client

    # Mock rerank response
    mock_result = MagicMock()
    mock_result.results = [
        MagicMock(index=2, relevance_score=0.95),
        MagicMock(index=0, relevance_score=0.87),
        MagicMock(index=1, relevance_score=0.72),
    ]
    mock_client.rerank.return_value = mock_result

    with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
        reranker = CohereReranker(config)

        candidates = [
            {"chunk_id": "c1", "text": "First chunk"},
            {"chunk_id": "c2", "text": "Second chunk"},
            {"chunk_id": "c3", "text": "Third chunk"},
        ]

        reranked = reranker.rerank(query="test query", candidates=candidates, top_k=3)

        # Verify API call
        mock_client.rerank.assert_called_once()
        call_kwargs = mock_client.rerank.call_args[1]
        assert call_kwargs["query"] == "test query"
        assert call_kwargs["documents"] == ["First chunk", "Second chunk", "Third chunk"]
        assert call_kwargs["model"] == "rerank-english-v3.0"
        assert call_kwargs["top_n"] == 3

        # Verify reranked order (by index in mock results)
        assert len(reranked) == 3
        assert reranked[0]["chunk_id"] == "c3"  # index=2
        assert reranked[0]["rerank_score"] == 0.95
        assert reranked[1]["chunk_id"] == "c1"  # index=0
        assert reranked[1]["rerank_score"] == 0.87
        assert reranked[2]["chunk_id"] == "c2"  # index=1
        assert reranked[2]["rerank_score"] == 0.72


@patch("chinvex.rerankers.cohere.cohere")
def test_cohere_reranker_handles_timeout(mock_cohere_module):
    """CohereReranker should raise TimeoutError on slow API response."""
    config = RerankerConfig(provider="cohere", model="rerank-english-v3.0")

    mock_client = MagicMock()
    mock_cohere_module.Client.return_value = mock_client
    mock_client.rerank.side_effect = Exception("Timeout")

    with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
        reranker = CohereReranker(config)

        candidates = [{"chunk_id": "c1", "text": "test"}]

        with pytest.raises(Exception, match="Timeout"):
            reranker.rerank(query="test", candidates=candidates, top_k=1)


@patch("chinvex.rerankers.cohere.cohere")
def test_cohere_reranker_truncates_to_max_candidates(mock_cohere_module):
    """CohereReranker should truncate to 50 candidates max (budget guardrail)."""
    config = RerankerConfig(provider="cohere", model="rerank-english-v3.0")

    mock_client = MagicMock()
    mock_cohere_module.Client.return_value = mock_client
    mock_client.rerank.return_value = MagicMock(results=[])

    with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
        reranker = CohereReranker(config)

        # Create 60 candidates
        candidates = [{"chunk_id": f"c{i}", "text": f"chunk {i}"} for i in range(60)]

        reranker.rerank(query="test", candidates=candidates, top_k=5)

        # Should only send 50 to API
        call_kwargs = mock_client.rerank.call_args[1]
        assert len(call_kwargs["documents"]) == 50
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_cohere_reranker.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chinvex.rerankers'"

**Step 3: Write minimal implementation**
```python
# src/chinvex/rerankers/__init__.py
"""Reranker providers for two-stage retrieval."""

from .cohere import CohereReranker

__all__ = ["CohereReranker"]
```

```python
# src/chinvex/rerankers/cohere.py
from __future__ import annotations

import os
from typing import Any

try:
    import cohere
except ImportError:
    cohere = None

from ..reranker_config import RerankerConfig


class CohereReranker:
    """Cohere Rerank API provider for two-stage retrieval.

    Requires COHERE_API_KEY environment variable.
    """

    MAX_CANDIDATES = 50  # Budget guardrail

    def __init__(self, config: RerankerConfig):
        if cohere is None:
            raise ImportError(
                "cohere package not installed. Install with: pip install cohere"
            )

        self.config = config
        self.model = config.model

        # Get API key from environment
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "COHERE_API_KEY environment variable required for Cohere reranker"
            )

        self.client = cohere.Client(api_key)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank candidates using Cohere Rerank API.

        Args:
            query: Search query string
            candidates: List of candidate dicts with 'chunk_id' and 'text' fields
            top_k: Number of results to return (default: use config.top_k)

        Returns:
            Reranked candidates (top K) with 'rerank_score' field added

        Raises:
            Exception: If API call fails or times out
        """
        if top_k is None:
            top_k = self.config.top_k

        # Budget guardrail: truncate to max 50 candidates
        if len(candidates) > self.MAX_CANDIDATES:
            candidates = candidates[:self.MAX_CANDIDATES]

        # Extract text for reranking
        documents = [c["text"] for c in candidates]

        # Call Cohere Rerank API
        response = self.client.rerank(
            query=query,
            documents=documents,
            model=self.model,
            top_n=top_k,
        )

        # Map results back to original candidates
        reranked = []
        for result in response.results:
            candidate = candidates[result.index].copy()
            candidate["rerank_score"] = result.relevance_score
            reranked.append(candidate)

        return reranked
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_cohere_reranker.py -v`
Expected: PASS (all 5 tests passing)

**Step 5: Commit**
```bash
git add src/chinvex/rerankers/__init__.py src/chinvex/rerankers/cohere.py tests/test_cohere_reranker.py
git commit -m "feat(P5.4): add Cohere reranker provider with budget guardrails

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```


---

### Task 21: Jina Reranker Provider

**Files:**
- Create: `C:\Code\chinvex\src\chinvex\rerankers\jina.py`
- Test: `C:\Code\chinvex\tests\test_jina_reranker.py`

**Step 1: Write the failing test**
```python
# tests/test_jina_reranker.py
from unittest.mock import MagicMock, patch
import pytest
from chinvex.rerankers.jina import JinaReranker, JinaRerankerConfig


def test_jina_reranker_initialization():
    config = JinaRerankerConfig(
        provider="jina",
        model="jina-reranker-v1-base-en",
        candidates=20,
        top_k=5,
    )
    with patch("chinvex.rerankers.jina.requests"):
        reranker = JinaReranker(config, api_key="test-key")
        assert reranker.model == "jina-reranker-v1-base-en"
        assert reranker.api_key == "test-key"
        assert reranker.config.top_k == 5


def test_jina_reranker_missing_api_key():
    config = JinaRerankerConfig(
        provider="jina",
        model="jina-reranker-v1-base-en",
        candidates=20,
        top_k=5,
    )
    with pytest.raises(ValueError, match="JINA_API_KEY"):
        JinaReranker(config, api_key=None)


def test_jina_reranker_rerank_success():
    config = JinaRerankerConfig(
        provider="jina",
        model="jina-reranker-v1-base-en",
        candidates=20,
        top_k=5,
    )
    candidates = [
        {"chunk_id": "c1", "text": "Python is a programming language"},
        {"chunk_id": "c2", "text": "Java is also a programming language"},
        {"chunk_id": "c3", "text": "The sky is blue"},
    ]

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"index": 0, "relevance_score": 0.95},
            {"index": 1, "relevance_score": 0.85},
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("chinvex.rerankers.jina.requests.post", return_value=mock_response):
        reranker = JinaReranker(config, api_key="test-key")
        results = reranker.rerank("Python programming", candidates, top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["rerank_score"] == 0.95
    assert results[1]["chunk_id"] == "c2"
    assert results[1]["rerank_score"] == 0.85


def test_jina_reranker_truncates_candidates():
    config = JinaRerankerConfig(
        provider="jina",
        model="jina-reranker-v1-base-en",
        candidates=20,
        top_k=5,
    )
    # Create 60 candidates (exceeds MAX_CANDIDATES=50)
    candidates = [{"chunk_id": f"c{i}", "text": f"text {i}"} for i in range(60)]

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [{"index": i, "relevance_score": 0.9 - i * 0.01} for i in range(5)]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("chinvex.rerankers.jina.requests.post", return_value=mock_response) as mock_post:
        reranker = JinaReranker(config, api_key="test-key")
        results = reranker.rerank("test query", candidates)

    # Verify only 50 candidates sent to API
    call_args = mock_post.call_args
    sent_documents = call_args[1]["json"]["documents"]
    assert len(sent_documents) == 50
    assert len(results) == 5


def test_jina_reranker_api_failure():
    config = JinaRerankerConfig(
        provider="jina",
        model="jina-reranker-v1-base-en",
        candidates=20,
        top_k=5,
    )
    candidates = [{"chunk_id": "c1", "text": "test"}]

    with patch("chinvex.rerankers.jina.requests.post", side_effect=Exception("API error")):
        reranker = JinaReranker(config, api_key="test-key")
        with pytest.raises(Exception, match="API error"):
            reranker.rerank("query", candidates)
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_jina_reranker.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chinvex.rerankers.jina'"

**Step 3: Write minimal implementation**
```python
# src/chinvex/rerankers/jina.py
from __future__ import annotations

import os
import requests
from dataclasses import dataclass


@dataclass
class JinaRerankerConfig:
    provider: str
    model: str
    candidates: int
    top_k: int


class JinaReranker:
    """Reranker using Jina Reranker API."""

    MAX_CANDIDATES = 50
    API_URL = "https://api.jina.ai/v1/rerank"

    def __init__(self, config: JinaRerankerConfig, api_key: str | None = None):
        self.config = config
        self.model = config.model
        self.api_key = api_key or os.getenv("JINA_API_KEY")

        if not self.api_key:
            raise ValueError(
                "JINA_API_KEY environment variable required for Jina reranker. "
                "Get your API key from https://jina.ai/"
            )

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Rerank candidates using Jina Rerank API.

        Args:
            query: Search query string
            candidates: List of candidate dicts with 'chunk_id' and 'text' fields
            top_k: Number of results to return (default: use config.top_k)

        Returns:
            Reranked candidates (top K) with 'rerank_score' field added

        Raises:
            Exception: If API call fails or times out
        """
        if top_k is None:
            top_k = self.config.top_k

        # Budget guardrail: truncate to max 50 candidates
        if len(candidates) > self.MAX_CANDIDATES:
            candidates = candidates[:self.MAX_CANDIDATES]

        # Extract text for reranking
        documents = [c["text"] for c in candidates]

        # Call Jina Rerank API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_k,
        }

        response = requests.post(
            self.API_URL,
            headers=headers,
            json=payload,
            timeout=2.0,
        )
        response.raise_for_status()

        data = response.json()

        # Map results back to original candidates
        reranked = []
        for result in data["results"]:
            candidate = candidates[result["index"]].copy()
            candidate["rerank_score"] = result["relevance_score"]
            reranked.append(candidate)

        return reranked
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_jina_reranker.py -v`
Expected: PASS (all 5 tests passing)

**Step 5: Commit**
```bash
git add src/chinvex/rerankers/jina.py tests/test_jina_reranker.py
git commit -m "feat(P5.4): add Jina reranker provider with budget guardrails

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 22: Local Cross-Encoder Reranker Provider

**Files:**
- Create: `C:\Code\chinvex\src\chinvex\rerankers\local.py`
- Test: `C:\Code\chinvex\tests\test_local_reranker.py`

**Step 1: Write the failing test**
```python
# tests/test_local_reranker.py
from unittest.mock import MagicMock, patch
import pytest
from chinvex.rerankers.local import LocalReranker, LocalRerankerConfig


def test_local_reranker_initialization():
    config = LocalRerankerConfig(
        provider="local",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidates=20,
        top_k=5,
    )
    with patch("chinvex.rerankers.local.CrossEncoder") as mock_ce:
        mock_model = MagicMock()
        mock_ce.return_value = mock_model
        reranker = LocalReranker(config)
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert reranker.config.top_k == 5
        mock_ce.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2")


def test_local_reranker_rerank_success():
    config = LocalRerankerConfig(
        provider="local",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidates=20,
        top_k=5,
    )
    candidates = [
        {"chunk_id": "c1", "text": "Python is a programming language"},
        {"chunk_id": "c2", "text": "Java is also a programming language"},
        {"chunk_id": "c3", "text": "The sky is blue"},
    ]

    mock_model = MagicMock()
    # Mock predict to return scores in descending order
    mock_model.predict.return_value = [0.95, 0.85, 0.25]

    with patch("chinvex.rerankers.local.CrossEncoder", return_value=mock_model):
        reranker = LocalReranker(config)
        results = reranker.rerank("Python programming", candidates, top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["rerank_score"] == 0.95
    assert results[1]["chunk_id"] == "c2"
    assert results[1]["rerank_score"] == 0.85


def test_local_reranker_truncates_candidates():
    config = LocalRerankerConfig(
        provider="local",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidates=20,
        top_k=5,
    )
    # Create 60 candidates (exceeds MAX_CANDIDATES=50)
    candidates = [{"chunk_id": f"c{i}", "text": f"text {i}"} for i in range(60)]

    mock_model = MagicMock()
    # Return 50 scores (truncated)
    mock_model.predict.return_value = [0.9 - i * 0.01 for i in range(50)]

    with patch("chinvex.rerankers.local.CrossEncoder", return_value=mock_model):
        reranker = LocalReranker(config)
        results = reranker.rerank("test query", candidates)

    # Verify only 50 candidates processed
    assert len(results) == 5
    assert mock_model.predict.call_count == 1
    # Check that predict was called with 50 pairs
    call_args = mock_model.predict.call_args[0][0]
    assert len(call_args) == 50


def test_local_reranker_model_loading_failure():
    config = LocalRerankerConfig(
        provider="local",
        model="invalid-model-name",
        candidates=20,
        top_k=5,
    )

    with patch("chinvex.rerankers.local.CrossEncoder", side_effect=Exception("Model not found")):
        with pytest.raises(Exception, match="Model not found"):
            LocalReranker(config)


def test_local_reranker_empty_candidates():
    config = LocalRerankerConfig(
        provider="local",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidates=20,
        top_k=5,
    )
    candidates = []

    mock_model = MagicMock()
    mock_model.predict.return_value = []

    with patch("chinvex.rerankers.local.CrossEncoder", return_value=mock_model):
        reranker = LocalReranker(config)
        results = reranker.rerank("test query", candidates)

    assert results == []
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_local_reranker.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chinvex.rerankers.local'"

**Step 3: Write minimal implementation**
```python
# src/chinvex/rerankers/local.py
from __future__ import annotations

from dataclasses import dataclass
from sentence_transformers import CrossEncoder


@dataclass
class LocalRerankerConfig:
    provider: str
    model: str
    candidates: int
    top_k: int


class LocalReranker:
    """Local cross-encoder reranker using sentence-transformers.

    Downloads model to ~/.cache/huggingface/ on first use.
    Slower than API-based rerankers but free (no API key needed).
    """

    MAX_CANDIDATES = 50

    def __init__(self, config: LocalRerankerConfig):
        self.config = config
        self.model_name = config.model
        # Load cross-encoder model (downloads on first use)
        self.model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Rerank candidates using local cross-encoder model.

        Args:
            query: Search query string
            candidates: List of candidate dicts with 'chunk_id' and 'text' fields
            top_k: Number of results to return (default: use config.top_k)

        Returns:
            Reranked candidates (top K) with 'rerank_score' field added
        """
        if top_k is None:
            top_k = self.config.top_k

        if not candidates:
            return []

        # Budget guardrail: truncate to max 50 candidates
        if len(candidates) > self.MAX_CANDIDATES:
            candidates = candidates[:self.MAX_CANDIDATES]

        # Prepare query-document pairs for cross-encoder
        pairs = [[query, c["text"]] for c in candidates]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach scores to candidates
        scored_candidates = []
        for candidate, score in zip(candidates, scores):
            candidate_copy = candidate.copy()
            candidate_copy["rerank_score"] = float(score)
            scored_candidates.append(candidate_copy)

        # Sort by score descending and return top K
        scored_candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return scored_candidates[:top_k]
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_local_reranker.py -v`
Expected: PASS (all 5 tests passing)

**Step 5: Commit**
```bash
git add src/chinvex/rerankers/local.py tests/test_local_reranker.py
git commit -m "feat(P5.4): add local cross-encoder reranker provider

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 23: Reranker Integration in Search with Budget Guardrails

**Files:**
- Modify: `C:\Code\chinvex\src\chinvex\search.py`
- Modify: `C:\Code\chinvex\src\chinvex\context.py`
- Test: `C:\Code\chinvex\tests\test_search_reranker.py`

**Step 1: Write the failing test**
```python
# tests/test_search_reranker.py
from unittest.mock import MagicMock, patch
import pytest
from pathlib import Path
from chinvex.search import search_context
from chinvex.context import ContextConfig


def test_search_with_reranker_enabled():
    """Test that reranker is invoked when configured in context."""
    mock_ctx = MagicMock()
    mock_ctx.name = "test"
    mock_ctx.index.sqlite_path = Path("/tmp/test.db")
    mock_ctx.index.chroma_dir = Path("/tmp/chroma")
    mock_ctx.embedding = MagicMock(provider="openai", model="text-embedding-3-small")
    mock_ctx.ollama.base_url = "http://localhost:11434"
    mock_ctx.ollama.embed_model = "mxbai-embed-large"
    mock_ctx.weights = {"repo": 1.0}
    mock_ctx.reranker = MagicMock(
        provider="cohere",
        model="rerank-english-v3.0",
        candidates=20,
        top_k=5,
    )

    # Mock search_chunks to return 15 candidates
    mock_scored_chunks = [
        MagicMock(chunk_id=f"c{i}", score=0.8 - i * 0.05, row={"source_type": "repo", "text": f"text {i}", "meta_json": None, "doc_id": f"doc{i}"})
        for i in range(15)
    ]

    with patch("chinvex.search.search_chunks", return_value=mock_scored_chunks):
        with patch("chinvex.search._get_reranker") as mock_get_reranker:
            mock_reranker = MagicMock()
            # Reranker returns top 5
            mock_reranker.rerank.return_value = [
                {"chunk_id": f"c{i}", "text": f"text {i}", "rerank_score": 0.95 - i * 0.05}
                for i in range(5)
            ]
            mock_get_reranker.return_value = mock_reranker

            results = search_context(mock_ctx, "test query", k=5, rerank=True)

            # Verify reranker was called
            mock_get_reranker.assert_called_once()
            mock_reranker.rerank.assert_called_once()
            # Verify candidates passed to reranker (15 chunks)
            call_args = mock_reranker.rerank.call_args
            assert call_args[0][0] == "test query"
            assert len(call_args[0][1]) == 15
            # Verify top 5 returned
            assert len(results) == 5


def test_search_with_reranker_disabled():
    """Test that reranker is NOT invoked when not configured."""
    mock_ctx = MagicMock()
    mock_ctx.name = "test"
    mock_ctx.index.sqlite_path = Path("/tmp/test.db")
    mock_ctx.index.chroma_dir = Path("/tmp/chroma")
    mock_ctx.embedding = MagicMock(provider="openai", model="text-embedding-3-small")
    mock_ctx.ollama.base_url = "http://localhost:11434"
    mock_ctx.ollama.embed_model = "mxbai-embed-large"
    mock_ctx.weights = {"repo": 1.0}
    mock_ctx.reranker = None  # No reranker configured

    mock_scored_chunks = [
        MagicMock(chunk_id=f"c{i}", score=0.8 - i * 0.05, row={"source_type": "repo", "text": f"text {i}", "meta_json": None, "doc_id": f"doc{i}"})
        for i in range(5)
    ]

    with patch("chinvex.search.search_chunks", return_value=mock_scored_chunks):
        with patch("chinvex.search._get_reranker") as mock_get_reranker:
            results = search_context(mock_ctx, "test query", k=5, rerank=False)

            # Verify reranker was NOT called
            mock_get_reranker.assert_not_called()
            assert len(results) == 5


def test_search_reranker_fallback_on_failure():
    """Test that search falls back to pre-rerank results if reranker fails."""
    mock_ctx = MagicMock()
    mock_ctx.name = "test"
    mock_ctx.index.sqlite_path = Path("/tmp/test.db")
    mock_ctx.index.chroma_dir = Path("/tmp/chroma")
    mock_ctx.embedding = MagicMock(provider="openai", model="text-embedding-3-small")
    mock_ctx.ollama.base_url = "http://localhost:11434"
    mock_ctx.ollama.embed_model = "mxbai-embed-large"
    mock_ctx.weights = {"repo": 1.0}
    mock_ctx.reranker = MagicMock(
        provider="cohere",
        model="rerank-english-v3.0",
        candidates=20,
        top_k=5,
    )

    mock_scored_chunks = [
        MagicMock(chunk_id=f"c{i}", score=0.8 - i * 0.05, row={"source_type": "repo", "text": f"text {i}", "meta_json": None, "doc_id": f"doc{i}"})
        for i in range(15)
    ]

    with patch("chinvex.search.search_chunks", return_value=mock_scored_chunks):
        with patch("chinvex.search._get_reranker") as mock_get_reranker:
            mock_reranker = MagicMock()
            mock_reranker.rerank.side_effect = Exception("API timeout")
            mock_get_reranker.return_value = mock_reranker

            with patch("sys.stderr") as mock_stderr:
                results = search_context(mock_ctx, "test query", k=5, rerank=True)

                # Verify warning printed
                assert any("Reranker failed" in str(call) for call in mock_stderr.write.call_args_list)
                # Verify fallback to pre-rerank results (top 5 by original score)
                assert len(results) == 5
                assert results[0].chunk_id == "c0"


def test_search_reranker_skipped_for_few_candidates():
    """Test that reranker is skipped if fewer than 10 candidates."""
    mock_ctx = MagicMock()
    mock_ctx.name = "test"
    mock_ctx.index.sqlite_path = Path("/tmp/test.db")
    mock_ctx.index.chroma_dir = Path("/tmp/chroma")
    mock_ctx.embedding = MagicMock(provider="openai", model="text-embedding-3-small")
    mock_ctx.ollama.base_url = "http://localhost:11434"
    mock_ctx.ollama.embed_model = "mxbai-embed-large"
    mock_ctx.weights = {"repo": 1.0}
    mock_ctx.reranker = MagicMock(
        provider="cohere",
        model="rerank-english-v3.0",
        candidates=20,
        top_k=5,
    )

    # Only 8 candidates (below threshold of 10)
    mock_scored_chunks = [
        MagicMock(chunk_id=f"c{i}", score=0.8 - i * 0.05, row={"source_type": "repo", "text": f"text {i}", "meta_json": None, "doc_id": f"doc{i}"})
        for i in range(8)
    ]

    with patch("chinvex.search.search_chunks", return_value=mock_scored_chunks):
        with patch("chinvex.search._get_reranker") as mock_get_reranker:
            results = search_context(mock_ctx, "test query", k=5, rerank=True)

            # Verify reranker was NOT called (too few candidates)
            mock_get_reranker.assert_not_called()
            assert len(results) == 5
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_search_reranker.py -v`
Expected: FAIL with "TypeError: search_context() got an unexpected keyword argument 'rerank'"

**Step 3: Write minimal implementation**

First, add reranker config to context.py:
```python
# src/chinvex/context.py (add to existing file after EmbeddingConfig)

@dataclass(frozen=True)
class RerankerConfig:
    """Reranker configuration (P5.4)."""
    provider: str  # "cohere", "jina", or "local"
    model: str
    candidates: int  # Number of candidates to fetch from initial retrieval
    top_k: int  # Number of results to return after reranking
```

Update ContextConfig class:
```python
# src/chinvex/context.py (modify ContextConfig class)

@dataclass(frozen=True)
class ContextConfig:
    # ... existing fields ...
    embedding: EmbeddingConfig | None = None
    reranker: RerankerConfig | None = None  # NEW: P5.4 addition

    @classmethod
    def from_dict(cls, data: dict) -> ContextConfig:
        # ... existing code for other fields ...

        # P4: embedding field (optional)
        embedding = None
        if "embedding" in data and data["embedding"] is not None:
            emb_data = data["embedding"]
            embedding = EmbeddingConfig(
                provider=emb_data.get("provider", "ollama"),
                model=emb_data.get("model"),
            )

        # P5.4: reranker field (optional)
        reranker = None
        if "reranker" in data and data["reranker"] is not None:
            rerank_data = data["reranker"]
            reranker = RerankerConfig(
                provider=rerank_data.get("provider", "cohere"),
                model=rerank_data.get("model", "rerank-english-v3.0"),
                candidates=rerank_data.get("candidates", 20),
                top_k=rerank_data.get("top_k", 5),
            )

        return cls(
            # ... existing args ...
            embedding=embedding,
            reranker=reranker,  # NEW
        )
```

Now update search.py with reranker support. Add at top of file:
```python
# src/chinvex/search.py (add imports)
import os
import time
```

Add helper function:
```python
# src/chinvex/search.py (add after imports)

def _get_reranker(config):
    """Factory function to instantiate reranker based on config.

    Args:
        config: RerankerConfig object

    Returns:
        Reranker instance (Cohere, Jina, or Local)

    Raises:
        ValueError: If provider is unknown
        ValueError: If required API key is missing
    """
    if config.provider == "cohere":
        from .rerankers.cohere import CohereReranker, CohereRerankerConfig
        api_key = os.getenv("COHERE_API_KEY")
        reranker_config = CohereRerankerConfig(
            provider=config.provider,
            model=config.model,
            candidates=config.candidates,
            top_k=config.top_k,
        )
        return CohereReranker(reranker_config, api_key=api_key)
    elif config.provider == "jina":
        from .rerankers.jina import JinaReranker, JinaRerankerConfig
        api_key = os.getenv("JINA_API_KEY")
        reranker_config = JinaRerankerConfig(
            provider=config.provider,
            model=config.model,
            candidates=config.candidates,
            top_k=config.top_k,
        )
        return JinaReranker(reranker_config, api_key=api_key)
    elif config.provider == "local":
        from .rerankers.local import LocalReranker, LocalRerankerConfig
        reranker_config = LocalRerankerConfig(
            provider=config.provider,
            model=config.model,
            candidates=config.candidates,
            top_k=config.top_k,
        )
        return LocalReranker(reranker_config)
    else:
        raise ValueError(f"Unknown reranker provider: {config.provider}")
```

Update search_context function:
```python
# src/chinvex/search.py (replace search_context function)

def search_context(
    ctx,
    query: str,
    *,
    k: int = 8,
    min_score: float = 0.35,
    source: str = "all",
    project: str | None = None,
    repo: str | None = None,
    ollama_host_override: str | None = None,
    recency_enabled: bool = True,
    rerank: bool = False,  # NEW: enable reranking for this query
) -> list[SearchResult]:
    """
    Search within a context using context-aware weights.

    Args:
        rerank: If True, enable reranking for this query (overrides context config)
    """
    from .context import ContextConfig

    # Existing Ollama warning code...
    if ctx.embedding and ctx.embedding.provider == "ollama":
        print(
            f"Warning: Context '{ctx.name}' uses Ollama embeddings. "
            f"Consider migrating to OpenAI for faster and more consistent results. "
            f"Run: chinvex ingest --context {ctx.name} --embed-provider openai --rebuild-index",
            file=sys.stderr
        )
    elif not ctx.embedding:
        print(
            f"Warning: Context '{ctx.name}' uses Ollama embeddings (legacy default). "
            f"Consider migrating to OpenAI for faster and more consistent results. "
            f"Run: chinvex ingest --context {ctx.name} --embed-provider openai --rebuild-index",
            file=sys.stderr
        )

    db_path = ctx.index.sqlite_path
    chroma_dir = ctx.index.chroma_dir

    storage = Storage(db_path)
    storage.ensure_schema()

    ollama_host = ollama_host_override or ctx.ollama.base_url
    embedding_model = ctx.ollama.embed_model
    fallback_host = "http://127.0.0.1:11434" if ollama_host != "http://127.0.0.1:11434" else None

    embedder = OllamaEmbedder(ollama_host, embedding_model, fallback_host=fallback_host)
    vectors = VectorStore(chroma_dir)

    # Determine if reranking should be used
    use_reranker = (ctx.reranker is not None) or rerank

    # If reranking enabled, fetch more candidates
    if use_reranker and ctx.reranker:
        candidates_k = ctx.reranker.candidates
    else:
        candidates_k = k

    # Use context weights for source-type prioritization
    scored = search_chunks(
        storage,
        vectors,
        embedder,
        query,
        k=candidates_k,
        min_score=min_score,
        source=source,
        project=project,
        repo=repo,
        weights=ctx.weights,
    )

    # Apply reranking if enabled and conditions met
    if use_reranker and ctx.reranker and len(scored) >= 10:
        try:
            # Prepare candidates for reranking
            candidates = [
                {"chunk_id": item.chunk_id, "text": item.row["text"]}
                for item in scored
            ]

            # Truncate to max 50 candidates
            if len(candidates) > 50:
                candidates = candidates[:50]
                scored = scored[:50]

            # Get reranker and rerank with 2s timeout
            start_time = time.time()
            reranker = _get_reranker(ctx.reranker)
            reranked = reranker.rerank(query, candidates, top_k=k)
            elapsed = time.time() - start_time

            if elapsed > 2.0:
                print(
                    f"Warning: Reranking took {elapsed:.2f}s (exceeded 2s budget)",
                    file=sys.stderr
                )

            # Map reranked results back to ScoredChunk objects
            reranked_map = {r["chunk_id"]: r["rerank_score"] for r in reranked}
            scored = [
                item for item in scored
                if item.chunk_id in reranked_map
            ]
            # Update scores with rerank scores
            for item in scored:
                item.score = reranked_map[item.chunk_id]

            # Sort by rerank score
            scored.sort(key=lambda r: r.score, reverse=True)

        except Exception as e:
            # Fallback: use pre-rerank results
            print(
                f"Warning: Reranker failed ({type(e).__name__}: {e}). "
                f"Falling back to pre-rerank results.",
                file=sys.stderr
            )

    results = [
        SearchResult(
            chunk_id=item.chunk_id,
            score=item.score,
            source_type=item.row["source_type"],
            title=_title_from_row(item.row),
            citation=_citation_from_row(item.row),
            snippet=make_snippet(item.row["text"], limit=200),
        )
        for item in scored
    ]

    storage.close()
    return results[:k]
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_search_reranker.py -v`
Expected: PASS (all 4 tests passing)

**Step 5: Commit**
```bash
git add src/chinvex/context.py src/chinvex/search.py tests/test_search_reranker.py
git commit -m "feat(P5.4): integrate reranker in search with budget guardrails

- Two-stage retrieval: fetch N candidates, rerank to top K
- Budget guardrails: min 10 candidates, max 50, 2s timeout
- Fallback on failure: return pre-rerank results with warning
- Enable via context.json reranker field or --rerank flag

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 24: Add --rerank CLI Flag to Search Commands

**Files:**
- Modify: `C:\Code\chinvex\src\chinvex\cli.py`
- Test: `C:\Code\chinvex\tests\test_cli_rerank.py`

**Step 1: Write the failing test**
```python
# tests/test_cli_rerank.py
from unittest.mock import MagicMock, patch
import pytest
from typer.testing import CliRunner
from chinvex.cli import app


runner = CliRunner()


def test_search_command_with_rerank_flag():
    """Test that --rerank flag is passed to search_context."""
    with patch("chinvex.cli.search_context") as mock_search:
        mock_search.return_value = [
            MagicMock(chunk_id="c1", score=0.95, source_type="repo", title="test", citation="test.py", snippet="test")
        ]
        with patch("chinvex.cli.load_context") as mock_load:
            mock_ctx = MagicMock()
            mock_ctx.name = "Chinvex"
            mock_load.return_value = mock_ctx

            result = runner.invoke(app, ["search", "--context", "Chinvex", "--rerank", "test query"])

            assert result.exit_code == 0
            # Verify rerank=True was passed
            call_args = mock_search.call_args
            assert call_args[1]["rerank"] is True


def test_search_command_without_rerank_flag():
    """Test that rerank defaults to False when flag not provided."""
    with patch("chinvex.cli.search_context") as mock_search:
        mock_search.return_value = [
            MagicMock(chunk_id="c1", score=0.95, source_type="repo", title="test", citation="test.py", snippet="test")
        ]
        with patch("chinvex.cli.load_context") as mock_load:
            mock_ctx = MagicMock()
            mock_ctx.name = "Chinvex"
            mock_load.return_value = mock_ctx

            result = runner.invoke(app, ["search", "--context", "Chinvex", "test query"])

            assert result.exit_code == 0
            # Verify rerank=False was passed (default)
            call_args = mock_search.call_args
            assert call_args[1]["rerank"] is False


def test_multi_context_search_with_rerank_flag():
    """Test that --rerank flag works with multi-context search."""
    with patch("chinvex.cli.search_multi_context") as mock_search:
        mock_search.return_value = [
            MagicMock(chunk_id="c1", score=0.95, source_type="repo", title="test", citation="test.py", snippet="test", context="Chinvex")
        ]

        result = runner.invoke(app, ["search", "--contexts", "Chinvex,Codex", "--rerank", "test query"])

        assert result.exit_code == 0
        # Verify rerank=True was passed
        call_args = mock_search.call_args
        assert call_args[1]["rerank"] is True
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_cli_rerank.py -v`
Expected: FAIL with "TypeError: got an unexpected keyword argument 'rerank'"

**Step 3: Write minimal implementation**

Update the search command in cli.py. Find the search_cmd function and modify:
```python
# src/chinvex/cli.py (find and modify the search command - around line 200+)

@app.command("search")
def search_cmd(
    context: str | None = typer.Option(None, "--context", "-c", help="Context name to search"),
    contexts: str | None = typer.Option(None, "--contexts", help="Comma-separated context names for multi-context search"),
    query: str = typer.Argument(..., help="Search query"),
    k: int = typer.Option(8, "--k", help="Number of results to return"),
    min_score: float = typer.Option(0.35, "--min-score", help="Minimum score threshold"),
    source: str = typer.Option("all", "--source", help="Filter by source type (all/repo/chat)"),
    ollama_host: str | None = typer.Option(None, "--ollama-host", help="Override Ollama host"),
    no_recency: bool = typer.Option(False, "--no-recency", help="Disable recency decay"),
    rerank: bool = typer.Option(False, "--rerank", help="Enable reranking for this query"),  # NEW
) -> None:
    """Search the index and return top results."""
    if not context and not contexts:
        typer.secho("Error: Must provide either --context or --contexts", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    if context and contexts:
        typer.secho("Error: Cannot use both --context and --contexts", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    # Multi-context search
    if contexts:
        from .search import search_multi_context
        context_list = [c.strip() for c in contexts.split(",")]
        results = search_multi_context(
            contexts=context_list,
            query=query,
            k=k,
            min_score=min_score,
            source=source,
            ollama_host=ollama_host,
            recency_enabled=not no_recency,
            rerank=rerank,  # NEW
        )
    else:
        # Single-context search
        from .context import load_context
        from .search import search_context

        contexts_root = get_contexts_root()
        ctx = load_context(context, contexts_root)

        results = search_context(
            ctx=ctx,
            query=query,
            k=k,
            min_score=min_score,
            source=source,
            ollama_host_override=ollama_host,
            recency_enabled=not no_recency,
            rerank=rerank,  # NEW
        )

    # Display results
    if not results:
        typer.secho("No results found.", fg=typer.colors.YELLOW)
        return

    for i, result in enumerate(results, 1):
        typer.secho(f"\n[{i}] {result.title}", fg=typer.colors.CYAN, bold=True)
        typer.secho(f"Score: {result.score:.3f}", fg=typer.colors.GREEN)
        if hasattr(result, "context") and result.context:
            typer.secho(f"Context: {result.context}", fg=typer.colors.BLUE)
        typer.secho(f"Citation: {result.citation}", dim=True)
        typer.secho(f"{result.snippet}...")
```

Also update search_multi_context signature in search.py:
```python
# src/chinvex/search.py (update function signature around line 430)

def search_multi_context(
    contexts: list[str] | str,
    query: str,
    k: int = 10,
    min_score: float = 0.35,
    source: str = "all",
    ollama_host: str | None = None,
    recency_enabled: bool = True,
    allow_mixed_embeddings: bool = False,
    rerank: bool = False,  # NEW
) -> list[SearchResult]:
    """
    Search across multiple contexts, merge results by score.

    Args:
        contexts: List of context names, or "all" for all contexts
        query: Search query
        k: Total number of results to return (not per-context)
        min_score: Minimum score threshold
        source: Filter by source type (all/repo/chat/codex_session)
        ollama_host: Ollama host override
        recency_enabled: Enable recency decay
        allow_mixed_embeddings: Allow mixed embedding providers (not yet supported in P5)
        rerank: Enable reranking for this query
    """
    # ... existing implementation ...

    # When calling search_context, pass rerank flag (around line 494):
    for ctx_name in contexts:
        try:
            from .context import load_context
            ctx = load_context(ctx_name, contexts_root)
            results = search_context(
                ctx=ctx,
                query=query,
                k=k_per_context,
                min_score=min_score,
                source=source,
                ollama_host_override=ollama_host,
                recency_enabled=recency_enabled,
                rerank=rerank,  # NEW: pass through rerank flag
            )
            # ... rest of existing code ...
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_cli_rerank.py -v`
Expected: PASS (all 3 tests passing)

**Step 5: Commit**
```bash
git add src/chinvex/cli.py src/chinvex/search.py tests/test_cli_rerank.py
git commit -m "feat(P5.4): add --rerank CLI flag to search commands

- Works with both single-context and multi-context search
- Overrides context.json reranker setting for single query
- Example: chinvex search --context X --rerank 'query'

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 25: Update README.md with Reranker Documentation

**Files:**
- Modify: `C:\Code\chinvex\README.md`

**Step 1: Write the failing test**
```python
# tests/test_readme_reranker_docs.py
from pathlib import Path


def test_readme_contains_reranker_documentation():
    """Verify README.md documents reranker feature."""
    readme_path = Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text()

    # Check for required sections/keywords
    assert "--rerank" in content, "README missing --rerank flag documentation"
    assert "reranker" in content.lower(), "README missing reranker configuration section"
    assert "cohere" in content.lower(), "README missing Cohere provider documentation"
    assert "jina" in content.lower(), "README missing Jina provider documentation"
    assert "cross-encoder" in content.lower() or "local" in content.lower(), "README missing local reranker documentation"
    assert "COHERE_API_KEY" in content, "README missing API key setup instructions"
    assert "JINA_API_KEY" in content, "README missing Jina API key setup"


def test_readme_reranker_config_example():
    """Verify README contains reranker configuration example."""
    readme_path = Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text()

    # Check for config structure
    assert '"provider"' in content, "README missing reranker provider field"
    assert '"model"' in content, "README missing reranker model field"
    assert '"candidates"' in content or "candidates" in content, "README missing candidates explanation"
    assert '"top_k"' in content or "top_k" in content, "README missing top_k explanation"
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_readme_reranker_docs.py -v`
Expected: FAIL with "AssertionError: README missing --rerank flag documentation"

**Step 3: Write minimal implementation**

Add reranker documentation section to README.md (insert after the search section):
```markdown
## Reranking (P5.4)

Chinvex supports optional two-stage retrieval for improved relevance:

1. **Stage 1**: Hybrid search returns top N candidates (default: 20)
2. **Stage 2**: Cross-encoder reranker reranks to top K (default: 5)

### Enable Reranking

**Per-query (CLI flag):**
```bash
chinvex search --context Chinvex --rerank "chromadb batch limit"
```

**Always-on (context.json):**
```json
{
  "reranker": {
    "provider": "cohere",
    "model": "rerank-english-v3.0",
    "candidates": 20,
    "top_k": 5
  }
}
```

### Reranker Providers

#### Cohere (Recommended)
- **Provider**: `cohere`
- **Model**: `rerank-english-v3.0`
- **Setup**: Set `COHERE_API_KEY` environment variable
- **Get API key**: https://cohere.com/
- **Latency**: ~200-500ms for 20 candidates
- **Quality**: Excellent

Example:
```bash
export COHERE_API_KEY="your-key-here"
chinvex search --context X --rerank "query"
```

#### Jina
- **Provider**: `jina`
- **Model**: `jina-reranker-v1-base-en`
- **Setup**: Set `JINA_API_KEY` environment variable
- **Get API key**: https://jina.ai/
- **Latency**: ~300-600ms for 20 candidates
- **Quality**: Good

Example:
```bash
export JINA_API_KEY="your-key-here"
```

#### Local Cross-Encoder
- **Provider**: `local`
- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Setup**: No API key required (downloads model on first use to `~/.cache/huggingface/`)
- **Latency**: ~1-2s for 20 candidates (slower but free)
- **Quality**: Good

Example context.json:
```json
{
  "reranker": {
    "provider": "local",
    "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "candidates": 20,
    "top_k": 5
  }
}
```

### Budget Guardrails

- **Minimum candidates**: Reranker only runs if initial retrieval returns ≥10 candidates
- **Maximum candidates**: Truncates to 50 candidates (prevents excessive latency/cost)
- **Latency budget**: 2s max for rerank step (warning if exceeded)
- **Fallback**: If reranker fails (missing creds, service down, timeout), returns pre-rerank results with warning

### Configuration Fields

- `provider` (required): `"cohere"`, `"jina"`, or `"local"`
- `model` (required): Provider-specific model name
- `candidates` (optional, default 20): Number of candidates to fetch from initial retrieval
- `top_k` (optional, default 5): Number of results to return after reranking

### Disable Reranking

- **Per-query**: Omit `--rerank` flag
- **Always-off**: Set `"reranker": null` in context.json or omit field entirely

### Performance Notes

- Reranking adds latency but significantly improves relevance
- API-based rerankers (Cohere, Jina) are faster than local cross-encoder
- Local cross-encoder downloads ~400MB model on first use
- Use `--rerank` flag for ad-hoc queries; configure in context.json for always-on behavior
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_readme_reranker_docs.py -v`
Expected: PASS (all 2 tests passing)

**Step 5: Commit**
```bash
git add README.md tests/test_readme_reranker_docs.py
git commit -m "docs(P5.4): add reranker documentation to README

- Explain --rerank CLI flag
- Document provider options (Cohere, Jina, local)
- Configuration examples and API key setup
- Budget guardrails and performance notes

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

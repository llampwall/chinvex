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

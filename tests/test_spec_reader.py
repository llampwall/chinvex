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
    src_dir = repo / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "code.py").write_text("# code")

    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True)

    # Modify spec and code
    (specs_dir / "P1_SPEC.md").write_text("# P1 Spec Updated")
    (src_dir / "code.py").write_text("# code updated")

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

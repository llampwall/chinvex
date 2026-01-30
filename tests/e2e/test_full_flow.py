"""End-to-end test of full auto-ingest flow."""
import pytest
import time
import subprocess
import json
import os
from pathlib import Path


@pytest.mark.slow
def test_full_auto_ingest_flow(tmp_path: Path):
    """
    E2E: Edit file → ingest → STATUS.json updated → GLOBAL_STATUS reflects change

    This test validates STATUS.json updates and GLOBAL_STATUS.md generation.
    Note: Auto-watch file detection is tested in Task 26 kill-switch test.
    """
    # Setup
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"
    repo_root = tmp_path / "test_repo"
    state_dir = tmp_path / "state"

    contexts_root.mkdir()
    indexes_root.mkdir()
    repo_root.mkdir()
    state_dir.mkdir()

    # Set environment variables for this test
    env = os.environ.copy()
    env["CHINVEX_CONTEXTS_ROOT"] = str(contexts_root)
    env["CHINVEX_INDEXES_ROOT"] = str(indexes_root)
    env["CHINVEX_STATE_DIR"] = str(state_dir)

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=repo_root, check=True, env=env)
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_root,
        check=True,
        env=env
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_root,
        check=True,
        env=env
    )

    # Create initial file and commit
    test_file = repo_root / "main.py"
    initial_content = '''"""Test module."""


def function_one():
    """First function."""
    return 1
'''
    test_file.write_text(initial_content)
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True, env=env)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_root,
        check=True,
        env=env
    )

    # Create context for this repo
    result = subprocess.run(
        [
            "chinvex", "ingest",
            "--context", "test_repo",
            "--repo", str(repo_root)
        ],
        capture_output=True,
        text=True,
        env=env
    )
    assert result.returncode == 0, f"Initial ingest failed: {result.stderr}\nStdout: {result.stdout}"

    # Start watcher
    result = subprocess.run(
        [
            "chinvex", "sync", "start"
        ],
        capture_output=True,
        text=True,
        env=env
    )
    assert result.returncode == 0, f"Watcher start failed: {result.stderr}"

    try:
        # Get initial STATUS.json state
        status_json = contexts_root / "test_repo" / "STATUS.json"
        assert status_json.exists(), "STATUS.json should exist after initial ingest"

        initial_status = json.loads(status_json.read_text())
        initial_chunks = initial_status["chunks"]

        # Make a change to the file - add more functions
        updated_content = '''"""Test module."""


def function_one():
    """First function."""
    return 1


def function_two():
    """Second function."""
    return 2


def function_three():
    """Third function."""
    return 3


class TestClass:
    """Test class with methods."""

    def method_one(self):
        """First method."""
        return "one"

    def method_two(self):
        """Second method."""
        return "two"
'''
        test_file.write_text(updated_content)
        subprocess.run(["git", "add", "."], cwd=repo_root, check=True, env=env)
        subprocess.run(
            ["git", "commit", "-m", "Add more functions and class"],
            cwd=repo_root,
            check=True,
            env=env
        )

        # Manually trigger ingest to verify STATUS.json update flow
        # (Full auto-watch with file changes tested in Task 26)
        result = subprocess.run(
            [
                "chinvex", "ingest",
                "--context", "test_repo"
            ],
            capture_output=True,
            text=True,
            env=env
        )
        assert result.returncode == 0, f"Second ingest failed: {result.stderr}"

        # Verify STATUS.json was updated (last_sync should change)
        final_status = json.loads(status_json.read_text())
        assert final_status["last_sync"] != initial_status["last_sync"], "STATUS.json should be updated after re-ingest"
        assert final_status["chunks"] >= initial_chunks, "Chunks should not decrease"

        # Generate GLOBAL_STATUS.md
        result = subprocess.run(
            ["chinvex", "status", "--regenerate"],
            capture_output=True,
            text=True,
            env=env
        )
        assert result.returncode == 0, f"Status regeneration failed: {result.stderr}"

        # Verify GLOBAL_STATUS.md exists and reflects the change
        global_status_md = contexts_root / "GLOBAL_STATUS.md"
        assert global_status_md.exists(), "GLOBAL_STATUS.md should exist after regeneration"

        global_content = global_status_md.read_text()
        assert "test_repo" in global_content, "GLOBAL_STATUS.md should mention test_repo"
        assert str(final_status["chunks"]) in global_content, "GLOBAL_STATUS.md should show updated chunk count"

    finally:
        # Cleanup: stop watcher
        subprocess.run(
            ["chinvex", "sync", "stop"],
            capture_output=True,
            env=env
        )

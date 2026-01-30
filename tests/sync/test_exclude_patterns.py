# tests/sync/test_exclude_patterns.py
import pytest
from pathlib import Path
from chinvex.sync.patterns import should_exclude


def test_exclude_exact_filename():
    """STATUS.json files should be excluded"""
    assert should_exclude("contexts/Chinvex/STATUS.json", watch_root=Path("/root"))
    assert should_exclude("STATUS.json", watch_root=Path("/root"))


def test_exclude_recursive_pattern():
    """**/.git/** should exclude all .git subdirs"""
    assert should_exclude("repo/.git/HEAD", watch_root=Path("/root"))
    assert should_exclude("repo/sub/.git/config", watch_root=Path("/root"))
    assert should_exclude("repo/.git/objects/ab/cd1234", watch_root=Path("/root"))


def test_exclude_wildcard_pattern():
    """**/*_BRIEF.md should exclude all _BRIEF.md files"""
    assert should_exclude("MORNING_BRIEF.md", watch_root=Path("/root"))
    assert should_exclude("subdir/SESSION_BRIEF.md", watch_root=Path("/root"))
    assert should_exclude("deep/nested/PROJECT_BRIEF.md", watch_root=Path("/root"))


def test_dont_exclude_normal_files():
    """Normal source files should not be excluded"""
    assert not should_exclude("src/main.py", watch_root=Path("/root"))
    assert not should_exclude("README.md", watch_root=Path("/root"))
    assert not should_exclude("test/test_file.py", watch_root=Path("/root"))


def test_exclude_case_insensitive_windows():
    """Pattern matching should be case-insensitive on Windows"""
    import platform
    if platform.system() == "Windows":
        assert should_exclude("contexts/chinvex/status.json", watch_root=Path("/root"))
        assert should_exclude("REPO/.GIT/config", watch_root=Path("/root"))


def test_exclude_home_chinvex_internals():
    """~/.chinvex internals should be excluded"""
    import os
    home = Path.home()
    assert should_exclude(str(home / ".chinvex/sync.log"), watch_root=home)
    assert should_exclude(str(home / ".chinvex/sync.pid"), watch_root=home)
    assert should_exclude(str(home / ".chinvex/sync_heartbeat.json"), watch_root=home)

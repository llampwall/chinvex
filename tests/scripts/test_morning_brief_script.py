# tests/scripts/test_morning_brief.py
import pytest
import subprocess
from pathlib import Path


def test_morning_brief_script_exists():
    """Morning brief script should exist"""
    script_path = Path("scripts/morning_brief.ps1")
    assert script_path.exists()


def test_morning_brief_syntax_valid():
    """PowerShell script should have valid syntax"""
    script_path = Path("scripts/morning_brief.ps1")

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(script_path), "-WhatIf"],
        capture_output=True,
        text=True
    )

    assert "ParserError" not in result.stderr


def test_morning_brief_requires_contexts_root():
    """Script should require ContextsRoot parameter"""
    script_path = Path("scripts/morning_brief.ps1")

    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(script_path)],
        capture_output=True,
        text=True
    )

    assert result.returncode != 0
    assert "ContextsRoot" in result.stderr or "parameter" in result.stderr.lower()

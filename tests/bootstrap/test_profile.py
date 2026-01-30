# tests/bootstrap/test_profile.py
import pytest
from pathlib import Path
from chinvex.bootstrap.profile import inject_dual_function, remove_dual_function, DUAL_FUNCTION_TEMPLATE


def test_inject_dual_function_creates_profile(tmp_path: Path):
    """Should create PowerShell profile if it doesn't exist"""
    profile_path = tmp_path / "Microsoft.PowerShell_profile.ps1"

    inject_dual_function(profile_path)

    assert profile_path.exists()
    content = profile_path.read_text()
    assert "function dual {" in content
    assert '"track"' in content
    assert '"brief"' in content
    assert "Set-Alias dn dual" in content


def test_inject_dual_function_appends_to_existing(tmp_path: Path):
    """Should append to existing profile without duplicating"""
    profile_path = tmp_path / "Microsoft.PowerShell_profile.ps1"
    profile_path.write_text("# Existing content\nSet-Alias ll ls\n")

    inject_dual_function(profile_path)

    content = profile_path.read_text()
    assert "# Existing content" in content
    assert "function dual {" in content


def test_inject_dual_function_idempotent(tmp_path: Path):
    """Should not duplicate if dual function already exists"""
    profile_path = tmp_path / "Microsoft.PowerShell_profile.ps1"

    inject_dual_function(profile_path)
    first_content = profile_path.read_text()

    inject_dual_function(profile_path)
    second_content = profile_path.read_text()

    assert first_content == second_content
    assert first_content.count("function dual {") == 1


def test_remove_dual_function(tmp_path: Path):
    """Should remove dual function from profile"""
    profile_path = tmp_path / "Microsoft.PowerShell_profile.ps1"
    profile_path.write_text("""# Existing content
Set-Alias ll ls

# Chinvex dual function - DO NOT EDIT
function dual {
    # ... function body
}
# End Chinvex dual function

# More content
""")

    remove_dual_function(profile_path)

    content = profile_path.read_text()
    assert "function dual {" not in content
    assert "# Existing content" in content
    assert "# More content" in content

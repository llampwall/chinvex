"""PowerShell profile injection."""
from __future__ import annotations

import re
from pathlib import Path

DUAL_FUNCTION_TEMPLATE = """
# Chinvex dual function - DO NOT EDIT
function dual {
    param([string]$cmd, [string]$arg)
    switch ($cmd) {
        "brief"  { chinvex brief --all-contexts }
        "track"  {
            $repo = if ($arg) { Resolve-Path $arg } else { Get-Location }
            $name = (Split-Path $repo -Leaf).ToLower() -replace '[^a-z0-9-]', '-'

            # Check if context exists
            $existing = chinvex context list --format json | ConvertFrom-Json | Where-Object { $_.name -eq $name }

            if ($existing) {
                # Check if same repo
                $existingRepo = $existing.sources | Where-Object { $_.type -eq "repo" } | Select-Object -First 1
                if ($existingRepo -and (Resolve-Path $existingRepo.path) -eq $repo) {
                    Write-Host "Already tracking $repo in context '$name'"
                    return
                }
                # Different repo, need unique name
                $i = 2
                while ($true) {
                    $newName = "$name-$i"
                    $check = chinvex context list --format json | ConvertFrom-Json | Where-Object { $_.name -eq $newName }
                    if (-not $check) { $name = $newName; break }
                    $i++
                }
            }

            # Create context and add source
            chinvex ingest --context $name --repo $repo

            # Add to sync watcher if running
            chinvex sync reconcile-sources 2>$null

            Write-Host "Tracking $repo in context '$name'"
        }
        "status" { chinvex status }
        default  { Write-Host "Usage: dual [brief|track|status]" }
    }
}

Set-Alias dn dual
# End Chinvex dual function
""".strip()

START_MARKER = "# Chinvex dual function - DO NOT EDIT"
END_MARKER = "# End Chinvex dual function"


def inject_dual_function(profile_path: Path) -> None:
    """
    Inject dual function into PowerShell profile.

    Idempotent: won't duplicate if already exists.

    Args:
        profile_path: Path to PowerShell profile
    """
    # Read existing content
    if profile_path.exists():
        content = profile_path.read_text(encoding="utf-8")
    else:
        content = ""
        profile_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if dual function already exists
    if START_MARKER in content:
        return  # Already injected

    # Append dual function
    if content and not content.endswith("\n"):
        content += "\n"

    content += "\n" + DUAL_FUNCTION_TEMPLATE + "\n"

    profile_path.write_text(content, encoding="utf-8")


def remove_dual_function(profile_path: Path) -> None:
    """
    Remove dual function from PowerShell profile.

    Args:
        profile_path: Path to PowerShell profile
    """
    if not profile_path.exists():
        return

    content = profile_path.read_text(encoding="utf-8")

    # Remove section between markers
    pattern = re.compile(
        rf"^{re.escape(START_MARKER)}.*?^{re.escape(END_MARKER)}\n?",
        re.MULTILINE | re.DOTALL
    )
    content = pattern.sub("", content)

    profile_path.write_text(content, encoding="utf-8")

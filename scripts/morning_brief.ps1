<#
.SYNOPSIS
    Generate and send morning brief with context status

.DESCRIPTION
    Uses chinvex.morning_brief Python module to generate brief with
    active project objectives and send ntfy push.

.PARAMETER ContextsRoot
    Path to contexts root directory

.PARAMETER NtfyTopic
    ntfy.sh topic for morning brief

.PARAMETER NtfyServer
    ntfy server URL (default: https://ntfy.sh)

.PARAMETER OutputPath
    Path to write MORNING_BRIEF.md (default: contexts root parent)

.EXAMPLE
    .\morning_brief.ps1 -ContextsRoot "P:\ai_memory\contexts" -NtfyTopic "morning-brief"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$ContextsRoot,

    [Parameter(Mandatory=$false)]
    [string]$NtfyTopic = "",

    [Parameter(Mandatory=$false)]
    [string]$NtfyServer = "https://ntfy.sh",

    [Parameter(Mandatory=$false)]
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"

function Invoke-Hidden {
    param(
        [string]$Command,
        [string[]]$Arguments
    )
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Command

    # Properly escape and join arguments
    $escapedArgs = $Arguments | ForEach-Object {
        if ($_ -match '\s|"') {
            # Quote arguments that contain spaces or quotes
            '"{0}"' -f ($_ -replace '"', '\"')
        } else {
            $_
        }
    }
    $psi.Arguments = $escapedArgs -join " "

    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi
    $process.Start() | Out-Null
    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit()

    return @{
        ExitCode = $process.ExitCode
        Output = $stdout
        Error = $stderr
    }
}

# Default output path
if (-not $OutputPath) {
    $OutputPath = Join-Path (Split-Path $ContextsRoot -Parent) "MORNING_BRIEF.md"
}

# Call Python module to generate brief
$pythonCode = @"
import sys
from pathlib import Path
from chinvex.morning_brief import generate_morning_brief

contexts_root = Path(r'$ContextsRoot')
output_path = Path(r'$OutputPath')

brief_text, ntfy_body = generate_morning_brief(contexts_root, output_path)

# Print ntfy body to stdout for PowerShell to capture
print(ntfy_body, end='')
"@

try {
    # Run Python code and capture ntfy body using hidden process
    $result = Invoke-Hidden -Command "python" -Arguments @("-c", $pythonCode)

    if ($result.ExitCode -ne 0) {
        throw "Python script failed: $($result.Error)"
    }

    $ntfyBody = $result.Output

    Write-Host "Generated morning brief at $OutputPath"

    # Send ntfy push if topic is configured
    if ($NtfyTopic) {
        $title = "Morning Brief"

        $url = "$NtfyServer/$NtfyTopic"
        $headers = @{
            "Title" = $title
            "Tags" = "sunrise,calendar"
        }

        Invoke-RestMethod -Uri $url -Method Post -Body $ntfyBody -Headers $headers | Out-Null
        Write-Host "Sent morning brief push to $NtfyTopic"
    }
} catch {
    Write-Error "Failed to generate morning brief: $_"
    exit 1
}

<#
.SYNOPSIS
    Test the REAL fix - VBScript launcher for scheduled task

.DESCRIPTION
    This test will:
    1. Show you the current task configuration
    2. Manually trigger the scheduled task
    3. Wait for you to watch for popups
    4. Check the results

    The fix uses a VBScript launcher that completely hides all windows
    by using WScript.Shell with window style 0.
#>

Write-Host @"

╔══════════════════════════════════════════════════════════════╗
║             SCHEDULED TASK FIX - FINAL TEST                 ║
╔══════════════════════════════════════════════════════════════╗

The previous fix addressed Python subprocess calls, but Task Scheduler
itself was still flashing windows when launching PowerShell.

THE REAL FIX:
- Uses a VBScript launcher (scheduled_sweep_launcher.vbs)
- VBScript launches PowerShell with window style 0 (completely hidden)
- This prevents ANY window from appearing when Task Scheduler runs

"@ -ForegroundColor Cyan

# Show current task configuration
Write-Host "`nCurrent task configuration:" -ForegroundColor Yellow
Get-ScheduledTask -TaskName "ChinvexSweep" | Select-Object -ExpandProperty Actions | Format-List

Write-Host "`nThe task now uses:" -ForegroundColor Green
Write-Host "  - wscript.exe (Windows Script Host)" -ForegroundColor Green
Write-Host "  - scheduled_sweep_launcher.vbs" -ForegroundColor Green
Write-Host "  - //B //Nologo flags (batch mode, no logo)" -ForegroundColor Green

Read-Host "`nPress ENTER to trigger the scheduled task and watch for popups"

Write-Host "`n╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Red
Write-Host "║          >>> WATCH YOUR SCREEN CAREFULLY <<<                ║" -ForegroundColor Red
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Red

Start-Sleep -Seconds 2

# Trigger the task
Write-Host "`nTriggering ChinvexSweep task..." -ForegroundColor Yellow
schtasks /Run /TN "ChinvexSweep"

Write-Host "Task triggered! Waiting 10 seconds for it to complete..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check results
Write-Host "`nChecking task results..." -ForegroundColor Yellow
$taskInfo = schtasks /Query /TN "ChinvexSweep" /FO LIST /V | Select-String "Last Run Time|Last Result"
Write-Host $taskInfo -ForegroundColor Green

$watcherStatus = chinvex sync status
Write-Host "`nWatcher status:" -ForegroundColor Yellow
Write-Host $watcherStatus -ForegroundColor Green

Write-Host @"

╔══════════════════════════════════════════════════════════════╗
║                        RESULTS                               ║
╔══════════════════════════════════════════════════════════════╗

"@ -ForegroundColor Cyan

$answer = Read-Host "Did you see ANY console window flash or popup? (yes/no)"

if ($answer -match "^n") {
    Write-Host @"

╔══════════════════════════════════════════════════════════════╗
║                    ✓✓✓ SUCCESS! ✓✓✓                        ║
╔══════════════════════════════════════════════════════════════╗

The fix is working! Your scheduled sweep will now run completely
silently every 30 minutes without any window popups or focus stealing.

Technical details:
- VBScript launcher prevents Task Scheduler window flashing
- Python subprocess calls use CREATE_NO_WINDOW flag
- PowerShell runs with -WindowStyle Hidden -NonInteractive
- Triple-layer window hiding ensures complete silence

"@ -ForegroundColor Green
} else {
    Write-Host @"

╔══════════════════════════════════════════════════════════════╗
║                      NEEDS ATTENTION                         ║
╔══════════════════════════════════════════════════════════════╗

Please describe what you saw:
- Was it a black console window?
- A PowerShell blue window?
- An error dialog?
- How long did it appear?

"@ -ForegroundColor Red

    $description = Read-Host "What did you see"
    Write-Host "`nNoted: $description" -ForegroundColor Yellow
}

Write-Host "`nNext scheduled run:" -ForegroundColor Cyan
schtasks /Query /TN "ChinvexSweep" /FO LIST /V | Select-String "Next Run Time"

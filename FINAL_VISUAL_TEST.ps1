<#
.SYNOPSIS
    Final visual test for window popup fix

.DESCRIPTION
    This script will trigger the same operations that the scheduled sweep does,
    allowing you to visually confirm that NO console windows pop up.

    Watch your screen carefully during the "WATCH NOW" sections!
#>

Write-Host @"

╔══════════════════════════════════════════════════════════════╗
║                 WINDOW POPUP FIX - VISUAL TEST              ║
╔══════════════════════════════════════════════════════════════╗

This test will simulate what the scheduled sweep does every 30 minutes.

INSTRUCTIONS:
1. Keep your eyes on the screen during "WATCH NOW" sections
2. If you see ANY console window flash/popup, the fix didn't work
3. If you see NOTHING popup, the fix is working correctly!

"@ -ForegroundColor Cyan

Read-Host "Press ENTER to begin the test"

# Test 1: Stop the watcher (triggers taskkill at line 87)
Write-Host "`n[TEST 1] Stopping watcher daemon..." -ForegroundColor Yellow
Write-Host ">>> WATCH NOW FOR WINDOW POPUPS <<<" -ForegroundColor Red -BackgroundColor White
Start-Sleep -Seconds 2

$startTime = Get-Date
chinvex sync stop 2>&1 | Out-Null
$elapsed = ((Get-Date) - $startTime).TotalMilliseconds

Write-Host "✓ Stop completed in $($elapsed)ms - Did you see a window popup? (You shouldn't have!)" -ForegroundColor Green
Start-Sleep -Seconds 2

# Test 2: Start the watcher
Write-Host "`n[TEST 2] Starting watcher daemon..." -ForegroundColor Yellow
chinvex sync start 2>&1 | Out-Null
Start-Sleep -Seconds 2
Write-Host "✓ Start completed" -ForegroundColor Green

# Test 3: Reconcile sources (triggers stop/start internally, uses taskkill at line 210)
Write-Host "`n[TEST 3] Reconciling sources (restarts watcher)..." -ForegroundColor Yellow
Write-Host ">>> WATCH NOW FOR WINDOW POPUPS <<<" -ForegroundColor Red -BackgroundColor White
Start-Sleep -Seconds 2

$startTime = Get-Date
chinvex sync reconcile-sources 2>&1 | Out-Null
$elapsed = ((Get-Date) - $startTime).TotalMilliseconds

Write-Host "✓ Reconcile completed in $($elapsed)ms - Did you see a window popup? (You shouldn't have!)" -ForegroundColor Green
Start-Sleep -Seconds 2

# Test 4: Run the actual scheduled sweep script
Write-Host "`n[TEST 4] Running the ACTUAL scheduled sweep script..." -ForegroundColor Yellow
Write-Host ">>> WATCH CAREFULLY - This is the real thing! <<<" -ForegroundColor Red -BackgroundColor White
Start-Sleep -Seconds 3

$startTime = Get-Date
& "$PSScriptRoot\scripts\scheduled_sweep.ps1" -ContextsRoot "P:\ai_memory\contexts" -NtfyTopic "dual-nature" 2>&1 | Out-Null
$elapsed = ((Get-Date) - $startTime).TotalSeconds

Write-Host "✓ Sweep completed in $([math]::Round($elapsed, 1))s" -ForegroundColor Green

# Final status check
Write-Host "`n[TEST 5] Verifying watcher is still running..." -ForegroundColor Yellow
$status = chinvex sync status
Write-Host $status -ForegroundColor Green

# Results
Write-Host @"

╔══════════════════════════════════════════════════════════════╗
║                        TEST RESULTS                          ║
╔══════════════════════════════════════════════════════════════╗

Did you see ANY console windows popup during the tests above?

"@ -ForegroundColor Cyan

$answer = Read-Host "Did you see any popups? (yes/no)"

if ($answer -match "^n") {
    Write-Host @"

╔══════════════════════════════════════════════════════════════╗
║                    ✓ FIX VERIFIED ✓                         ║
╔══════════════════════════════════════════════════════════════╗

The window popup issue is FIXED!

Your scheduled sweep will now run silently in the background
every 30 minutes without stealing focus or showing windows.

"@ -ForegroundColor Green
} else {
    Write-Host @"

╔══════════════════════════════════════════════════════════════╗
║                    ✗ FIX FAILED ✗                           ║
╔══════════════════════════════════════════════════════════════╗

If you saw popups, please report:
1. Which test number showed the popup
2. What the popup looked like (console window? error dialog?)
3. Any error messages

"@ -ForegroundColor Red
}

Write-Host "`nNext scheduled sweep: " -NoNewline
schtasks /Query /TN "ChinvexSweep" /FO LIST /V | Select-String "Next Run Time"

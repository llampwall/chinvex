# Test script to verify window hiding works
# This simulates what the scheduled sweep does

Write-Host "Testing window hiding fix..." -ForegroundColor Cyan

# 1. Start the watcher if not running
Write-Host "`n1. Starting watcher..." -ForegroundColor Yellow
chinvex sync start 2>&1 | Out-Null
Start-Sleep -Seconds 1

# 2. Check status
Write-Host "2. Checking status..." -ForegroundColor Yellow
chinvex sync status

# 3. Test stop command (this should NOT show a window)
Write-Host "`n3. Testing stop command (should be silent)..." -ForegroundColor Yellow
Write-Host "   Watch carefully for any console windows popping up!" -ForegroundColor Red
Start-Sleep -Seconds 2

chinvex sync stop
Write-Host "   Stop command completed" -ForegroundColor Green
Start-Sleep -Seconds 1

# 4. Restart
Write-Host "`n4. Restarting watcher..." -ForegroundColor Yellow
chinvex sync start
Start-Sleep -Seconds 1

# 5. Test reconcile-sources (this also triggers stop/start internally)
Write-Host "`n5. Testing reconcile-sources (should be silent)..." -ForegroundColor Yellow
Write-Host "   Watch carefully for any console windows popping up!" -ForegroundColor Red
Start-Sleep -Seconds 2

chinvex sync reconcile-sources
Write-Host "   Reconcile command completed" -ForegroundColor Green

Write-Host "`n=== Test complete ===" -ForegroundColor Cyan
Write-Host "If you saw NO console windows flash during steps 3 and 5, the fix is working!" -ForegroundColor Green

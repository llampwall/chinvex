# Gap Fixes and Missing Tasks - Summary

## All Fixes and Additions COMPLETED ✓

### 1. Task 3 (Portalocker Integration) ✓
**Issue:** Verify OS-level locking with fail_when_locked
**Fix:** Already correct - implementation uses `portalocker.Lock(timeout=0, fail_when_locked=True)`

### 2. Task 4 (ChangeAccumulator) ✓
**Issue:** Missing 5-minute max debounce cap
**Fix Added:**
- Added `MAX_DEBOUNCE_SECONDS = 300` constant
- Track `_first_change_time` to monitor total accumulation time
- `is_ready()` now returns True if >5min elapsed, even if debounce not satisfied
- Added test `test_accumulator_max_debounce_cap()`

### 3. Task 7 (WatcherProcess) ✓
**Issue:** Heartbeat writes every 1s instead of every 30s
**Fix Added:**
- Added `heartbeat_counter` in main loop
- Writes heartbeat only when counter reaches 30 (every 30s)
- Resets counter after write

### 4. Task 7→9 Connection ✓
**Issue:** Clarify that Task 9 fills in _trigger_ingest TODOs
**Fix Added:**
- Updated TODO comments: `# TODO (Task 9): Call chinvex ingest...`
- Added comment in docstring: "Full implementation in Task 9"

### 5. Task 5 (Sync Commands) ✓
**Issue:** Missing `sync reconcile-sources` command
**Fix Added:**
- Added `sync_reconcile_sources_cmd()` function
- Restarts watcher to pick up new sources from contexts
- Called by `dual track` after adding new context
- Added CLI binding: `@sync_app.command("reconcile-sources")`

### 6. Task 17 (ntfy Push) ✓
**Issue:** Missing push dedup log
**Fix Added:**
- Added `should_send_stale_alert(context_name, log_file)` - checks dedup
- Added `log_push(context_name, push_type, log_file)` - writes to JSONL
- Added `send_stale_alert()` helper - PowerShell-callable
- Push log schema: `{timestamp, context, type, date}`
- Dedup key: `(context, type, date)` - max 1 stale alert per context per day
- Log file: `~/.chinvex/push_log.jsonl`
- Added tests for dedup logic

### 7. Task 15B (Login Trigger) ✓
**Issue:** Missing login-trigger scheduled task (primary watcher start mechanism)
**Fix Added:**
- New task "Task 15B: Login-Trigger Scheduled Task"
- Registers `ChinvexWatcherStart` task (runs at user login)
- Calls `chinvex sync ensure-running` on login
- PRIMARY mechanism (sweep is backup)
- Added `register_login_trigger_task()` and `unregister_login_trigger_task()`
- Generates Task Scheduler XML with LogonTrigger

### 8. Task 14 (Stale Context Push) ✓
**Issue:** Spec section 3.3 requires stale context push during sweep
**Fix Added:**
- Added step 3.5 to scheduled_sweep.ps1
- Iterates contexts, checks STATUS.json for `is_stale` flag
- Calls Python helper `send_stale_alert()` for dedup + send
- Added step 4 to regenerate GLOBAL_STATUS.md
- Updated commit message to reflect additions

### 9. Task 24 (_global Context Creation) ✓
**Issue:** Bootstrap install should create `_global` context
**Fix Added:**
- Added `_create_global_context()` helper function
- Creates context.json with constraints:
  - `max_chunks: 10000`
  - `age_threshold_days: 90`
  - `stale_after_hours: 24`
  - `archive.enabled: True`
- Invoked in step 1.5 of bootstrap_install()
- Idempotent (checks if already exists)

### 10. Task 14 (GLOBAL_STATUS.md Generation) ✓
**Issue:** When/how is GLOBAL_STATUS.md generated?
**Fix Added:**
- Added step 4 to scheduled_sweep.ps1:
  ```powershell
  chinvex status --regenerate --contexts-root $ContextsRoot
  ```
- Runs after ingest sweep and stale checks
- Generates fresh GLOBAL_STATUS.md from all STATUS.json files
- Also callable manually via `chinvex status --regenerate`

## Summary

**All 10 issues/gaps FIXED ✓**

**Files Modified:**
- `docs/plans/2026-01-29-bootstrapping-completion.md` (Tasks 4, 5, 7, 14, 15B, 17, 24 updated)

**New Code Added:**
- Task 4: 5-min max debounce cap in ChangeAccumulator
- Task 5: `sync reconcile-sources` command
- Task 7: Heartbeat counter (30s interval)
- Task 14: Stale context push + GLOBAL_STATUS.md generation in sweep
- Task 15B: Login-trigger scheduled task (NEW TASK)
- Task 17: Push dedup log + `send_stale_alert()` helper
- Task 24: `_create_global_context()` helper

**Ready for Execution:**
Plan is now complete with all identified gaps filled. Ready to proceed with:
```
/sequential-batch-execution docs/plans/2026-01-29-bootstrapping-completion.md
```

**Review Checklist:**
- [x] All portalocker calls use `fail_when_locked=True`
- [x] ChangeAccumulator caps debounce at 5 minutes
- [x] Heartbeat writes every 30s (not 1s)
- [x] Task 9 connection clarified
- [x] `sync reconcile-sources` implemented
- [x] Push dedup log with `push_log.jsonl`
- [x] Login-trigger task (primary watcher start)
- [x] Stale context push in sweep
- [x] `_global` context creation in bootstrap
- [x] GLOBAL_STATUS.md generation in sweep

All feedback addressed. Plan is production-ready.

# Final Fixes Summary

## All 6 Additional Issues FIXED ✓

### 1. Task 19 (Watch Hit Push) ✓
**Issue:** Placeholder with TODO comments instead of actual integration
**Fix Added:**
- Replaced placeholder with actual implementation in `post_ingest_hook()`
- After watches run, send ntfy notification with watch hits
- Format: "{context}: {count} watch hits ({queries})"
- Uses first 3 queries in message, adds "and N more" if needed
- Integration point: `src/chinvex/hooks.py::post_ingest_hook()`

```python
# Send watch hit notifications
if watch_hits and context.notifications and context.notifications.enabled:
    ntfy_config = NtfyConfig(...)
    queries = ', '.join(h.query for h in watch_hits[:3])
    more = f" (and {len(watch_hits) - 3} more)" if len(watch_hits) > 3 else ""
    send_ntfy_push(ntfy_config, f"{context.name}: {len(watch_hits)} watch hits...")
```

### 2. Task 10 (STATUS.json Freshness) ✓
**Issue:** Hardcoded `stale_after_hours=6` instead of reading from context.json
**Fix Added:**
- Added `stale_after_hours` parameter to `write_status_json()`
- Read value from `context.constraints.stale_after_hours` in ingest.py
- Falls back to 6 if not specified in config
- Allows per-context staleness thresholds

```python
def write_status_json(..., stale_after_hours: int = 6):
    freshness = compute_freshness(last_sync, stale_after_hours=stale_after_hours)

# In ingest.py:
stale_after_hours = 6  # Default
if hasattr(ctx_config, 'constraints') and ctx_config.constraints:
    stale_after_hours = ctx_config.constraints.get('stale_after_hours', 6)
```

### 3. Task 18 (morning_brief.ps1 Emoji) ✓
**Issue:** Used emoji (`⚠` and `✓`) instead of ASCII
**Fix Added:**
- Replaced `"⚠"` with `"[STALE]"`
- Replaced `"✓"` with `"[OK]"`
- ASCII-safe for all terminals and notifications

### 4. Task 21 (Status Output Emoji) ✓
**Issue:** Used emoji (`✓` and `⚠`) instead of ASCII
**Fix Added:**
- Replaced `"✓"` with `"[OK]"`
- Replaced `"⚠"` with `"[STALE]"`
- Updated all 3 test cases to expect ASCII indicators

### 5. Task 14 (State-Dir Consistency) ✓
**Issue:** Heartbeat check might use different state dir than sync commands
**Fix Added:**
- Added `StateDir` parameter to `scheduled_sweep.ps1`
- Defaults to `~/.chinvex` (standard location)
- All sync commands now explicitly pass `--state-dir $StateDir`
- Ensures heartbeat and watcher use same state directory

```powershell
param(
    ...
    [string]$StateDir = (Join-Path $env:USERPROFILE ".chinvex")
)

chinvex sync status --state-dir $StateDir
chinvex sync start --state-dir $StateDir
chinvex sync stop --state-dir $StateDir
```

### 6. Task 15 (Missing register_morning_brief_task) ✓
**Issue:** Function imported in cli.py but never implemented
**Fix Added:**
- Added `register_morning_brief_task()` to scheduler.py
- Generates Task Scheduler XML with daily trigger
- Parses time parameter (HH:MM) for StartBoundary
- Added `_generate_morning_brief_xml()` helper
- Added `unregister_morning_brief_task()` for cleanup
- Created `tasks.py` wrapper module for clean imports:
  - Re-exports all task functions from scheduler.py
  - Provides `unregister_task(name)` dispatcher
  - Provides `check_task_exists(name)` checker

```python
def register_morning_brief_task(contexts_root, ntfy_topic, time="07:00"):
    """Register morning brief task (daily at specified time)."""
    xml_content = _generate_morning_brief_xml(...)
    # Register ChinvexMorningBrief task

# src/chinvex/bootstrap/tasks.py
from .scheduler import (
    register_sweep_task,
    register_login_trigger_task,
    register_morning_brief_task,
    ...
)

def unregister_task(task_name: str):
    """Dispatcher for unregistering tasks."""
    if task_name == "ChinvexSweep":
        unregister_sweep_task()
    elif task_name == "ChinvexMorningBrief":
        unregister_morning_brief_task()
    ...
```

## Summary

**All 6 issues FIXED ✓**

**Files Modified:**
- Task 19: Watch hit push implementation in hooks.py
- Task 10: STATUS.json freshness parameter from context config
- Task 18: ASCII indicators in morning_brief.ps1
- Task 21: ASCII indicators in cli_status.py (+ test updates)
- Task 14: StateDir parameter in scheduled_sweep.ps1
- Task 15: register_morning_brief_task() + tasks.py wrapper module

**Ready for Execution:**
All identified gaps are now filled. Plan is complete and ready for:
```
/sequential-batch-execution docs/plans/2026-01-29-bootstrapping-completion.md
```

**Review Checklist:**
- [x] Watch hit push integrated in post_ingest_hook
- [x] STATUS.json reads stale_after_hours from context config
- [x] morning_brief.ps1 uses ASCII indicators
- [x] status CLI uses ASCII indicators
- [x] scheduled_sweep.ps1 uses consistent state dir
- [x] register_morning_brief_task() implemented
- [x] tasks.py wrapper module created

All feedback addressed. Plan is production-ready.

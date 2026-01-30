# Bootstrapping Completion Spec

**Headline: Turn Dual Nature from a system you operate into an appliance that operates on you.**

## The Test

> "If I have to remember to use it, it fails."

Everything in this spec serves one outcome: the system keeps Jordan on track without Jordan having to think about it.

## Non-Goals

- Smart agent / LLM-based prioritization (future phase)
- Learning from ignored nudges (future phase)
- Cross-context summarization (future phase)
- Perfect natural language interface (two verbs is enough)

---

## Property 1: Always Fresh

### Outcome

All contexts are automatically synced. Zero manual `chinvex ingest` ever again.

### Implementation

#### 1.1 File Watcher (primary for active work)

Long-running process that watches registered repos for changes.

```
chinvex sync start              # Start daemon
chinvex sync stop               # Stop daemon
chinvex sync status             # Show what's being watched
chinvex sync ensure-running     # Start if not running (idempotent)
chinvex sync reconcile-sources  # Update watcher to match context.json sources
```

**Note:** `chinvex watch` is reserved for watch-queries (semantic alerts). File watching uses `chinvex sync`.

**Behavior:**
- Debounce: Wait 30s after last file change before triggering ingest
- Scope: Only ingest changed files (delta ingest)
- Trigger: File create/modify/delete in watched paths

**Debounce semantics:**
- During debounce window, accumulate changed paths in memory
- After 30s quiet, ingest runs with accumulated path set
- Call: `chinvex ingest --context X --paths <file1> <file2> ...`
- Hard cap: if >500 paths accumulated, fall back to full ingest for that context
- Path set cleared after successful ingest (or on failure, retained for retry)

**Exclude list (CRITICAL - prevents infinite loops):**

Patterns use fnmatch glob syntax, matched against paths relative to watched root:

```
# Chinvex outputs (will cause ingest storm if watched)
contexts/**/STATUS.json
contexts/**/ingest_runs.jsonl
contexts/**/digests/**
**/MORNING_BRIEF.md
**/GLOBAL_STATUS.md
**/*_BRIEF.md
**/SESSION_BRIEF.md

# Per-repo chinvex artifacts
**/.chinvex/**
**/docs/memory/**   # Memory files are human-edited only

# Standard ignores
**/.git/**
**/node_modules/**
**/__pycache__/**
**/*.pyc
**/.venv/**
**/venv/**

# Chinvex internals (absolute paths)
~/.chinvex/*.log
~/.chinvex/*.pid
~/.chinvex/*.json
~/.chinvex/*.jsonl
```

**Matching rules:**
- `**` matches any directory depth
- `*` matches any characters within a path segment
- Patterns are case-insensitive on Windows
- Use `fnmatch` library for matching

**Critical rule:** Sync daemon only watches **external sources** (repos, inbox paths), never context directories themselves. The status example showing `P:\ai_memory\contexts\Chinvex` as a source is **wrong** - that should never be a watched source.

**Memory files rule:** `docs/memory/` files (STATE.md, CONSTRAINTS.md, DECISIONS.md) are **human-edited only**. Generators never write into watched trees. If automation needs to update state, it writes to `contexts/X/` artifacts, not repo-local memory files.

**Concurrency rules (CRITICAL - prevents thrashing):**
- Single-writer lock per context: OS-level file lock (portalocker) on `contexts/X/.ingest.lock`
- If ingest running and new fs events arrive → set `pending=true`, do ONE rerun after completion
- Never run two ingests for the same context simultaneously
- Sweep skips contexts currently ingesting (checks lock without blocking)

**Multi-context repo handling:**
If a repo is a source for multiple contexts (e.g., monorepo):
- Each context gets its own ingest triggered by file changes
- Ingests run sequentially, not parallel (each acquires its own context lock)
- Git hook: if repo feeds N contexts, hook runs N sequential ingests
- Watcher: coalesces by context, so each context gets one debounced ingest

**Lock implementation:** Use `portalocker` library:
```python
import portalocker
with portalocker.Lock(lock_path, timeout=0, fail_when_locked=True):
    # do ingest
```
`timeout=0` means non-blocking - if lock held, skip or queue.

**Daemon management:**
- Runs as background process (not Windows service - keep it simple)
- Writes PID to `~/.chinvex/sync.pid`
- Writes heartbeat to `~/.chinvex/sync_heartbeat.json` every 30s
- Logs to `~/.chinvex/sync.log`
- Sweep checks heartbeat to detect zombie process

**Startup mechanism (two triggers):**
1. **Login trigger:** Task Scheduler task runs `chinvex sync start` at user login
2. **Sweep backup:** Scheduled sweep runs `chinvex sync ensure-running` every 30 min

Both are belt-and-suspenders. Login trigger is primary, sweep is recovery.

**Atomic start semantics (prevents double-start race):**
- `sync start`: Acquire OS-level file lock (portalocker) on `~/.chinvex/sync.lock`, check if PID file exists and process alive → refuse if already running, else start and write PID
- `sync ensure-running`: Same lock, atomic "check+start" - only starts if not running
- `sync stop`: Acquire lock, send termination signal, wait for exit, remove PID file
- Lock held only during start/stop operations, not while daemon runs

**Lock implementation:** Use `portalocker` library for OS-level file locks (works on Windows and Linux). Not just file existence checks.

#### 1.2 Git Post-Commit Hook (belt-and-suspenders)

For repos with git, trigger ingest on commit.

**Installation:**
```bash
chinvex hook install --context Chinvex --repo C:\Code\chinvex
```

**Conflict handling:**
- If `.git/hooks/post-commit` exists → backup to `.git/hooks/post-commit.bak`, then replace
- If backup already exists → append timestamp `.git/hooks/post-commit.bak.20260129`
- Print warning: "Existing hook backed up to .git/hooks/post-commit.bak"

**Generated hook (Windows-safe):**

Creates `.git/hooks/post-commit` that calls a PowerShell wrapper:
```bash
#!/bin/sh
# Generated by chinvex hook install
# Calls PowerShell for Windows compatibility
pwsh -NoProfile -ExecutionPolicy Bypass -File "C:/Code/chinvex/.chinvex/post-commit.ps1" &
exit 0
```

Creates `.chinvex/post-commit.ps1` in repo root:
```powershell
# Generated by chinvex hook install
# Context: Chinvex
# Python path resolved at install time
$ErrorActionPreference = "SilentlyContinue"
& "{{PYTHON_PATH}}" -m chinvex.cli ingest --context Chinvex --changed-only --quiet
```

**Python path resolution (at hook install time):**

Order of preference:
1. `{repo}\.venv\Scripts\python.exe` (repo-local venv)
2. `py -3` (Windows Python launcher)
3. `$env:CHINVEX_PYTHON` (explicit override)
4. `python` (PATH fallback)

Installer tests each option and uses first that resolves to working Python with chinvex installed.

**Why this complexity:**
- Git hooks run in Git Bash environment, not PowerShell
- PATH is not guaranteed to include chinvex or python
- Using absolute python path from venv ensures it works
- PowerShell wrapper handles Windows-specific behavior
- `&` in sh + `exit 0` ensures non-blocking

**Behavior:**
- Non-blocking (backgrounded)
- `--changed-only` uses `git diff HEAD~1 --name-only` to get changed file list
- Internally converts to `--paths <file1> <file2> ...` call
- Fails silently if chinvex not available (don't break git)
- Hook checks `.chinvex/post-commit.ps1` exists before calling

**Ingest path modes (for clarity):**

| Flag | Source | Use case |
|------|--------|----------|
| `--paths <f1> <f2>` | Explicit file list | Watcher (accumulated changes) |
| `--changed-only` | Git diff HEAD~1 | Git hook |
| (no flag) | Full source scan | Manual, sweep fallback |

#### 1.3 Scheduled Sweep (fallback)

Windows Task Scheduler job that runs every 30 minutes.

**Purpose:**
- Catch anything the watcher missed
- Ensure watcher is running and watching correct sources
- Sync contexts without file watchers (chat exports, etc.)
- Archive pass for `_global` context

**Script:** `scripts/scheduled_sweep.ps1`
```powershell
# Ensure watcher is running and sources are current
chinvex sync ensure-running
chinvex sync reconcile-sources

# Check heartbeat (detect zombie process)
$heartbeat = Get-Content ~/.chinvex/sync_heartbeat.json -ErrorAction SilentlyContinue | ConvertFrom-Json
$heartbeatAge = (Get-Date) - [datetime]$heartbeat.timestamp
if ($heartbeatAge.TotalMinutes -gt 5) {
    Write-Warning "Watcher heartbeat stale, restarting..."
    chinvex sync stop
    chinvex sync start
}

# Sweep all contexts (skip those currently ingesting)
chinvex ingest --all-contexts --changed-only --quiet --skip-locked

# Run archive pass on _global if needed
chinvex archive --context _global --apply-constraints --quiet
```

**Task Scheduler config:**
- Trigger: Every 30 minutes
- Run whether user logged in or not
- Don't start new instance if already running
- **Critical:** Task passes config explicitly (env vars unreliable in non-interactive context):
  ```
  pwsh -NoProfile -File scripts/scheduled_sweep.ps1 `
    -ContextsRoot "P:\ai_memory\contexts" `
    -NtfyTopic "chinvex-alerts"
  ```

#### 1.4 Auto-Register Repos

When a repo is tracked, it's automatically added to the sync watcher.

```bash
chinvex context add-source --context Chinvex --repo C:\Code\chinvex
# Automatically adds to sync watcher if daemon is running
```

Or via alias (handles naming):
```bash
dual track C:\Code\chinvex
# Creates context "chinvex" (lowercase leaf), adds to watcher
```

**Naming collision handling:**
- Context name = lowercase folder leaf
- If context exists pointing elsewhere → create `name-2`, `name-3`, etc.
- Normalize: replace spaces with `-`, strip special chars

#### 1.5 Global Catch-All Context

A special context called `_global` for stuff that doesn't belong elsewhere.

**Sources:**
- `P:\ai_memory\inbox\` - Drop zone for random files
- Chat exports that aren't project-specific
- Browser highlights / Readwise exports (future)

**Constraints (critical - prevents garbage heap):**
- Max 10,000 chunks (archive oldest when exceeded)
- Auto-archive after 90 days untouched
- Promotion mechanism: `chinvex promote --from _global --to ProjectX --query "search terms"`
- Constraints enforced during scheduled sweep (see 1.3)

**Constraint enforcement details:**
- "Oldest" = by `updated_at` timestamp at chunk level
- "Archive" = set `archived=1` in SQLite `chunks` table (column added in P3 schema v2)
- Archived chunks excluded from search via `WHERE archived=0` filter
- Enforcement: `chinvex archive --context _global --apply-constraints` runs during sweep
- Order: archive by age first (>90 days), then by count (oldest beyond 10k)

**Schema note:** P3 added `archived INTEGER DEFAULT 0` column to chunks table. If upgrading from earlier version, run `chinvex db migrate` or column is auto-added on first archive operation.

**Config:** `contexts/_global/context.json`
```json
{
  "name": "_global",
  "type": "catch-all",
  "freshness": {
    "stale_after_hours": 24
  },
  "constraints": {
    "max_chunks": 10000,
    "archive_after_days": 90
  },
  "includes": {
    "inbox": ["P:\\ai_memory\\inbox"]
  }
}
```

**Per-context freshness override:**

Any context can override the default 6-hour stale threshold:
```json
{
  "name": "Chinvex",
  "freshness": {
    "stale_after_hours": 1
  }
}
```

- Default: 6 hours
- `_global`: 24 hours (only syncs when inbox changes)
- Active repos: 1-6 hours depending on velocity
```

### Acceptance

- [ ] File watcher detects changes within 5s, ingests within 60s
- [ ] Git hook fires on commit without blocking (Windows-safe)
- [ ] Scheduled sweep runs every 30 min, visible in Task Scheduler
- [ ] New repos added via `add-source` automatically watched
- [ ] `_global` context exists with constraints enforced
- [ ] Concurrent ingests blocked by lock file
- [ ] Watcher excludes own outputs (no infinite loops)
- [ ] Per-context stale threshold respected

---

## Property 2: Always Visible

### Outcome

You can instantly see whether the system is healthy and current without running commands.

### Implementation

#### 2.1 Status Command

```bash
chinvex status [--context X]
```

**Output (no args):**
```
Chinvex Status
══════════════════════════════════════════════════
Watcher:     RUNNING (pid 12345, watching 4 repos)
Last sweep:  3 min ago

Context          Last Sync    Chunks    Watches    Status
─────────────────────────────────────────────────────────
Chinvex          2 min ago    69,171    3 active   ✓ fresh
Streamside       2 min ago    12,847    1 active   ✓ fresh
Godex            6 hours ago   4,221    0 active   ⚠ stale
_global          1 day ago       847    0 active   ⚠ stale

Pending watch hits: 2 (Chinvex: "retry logic", "embeddings")
```

**Output (with --context):**
```
Context: Chinvex
══════════════════════════════════════════════════
Last ingest:     2 min ago (run_id: abc123)
Chunks:          69,171 (delta: +12 since yesterday)
Embedding:       ollama/mxbai-embed-large (1024 dims)
Watches:         3 active, 2 pending hits

Sources:
  repo: C:\Code\chinvex (watching: yes)

Recent ingests:
  2026-01-29 14:32 - succeeded (12 files, 47 chunks)
  2026-01-29 12:15 - succeeded (3 files, 8 chunks)
  2026-01-29 09:00 - succeeded (sweep, no changes)
```

#### 2.2 Status Artifacts

Written automatically after each ingest/digest.

**Per-context:** `contexts/X/STATUS.json`
```json
{
  "context": "Chinvex",
  "last_ingest": "2026-01-29T14:32:00Z",
  "last_ingest_run_id": "abc123",
  "last_ingest_status": "succeeded",
  "last_digest": "2026-01-29T07:00:00Z",
  "chunks": 69171,
  "watches_active": 3,
  "watches_pending_hits": 2,
  "freshness": {
    "stale_after_hours": 6,
    "is_stale": false,
    "hours_since_sync": 0.05
  },
  "sources": [
    {"type": "repo", "path": "C:/Code/chinvex", "watching": true}
  ],
  "embedding": {
    "provider": "ollama",
    "model": "mxbai-embed-large",
    "dimensions": 1024
  }
}
```

**Note:** Sources are external (repos, inbox paths). Context directories (`contexts/X/`) are never listed as sources.
```

**Global:** `GLOBAL_STATUS.md` (in ai_memory root)
```markdown
# Dual Nature Status
Generated: 2026-01-29T14:35:00Z

## Health
- Watcher: RUNNING
- Last sweep: 3 min ago
- Stale contexts: 1 (Godex)

## Contexts
| Context    | Last Sync | Chunks | Watch Hits | Status |
|------------|-----------|--------|------------|--------|
| Chinvex    | 2m ago    | 69,171 | 2 pending  | ✓      |
| Streamside | 2m ago    | 12,847 | 0          | ✓      |
| Godex      | 6h ago    |  4,221 | 0          | ⚠ stale |
| _global    | 1d ago    |    847 | 0          | ⚠ stale |

## Pending Watch Hits
- **Chinvex**: "retry logic" (2 hits), "embeddings" (1 hit)
```

#### 2.3 Freshness in Morning Push

The morning brief (see Property 3) includes sync status so you see it passively.

### Acceptance

- [ ] `chinvex status` shows all contexts with freshness
- [ ] `STATUS.json` written after every ingest
- [ ] `GLOBAL_STATUS.md` regenerated after every sweep
- [ ] Stale threshold respects per-context `freshness.stale_after_hours` (default 6h)

---

## Property 3: Always In Your Face

### Outcome

The system reaches out to you. You don't have to remember to check it.

### Implementation

#### 3.1 Morning Brief Push

Scheduled task that runs at configured time (default 7:00 AM local).

**Configuration:**
- `CHINVEX_MORNING_BRIEF_TIME=07:00` (env var, 24h format)
- Task Scheduler reads this at install time

**Script:** `scripts/morning_brief.ps1`
```powershell
param(
    [Parameter(Mandatory=$true)][string]$ContextsRoot,
    [Parameter(Mandatory=$true)][string]$NtfyTopic,
    [string]$NtfyServer = "https://ntfy.sh",
    [string]$OutputPath = "P:\ai_memory\MORNING_BRIEF.md"
)

# Generate briefs for all active contexts
$briefs = chinvex brief --all-contexts --contexts-root $ContextsRoot --format json | ConvertFrom-Json

# Build notification
$stale = $briefs | Where-Object { $_.status -eq "stale" }
$watchHits = ($briefs | ForEach-Object { $_.watch_hits } | Measure-Object -Sum).Sum

$title = "Dual Nature Morning Brief"
$body = @"
Contexts: $($briefs.Count) ($($stale.Count) stale)
Watch hits: $watchHits pending
"@

if ($stale.Count -gt 0) {
    $body += "`nStale: $($stale.context -join ', ')"
}

# Truncate body if too long (ntfy limit)
if ($body.Length -gt 500) {
    $body = $body.Substring(0, 497) + "..."
}

# Push via ntfy
Invoke-RestMethod -Uri "$NtfyServer/$NtfyTopic" -Method Post -Body $body -Headers @{
    "Title" = $title
    "Priority" = "default"
    "Tags" = "brain,clipboard"
}

# Also write to file for Claude session start
chinvex brief --all-contexts --contexts-root $ContextsRoot --output $OutputPath
```

**Task Scheduler config:**
```
pwsh -NoProfile -File scripts/morning_brief.ps1 `
    -ContextsRoot "P:\ai_memory\contexts" `
    -NtfyTopic "chinvex-alerts" `
    -NtfyServer "https://ntfy.sh"
```

All config passed explicitly - no env var reads in scripts (unreliable in non-interactive context).

**Notification content:**
```
Dual Nature Morning Brief
Contexts: 4 (1 stale)
Watch hits: 3 pending
Stale: Godex
```

Tapping the notification opens browser to ntfy.sh topic (or self-hosted server).

#### 3.2 Watch Hit Push

Triggered immediately after ingest if watches matched. Coalesced per ingest run.

**In ingest completion logic:**
```python
if watch_hits:
    # Coalesce all hits from this ingest run into one notification
    queries = ', '.join(h.query for h in watch_hits[:3])
    if len(watch_hits) > 3:
        queries += f" +{len(watch_hits) - 3} more"
    push_ntfy(
        title="Watch hit",
        body=f"{context}: {len(watch_hits)} hits ({queries})",
        priority="default"
    )
```

**Notification:**
```
Watch hit
Chinvex: 2 hits (retry logic, embeddings)
```

**Coalescing:** All hits from a single ingest run → one notification. No dedup beyond that (watch hits are always interesting).

#### 3.3 Stale Context Push

If scheduled sweep detects a context hasn't synced within its configured threshold.

**In sweep logic:**
```python
for ctx in contexts:
    threshold = ctx.config.get("freshness", {}).get("stale_after_hours", 6)
    if hours_since_sync(ctx) > threshold:
        push_ntfy(
            title="Stale context",
            body=f"{ctx.name}: last sync {humanize(ctx.last_sync)}",
            priority="low"
        )
```

**Notification:**
```
Stale context
Godex: last sync 6h ago
```

**Dedup:** Only push once per context per day (track in `~/.chinvex/push_log.jsonl`).

**Push log schema:**
```json
{"timestamp": "2026-01-29T14:00:00Z", "context": "Godex", "type": "stale", "date": "2026-01-29"}
```
Dedup key: `(context, type, date)`. Same context+type on same calendar day = skip.

**Emoji note:** All notifications use plain ASCII text (no emojis). Windows terminal and some ntfy clients handle emojis inconsistently.

#### 3.4 Ntfy Configuration

Environment variables (consistent with existing pattern):
- `CHINVEX_NTFY_TOPIC` (required for push features)
- `CHINVEX_NTFY_SERVER` (default: `https://ntfy.sh`)

**Graceful degradation:** If `CHINVEX_NTFY_TOPIC` not set, skip push but still generate artifacts.

### Acceptance

- [ ] Morning brief push arrives at 7 AM
- [ ] Watch hit push arrives within 60s of ingest completion
- [ ] Stale context push arrives (max 1x per context per day)
- [ ] All pushes work with ntfy.sh (no self-hosted requirement)
- [ ] Missing ntfy config = silent skip, not error

---

## Property 4: Zero Commands

### Outcome

In daily use, you never type a chinvex command.

### Implementation

#### 4.1 Two Aliases (escape hatch)

For the rare cases when you need manual control.

**PowerShell profile (`$PROFILE`):**
```powershell
function dual {
    param([string]$cmd, [string]$arg)
    switch ($cmd) {
        "brief"  { chinvex brief --all-contexts }
        "track"  { 
            $repo = if ($arg) { Resolve-Path $arg } else { Get-Location }
            # Name = lowercase folder leaf, normalized
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

# Optional shorter alias
Set-Alias dn dual
```

**Usage:**
```powershell
dual brief          # Show all context briefs
dual track          # Track current directory
dual track C:\Code\newproject  # Track specific path
dual status         # Show system health
dn brief            # Short form
```

**`dual track` behavior:**
- Name = lowercase folder leaf, non-alphanumeric replaced with `-`
- If name exists pointing to same repo (resolved path match) → no-op
- If name exists pointing elsewhere → `name-2`, `name-3`, etc.
- Uses `chinvex ingest --context X --repo Y` (P4.1 inline context creation)
- Triggers `sync reconcile-sources` to update watcher

#### 4.2 Session Start Protocol (already exists)

From P4: `CLAUDE.md` / `AGENTS.md` contains instruction for Claude to run `chinvex brief` on session start.

**Enhancement:** Point to combined brief:
```markdown
## Session Start Protocol
On session start, read `P:\ai_memory\MORNING_BRIEF.md` for current state.
If stale (>6 hours), run: `chinvex brief --all-contexts --output P:\ai_memory\MORNING_BRIEF.md`
```

#### 4.3 MCP Tools (already exists)

From P4: Claude Code can query via MCP without commands.

#### 4.4 ChatGPT Actions (already exists)

From P3: ChatGPT can query via gateway API.

#### 4.5 Skills (future, not required for bootstrap)

Natural language like "make sure this repo is tracked" → skill calls `dual track`.

**Not in scope for bootstrapping.** The two aliases + session protocol are sufficient.

### Acceptance

- [ ] `dual brief` works from any PowerShell prompt
- [ ] `dual track` registers repo and adds to watcher
- [ ] `dual status` shows system health
- [ ] Session start loads brief automatically (Claude Code + ChatGPT)
- [ ] No other commands needed in daily workflow

---

## Implementation Order

1. **File watcher / sync daemon** (1.1) - Core of "always fresh", generates real ingest runs
2. **Scheduled sweep** (1.3) - Belt-and-suspenders, recovery path
3. **Status artifacts** (2.2) - Now has real data to report
4. **Status command** (2.1) - Consume the artifacts
5. **Git hooks** (1.2) - Extra reliability for git repos
6. **Morning brief push** (3.1) - First "in your face" feature
7. **Watch hit push** (3.2) - Immediate feedback loop
8. **Stale context push** (3.3) - Safety net
9. **Aliases** (4.1) - Escape hatch
10. **Global catch-all** (1.5) - Last because it needs constraints thought through
11. **Bootstrap installer** - Wraps it all up

---

## Migration / Setup

### One-command installer

```bash
chinvex bootstrap install
```

**Interactive prompts:**
- ntfy topic (required): `CHINVEX_NTFY_TOPIC`
- Morning brief time (default 07:00): `CHINVEX_MORNING_BRIEF_TIME`
- Contexts root (default P:\ai_memory\contexts): `CHINVEX_CONTEXTS_ROOT`

**Actions performed:**
1. Create `_global` context with constraints
2. Install scheduled task: sweep (every 30 min)
3. Install scheduled task: morning brief (at configured time)
4. Add `dual` function to PowerShell profile
5. Set environment variables (user scope)
6. Start sync watcher
7. Print status summary

**Other bootstrap commands:**
```bash
chinvex bootstrap status      # Show what's installed, what's running
chinvex bootstrap uninstall   # Remove tasks, stop watcher, remove env vars
chinvex bootstrap reinstall   # Uninstall + install (for config changes)
```

### Manual setup (if bootstrap fails)

1. Create `_global` context with constraints
2. Install scheduled task for sweep
3. Install scheduled task for morning brief
4. Add aliases to PowerShell profile
5. Set `CHINVEX_NTFY_TOPIC` env var
6. Run `chinvex sync start`

### Existing contexts

- No schema changes required
- STATUS.json created automatically on next ingest
- Sync watcher auto-discovers sources from context.json
- Ingest run log format (started/succeeded/failed) already defined in P4 spec

---

## Testing

### Unit

- [ ] Status artifact generation
- [ ] Freshness calculation (per-context stale threshold)
- [ ] Push deduplication logic
- [ ] Path watching / debounce
- [ ] Exclude list matching
- [ ] Context name collision handling (`name-2`, `name-3`)
- [ ] Lock file acquire/release
- [ ] Hook python path resolution order

### Integration

- [ ] Watcher detects file change → ingest triggers
- [ ] Watcher ignores excluded paths (no infinite loop)
- [ ] Concurrent file changes → single coalesced ingest
- [ ] Git commit → hook fires → ingest runs (Windows)
- [ ] Sweep runs → stale contexts identified
- [ ] Sweep skips locked contexts
- [ ] Ntfy push delivery (mock server)
- [ ] `chinvex sync reconcile-sources` updates watcher
- [ ] `chinvex sync start` refuses if already running (atomic)
- [ ] Two concurrent `sync ensure-running` → only one daemon starts

### E2E

- [ ] Full flow: edit file → watcher ingests → watch hits → push arrives
- [ ] Morning brief: scheduled task runs → brief generated → push arrives
- [ ] `dual track .` → context created → watcher starts → ingest runs
- [ ] `chinvex bootstrap install` → all components configured
- [ ] `chinvex bootstrap status` → accurate health report

### Kill-Switch Test (appliance proof)

**The definitive test that this is an appliance, not a system you operate:**

1. Stop sync daemon manually
2. Delete `~/.chinvex/sync_heartbeat.json`
3. Reboot machine
4. Do nothing for 35 minutes

**Expected outcome (no human intervention):**
- [ ] Sweep task detects missing heartbeat
- [ ] Sweep restarts sync daemon
- [ ] GLOBAL_STATUS.md regenerated
- [ ] Stale context push sent (if any context >threshold)
- [ ] Morning brief fires at 07:00 the next day

If this passes, you've built an appliance.

---

## Success Criteria

**After this phase, the following is true:**

1. Jordan never runs `chinvex ingest` manually
2. Jordan can see system health in <5 seconds (status command or GLOBAL_STATUS.md)
3. Jordan receives morning brief without asking
4. Jordan receives watch hits without asking
5. Jordan is alerted if something goes stale
6. Jordan can say "dual brief" or "dual track" if needed, but rarely does

**What this unlocks:**

- Trust in the data (it's always fresh)
- Passive awareness (nudges come to you)
- Foundation for smart agent (future phase)

---

## Future Phase: Smart Agent

NOT in scope for bootstrapping, but this is where it goes:

- Cron agent that reads digests and decides what to surface
- Priority ranking based on urgency, ROI, deadlines
- Learning from ignored nudges (deprioritize)
- Cross-context insights ("Chinvex watch hit relates to Streamside blocker")
- Proactive suggestions ("you haven't touched X in 2 weeks, archive?")

This only works if Bootstrapping Completion is rock solid. Data must be fresh and trusted first.

---

## Implementation Notes

**Gametime decisions (handle during coding, not spec):**

| Area | Guidance |
|------|----------|
| Malformed context.json | Skip with warning, continue to next context |
| Git hook edge cases | Initial commit (no HEAD~1): skip diff, run full ingest. Merge commits: use `git diff HEAD^ --name-only` |
| Hook failure modes | All failures silent except lock conflicts (log to ~/.chinvex/hook_errors.log) |
| Notification truncation | Max 500 chars for ntfy body |
| Humanize timestamps | Use "Xh ago" format (e.g., "6h ago", "2d ago") |
| Delta ingest failure | Log error, do NOT fall back to full ingest (could mask the problem) |
| Debounce max duration | If files change continuously for >5 minutes, force ingest anyway |
| Path accumulator on crash | Lost (in-memory only). Sweep will catch up within 30 min. |
| Continuous debounce | Cap at 5 min total debounce time, then force ingest regardless |
| Status "recent ingests" | Show last 5 runs |
| "Delta since yesterday" | Literal 24h ago, not calendar day |
| Watch query storage | Already exists: `contexts/X/watch.json` |
| Profile doesn't exist | Bootstrap installer creates it with just the dual function |
| Stale context push threshold | Respects per-context `stale_after_hours`, not hardcoded 6h |
| ntfy.sh privacy | Warn during bootstrap install that context names may be visible; suggest self-host for sensitive data |
| Brief vs digest cost | `brief --all-contexts` reads STATUS.json, does NOT regenerate digests |

**Dependencies to add:**
```
portalocker>=2.0.0   # OS-level file locks
watchdog>=3.0.0      # File system events
```

---

## Locked Design Decisions

| Question | Decision |
|----------|----------|
| Watcher architecture | Long-running Python process (watchdog), started at login via Task Scheduler + sweep ensure-running backup. Heartbeat file for health check. Not a Windows service. |
| Sweep frequency | 30 minutes. Watcher is primary; sweep is recovery. |
| Stale threshold | Default 6 hours, per-context override via `freshness.stale_after_hours` |
| `_global` constraints | 10K chunks / 90 days. Oldest by `updated_at`. Archive = `archived=1` column in SQLite chunks table. Enforced during sweep. |
| Morning brief time | Configurable via CLI arg to scheduled task (default 07:00) |
| Push dedup | Stale alerts: 1x per context per day. Watch hits: no dedup, but coalesced per ingest run. |
| Alias name | `dual` primary, `dn` as shorter alias |
| Command namespace | `chinvex sync` for file watcher daemon. `chinvex watch` reserved for watch-queries. |
| Debounce | 30s quiet window, accumulate paths, cap at 500 paths then full ingest |
| Delta ingest | Watcher uses `--paths`, git hook uses `--changed-only` (git diff) |
| Watched sources | External sources only (repos, inbox). Never context directories. |
| Memory files | Human-edited only. Generators never write to `docs/memory/`. |
| Task config | Pass explicitly via CLI args, not env vars (non-interactive reliability). |
| Emojis | ntfy notifications only (ASCII text, no emoji - Windows terminal safe). CLI uses ASCII. |
| Hook python path | Resolved at install: repo venv → `py -3` → `CHINVEX_PYTHON` → PATH `python` |
| Sync start atomicity | OS-level file lock via portalocker on `sync.lock` during start/stop. `ensure-running` is atomic check+start. |
| Exclude patterns | fnmatch glob syntax with `**` for recursive matching |
| Hook conflicts | Backup existing hook to `.bak`, then replace |
| Multi-context repos | Sequential ingests per context, each with own lock |

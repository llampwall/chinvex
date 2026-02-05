# Project Memory System Spec

> **Status:** Final
> **Purpose:** Define the 3-file project memory system and its maintenance
> **Changes from v0.2:**
> - Added decision logging threshold (what counts as a decision)
> - Added CONSTRAINTS dedupe rule (search before adding, edit in place)
> - Added `## Quick Reference` section to STATE.md template
> - Clarified coverage anchor semantics (HEAD-inclusive)
> - Extended "Unknown (needs human)" pattern to Active Work and Blockers
> - Added DO/DON'T inline headers to each file template
> - Updated brief generation to include Quick Reference (comes free with STATE)

---

## Problem Statement

The original 5-file system (`operating_brief.md`, `key_facts.md`, `adrs.md`, `bugs.md`, `worklog.md`) failed because:

1. **ADRs never got updated** — format too ceremonial, agents couldn't decide what qualified
2. **Multiple files competed** — agents spread updates thin, missed important ones
3. **Unclear triggers** — "when should I write an ADR?" has no crisp answer
4. **Too much structure** — high risk of rewriting history, so agents avoided touching files

The 3-file system fixes this by:
- Consolidating related concerns (key_facts + adrs + hazards → CONSTRAINTS)
- Clear update modes (rewrite vs merge vs append)
- Trigger based on "did something become true?" not "should I write an ADR?"

---

## File Structure

```
{repo_root}/docs/memory/
├── STATE.md        # Rewrite allowed
├── CONSTRAINTS.md  # Merge-only (add/update, don't delete)
└── DECISIONS.md    # Append-only
```

**Location:** Always `{repo_root}/docs/memory/` (per-repo, not per-context).

---

## File Specifications

### STATE.md

**Purpose:** "Load me into Claude's head" — what's true right now.

**Update mode:** Full rewrite allowed. This is the only file that should feel "current."

**Required sections:**
```markdown
<!-- DO: Rewrite freely. Keep under 30 lines. Current truth only. -->
<!-- DON'T: Add history, rationale, or speculation. No "we used to..." -->

# State

## Current Objective
[One line: what we're trying to accomplish — or "Unknown (needs human)"]

## Active Work
- [1–3 bullets of in-progress items — or "Unknown (needs human)"]

## Blockers
[None, or 1–3 bullets — or "Needs triage"]

## Next Actions
- [ ] [Max 5 checkboxes of immediate next steps]

## Quick Reference
- Run: `<command to start/run>`
- Test: `<command to run tests>`
- Entry point: `<main file or module>`

## Out of Scope (for now)
- [Max 3 bullets of things explicitly deferred]

---
Last memory update: YYYY-MM-DD
Commits covered through: <hash>
```

**Rules:**
- Keep it SHORT. Target ~30 lines, hard cap 45.
- Each section: 1–3 bullets max. Next Actions: max 5 checkboxes. Out of Scope: max 3 bullets.
- No history. No rationale. Just current truth.
- **No "should" language.** STATE is truth, not intention. If it reads like "we should probably…" it's drifting into planning/hallucination. Rewrite as a fact or a Next Action.
- Update every session or when objective changes.
- **Quick Reference:** 3–5 lines max. Only the commands/URLs/paths needed to start working. If a value changes, update it here (it's current truth, not a constraint).
- **Coverage anchor required:** Footer must include last update date and ending commit hash. The footer hash is the HEAD commit that this memory update is current through (inclusive). Next maintainer run processes `git log <footer_hash>..HEAD`.
- **Failure mode:** If maintainer cannot infer a section with confidence, use the unknown patterns shown in the template above (`Unknown (needs human)` / `Needs triage`). This prevents confident hallucinated state — the worst failure mode.

---

### CONSTRAINTS.md

**Purpose:** What must not change — rules, hazards, infrastructure facts that constrain future work.

**Update mode:** Merge-only. Add/update bullets, don't delete unless explicitly told.

**Sections:**
```markdown
<!-- DO: Add bullets. Edit existing bullets in place with (updated YYYY-MM-DD). -->
<!-- DON'T: Delete bullets. Don't write prose. Don't duplicate — search first. -->

# Constraints

## Infrastructure
- [Technical limits, batch sizes, ports, paths]

## Rules
- [Invariants, "don't do X because Y"]

## Key Facts
- [Lookup values: URLs, env var names, commands]

## Hazards
- [Things that bite you if you forget]

## Superseded
- (Superseded YYYY-MM-DD) [Old constraint that no longer applies, with reason]
```

**Rules:**
- Bullets only. No prose. One constraint per bullet (atomic).
- **Core sections (always present):** Infrastructure, Rules, Key Facts, Hazards, Superseded
- **Optional sections (add as needed):** APIs, Performance, Security, Dependencies
- Trigger: "learned something the hard way"
- **Dedupe rule:** Before adding a bullet, search the file for an existing bullet covering the same concern. If found, **edit it in place** (same section) and append `(updated YYYY-MM-DD)`. Do not create duplicates.
- **Supersede, don't delete:** When a constraint no longer applies, move it to `## Superseded` with date and reason. Never delete outright — preserves history while keeping hot path scannable.

---

### DECISIONS.md

**Purpose:** Audit trail — how did we get here?

**Update mode:** Append-only for entries. Top rollup section can be rewritten.

**Structure:**
```markdown
<!-- DO: Append new entries to current month. Rewrite Recent rollup. -->
<!-- DON'T: Edit or delete old entries. Don't log trivial changes. -->

# Decisions

## Recent (last 30 days)
- [5–10 bullet summary of recent decisions — rewritable]

## 2026-01
[Monthly sections for append-only entries]

### YYYY-MM-DD — [Decision title]

- **Why:** [Reason for the decision]
- **Impact:** [What changed as a result]
- **Evidence:** [commit hash or PR link preferred; file path only for on-disk invariants not tied to a specific commit]
```

**Bug fix entries** (when decision is a bug fix, use this sub-shape):
```markdown
### YYYY-MM-DD — Fixed [bug description]

- **Symptom:** [What you observed]
- **Root cause:** [Why it happened]
- **Fix:** [What you did]
- **Prevention:** [How to avoid in future]
- **Evidence:** [commit hash]
```

**Decision logging threshold — what counts:**

Log a decision if it changes any of:
- Interfaces (API shape, CLI flags, config format)
- Storage (schema, file layout, database changes)
- Workflow (build process, deploy pipeline, CI changes)
- Security posture (auth, permissions, secrets handling)
- Performance tradeoffs (caching strategy, batch sizes, timeouts)
- Dependencies (added, removed, or version-pinned for a reason)
- Deployment (infra changes, new services, port assignments)
- User-facing behavior (features, breaking changes, defaults)

Do **not** log:
- Refactors that don't change behavior
- Renames, formatting, lint fixes
- Dependency bumps unless they change behavior or constraints
- "Fixed typo" / "cleaned up comments"

When in doubt: if future-you would say "why did we do it this way?", log it.

**Rules:**
- Trigger: "something became true because of a change"
- Captures: architectural choices, bug resolutions, lessons learned
- No ADR-001 numbering. Dates only.
- **Monthly sections:** Organize by `## YYYY-MM` to keep inserts localized
- **Rollup section:** Top "Recent (last 30 days)" is rewritable for quick scanning
- **Bug playbook preserved:** Bug fixes use extended format so playbook value isn't lost
- **Evidence required:** Every entry must include a **commit hash or PR link** whenever possible. File path alone is allowed only for on-disk invariants not tied to a specific commit window.
- Recent entries (last 7 days) feed into session briefs.

---

## Maintainer System

### Trigger: Scheduled Batch (NOT per-commit)

**Rationale:**
- Claude/Codex commit frequently — per-commit hooks are too expensive
- Batch processing lets agent see patterns across multiple commits
- Can be triggered on-demand when needed
- Requires good commit messages and specs/plans as source material

**Options:**

| Mode | Trigger | Use case |
|------|---------|----------|
| Scheduled | Daily/twice-daily cron | Hands-off maintenance |
| On-demand | `chinvex update-memory --context X` | Before starting work |
| Session-end | Manual or hook | After significant work sessions |

### Input Sources (for maintainer agent)

1. **Git log** — commits since last update
2. **Specs/plans** — `specs/*.md`, `docs/plans/*.md`
3. **Current memory files** — to avoid duplicating existing content
4. **Diff of changed files** — to understand what actually changed

### Update Logic

```
For each update run:
1. Read STATE.md footer to get last processed commit hash
2. Gather commits: git log <last_hash>..HEAD
3. Capture current HEAD hash for footer update
4. Read current STATE.md, CONSTRAINTS.md, DECISIONS.md
5. Read any referenced specs/plans from commits
6. For STATE.md:
   - Rewrite based on current project state
   - Use latest specs/plans to determine objective
   - If a section cannot be inferred with confidence, use "Unknown (needs human)" pattern
   - Update Quick Reference if run/test/entry point commands have changed
   - Update footer with new date and ending commit hash (HEAD-inclusive)
7. For CONSTRAINTS.md:
   - Check if any commits reveal new constraints
   - Search for existing bullets before adding (dedupe rule)
   - Edit existing bullets in place with (updated YYYY-MM-DD) if updating
   - Merge new bullets into appropriate sections
   - Move obsolete constraints to Superseded section (never delete)
8. For DECISIONS.md:
   - Check if any commits represent significant decisions (apply logging threshold)
   - Append new entries to current month section
   - Update Recent rollup with last 30 days summary
   - Use bug-fix format when applicable
   - Require at least one Evidence pointer per entry
```

---

## Integration with Chinvex

### Brief Generation

`chinvex brief --context X` assembles:
1. Full content of `STATE.md` (includes Quick Reference)
2. `CONSTRAINTS.md`: Infrastructure + Rules + Hazards sections only (skip Key Facts, Superseded)
3. `DECISIONS.md`: Recent rollup section + entries from last 7 days
4. Latest digest (watch hits, recent activity)

**Note:** Quick Reference (run/test/entry commands) is included automatically since it's part of STATE. Key Facts are excluded from briefs to keep them tight — if you need URLs/env vars/commands beyond Quick Reference, open `docs/memory/CONSTRAINTS.md#key-facts` directly.

### Digest Generation

`chinvex digest --context X` produces daily delta:
- Watch hits
- Files changed
- Ingest stats

**Digest does NOT replace memory files** — it's a delta, not the canon.

---

## Acceptance Criteria

- [ ] STATE.md reflects current project state after each maintainer run
- [ ] STATE.md uses "Unknown (needs human)" when objective/work/blockers can't be inferred
- [ ] Quick Reference section contains working run/test commands
- [ ] CONSTRAINTS.md accumulates learnings without losing history
- [ ] CONSTRAINTS.md has no duplicate bullets (dedupe rule enforced)
- [ ] DECISIONS.md provides traceable audit trail
- [ ] Every DECISIONS entry has at least one Evidence pointer
- [ ] DECISIONS entries pass the logging threshold (no trivial changes logged)
- [ ] Brief generation produces useful session-start context (includes Quick Reference)
- [ ] Maintainer doesn't block active development work
- [ ] Files stay short and scannable (STATE ≤ 45 lines hard cap, CONSTRAINTS < 100 lines)
- [ ] DO/DON'T headers present in all three file templates

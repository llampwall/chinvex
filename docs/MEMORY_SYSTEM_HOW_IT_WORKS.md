# Memory System — How It Works

> **Last updated:** 2026-02-05
> **Status:** Production-ready, deployed across repos

---

## Overview

The memory system gives every repo a 3-file knowledge base (`STATE.md`, `CONSTRAINTS.md`, `DECISIONS.md`) that agents read on session start and update as work progresses. The system is fully automated from ingestion through session use — the only human behavior required is committing code.

---

## The Three Files

All live at `{repo_root}/docs/memory/`.

| File | Purpose | Update mode |
|------|---------|-------------|
| `STATE.md` | What's true right now | Full rewrite |
| `CONSTRAINTS.md` | What must not change | Merge-only (add/edit, never delete) |
| `DECISIONS.md` | How we got here | Append-only (+ rewritable rollup) |

Each file has inline `DO/DON'T` headers so agents editing them follow the rules without needing the spec loaded.

**Spec:** `docs/PROJECT_MEMORY_SPEC.md` in the chinvex repo defines the full contract.

---

## End-to-End Flow

```
Repo ingested by chinvex
        ↓
Bootstrap templates written to docs/memory/
        ↓
.claude/settings.json installed with SessionStart hook
        ↓
Agent opens repo → SessionStart → chinvex brief --context <name>
        ↓
┌─ Files populated?
│   YES → Brief assembles STATE + CONSTRAINTS (partial) + DECISIONS (recent)
│         Agent starts work with full context
│
│   NO  → Brief shows "ACTION REQUIRED: run /update-memory"
│         Agent runs skill → reads git log → generates files → commits
│         Next session picks up real context
└─
        ↓
Work happens → commits land
        ↓
Next session or /update-memory run
        ↓
Git log since last coverage anchor processed
Memory files updated respecting update modes
Footer hash advanced to new HEAD
```

---

## Components

### 1. Ingestion & Bootstrap (`chinvex ingest`)

When chinvex ingests a repo:
- Creates `docs/memory/` directory
- Writes bootstrap templates (empty structure matching spec v0.3)
- Installs `.claude/settings.json` with SessionStart hook

Bootstrap templates contain the correct section headers and DO/DON'T comments but no real content.

**Source:** `src/chinvex/memory_templates.py`

### 2. Brief Generation (`chinvex brief --context <name>`)

Called by the SessionStart hook every time an agent opens the repo.

**What it assembles:**
- Full content of `STATE.md` (includes Quick Reference)
- `CONSTRAINTS.md`: Infrastructure + Rules + Hazards sections only
- `DECISIONS.md`: Recent rollup + entries from last 7 days
- Latest digest (watch hits, recent activity)

**Uninitialized detection:** If STATE.md contains bootstrap template content or `Unknown (needs human)` with no real active work, the brief returns an `ACTION REQUIRED` warning directing the agent to run `/update-memory`.

**Source:** `src/chinvex/brief.py`

### 3. Update-Memory Skill (`/update-memory`)

An agent-driven skill (no direct API calls). The agent IS the inference layer.

**What it does:**
1. Reads `STATE.md` footer to get last processed commit hash
2. Runs `git log <hash>..HEAD` to gather new commits
3. Reads current memory files
4. Reads any specs/plans referenced in commits
5. Analyzes changes and generates updates:
   - **STATE.md:** Full rewrite based on current project state
   - **CONSTRAINTS.md:** Merges new bullets, dedupes against existing, supersedes obsolete
   - **DECISIONS.md:** Appends entries to current month, rewrites Recent rollup
6. Writes files and updates coverage anchor (footer hash = new HEAD)
7. Commits changes

**Decision logging threshold:** Only logs changes to interfaces, storage, workflow, security, performance tradeoffs, dependencies, deployment, or user-facing behavior. Ignores refactors, renames, lint fixes, trivial bumps.

**Evidence requirement:** Every DECISIONS entry must include a commit hash or PR link.

**Source:** `skills/update-memory/SKILL.md`

### 4. Coverage Anchor

The footer of `STATE.md` tracks what's been processed:

```markdown
---
Last memory update: 2026-02-05
Commits covered through: da436974e6a35c5b1e39e769d4e67672fb312ea0
```

- Footer hash = HEAD commit at time of last update (inclusive)
- Next run processes `git log <footer_hash>..HEAD`
- This prevents re-processing old commits and ensures nothing is missed

---

## Trigger Modes

| Mode | Trigger | Use case |
|------|---------|----------|
| Post-ingestion | `chinvex ingest` bootstraps templates | First-time setup |
| SessionStart | `chinvex brief` detects empty files → agent runs `/update-memory` | First real population |
| On-demand | Agent runs `/update-memory` during work | Before starting or after significant changes |
| Scheduled | Daily/twice-daily cron (planned) | Hands-off maintenance |

---

## File Locations

| What | Where |
|------|-------|
| Memory files (per repo) | `{repo_root}/docs/memory/STATE.md`, `CONSTRAINTS.md`, `DECISIONS.md` |
| SessionStart hook (per repo) | `{repo_root}/.claude/settings.json` |
| Bootstrap templates | `src/chinvex/memory_templates.py` |
| Brief generation | `src/chinvex/brief.py` |
| Update-memory skill | `skills/update-memory/SKILL.md` |
| Spec | `docs/PROJECT_MEMORY_SPEC.md` |

---

## Key Design Decisions

- **Agent-driven inference, not API calls.** The skill runs inside an active Claude Code session. The agent does the reasoning about what changed and what matters. No Anthropic SDK dependency in chinvex.
- **Mechanical + inference separation.** Git log extraction, file I/O, and update mode enforcement are code. Interpretation of what changed is the agent's job.
- **Brief reads, skill writes.** `chinvex brief` never modifies memory files. `/update-memory` is the only write path.
- **Bootstrap → detect → prompt pattern.** Empty files aren't an error state — they're a trigger for the agent to populate them on first session.
- **Coverage anchor chains.** Each run picks up exactly where the last left off. No gaps, no re-processing.
